# NOTES TO SELF:
# There are a bunch of examples where we artificially miss frames or split up
# what should be a single track because a detection is deemed to be "too far
# away" from the previous one. Turning down the jaccard index could fix this,
# but it also ends up adding a bunch of false positives that I think we want to
# avoid.
# Maybe want to experiment with the other params as well? Missed frames? Minimum length?
# Maybe add some new params?
#  - min resolution
#  - filter out blurry results?
#  - filter out jumpy results?
#  - I think you can adjust the confidence level of the face detector?
#  - Maybe try some different face detectors?
# TODO: Should we add some kind of safeguard if the stabilized bbox drifts too
# far from the original bbox? And just throw it out in that case?

import numpy as np
import cv2
import math
import sys
import os
from itertools import cycle
import argparse
import copy
import time
import pprint
from string import Template
import webbrowser


# Params for algorithm
parser = argparse.ArgumentParser()
# Input/output params
parser.add_argument("--input_directory", required=True, help="Directory to look for input")
parser.add_argument("--output_directory", required=True, help="Directory to place output")
parser.add_argument("--visualization_directory", default="", help="Directory to place visualization output; if empty string, will skip generating this output")
parser.add_argument("--max_consecutive_errors", type=int, default=10, help="This many consecutive errors will halt processing")
parser.add_argument("--update_frequency", type=float, default=15.0, help="Update frequency in seconds")
# Params for the Haar Cascade Classifier
parser.add_argument("--opencv_data_dir", default="/home/sandro/opencv-3.1.0/opencv-3.1.0/data/", help="Directory from which to load classifier config file")
parser.add_argument("--classifier_config_file", default="haarcascades/haarcascade_frontalface_alt2.xml", help="Classifier config file")
parser.add_argument("--classifier_scale_factor", type=float, default=1.1, help="cc.detectMultiScale(scaleFactor)")
parser.add_argument("--classifier_min_neighbors", type=int, default=4, help="cc.detectMultiScale(minNeighbors)")
parser.add_argument("--classifier_min_size", type=int, default=32, help="cc.detectMultiScale(minSize)")
parser.add_argument("--classifier_max_size_factor", type=float, default=1.0, help="Multiplier on image side length to determine cc.detectMultiScale(maxSize)")
parser.add_argument("--classifier_flags", type=int, default=cv2.CASCADE_DO_CANNY_PRUNING, help="cc.detectMultiScale(flags)")
# Params for determining how to stitch frames together
parser.add_argument("--min_jaccard", type=float, default=0.60, help="Minimum jaccard index between adjacent bounding boxes in a track")
parser.add_argument("--max_skip", type=int, default=6, help="Maximium number of consecutive missing detections in a track")
parser.add_argument("--min_frame_count", type=int, default=20, help="Minimum number of consecutive frames required to form a track")
parser.add_argument("--min_total_detections", type=int, default=10, help="Minimum number of total detections in a track")
# Params for determining how to size the bounding box around the face
parser.add_argument("--target_width", type=int, default=256, help="Width of the cropped face video")
parser.add_argument("--target_height", type=int, default=256, help="Height of the cropped face video")
parser.add_argument("--bounding_box_scaling_factor", type=float, default=1.0, help="Amount to scale the haar cascade bounding box before cropping")
# Params for determining how statistics are collected
parser.add_argument("--hst_jaccard_bin_size", type=float, default=0.01, help="Size of a bin in the jaccard index histogram")
parser.add_argument("--hst_time_bin_size", type=float, default=0.1, help="Size of a bin in the time histogram")
# Params for stabilization - finding features
parser.add_argument("--feature_track_max_corners", type=int, default=1000, help="cv2.goodFeaturesToTrack(maxCorners)")
parser.add_argument("--feature_track_quality_level", type=float, default=0.01, help="cv2.goodFeaturesToTrack(qualityLevel)")
parser.add_argument("--feature_track_min_distance", type=int, default=8, help="cv2.goodFeaturesToTrack(minDistance)")
parser.add_argument("--feature_track_block_size", type=int, default=10, help="cv2.goodFeaturesToTrack(blockSize)") # Originally 19
# Params for stabilization - sparse optical flow
parser.add_argument("--optical_flow_win_size", type=int, default=19, help="Used to determine cv2.calcOpticalFlowPyrLK(winSize)")
parser.add_argument("--optical_flow_max_level", type=int, default=2, help="cv2.calcOpticalFlowPyrLK(maxLevel)")
# Yeah, yeah, I know, "eval is evil". Well suck it, cause I don't care.
parser.add_argument("--optical_flow_criteria", type=eval, default=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03), help="cv2.calcOpticalFlowPyrLK(criteria)")
parser.add_argument("--resize_interpolation_method", type=eval, default=cv2.INTER_LINEAR, help="cv2.resize(interpolation)")


# Global fourcc
fourcc = 0x20

# Global colour list for visualization
colours = [
    # in BGR space
    np.array([0,0,255]), # red
    np.array([0,255,0]), # green
    np.array([255,0,0]), # blue
    np.array([0,255,255]), # yellow
    np.array([255,0,255]), # magenta
    np.array([255,255,0]), # cyan
]


class Detection:
    def __repr__(self):
        return "(%d: <(%d, %d), (%d, %d)>)" % (
            self.frame_number,
            self.x1,
            self.y1,
            self.x2,
            self.y2)
    def __init__(self, rect, frame_number, interpolated=False):
        # rect is a tuple (x1, y1, width, height)
        self.x1 = rect[0]
        self.y1 = rect[1]
        self.x2 = rect[0] + rect[2]
        self.y2 = rect[1] + rect[3]
        self.frame_number = frame_number
        self.interpolated = interpolated
        self.too_big = False
    @property
    def height(self):
        return self.y2 - self.y1
    @property
    def width(self):
        return self.x2 - self.x1
    def as_vec(self):
        return np.array([self.x1, self.y1, self.width, self.height])


# Basically just a struct binding together a bunch of stats
class Stats:
    def __init__(self):
        # Drop counters (in priority order, higher to lower)
        self.cnt_drop_because_low_frame_count = 0
        self.cnt_drop_because_low_total_detections = 0
        self.cnt_drop_because_expanded_bb_too_big = 0
        self.cnt_drop_because_optical_flow_bb_too_big = 0
        self.cnt_drop_because_no_feature_points = 0
        self.cnt_drop_because_failed_optical_flow = 0
        self.cnt_drop_because_no_rigid_transform = 0
        self.cnt_drop_because_stabilized_bb_too_big = 0

        # Truncate counters (in priority order, higher to lower)
        self.cnt_truncate_because_optical_flow_bb_too_big = 0
        self.cnt_truncate_because_no_feature_points = 0
        self.cnt_truncate_because_failed_optical_flow = 0
        self.cnt_truncate_because_no_rigid_transform = 0
        self.cnt_truncate_because_stabilized_bb_too_big = 0

        # Other counters
        #   Number of initial face detections
        self.cnt_total_detections = 0
        #   Number of those detections actually used
        self.cnt_detections_kept = 0
        #   Number of face detections written to file (i.e the number above plus
        #   the number of interpolated detections).
        self.cnt_detections_written = 0
        #   Number of tracks initially (before any filtering)
        self.cnt_initial_tracks = 0
        #   Number of tracks actually written to file
        self.cnt_final_tracks = 0
        #   Number of files where there were erros
        self.cnt_errors = 0
        #   Total number of input files
        self.cnt_input_files = 0
        #   Total number of processed files so far
        self.cnt_processed_files = 0
        #   Total number of tracks with at least one interpolated frame
        self.cnt_tracks_with_interpolation = 0

        # Histograms
        self.hst_skip_raw = {}
        self.hst_skip_used = {}
        self.hst_jaccard_raw = {}
        self.hst_jaccard_used = {}
        self.hst_track_len_raw = {}
        self.hst_track_len_used = {}
        self.hst_num_feature_points = {}
        self.hst_num_tracked_feature_points = {}
        self.hst_time = {}
        self.hst_frame_height_raw = {}
        self.hst_frame_width_raw = {}
        self.hst_frame_height_used = {}
        self.hst_frame_width_used = {}
        self.hst_num_crops_per_video = {}

        self.start_time = 0
        self.start_time_str = ""

# Main function to process a gif
def process(f, args, stats):
    start = time.time()

    # Get the full path to input file
    path = os.path.join(args.input_directory, f)

    # First, just get face detections and initial face tracks
    cap = cv2.VideoCapture(path)
    (initial_tracks, frame_size) = get_initial_tracks(cap, args, stats)

    # Then discard those that don't meet length or other requirements
    (valid_tracks, untracked_detections) = discard_invalid_tracks(
        initial_tracks, args, stats)

    # Now interpolate any missing bounding boxes
    interpolated_tracks = interpolate_missing_frames(valid_tracks, args, stats)

    # Now we want to adjust all of the rectangles so that they have the correct
    # aspect ratio. We do this by expanding the bounding boxes, never
    # contracting (since we don't want to throw away anything that was inside
    # the original bounding box). This means that some bounding boxes may get so
    # large that they extend outside the original image. We will discard these
    # tracks.
    (expanded_tracks, oversize_tracks) = expand_bounding_boxes(
        interpolated_tracks, frame_size, args, stats)

    # Now we will perform stabilization. The new stabilized track will have
    # bounding boxes in slightly different locations from the original track. It
    # is possible that the new bounding boxes may go outside of the original
    # image. If this happens, we will discard the track. Stabilization needs to
    # see the frames again, so we reread them from file. TODO: would it be
    # better to keep them all in memory? Probably doesn't make too much of a
    # difference since runtime is not dominated by file access.
    cap = cv2.VideoCapture(path)
    stabilized_tracks = stabilize_tracks(cap, expanded_tracks, frame_size, args, stats)

    # Now crop out the faces
    cap = cv2.VideoCapture(path)
    (filename, extension) = os.path.splitext(f)
    crop_base = os.path.join(args.output_directory, filename)
    crop_faces(cap, crop_base, stabilized_tracks, args, stats)

    # Generate a debug video, if required
    if args.visualization_directory:
        cap = cv2.VideoCapture(path)
        viz_file = os.path.join(args.visualization_directory, f)
        generate_visualization(cap, viz_file, stabilized_tracks, expanded_tracks,
                               oversize_tracks, untracked_detections, frame_size, args, stats)

    end = time.time()
    duration = end - start
    time_bin = round(duration / args.hst_time_bin_size) * args.hst_time_bin_size
    inc(stats.hst_time, time_bin)


# Stages of processing a gif


# Arguments:
#  - cap, opencv reader
#  - args, command line params
#  - stats, object to store statistics
# Returns:
#  - initial_tracks, a list of lists of Detection objects
#  - frame_size, the (width, height) of the frames in f
def get_initial_tracks(cap, args, stats):
    # Initialize the classifier
    config = os.path.join(args.opencv_data_dir, args.classifier_config_file)
    cc = cv2.CascadeClassifier(config)

    tracks = []
    frame_number = 0
    frame_size = None
    # Iterate over the frames
    while(cap.isOpened()):
        ret, im = cap.read()
        if not ret:
            break
        if not frame_size:
            frame_size = (im.shape[1], im.shape[0])

        # Get face detections on this frame
        side = math.sqrt(im.shape[0] * im.shape[1])
        minlen = args.classifier_min_size
        maxlen = int(side * args.classifier_max_size_factor)
        features = cc.detectMultiScale(
            im, args.classifier_scale_factor, args.classifier_min_neighbors,
            args.classifier_flags, (minlen, minlen), (maxlen, maxlen))
        current_detections = [Detection(rect, frame_number) for rect in features]
        current_detections = set(current_detections)
        stats.cnt_total_detections += len(current_detections)

        # Score how much each detection overlaps with previous tracks
        scored_matches = []
        for current in current_detections:
            for track in tracks:
                previous = track[-1]
                j = jaccard_index(current, previous)
                skip = frame_number - previous.frame_number - 1
                if j > 0 and skip <= args.max_skip:
                    jaccard_bin = round(j / args.hst_jaccard_bin_size) * args.hst_jaccard_bin_size
                    inc(stats.hst_jaccard_raw, jaccard_bin)
                if j >= args.min_jaccard:
                    inc(stats.hst_skip_raw, skip)
                if j >= args.min_jaccard and skip <= args.max_skip:
                    scored_matches.append( (j, (current, track, skip)) )
        scored_matches.sort(reverse=True, key=lambda x: x[0])

        # Try to match each detection with the best track
        for (j, (current, track, skip)) in scored_matches:
            if current not in current_detections:
                # We already matched this with another track
                continue
            track.append(current)
            current_detections.remove(current)
            # Update histograms
            inc(stats.hst_skip_used, skip)
            jaccard_bin = round(j / args.hst_jaccard_bin_size) * args.hst_jaccard_bin_size
            inc(stats.hst_jaccard_used, jaccard_bin)

        # Everything that wasn't paired becomes a new track
        for current in current_detections:
            tracks.append([current])

        frame_number += 1

    # Update some stats
    stats.cnt_initial_tracks += len(tracks)
    for t in tracks:
        inc(stats.hst_track_len_raw, len(t))
        for d in t:
            inc(stats.hst_frame_height_raw, d.height)
            inc(stats.hst_frame_width_raw, d.width)

    assert(frame_size is not None)
    return (tracks, frame_size)


# Arguments:
#  - tracks, list of lists of Detection objects
#  - args, command line params
#  - stats, object to store statistics
# Returns:
#  - valid_tracks, a list of lists of Detection objects
#  - untracked_detections, a list of Detection objects
def discard_invalid_tracks(tracks, args, stats):
    valid_tracks = []
    untracked_detections = []
    for track in tracks:
        # Drop track if overall frame count is too low
        frame_count = track[-1].frame_number - track[0].frame_number + 1
        if frame_count < args.min_frame_count:
            stats.cnt_drop_because_low_frame_count += 1
            untracked_detections.extend(copy.deepcopy(track))
            continue

        # Drop track if detection count is too low
        num_detections = len(track)
        if num_detections < args.min_total_detections:
            stats.cnt_drop_because_low_total_detections += 1
            untracked_detections.extend(copy.deepcopy(track))
            continue

        # Otherwise keep this track
        valid_tracks.append(copy.deepcopy(track))
    return (valid_tracks, untracked_detections)


# Arguments:
#  - tracks, list of lists of Detection objects
#  - args, command line params
#  - stats, object to store statistics
# Returns:
#  - interpolated_tracks, a list of lists of Detection objects
def interpolate_missing_frames(tracks, args, stats):
    output_tracks = []
    for track in tracks:
        interpolated_track = []
        # Iterate over each pair of adjacent frames
        for i in range(len(track)-1):
            d1 = track[i]
            d2 = track[i+1]
            interpolated_track.append(copy.copy(d1))
            frame_delta = d2.frame_number - d1.frame_number
            rect_delta = d2.as_vec() - d1.as_vec()
            # Iterate over any missing frames between this pair
            for frm in range(d1.frame_number+1,d2.frame_number):
                fraction = float(frm - d1.frame_number)/frame_delta
                new_rect = d1.as_vec() + fraction*rect_delta
                new_rect = np.round(new_rect).astype('int32')
                new_detection = Detection(new_rect, frm, interpolated=True)
                interpolated_track.append(new_detection)
        interpolated_track.append(copy.copy(track[-1]))
        output_tracks.append(interpolated_track)
    return output_tracks


# Arguments:
#  - tracks, list of lists of Detection objects
#  - frame_size, as (width, height)
#  - args, command line params
#  - stats, object to store statistics
# Returns:
#  - expanded_tracks, a list of lists of Detection objects
#  - oversize_tracks, a list of lists of Detection objects
def expand_bounding_boxes(tracks, frame_size, args, stats):
    (width, height) = frame_size
    expanded_tracks = []
    oversize_tracks = []
    for track in tracks:
        drop_track = False
        new_track = []
        for d in track:
            # Expand the box along one axis so the aspect ratio is correct
            required_aspect_ratio = float(args.target_width)/float(args.target_height)
            actual_aspect_ratio = float(d.width)/float(d.height)
            scaling = required_aspect_ratio / actual_aspect_ratio
            x_scaling = scaling if scaling > 1.0 else 1.0
            y_scaling = 1.0/scaling if scaling <= 1.0 else 1.0
            # Find the centre of the box, and expand outward from there
            centre_x = (d.x1 + d.x2)/2.0
            centre_y = (d.y1 + d.y2)/2.0
            assert(centre_x >= 0 and centre_x < width)
            assert(centre_y >= 0 and centre_y < height)
            # We expand to get the right aspect ratio, and on top of that, we
            # add an additional command-line-specified scaling factor
            f = args.bounding_box_scaling_factor
            x1 = int(round(x_scaling * f * (d.x1 - centre_x) + centre_x))
            y1 = int(round(y_scaling * f * (d.y1 - centre_y) + centre_y))
            x2 = int(round(x_scaling * f * (d.x2 - centre_x) + centre_x))
            y2 = int(round(y_scaling * f * (d.y2 - centre_y) + centre_y))
            # If we find that the box has now expanded past the edges of the
            # image, we throw away this entire track :(
            if x1 < 0 or y1 < 0 or x2 >= width or y2 >= height:
                drop_track = True
            new_d = Detection((x1, y1, x2-x1, y2-y1), d.frame_number,
                              interpolated=d.interpolated)
            new_track.append(new_d)
        if drop_track:
            stats.cnt_drop_because_expanded_bb_too_big += 1
            oversize_tracks.append(new_track)
        else:
            expanded_tracks.append(new_track)
    return (expanded_tracks, oversize_tracks)


# Arguments:
#  - cap, opencv reader
#  - tracks, list of lists of Detection objects
#  - frame_size, as (width, height)
#  - args, command line params
#  - stats, object to store statistics
# Returns:
#  - stabilized_tracks, a list of lists of Detection objects
def stabilize_tracks(cap, tracks, frame_size, args, stats):
    (width, height) = frame_size
    # First build parameter dictionaries for finding features and doing sparse
    # optical flow.
    feature_params = dict( maxCorners = args.feature_track_max_corners,
                           qualityLevel = args.feature_track_quality_level,
                           minDistance = args.feature_track_min_distance,
                           blockSize = args.feature_track_block_size )
    win_size = args.optical_flow_win_size
    lk_params = dict( winSize  = (win_size, win_size),
                      maxLevel = args.optical_flow_max_level,
                      criteria = args.optical_flow_criteria)

    # For each of the original tracks, we're going to create a map of
    # frame_number -> detection to make lookups easier for ourselves. We will
    # use this lookup to tell us when we need to run stabilization. Since we
    # don't run stabilization on the first detection in a track (we just copy it
    # verbatim), we remove that first detection.
    lookup = [{ d.frame_number : d for d in track[1:] } for track in tracks]

    # Our stabilized tracks start with exactly the same bounding box as our
    # original tracks, so let's copy over the first bounding boxes now.
    stable_tracks = [[copy.copy(track[0])] for track in tracks]

    # Let's also build a structure for ourselves to record which tracks have
    # gone offscreen and need to be thrown away
    onscreen = [True for _ in tracks]

    # Now we need to step through the video again
    frame_number = 0
    prev_frame = None
    while(cap.isOpened()):
        ret, im = cap.read()
        if not ret:
            break
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

        for track_id in range(len(tracks)):
            # First check if this track has gone offscreen, and if so bail
            if not onscreen[track_id]:
                continue
            if frame_number in lookup[track_id]:
                assert(prev_frame is not None)

                # Grab the previous (stabilized!) detection for this track
                d = stable_tracks[track_id][-1]
                # Get the current (unstabilized!) detection for this track
                next_d = lookup[track_id][frame_number]
                # Since the sizes may be different, resize next_d so that it
                # matches d's size.
                next_d = same_size_crop(next_d, d)
                # next_d may now have gone offscreen. Check for that.
                if (next_d.x1 < 0 or next_d.y1 < 0 or
                    next_d.x2 >= width or next_d.y2 >= height):
                    if len(stable_tracks[track_id]) >= args.min_frame_count:
                        stats.cnt_truncate_because_optical_flow_bb_too_big += 1
                    else:
                        stats.cnt_drop_because_optical_flow_bb_too_big += 1
                    onscreen[track_id] = False
                    continue

                # Now crop out the relevant regions from the previous and
                # current frame, so that we can run optical flow.
                prev_crop = prev_frame[d.y1:d.y2+1,d.x1:d.x2+1]
                crop = im[next_d.y1:next_d.y2+1,next_d.x1:next_d.x2+1]
                # Find features
                pnts = cv2.goodFeaturesToTrack(prev_crop, **feature_params)
                inc(stats.hst_num_feature_points, len(pnts))
                if len(pnts) == 0:
                    # Oops, failed to find tracking points. Record this failure
                    # and move on.
                    if len(stable_tracks[track_id]) >= args.min_frame_count:
                        stats.cnt_truncate_because_no_feature_points += 1
                    else:
                        stats.cnt_drop_because_no_feature_points += 1
                    onscreen[track_id] = False
                    continue

                # Run optical flow
                (pnts2, status, _) = cv2.calcOpticalFlowPyrLK(
                    prev_crop, crop, pnts, None, **lk_params)
                # Filter out the points where optical flow failed
                pnts = [p for (p, s) in zip(pnts, status) if s]
                pnts2 = [p for (p, s) in zip(pnts2, status) if s]
                inc(stats.hst_num_tracked_feature_points, len(pnts))
                if len(pnts) == 0 or len(pnts2) == 0:
                    # Oops, optical flow calculation failed. Record this failure
                    # and move on.
                    if len(stable_tracks[track_id]) >= args.min_frame_count:
                        stats.cnt_truncate_because_failed_optical_flow += 1
                    else:
                        stats.cnt_drop_because_failed_optical_flow += 1
                    onscreen[track_id] = False
                    continue

                # Estimate transform between the two frames
                pnts += np.array([d.x1,d.y1])
                pnts = np.swapaxes(pnts, 0, 1)
                pnts2 += np.array([next_d.x1,next_d.y1])
                pnts2 = np.swapaxes(pnts2, 0, 1)
                transformation = cv2.estimateRigidTransform(pnts,pnts2,fullAffine=False)
                if transformation is None:
                    # Oops, can't estimate transform. Record this failure and
                    # move on.
                    if len(stable_tracks[track_id]) >= args.min_frame_count:
                        stats.cnt_truncate_because_no_rigid_transform += 1
                    else:
                        stats.cnt_drop_because_no_rigid_transform += 1
                    onscreen[track_id] = False
                    continue

                # Apply the transform to calculate the next stabilized bounding
                # box.
                m = transformation[:,:2]
                b = transformation[:,2:3]
                new_d = copy.copy(next_d)
                # Kill the rotation, extract just the scale and offset
                x1y1 = m.dot(np.array([[d.x1],[d.y1]])) + b
                x2y2 = m.dot(np.array([[d.x2],[d.y2]])) + b
                centre = (x1y1 + x2y2) / 2.0
                diag = (x1y1 - x2y2)
                diag_len = math.sqrt(diag[0,0]**2 + diag[1,0]**2)
                old_diag_len = math.sqrt((d.x2-d.x1)**2 + (d.y2-d.y1)**2)
                scale = diag_len / old_diag_len
                aspect_ratio = float(args.target_width)/float(args.target_height)
                new_d.y1 = int(round(centre[1,0] - scale*(d.y2-d.y1)/2.0))
                new_d.y2 = int(round(centre[1,0] + scale*(d.y2-d.y1)/2.0))
                new_height = new_d.y2 - new_d.y1
                new_d.x1 = int(round(centre[0,0] - aspect_ratio*new_height/2.0))
                new_d.x2 = int(round(centre[0,0] + aspect_ratio*new_height/2.0))

                # The stabilized bounding box may now have gone offscreen, so
                # check that.
                if (new_d.x1 < 0 or new_d.y1 < 0 or
                    new_d.x2 >= width or new_d.y2 >= height):
                    if len(stable_tracks[track_id]) >= args.min_frame_count:
                        stats.cnt_truncate_because_stabilized_bb_too_big += 1
                    else:
                        stats.cnt_drop_because_stabilized_bb_too_big += 1
                    onscreen[track_id] = False
                    continue

                # Yay we made it through everything! We can add this stabilized
                # frame to the stabilized tracks.
                stable_tracks[track_id].append(new_d)
        # End of loop over track ids
        prev_frame = im
        frame_number += 1
    # End of loop over frames in video

    # We are going to output a list of stabilized tracks. We are going to keep
    # it in alignment with the input list, so if there was ever a track that
    # went offscreen during stabilization and we don't have enough frames to
    # justify keeping it, we will simply return None at that index.
    return [track if len(track) >= args.min_frame_count else None for track in stable_tracks]


# Arguments:
#  - cap, opencv reader
#  - crop_base, base filename for writing out crops
#  - stabilized_tracks, a list of lists of Detection objects which describes how
#    to construct the face crops.
#  - args, command line params
#  - stats, object to store statistics
def crop_faces(cap, crop_base, stabilized_tracks, args, stats):
    # Remove the empty tracks
    stabilized_tracks = [s for s in stabilized_tracks if s is not None]

    # Do some counter stuff
    stats.cnt_detections_kept += sum([1 for t in stabilized_tracks
                                        for d in t if not d.interpolated])
    stats.cnt_detections_written += sum([len(t) for t in stabilized_tracks])
    stats.cnt_final_tracks += len(stabilized_tracks)
    for t in stabilized_tracks:
        inc(stats.hst_track_len_used, len(t))
        for d in t:
            inc(stats.hst_frame_height_used, d.height)
            inc(stats.hst_frame_width_used, d.width)
    inc(stats.hst_num_crops_per_video, len(stabilized_tracks))
    stats.cnt_tracks_with_interpolation += sum(
        [1 for t in stabilized_tracks if any([d.interpolated for d in t])])

    # Get min width and height for each track
    widths = [min([d.width for d in t]) for t in stabilized_tracks]
    heights = [min([d.height for d in t]) for t in stabilized_tracks]

    # Open a writer for each track
    writers = [cv2.VideoWriter(crop_base + ("_%d_%d-by-%d.mp4" % (i, widths[i], heights[i])),
                               fourcc, 25.0, (args.target_width, args.target_height))
               for i in range(len(stabilized_tracks))]

    # Make cursors for each track
    cursors = [0 for _ in range(len(stabilized_tracks))]

    frame_number = 0
    while(cap.isOpened()):
        ret, im = cap.read()
        if not ret:
            break

        # Create the crop frames
        for i in range(len(stabilized_tracks)):
            cur = cursors[i]
            t = stabilized_tracks[i]
            w = writers[i]
            if cur < len(t) and t[cur].frame_number == frame_number:
                w.write(get_crop(im, t[cur], args))
                cursors[i] += 1
        frame_number += 1

    # close the writers
    for w in writers:
        w.release()


# Arguments:
#  - cap, opencv reader
#  - viz_file, filename to write output
#  - stabilized_tracks, expanded_tracks, oversize_tracks, untracked_detections,
#    the stuff to annotate onto the video file.
#  - frame_size, as (width, height)
#  - args, command line params
#  - stats, object to store statistics
def generate_visualization(cap, viz_file, stabilized_tracks, expanded_tracks,
                           oversize_tracks, untracked_detections, frame_size,
                           args, stats):
    # For visualization purposes, pair up each stabilized track with its
    # corresponding unstabilized track, and throw away the pairs where
    # stabilization failed.
    pairs = [(e, s) for (e, s) in zip(expanded_tracks, stabilized_tracks)
             if s is not None]
    # Now assign each pair a colour.
    coloured_tracks = [(e, s, c) for ((e, s), c) in zip(pairs, cycle(colours))]

    # Create a lookup of untracked detections per frame
    detections_per_frame = {}
    for d in untracked_detections:
        detections_per_frame.setdefault(d.frame_number, []).append(d)

    # Create a list of all the detections we threw away
    discarded_tracks = (oversize_tracks +
                        [e for (e, s) in zip(expanded_tracks, stabilized_tracks)
                         if s is None])
    discarded_detections_per_frame = {}
    for t in discarded_tracks:
        for d in t:
            discarded_detections_per_frame.setdefault(d.frame_number, []).append(d)

    assert(frame_size is not None)
    # Open a writer for the debug video
    out = cv2.VideoWriter(viz_file, fourcc, 25.0, frame_size)
    # Make cursors for each track - it's kinda silly that we use cursors for
    # these detections but a dictionary for the other detections ... oh well.
    cursors = [0 for _ in range(len(coloured_tracks))]

    frame_number = 0
    while(cap.isOpened()):
        ret, im = cap.read()
        if not ret:
            break

        # Draw the untracked detections
        detections = detections_per_frame.get(frame_number, [])
        for d in detections:
            # Make a white rectangle
            cv2.rectangle(im, (d.x1, d.y1), (d.x2, d.y2), (0, 0, 0), 4)
            cv2.rectangle(im, (d.x1, d.y1), (d.x2, d.y2), (255, 255, 255), 2)

        # Draw the discarded detections
        discarded = discarded_detections_per_frame.get(frame_number, [])
        for d in discarded:
            # Make a white rectangle
            cv2.rectangle(im, (d.x1, d.y1), (d.x2, d.y2), (0, 0, 0), 4)
            cv2.rectangle(im, (d.x1, d.y1), (d.x2, d.y2), (255, 255, 255), 2)
            # Draw a big 'X' in the rectangle
            cv2.line(im, (d.x1, d.y1), (d.x2, d.y2), (255, 255, 255), 2)
            cv2.line(im, (d.x1, d.y2), (d.x2, d.y1), (255, 255, 255), 2)

        # Draw the tracked detections
        for i in range(len(coloured_tracks)):
            cur = cursors[i]
            (expanded, stabilized, colour) = coloured_tracks[i]
            if cur < len(expanded) and expanded[cur].frame_number == frame_number:
                # Draw the original rectangle
                d = expanded[cur]
                cv2.rectangle(im, (d.x1, d.y1), (d.x2, d.y2), colour, 1)
                if cur < len(stabilized):
                    # Draw the stabilized rectangle
                    assert(stabilized[cur].frame_number == frame_number)
                    d = stabilized[cur]
                    cv2.rectangle(im, (d.x1, d.y1), (d.x2, d.y2), (0, 0, 0), 4)
                    cv2.rectangle(im, (d.x1, d.y1), (d.x2, d.y2), colour, 2)
                else:
                    # Otherwise, we lost this track partway through during
                    # stabilization, so let's notate that with an 'X'
                    cv2.line(im, (d.x1, d.y1), (d.x2, d.y2), colour, 1)
                    cv2.line(im, (d.x1, d.y2), (d.x2, d.y1), colour, 1)
                cursors[i] += 1

        frame_number += 1
        out.write(im)
    out.release()


# Write out the text and html stats
def write_stats(args, stats):
    # Write the .txt stats
    with open(os.path.join(args.output_directory, "stats.txt"), 'w') as f:
        f.write(pprint.pformat(stats.__dict__))

    # Write the .html stats

    # Read the template
    with open("graph_template.html", 'r') as f:
        template = Template(f.read())

    # Compute the values to fill in

    # Basic progress
    files_processed = stats.cnt_processed_files + stats.cnt_errors
    input_files = stats.cnt_input_files
    progress_percent = 0.0
    if stats.cnt_input_files > 0:
        progress_percent = "%.2f" % (100.0 * float(files_processed) /
                                     float(stats.cnt_input_files))
    total_faces = stats.cnt_final_tracks
    faces_per_gif_value = 0.0
    if files_processed > 0:
        faces_per_gif_value = (float(stats.cnt_final_tracks) /
                               float(files_processed))
    faces_per_gif = "%.2f" % faces_per_gif_value
    faces_at_completion = int(round(faces_per_gif_value * stats.cnt_input_files))
    start_time = stats.start_time_str
    time_per_gif = "0"
    if files_processed > 0:
        time_per_gif = "%.2f" % ((time.time() - stats.start_time) / files_processed)
    files_done = files_processed
    files_left = stats.cnt_input_files - files_processed
    time_remaining = "calculating..."
    if files_processed > 0:
        time_remaining_value = (time.time() - stats.start_time) * files_left / files_processed
        days = int(time_remaining_value / (60*60*24))
        hours = int(time_remaining_value / (60*60)) - days*24
        minutes = int(time_remaining_value / 60) - days*24*60 - hours*60
        seconds = int(time_remaining_value) - days*24*60*60 - hours*60*60 - minutes*60
        time_remaining = "%d day(s) %d hour(s) %d minute(s) %d second(s)" % (
            days, hours, minutes, seconds)
    success = stats.cnt_processed_files
    error = stats.cnt_errors

    # chart-time-hist
    chart_time_hist_buckets = [x/2.0 for x in range(0, 20, 1)] + [10, 15, 20, 25, 30, 45, 60]
    chart_time_hist_labels = ','.join(['"%s"' % v for v in chart_time_hist_buckets])
    chart_time_hist_colours = ''.join(["'rgba(0, 204, 255, 1.0)',\n"]*len(chart_time_hist_labels))
    chart_time_hist_data_values = [
        sum([v for (k, v) in stats.hst_time.iteritems() if lower <= k < upper])
        for (lower, upper) in zip(chart_time_hist_buckets,
                                  chart_time_hist_buckets[1:] + [float("inf")])]
    chart_time_hist_data = ','.join([str(v) for v in chart_time_hist_data_values])

    # chart-num-faces-hist
    chart_num_faces_hist_buckets = range(21)
    chart_num_faces_hist_labels = ','.join(['"%s"' % v for v in chart_num_faces_hist_buckets])
    chart_num_faces_hist_colours = ''.join(["'rgba(0, 204, 255, 1.0)',\n"]*len(chart_num_faces_hist_labels))
    chart_num_faces_hist_data_values = [
        sum([v for (k, v) in stats.hst_num_crops_per_video.iteritems() if lower <= k < upper])
        for (lower, upper) in zip(chart_num_faces_hist_buckets,
                                  chart_num_faces_hist_buckets[1:] + [float("inf")])]
    chart_num_faces_hist_data = ','.join([str(v) for v in chart_num_faces_hist_data_values])

    # chart-detection-breakdown
    interpolated_detections = stats.cnt_detections_written - stats.cnt_detections_kept
    kept_detections = stats.cnt_detections_kept
    dropped_detections = stats.cnt_total_detections - stats.cnt_detections_kept

    # chart-raw-jaccard-hist
    chart_raw_jaccard_hist_buckets = [x/100.0 for x in range(0, 100, 5)]
    chart_raw_jaccard_hist_labels = ','.join(['"%s"' % v for v in chart_raw_jaccard_hist_buckets])
    chart_raw_jaccard_hist_colours = ''.join(["'rgba(0, 204, 255, 1.0)',\n"]*len(chart_raw_jaccard_hist_labels))
    chart_raw_jaccard_hist_data_values = [
        sum([v for (k, v) in stats.hst_jaccard_raw.iteritems() if lower <= k < upper])
        for (lower, upper) in zip(chart_raw_jaccard_hist_buckets,
                                  chart_raw_jaccard_hist_buckets[1:] + [float("inf")])]
    chart_raw_jaccard_hist_data = ','.join([str(v) for v in chart_raw_jaccard_hist_data_values])

    # chart-used-jaccard-hist
    chart_used_jaccard_hist_buckets = [x/100.0 for x in range(0, 100, 5)]
    chart_used_jaccard_hist_labels = ','.join(['"%s"' % v for v in chart_used_jaccard_hist_buckets])
    chart_used_jaccard_hist_colours = ''.join(["'rgba(0, 204, 255, 1.0)',\n"]*len(chart_used_jaccard_hist_labels))
    chart_used_jaccard_hist_data_values = [
        sum([v for (k, v) in stats.hst_jaccard_used.iteritems() if lower <= k < upper])
        for (lower, upper) in zip(chart_used_jaccard_hist_buckets,
                                  chart_used_jaccard_hist_buckets[1:] + [float("inf")])]
    chart_used_jaccard_hist_data = ','.join([str(v) for v in chart_used_jaccard_hist_data_values])
    average_used_jaccard = "0.0"
    total = sum([v for (k, v) in stats.hst_jaccard_used.iteritems()])
    if total > 0:
        average_used_jaccard = "%.2f" % (
            sum([v*k for (k, v) in stats.hst_jaccard_used.iteritems()]) / total)

    # chart-raw-skip-hist
    chart_raw_skip_hist_buckets = range(26)
    chart_raw_skip_hist_labels = ','.join(['"%s"' % v for v in chart_raw_skip_hist_buckets])
    chart_raw_skip_hist_colours = ''.join(["'rgba(0, 204, 255, 1.0)',\n"]*len(chart_raw_skip_hist_labels))
    chart_raw_skip_hist_data_values = [
        sum([v for (k, v) in stats.hst_skip_raw.iteritems() if lower <= k < upper])
        for (lower, upper) in zip(chart_raw_skip_hist_buckets,
                                  chart_raw_skip_hist_buckets[1:] + [float("inf")])]
    chart_raw_skip_hist_data = ','.join([str(v) for v in chart_raw_skip_hist_data_values])

    # chart-raw-skip-hist
    chart_raw_skip_hist_detail_buckets = chart_raw_skip_hist_buckets[1:11]
    chart_raw_skip_hist_detail_labels = ','.join(['"%s"' % v for v in chart_raw_skip_hist_detail_buckets])
    chart_raw_skip_hist_detail_colours = ''.join(["'rgba(0, 204, 255, 1.0)',\n"]*len(chart_raw_skip_hist_detail_labels))
    chart_raw_skip_hist_detail_data_values = chart_raw_skip_hist_data_values[1:11]
    chart_raw_skip_hist_detail_data = ','.join([str(v) for v in chart_raw_skip_hist_detail_data_values])

    # chart-used-skip-hist
    chart_used_skip_hist_buckets = range(26)
    chart_used_skip_hist_labels = ','.join(['"%s"' % v for v in chart_used_skip_hist_buckets])
    chart_used_skip_hist_colours = ''.join(["'rgba(0, 204, 255, 1.0)',\n"]*len(chart_used_skip_hist_labels))
    chart_used_skip_hist_data_values = [
        sum([v for (k, v) in stats.hst_skip_used.iteritems() if lower <= k < upper])
        for (lower, upper) in zip(chart_used_skip_hist_buckets,
                                  chart_used_skip_hist_buckets[1:] + [float("inf")])]
    chart_used_skip_hist_data = ','.join([str(v) for v in chart_used_skip_hist_data_values])
    average_used_skip = "0.0"
    total = sum([v for (k, v) in stats.hst_skip_used.iteritems()])
    if total > 0:
        average_used_skip = "%.6f" % (
            sum([v*k for (k, v) in stats.hst_skip_used.iteritems()]) / float(total))

    # chart-num-fp-hist
    chart_num_fp_hist_buckets = range(0,105,5)
    chart_num_fp_hist_labels = ','.join(['"%s"' % v for v in chart_num_fp_hist_buckets])
    chart_num_fp_hist_colours = ''.join(["'rgba(0, 204, 255, 1.0)',\n"]*len(chart_num_fp_hist_labels))
    chart_num_fp_hist_data_values = [
        sum([v for (k, v) in stats.hst_num_feature_points.iteritems() if lower <= k < upper])
        for (lower, upper) in zip(chart_num_fp_hist_buckets,
                                  chart_num_fp_hist_buckets[1:] + [float("inf")])]
    chart_num_fp_hist_data = ','.join([str(v) for v in chart_num_fp_hist_data_values])
    average_num_fp = "0.0"
    total = sum([v for (k, v) in stats.hst_num_feature_points.iteritems()])
    if total > 0:
        average_num_fp = "%.1f" % (
            sum([v*k for (k, v) in stats.hst_num_feature_points.iteritems()]) / float(total))

    # chart-num-tracked-fp-hist
    chart_num_tracked_fp_hist_buckets = range(0,105,5)
    chart_num_tracked_fp_hist_labels = ','.join(['"%s"' % v for v in chart_num_tracked_fp_hist_buckets])
    chart_num_tracked_fp_hist_colours = ''.join(["'rgba(0, 204, 255, 1.0)',\n"]*len(chart_num_tracked_fp_hist_labels))
    chart_num_tracked_fp_hist_data_values = [
        sum([v for (k, v) in stats.hst_num_tracked_feature_points.iteritems() if lower <= k < upper])
        for (lower, upper) in zip(chart_num_tracked_fp_hist_buckets,
                                  chart_num_tracked_fp_hist_buckets[1:] + [float("inf")])]
    chart_num_tracked_fp_hist_data = ','.join([str(v) for v in chart_num_tracked_fp_hist_data_values])
    average_num_tracked_fp = "0.0"
    total = sum([v for (k, v) in stats.hst_num_tracked_feature_points.iteritems()])
    if total > 0:
        average_num_tracked_fp = "%.1f" % (
            sum([v*k for (k, v) in stats.hst_num_tracked_feature_points.iteritems()]) / float(total))

    # chart-raw-height-hist
    chart_raw_height_hist_buckets = range(20, 270, 10)
    chart_raw_height_hist_labels = ','.join(['"%s"' % v for v in chart_raw_height_hist_buckets])
    chart_raw_height_hist_colours = ''.join(["'rgba(0, 204, 255, 1.0)',\n"]*len(chart_raw_height_hist_labels))
    chart_raw_height_hist_data_values = [
        sum([v for (k, v) in stats.hst_frame_height_raw.iteritems() if lower <= k < upper])
        for (lower, upper) in zip(chart_raw_height_hist_buckets,
                                  chart_raw_height_hist_buckets[1:] + [float("inf")])]
    chart_raw_height_hist_data = ','.join([str(v) for v in chart_raw_height_hist_data_values])
    average_raw_height = "0"
    total = sum([v for (k, v) in stats.hst_frame_height_raw.iteritems()])
    if total > 0:
        average_raw_height = "%d" % (
            sum([v*k for (k, v) in stats.hst_frame_height_raw.iteritems()]) / total)

    # chart-used-height-hist
    chart_used_height_hist_buckets = range(20, 270, 10)
    chart_used_height_hist_labels = ','.join(['"%s"' % v for v in chart_used_height_hist_buckets])
    chart_used_height_hist_colours = ''.join(["'rgba(0, 204, 255, 1.0)',\n"]*len(chart_used_height_hist_labels))
    chart_used_height_hist_data_values = [
        sum([v for (k, v) in stats.hst_frame_height_used.iteritems() if lower <= k < upper])
        for (lower, upper) in zip(chart_used_height_hist_buckets,
                                  chart_used_height_hist_buckets[1:] + [float("inf")])]
    chart_used_height_hist_data = ','.join([str(v) for v in chart_used_height_hist_data_values])
    average_used_height = "0"
    total = sum([v for (k, v) in stats.hst_frame_height_used.iteritems()])
    if total > 0:
        average_used_height = "%d" % (
            sum([v*k for (k, v) in stats.hst_frame_height_used.iteritems()]) / total)

    # chart-track-breakdown
    truncated_tracks = (
        stats.cnt_truncate_because_optical_flow_bb_too_big +
        stats.cnt_truncate_because_no_feature_points +
        stats.cnt_truncate_because_failed_optical_flow +
        stats.cnt_truncate_because_no_rigid_transform +
        stats.cnt_truncate_because_stabilized_bb_too_big)
    dropped_tracks = (
        stats.cnt_drop_because_low_frame_count +
        stats.cnt_drop_because_low_total_detections +
        stats.cnt_drop_because_expanded_bb_too_big +
        stats.cnt_drop_because_optical_flow_bb_too_big +
        stats.cnt_drop_because_no_feature_points +
        stats.cnt_drop_because_failed_optical_flow +
        stats.cnt_drop_because_no_rigid_transform +
        stats.cnt_drop_because_stabilized_bb_too_big)
    kept_tracks = stats.cnt_final_tracks - truncated_tracks

    # chart-drop-breakdown and chart-drop-breakdown-detail
    drop_because_low_frame_count = stats.cnt_drop_because_low_frame_count
    drop_because_low_total_detections = stats.cnt_drop_because_low_total_detections
    drop_because_expanded_bb_too_big = stats.cnt_drop_because_expanded_bb_too_big
    drop_because_optical_flow_bb_too_big = stats.cnt_drop_because_optical_flow_bb_too_big
    drop_because_no_feature_points = stats.cnt_drop_because_no_feature_points
    drop_because_failed_optical_flow = stats.cnt_drop_because_failed_optical_flow
    drop_because_no_rigid_transform = stats.cnt_drop_because_no_rigid_transform
    drop_because_stabilized_bb_too_big = stats.cnt_drop_because_stabilized_bb_too_big

    # chart-truncate-breakdown
    truncate_because_optical_flow_bb_too_big = stats.cnt_truncate_because_optical_flow_bb_too_big
    truncate_because_no_feature_points = stats.cnt_truncate_because_no_feature_points
    truncate_because_failed_optical_flow = stats.cnt_truncate_because_failed_optical_flow
    truncate_because_no_rigid_transform = stats.cnt_truncate_because_no_rigid_transform
    truncate_because_stabilized_bb_too_big = stats.cnt_truncate_because_stabilized_bb_too_big

    # chart-interpolation-breakdown
    tracks_with_interpolation = stats.cnt_tracks_with_interpolation
    tracks_without_interpolation = stats.cnt_final_tracks - tracks_with_interpolation

    # chart-raw-track-len-hist
    chart_raw_track_len_hist_buckets = range(25) + range(25, 50, 5) + range(50, 250, 25)
    chart_raw_track_len_hist_labels = ','.join(['"%s"' % v for v in chart_raw_track_len_hist_buckets])
    chart_raw_track_len_hist_colours = ''.join(["'rgba(0, 204, 255, 1.0)',\n"]*len(chart_raw_track_len_hist_labels))
    chart_raw_track_len_hist_data_values = [
        sum([v for (k, v) in stats.hst_track_len_raw.iteritems() if lower <= k < upper])
        for (lower, upper) in zip(chart_raw_track_len_hist_buckets,
                                  chart_raw_track_len_hist_buckets[1:] + [float("inf")])]
    chart_raw_track_len_hist_data = ','.join([str(v) for v in chart_raw_track_len_hist_data_values])

    # chart-used-track-len-hist
    chart_used_track_len_hist_buckets = range(20,50) + range(50, 250, 25)
    chart_used_track_len_hist_labels = ','.join(['"%s"' % v for v in chart_used_track_len_hist_buckets])
    chart_used_track_len_hist_colours = ''.join(["'rgba(0, 204, 255, 1.0)',\n"]*len(chart_used_track_len_hist_labels))
    chart_used_track_len_hist_data_values = [
        sum([v for (k, v) in stats.hst_track_len_used.iteritems() if lower <= k < upper])
        for (lower, upper) in zip(chart_used_track_len_hist_buckets,
                                  chart_used_track_len_hist_buckets[1:] + [float("inf")])]
    chart_used_track_len_hist_data = ','.join([str(v) for v in chart_used_track_len_hist_data_values])
    total = sum([v for (k, v) in stats.hst_track_len_used.iteritems()])
    average_used_track_len = "0.0"
    if total > 0:
        average_used_track_len = "%.1f" % (
            sum([v*k for (k, v) in stats.hst_track_len_used.iteritems()]) / float(total))

    invokation_parameters = pprint.pformat(args.__dict__)

    # Fill in the template and write it out to file
    graph_html = template.substitute(**locals())
    graph_file = os.path.join(args.output_directory, "stats.html")
    with open(graph_file, 'w') as f:
        f.write(graph_html)

    return graph_file


# Helper functions


# Compute jaccard index between the rectangles a and b (which are both Detection
# objects)
def jaccard_index(a, b):
    intersection_width = min(a.x2, b.x2) - max(a.x1, b.x1)
    intersection_height = min(a.y2, b.y2) - max(a.y1, b.y1)
    if intersection_width <= 0.0 or intersection_height <= 0.0:
        return 0.0

    intersection_area = intersection_width * intersection_height
    total_area = a.height * a.width + b.height * b.width
    union_area = total_area - intersection_area

    return float(intersection_area) / float(union_area)


# Increment the given key in hist dictionary, and create it if necessary.
def inc(hist, key):
    hist[key] = hist.get(key, 0) + 1


# Extract the region specified by Detection d from image im. Resize to the
# common size args.target_width, args.target_height
def get_crop(im, d, args):
    crop = im[d.y1:d.y2+1,d.x1:d.x2+1]
    # Methods are cv2.INTER_CUBIC (slow) and cv2.INTER_LINEAR (fast but worse)
    res = cv2.resize(crop,(args.target_width,args.target_height), interpolation=args.resize_interpolation_method)
    return res

# Arguments:
#  - inpt, detection to resize
#  - target, detection from which to copy size
# Returns:
#  - resized detection
def same_size_crop(inpt, target):
    # Resize from the centre. First find the centre of both detections.
    inpt_x = int(round((inpt.x1 + inpt.x2) / 2.0))
    inpt_y = int(round((inpt.y1 + inpt.y2) / 2.0))
    target_x = int(round((target.x1 + target.x2) / 2.0))
    target_y = int(round((target.y1 + target.y2) / 2.0))
    # Copy inpt and then resize it
    new = copy.copy(inpt)
    new.x1 = target.x1 - target_x + inpt_x
    new.y1 = target.y1 - target_y + inpt_y
    new.x2 = target.x2 - target_x + inpt_x
    new.y2 = target.y2 - target_y + inpt_y
    return new


def main():
    args = parser.parse_args()
    stats = Stats()
    stats.start_time_str = time.strftime("%Y-%m-%d %H:%M:%S")
    stats.start_time = time.time()

    # Make output directory/directories
    if not os.path.exists(args.output_directory):
        os.makedirs(args.output_directory)
    if args.visualization_directory and not os.path.exists(args.visualization_directory):
        os.makedirs(args.visualization_directory)

    # Write out the args that we were called with
    with open(os.path.join(args.output_directory, "params.txt"), 'w') as f:
        f.write(pprint.pformat(args.__dict__))

    # Read input directory
    files = os.listdir(args.input_directory)
    files = [f for f in files if os.path.splitext(f)[1] == ".mp4"]
    stats.cnt_input_files = len(files)
    consecutive_errors = 0

    # Write the first update
    html_location = write_stats(args, stats)
    last_update = time.time()
    print "Live status available at:", html_location
    webbrowser.open(html_location)

    for f in files:
        try:
            process(f, args, stats)
            print "processed", f
            consecutive_errors = 0
            stats.cnt_processed_files += 1
        except Exception as e:
            print "error on", f, ":", e.message
            consecutive_errors += 1
            stats.cnt_errors += 1
            if consecutive_errors >= args.max_consecutive_errors:
                print "aborting because too many errors"
                break
        # Write out stats at regular intervals
        if time.time() - last_update >= args.update_frequency:
            write_stats(args, stats)
            last_update = time.time()

    # Do one last update before we exit
    write_stats(args, stats)


if __name__ == "__main__":
    main()
