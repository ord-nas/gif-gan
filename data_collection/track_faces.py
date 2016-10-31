# NOTES TO SELF:
# Probably want to turn up the jaccard index, there are a bunch of examples where we
# artificially miss frames or split up what should be a single track because a detection
# is deemed to be "too far away" from the previous one.
# Maybe want to experiment with the other params as well? Missed frames? Minimum length?
# Maybe add some new params?
#  - min resolution
#  - filter out blurry results?
#  - filter out jumpy results?
#  - I think you can adjust the confidence level of the face detector?
#  - Maybe try some different face detectors?
# Looks like we need to add stabilization. Thinking:
#  - Do optical flow inside face bounding box between consecutive frames
#  - From the point pairs, compute overall affine transform
#  - Make new bounding box be a weighted average of the detector result and the optical
#    flow result
#  - Maybe throw away results where the optical flow and detector disagree too strongly?
# Actually test counters are valid, lol
# Cleanup notes:
#  - Should do max_skip check at same point that we compute jaccard index
#  - Should assign each detection in a track a colour, instead of doing n^2 search
#  - Should split up into functions and things, lol
# TODO: TRY SETTING FEATURE TRACK BLOCK SIZE BACK TO 19 TO MATCH OPTICAL FLOW SETTING
# TODO: Should we add some kind of safeguard if the stabilized bbox drifts too
# far from the original bbox? And just throw it out in that case?
# TODO: what is with the crop for FsfczP3ESd5UA? It is garbage and also too
# short, it should have been thrown away ... :/

import numpy as np
import cv2
import math
import sys
import os
from itertools import cycle
import argparse
import copy

def jaccard_index(a, b):
    intersection_width = min(a.x2, b.x2) - max(a.x1, b.x1)
    intersection_height = min(a.y2, b.y2) - max(a.y1, b.y1)
    if intersection_width <= 0.0 or intersection_height <= 0.0:
        return 0.0

    intersection_area = intersection_width * intersection_height
    total_area = a.height * a.width + b.height * b.width
    union_area = total_area - intersection_area

    return float(intersection_area) / float(union_area)


def inc(hist, key):
    hist[key] = hist.get(key, 0) + 1


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


# Basically just a struct binding together a bunch of outputs
class Output:
    def __init__(self):
        # Drop counters (in priority order, higher to lower)
        self.cnt_drop_because_low_frame_count = 0
        self.cnt_drop_because_low_total_detections = 0
        self.cnt_drop_because_expanded_bb_too_big = 0
        self.cnt_drop_because_optical_flow_bb_too_big = 0
        self.cnt_drop_because_stabilized_bb_too_big = 0
        self.cnt_drop_because_no_feature_points = 0
        self.cnt_drop_because_failed_optical_flow = 0
        self.cnt_drop_because_no_rigid_transform = 0

        # Histograms
        self.hst_frame_count = {}
        self.hst_total_detections = {}
        self.hst_skip = {}
        self.hst_jaccard = {}

# Params for algorithm
parser = argparse.ArgumentParser()
# Params for the Haar Cascade Classifier
parser.add_argument("--opencv_data_dir", default="/home/sandro/opencv-3.1.0/opencv-3.1.0/data/", help="Directory from which to load classifier config file")
parser.add_argument("--classifier_config_file", default="haarcascades/haarcascade_frontalface_alt2.xml", help="Classifier config file")
parser.add_argument("--classifier_scale_factor", type=float, default=1.1, help="cc.detectMultiScale(scaleFactor)")
parser.add_argument("--classifier_min_neighbors", type=int, default=4, help="cc.detectMultiScale(minNeighbors)")
parser.add_argument("--classifier_min_size_factor", type=float, default=1/20.0, help="Multiplier on image side length to determine cc.detectMultiScale(minSize)")
parser.add_argument("--classifier_max_size_factor", type=float, default=1/2.0, help="Multiplier on image side length to determine cc.detectMultiScale(maxSize)")
parser.add_argument("--classifier_flags", type=int, default=cv2.CASCADE_DO_CANNY_PRUNING, help="cc.detectMultiScale(flags)")
# Params for determining how to stitch frames together
parser.add_argument("--min_jaccard", type=float, default=0.60, help="Minimum jaccard index between adjacent bounding boxes in a track")
parser.add_argument("--max_skip", type=int, default=6, help="Maximium number of consecutive missing detections in a track")
parser.add_argument("--min_frame_count", type=int, default=20, help="Minimum number of consecutive frames required to form a track")
parser.add_argument("--min_total_detections", type=int, default=10, help="Minimum number of total detections in a track")
# Params for determining how to size the bounding box around the face
parser.add_argument("--target_width", type=int, default=500, help="Width of the cropped face video")
parser.add_argument("--target_height", type=int, default=500, help="Height of the cropped face video")
parser.add_argument("--bounding_box_scaling_factor", type=float, default=1.0, help="Amount to scale the haar cascade bounding box before cropping")
# Params for determining how statistics are collected
parser.add_argument("--hst_jaccard_bin_size", type=float, default=0.01, help="Size of a bin in the jaccard index histogram")
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

colours = [
    # in BGR space
    np.array([0,0,255]), # red
    np.array([0,255,0]), # green
    np.array([255,0,0]), # blue
    np.array([0,255,255]), # yellow
    np.array([255,0,255]), # magenta
    np.array([255,255,0]), # cyan
]


def get_crop(im, d):
    # # Expand the box along one axis so the aspect ratio is correct
    # required_aspect_ratio = float(target_width)/float(target_height)
    # actual_aspect_ratio = float(d.width)/float(d.height)
    # scaling = required_aspect_ratio / actual_aspect_ratio
    # x_scaling = scaling if scaling > 1.0 else 1.0
    # y_scaling = 1.0/scaling if scaling <= 1.0 else 1.0
    # # Expand the box by a scaling factor
    # centre_x = (d.x1 + d.x2)/2
    # centre_y = (d.y1 + d.y2)/2
    # x1 = x_scaling * bounding_box_scaling_factor * (d.x1 - centre_x) + centre_x
    # y1 = y_scaling * bounding_box_scaling_factor * (d.y1 - centre_y) + centre_y
    # x2 = x_scaling * bounding_box_scaling_factor * (d.x2 - centre_x) + centre_x
    # y2 = y_scaling * bounding_box_scaling_factor * (d.y2 - centre_y) + centre_y
    # pass
    # Methods are cv2.INTER_CUBIC (slow) and cv2.INTER_LINEAR (fast but worse)
    crop = im[d.y1:d.y2+1,d.x1:d.x2+1]
    res = cv2.resize(crop,(args.target_width,args.target_height), interpolation = cv2.INTER_LINEAR)
    #return (np.ones([target_height, target_width, 3]) * c).astype('uint8')
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

def process(f):
    global cnt_drop_because_low_total_detections
    global cnt_drop_because_low_frame_count
    global cnt_drop_because_expanded_bb_too_big
    # Temp
    global coloured_tracks
    global detections_per_frame
    global expanded_tracks
    
    cap = cv2.VideoCapture("/home/sandro/Documents/ECE496/gif-gan/data_collection/gifs/" + f + ".mp4")
    
    tracks = []
    detections_per_frame = []
    frame_number = 0
    frame_size = None
    while(cap.isOpened()):
        #print
        #print
        
        ret, im = cap.read()
        if not ret:
            break
        if not frame_size:
            frame_size = (im.shape[1], im.shape[0])

        side = math.sqrt(im.size)
        minlen = int(side * args.classifier_min_size_factor)
        maxlen = int(side * args.classifier_max_size_factor)

        features = cc.detectMultiScale(
            im, args.classifier_scale_factor, args.classifier_min_neighbors,
            args.classifier_flags, (minlen, minlen), (maxlen, maxlen))
        current_detections = [Detection(rect, frame_number) for rect in features]
        detections_per_frame.append(current_detections)
        current_detections = set(current_detections)

        scored_matches = []
        for current in current_detections:
            for track in tracks:
                previous = track[-1]
                j = jaccard_index(current, previous)
                if j >= args.min_jaccard:
                    scored_matches.append( (j, (current, track)) )
        scored_matches.sort(reverse=True)
        #print scored_matches
        for (j, (current, track)) in scored_matches:
            if current not in current_detections:
                # We already matched this with someone
                continue
            skip = frame_number - track[-1].frame_number - 1
            #print "track",track,"skip", skip
            if skip <= args.max_skip:
                #print "assigned",current,"to",track
                track.append(current)
                current_detections.remove(current)
                # Update histograms
                inc(hst_skip, skip)
                jaccard_bin = round(j / args.hst_jaccard_bin_size) * args.hst_jaccard_bin_size
                inc(hst_jaccard, jaccard_bin)

        # Everything that wasn't paired becomes a new track
        for current in current_detections:
            #print "new track for",current
            tracks.append([current])

        #out.write(im)
        frame_number += 1
        #print "frame",frame_number

    # Do a last pass over the tracks, filtering out the bad ones and interpolating
    # missing frames
    valid_tracks = []
    for track in tracks:
        # Drop track if invalid
        frame_count = track[-1].frame_number - track[0].frame_number + 1
        if frame_count < args.min_frame_count:
            #print "dropped track because frame count! Was",frame_count,"not",min_frame_count
            cnt_drop_because_low_frame_count += 1
            continue
        num_detections = len(track)
        if num_detections < args.min_total_detections:
            #print "dropped track because total detections! Was",num_detections,"not",min_total_detections
            cnt_drop_because_low_total_detections += 1
            continue

        # Okay track is valid. Interpolate missing frames.
        interpolated_track = []
        for i in range(len(track)-1):
            d1 = track[i]
            d2 = track[i+1]
            interpolated_track.append(d1)
            frame_delta = d2.frame_number - d1.frame_number
            rect_delta = d2.as_vec() - d1.as_vec()
            for frm in range(d1.frame_number+1,d2.frame_number):
                fraction = float(frm - d1.frame_number)/frame_delta
                new_rect = d1.as_vec() + fraction*rect_delta
                new_rect = np.round(new_rect).astype('int32')
                new_detection = Detection(new_rect, frm, interpolated=True)
                interpolated_track.append(new_detection)
                detections_per_frame[frm].append(new_detection)
                #print "interpolated",d1,"+",d2,"@",frm,"=>",new_detection
        interpolated_track.append(track[-1])
                
        inc(hst_frame_count, frame_count)
        inc(hst_total_detections, num_detections)
        valid_tracks.append(interpolated_track)

    # Okay we need to scale all the rectangles and potentially throw out
    # some stuff if the rectangles get too big, lol.
    expanded_tracks = []
    for track in valid_tracks:
        drop_track = False
        new_track = []
        print "track"
        for d in track:
            # Expand the box along one axis so the aspect ratio is correct
            required_aspect_ratio = float(args.target_width)/float(args.target_height)
            actual_aspect_ratio = float(d.width)/float(d.height)
            scaling = required_aspect_ratio / actual_aspect_ratio
            x_scaling = scaling if scaling > 1.0 else 1.0
            y_scaling = 1.0/scaling if scaling <= 1.0 else 1.0
            # Expand the box by a scaling factor
            centre_x = (d.x1 + d.x2)/2.0
            centre_y = (d.y1 + d.y2)/2.0
            width = frame_size[0]
            height = frame_size[1]
            assert(centre_x >= 0 and centre_x < width)
            assert(centre_y >= 0 and centre_y < height)
            x1 = int(round(x_scaling * args.bounding_box_scaling_factor * (d.x1 - centre_x) + centre_x))
            y1 = int(round(y_scaling * args.bounding_box_scaling_factor * (d.y1 - centre_y) + centre_y))
            x2 = int(round(x_scaling * args.bounding_box_scaling_factor * (d.x2 - centre_x) + centre_x))
            y2 = int(round(y_scaling * args.bounding_box_scaling_factor * (d.y2 - centre_y) + centre_y))
            if x1 < 0 or y1 < 0 or x2 >= width or y2 >= height:
                cnt_drop_because_expanded_bb_too_big += 1
                drop_track = True
                break
            new_d = Detection((x1, y1, x2-x1, y2-y1), d.frame_number, interpolated=d.interpolated)
            print new_d.height, new_d.width
            new_track.append(new_d)
        print "done track"
        if drop_track:
            for d in track:
                d.too_big = True
        else:
            expanded_tracks.append(new_track)
    #valid_tracks = expanded_tracks

    # # TEMP
    # # Okay let's try to follow the face just using dense optical flow
    # flow_tracks = []
    # for track in valid_tracks:
    #     cap = cv2.VideoCapture("/home/sandro/Documents/ECE496/gif-gan/data_collection/gifs/" + f + ".mp4")
    #     new_track = []
    #     d = Detection(track[0].as_vec(), track[0].frame_number, interpolated=track[0].interpolated)
    #     new_track.append(d)
    #     frame_number = d.frame_number
    #     prev_frame = None
    #     while(cap.isOpened()):
    #         ret, im = cap.read()
    #         if not ret:
    #             break
    #         im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    #         if prev_frame is not None:
    #             print "gray shape:",im.shape
    #             prev_crop = prev_frame[d.y1:d.y2+1,d.x1:d.x2+1]
    #             crop = im[d.y1:d.y2+1,d.x1:d.x2+1]
    #             flow = cv2.calcOpticalFlowFarneback(prev_crop, crop, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    #             print "flow shape:",flow.shape
    #             print "flow type:",flow.dtype
    #             shape = (1, flow.shape[0] * flow.shape[1], 2)
    #             source = np.zeros(shape, dtype=np.float32)
    #             target = np.zeros(shape, dtype=np.float32)
    #             for r in range(flow.shape[0]):
    #                 for c in range(flow.shape[1]):
    #                     source[0,c+r*flow.shape[1]] = [c + d.x1, r + d.y1]
    #                     target[0,c+r*flow.shape[1]] = [c + d.x1 + flow[r,c][0], r + d.y1 + flow[r,c][1]]
    #             # source = prev_crop
    #             # target = crop
    #             transformation = cv2.estimateRigidTransform(source, target, fullAffine=False)
    #             m = transformation[:,:2]
    #             b = transformation[:,2]
    #             print "transform",transformation
    #             new_d = Detection((0,0,0,0), frame_number+1)
    #             x1y1 = m.dot(np.array([[d.x1],[d.y1]])) + b
    #             new_d.x1 = x1y1[0,0]
    #             new_d.y1 = x1y1[1,0]
    #             x2y2 = m.dot(np.array([[d.x2],[d.y2]])) + b
    #             new_d.x2 = x2y2[0,0]
    #             new_d.y2 = x2y2[1,0]
    #             d = new_d
    #             print "new_d",d
    #             if new_d.x1 < 0 or new_d.y1 < 0: break
    #             detections_per_frame[frame_number].append(d)
    #             new_track.append(d)
    #             frame_number += 1
    #         prev_frame = im
    #     for d in new_track:
    #         d.x1 = int(round(d.x1))
    #         d.y1 = int(round(d.y1))
    #         d.x2 = int(round(d.x2))
    #         d.y2 = int(round(d.y2))
    #     flow_tracks.append(new_track)
    #         # shape = (1, 10, 2) # Needs to be a 3D array
    #         # source = np.random.randint(0, 100, shape).astype(np.int)
    #         # target = source + np.array([1, 0]).astype(np.int)
    #         # transformation = cv2.estimateRigidTransform(source, target, False)
    # valid_tracks.extend(flow_tracks)
    # # END TEMP

    # TEMP
    # Okay let's try to follow the face just using sparse optical flow
    feature_params = dict( maxCorners = 1000,
                           qualityLevel = 0.01,
                           minDistance = 8,
                           blockSize = 10 ) # originally 19
    lk_params = dict( winSize  = (19, 19),
                      maxLevel = 2,
                      criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    flow_tracks = []
    for track in valid_tracks:
        cap = cv2.VideoCapture("/home/sandro/Documents/ECE496/gif-gan/data_collection/gifs/" + f + ".mp4")
        new_track = []
        d = Detection(track[0].as_vec(), track[0].frame_number, interpolated=track[0].interpolated)
        new_track.append(d)
        frame_number = d.frame_number
        prev_frame = None
        while(cap.isOpened()):
            ret, im = cap.read()
            if not ret:
                break
            im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            if prev_frame is not None:
                print "gray shape:",im.shape
                prev_crop = prev_frame[d.y1:d.y2+1,d.x1:d.x2+1]#prev_frame[d.y1:d.y2+1,d.x1:d.x2+1]
                next_d = same_size_crop(track[len(new_track)], d)
                crop = im[next_d.y1:next_d.y2+1,next_d.x1:next_d.x2+1]#im[d.y1:d.y2+1,d.x1:d.x2+1]
                pnts = cv2.goodFeaturesToTrack(prev_crop, **feature_params)
                #print "before",pnts
                #print "after",pnts
                (pnts2, status, _) = cv2.calcOpticalFlowPyrLK(prev_crop, crop, pnts, None, **lk_params)
                #print "pnts",pnts
                #print "pnts2",pnts2
                #print "status",status
                pnts = [p for (p, s) in zip(pnts, status) if s]
                pnts2 = [p for (p, s) in zip(pnts2, status) if s]
                if len(pnts) == 0 or len(pnts2) == 0:
                    print "Oh Noes!"
                    print "pnts",pnts
                    print "pnts2",pnts2
                    cv2.imwrite("/home/sandro/Documents/ECE496/gif-gan/data_collection/tmp_crop.png", crop)
                    cv2.imwrite("/home/sandro/Documents/ECE496/gif-gan/data_collection/tmp_prev_crop.png", prev_crop)
                    raise Exception("Couldn't find tracking points")
                pnts += np.array([d.x1,d.y1])
                pnts2 += np.array([next_d.x1,next_d.y1])
                transformation = cv2.estimateRigidTransform(pnts,pnts2,fullAffine=False)
                if transformation is None:
                    print "Oh Noes!"
                    print "pnts",pnts
                    print "pnts2",pnts2
                    raise Exception("Couldn't find transformation")
                m = transformation[:,:2]
                b = transformation[:,2:3]
                #print "b",b
                print "transform",transformation
                new_d = Detection((0,0,0,0), frame_number+1)
                # Kill the rotation, extract just the scale and offset
                x1y1 = m.dot(np.array([[d.x1],[d.y1]])) + b
                x2y2 = m.dot(np.array([[d.x2],[d.y2]])) + b
                centre = (x1y1 + x2y2) / 2.0
                diag = (x1y1 - x2y2)
                diag_len = math.sqrt(diag[0,0]**2 + diag[1,0]**2)
                old_diag_len = math.sqrt((d.x2-d.x1)**2 + (d.y2-d.y1)**2)
                scale = diag_len / old_diag_len
                new_d.x1 = centre[0,0] - scale*(d.x2-d.x1)/2.0
                new_d.x2 = centre[0,0] + scale*(d.x2-d.x1)/2.0
                new_d.y1 = centre[1,0] - scale*(d.y2-d.y1)/2.0
                new_d.y2 = centre[1,0] + scale*(d.y2-d.y1)/2.0
                # x1y1 = m.dot(np.array([[d.x1],[d.y1]])) + b
                # new_d.x1 = x1y1[0,0]
                # new_d.y1 = x1y1[1,0]
                # x2y2 = m.dot(np.array([[d.x2],[d.y2]])) + b
                # new_d.x2 = x2y2[0,0]
                # new_d.y2 = x2y2[1,0]
                d = new_d
                print "new_d",d
                if new_d.x1 < 0 or new_d.y1 < 0: break
                if new_d.x2 >= frame_size[0] or new_d.y2 >= frame_size[1]: break
                detections_per_frame[frame_number].append(d)
                new_track.append(d)
                frame_number += 1
            prev_frame = im
        for d in new_track:
            #print "aspect ratio (before):", float(d.x2-d.x1) / float(d.y2 - d.y1)
            d.x1 = int(round(d.x1))
            d.y1 = int(round(d.y1))
            d.x2 = int(round(d.x2))
            d.y2 = int(round(d.y2))
            #print "aspect ratio (after):", float(d.x2-d.x1) / float(d.y2 - d.y1)
        flow_tracks.append(new_track)
            # shape = (1, 10, 2) # Needs to be a 3D array
            # source = np.random.randint(0, 100, shape).astype(np.int)
            # target = source + np.array([1, 0]).astype(np.int)
            # transformation = cv2.estimateRigidTransform(source, target, False)
    valid_tracks.extend(flow_tracks)
    # END TEMP

    # Do another run through the video, and colour the detections.
    # White for dropped detections, and other colours for tracks.

    # First colour the tracks
    coloured_tracks = zip(valid_tracks, cycle(colours))

    assert(frame_size is not None)
    # Open the reader again
    cap = cv2.VideoCapture("/home/sandro/Documents/ECE496/gif-gan/data_collection/gifs/" + f + ".mp4")
    # Open a writer for the debug video
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(
        "/home/sandro/Documents/ECE496/gif-gan/data_collection/tracks_v1_interpolated/" + f + ".avi",
        fourcc, 25.0, frame_size)
    # Open a writer for each track
    writers = [cv2.VideoWriter("/home/sandro/Documents/ECE496/gif-gan/data_collection/crops/" + f + "_" + str(i) + ".avi",
                               fourcc, 25.0, (args.target_width, args.target_height))
               for i in range(len(flow_tracks))]
    # Make cursors for each track
    cursors = [0 for _ in range(len(flow_tracks))]

    
    frame_number = 0
    while(cap.isOpened()):
        ret, im = cap.read()
        if not ret:
            break

        # # First create the crop frames
        # for (w, (t, c)) in zip(writers, coloured_tracks):
        #     for d in t:
        #         #print "compare",d.frame_number,frame_number
        #         if d.frame_number == frame_number:
        #             #print "write"
        #             w.write(get_crop(im, d, c))
        #             break

        # First create the crop frames
        for i in range(len(flow_tracks)):#(w, (t, c), cur) in zip(writers, coloured_tracks, cursors):
            cur = cursors[i]
            t = flow_tracks[i]
            w = writers[i]
            if cur < len(t) and t[cur].frame_number == frame_number:
                w.write(get_crop(im, t[cur]))
                cursors[i] += 1

        # Now create the debug video frame
        detections = detections_per_frame[frame_number]
        #print "at frame",frame_number,"got detections",detections
        frame_number += 1
        for d in detections:
            #m_colour = np.array([255,255,255])
            # Make a white rectangle, assuming the detection is spurious
            cv2.rectangle(im, (d.x1, d.y1), (d.x2, d.y2), (0, 0, 0), 4)
            cv2.rectangle(im, (d.x1, d.y1), (d.x2, d.y2), (255, 255, 255), 2)
            for (track, colour) in coloured_tracks:
                if d in track:
                    # This detection is in a track! Colour it appropriately
                    #print "found!"
                    if d.interpolated:
                        cv2.rectangle(im, (d.x1, d.y1), (d.x2, d.y2), (255,255,255), 6)
                    cv2.rectangle(im, (d.x1, d.y1), (d.x2, d.y2), colour, 2)
                    if d.too_big:
                        cv2.line(im, (d.x1, d.y1), (d.x2, d.y2), colour, 2)
                        cv2.line(im, (d.x1, d.y2), (d.x2, d.y1), colour, 2)
                    #m_colour = colour
                    break

            # for scale in [1.0, 1.3, 1.6, 1.9, 2.2, 2.5]:
            #     # Expand the box along one axis so the aspect ratio is correct
            #     required_aspect_ratio = float(target_width)/float(target_height)
            #     actual_aspect_ratio = float(d.width)/float(d.height)
            #     scaling = required_aspect_ratio / actual_aspect_ratio
            #     x_scaling = scaling if scaling > 1.0 else 1.0
            #     y_scaling = 1.0/scaling if scaling <= 1.0 else 1.0
            #     # Expand the box by a scaling factor
            #     centre_x = (d.x1 + d.x2)/2.0
            #     centre_y = (d.y1 + d.y2)/2.0
            #     x1 = int(round(x_scaling * scale * (d.x1 - centre_x) + centre_x))
            #     y1 = int(round(y_scaling * scale * (d.y1 - centre_y) + centre_y))
            #     x2 = int(round(x_scaling * scale * (d.x2 - centre_x) + centre_x))
            #     y2 = int(round(y_scaling * scale * (d.y2 - centre_y) + centre_y))
            #     new_d = Detection((x1, y1, x2-x1, y2-y1), 0)
            #     cv2.rectangle(im, (new_d.x1, new_d.y1), (new_d.x2, new_d.y2), m_colour, 2)


        out.write(im)
    out.release()

    # close the writers
    for w in writers:
        w.release()
    
    print "done", f


# Arguments:
#  - cap, opencv reader
#  - args, command line params
#  - output, object to stuff statistics and other side outputs into
# Returns:
#  - initial_tracks, a list of lists of Detection objects
#  - frame_size, the (width, height) of the frames in f
def get_initial_tracks(cap, args, output):
    # Intiialize the classifier
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
        side = math.sqrt(im.size)
        minlen = int(side * args.classifier_min_size_factor)
        maxlen = int(side * args.classifier_max_size_factor)
        features = cc.detectMultiScale(
            im, args.classifier_scale_factor, args.classifier_min_neighbors,
            args.classifier_flags, (minlen, minlen), (maxlen, maxlen))
        current_detections = [Detection(rect, frame_number) for rect in features]
        current_detections = set(current_detections)

        # Score how much each detection overlaps with previous tracks
        scored_matches = []
        for current in current_detections:
            for track in tracks:
                previous = track[-1]
                j = jaccard_index(current, previous)
                skip = frame_number - previous.frame_number - 1
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
            inc(output.hst_skip, skip)
            jaccard_bin = round(j / args.hst_jaccard_bin_size) * args.hst_jaccard_bin_size
            inc(output.hst_jaccard, jaccard_bin)

        # Everything that wasn't paired becomes a new track
        for current in current_detections:
            tracks.append([current])

        frame_number += 1

    assert(frame_size is not None)
    return (tracks, frame_size)


# Arguments:
#  - tracks, list of lists of Detection objects
#  - args, command line params
#  - output, object to stuff statistics and other side outputs into
# Returns:
#  - valid_tracks, a list of lists of Detection objects
#  - untracked_detections, a list of Detection objects
def discard_invalid_tracks(tracks, args, output):
    valid_tracks = []
    untracked_detections = []
    for track in tracks:
        # Drop track if overall frame count is too low
        frame_count = track[-1].frame_number - track[0].frame_number + 1
        if frame_count < args.min_frame_count:
            output.cnt_drop_because_low_frame_count += 1
            print "output.cnt_drop_because_low_frame_count"
            untracked_detections.extend(copy.deepcopy(track))
            continue
        
        # Drop track if detection count is too low
        num_detections = len(track)
        if num_detections < args.min_total_detections:
            output.cnt_drop_because_low_total_detections += 1
            print "output.cnt_drop_because_low_total_detections"
            untracked_detections.extend(copy.deepcopy(track))
            continue

        # Otherwise keep this track
        inc(output.hst_frame_count, frame_count)
        inc(output.hst_total_detections, num_detections)
        valid_tracks.append(copy.deepcopy(track))
    return (valid_tracks, untracked_detections)


# Arguments:
#  - tracks, list of lists of Detection objects
#  - args, command line params
#  - output, object to stuff statistics and other side outputs into
# Returns:
#  - interpolated_tracks, a list of lists of Detection objects
def interpolate_missing_frames(tracks, args, output):
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
#  - output, object to stuff statistics and other side outputs into
# Returns:
#  - expanded_tracks, a list of lists of Detection objects
#  - oversize_tracks, a list of lists of Detection objects
def expand_bounding_boxes(tracks, frame_size, args, output):
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
            output.cnt_drop_because_expanded_bb_too_big += 1
            print "output.cnt_drop_because_expanded_bb_too_big"
            oversize_tracks.append(new_track)
        else:
            expanded_tracks.append(new_track)
    return (expanded_tracks, oversize_tracks)


# Arguments:
#  - cap, opencv reader
#  - tracks, list of lists of Detection objects
#  - frame_size, as (width, height)
#  - args, command line params
#  - output, object to stuff statistics and other side outputs into
# Returns:
#  - stabilized_tracks, a list of lists of Detection objects
def stabilize_tracks(cap, tracks, frame_size, args, output):
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
                    output.cnt_drop_because_optical_flow_bb_too_big += 1
                    print "output.cnt_drop_because_optical_flow_bb_too_big"
                    onscreen[track_id] = False
                    continue

                # Now crop out the relevant regions from the previous and
                # current frame, so that we can run optical flow.
                prev_crop = prev_frame[d.y1:d.y2+1,d.x1:d.x2+1]
                crop = im[next_d.y1:next_d.y2+1,next_d.x1:next_d.x2+1]
                # Find features
                pnts = cv2.goodFeaturesToTrack(prev_crop, **feature_params)
                if len(pnts) == 0:
                    # Oops, failed to find tracking points. Record this failure
                    # and move on.
                    output.cnt_drop_because_no_feature_points += 1
                    print "output.cnt_drop_because_no_feature_points"
                    onscreen[track_id] = False
                    continue

                # Run optical flow
                (pnts2, status, _) = cv2.calcOpticalFlowPyrLK(
                    prev_crop, crop, pnts, None, **lk_params)
                # Filter out the points where optical flow failed
                pnts = [p for (p, s) in zip(pnts, status) if s]
                pnts2 = [p for (p, s) in zip(pnts2, status) if s]
                if len(pnts) == 0 or len(pnts2) == 0:
                    # Oops, optical flow calculation failed. Record this failure
                    # and move on.
                    output.cnt_drop_because_failed_optical_flow += 1
                    print "output.cnt_drop_because_failed_optical_flow"
                    onscreen[track_id] = False
                    continue

                # Estimate transform between the two frames
                pnts += np.array([d.x1,d.y1])
                pnts2 += np.array([next_d.x1,next_d.y1])
                transformation = cv2.estimateRigidTransform(pnts,pnts2,fullAffine=False)
                if transformation is None:
                    # Oops, can't estimate transform. Record this failure and
                    # move on.
                    output.cnt_drop_because_no_rigid_transform += 1
                    print "output.cnt_drop_because_no_rigid_transform"
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
                new_d.x1 = int(round(centre[0,0] - scale*(d.x2-d.x1)/2.0))
                new_d.x2 = int(round(centre[0,0] + scale*(d.x2-d.x1)/2.0))
                new_d.y1 = int(round(centre[1,0] - scale*(d.y2-d.y1)/2.0))
                new_d.y2 = int(round(centre[1,0] + scale*(d.y2-d.y1)/2.0))

                # The stabilized bounding box may now have gone offscreen, so
                # check that.
                if (new_d.x1 < 0 or new_d.y1 < 0 or
                    new_d.x2 >= width or new_d.y2 >= height):
                    output.cnt_drop_because_stabilized_bb_too_big += 1
                    print "output.cnt_drop_because_stabilized_bb_too_big"
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
    # went offscreen during stabilization, we will simply return None at that
    # index.
    return [track if valid else None
            for (track, valid) in zip(stable_tracks, onscreen)]

def generate_visualization(cap, viz_file, stabilized_tracks, expanded_tracks,
                           oversize_tracks, untracked_detections, frame_size,
                           args, output):
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
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
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
            if cur < len(stabilized) and stabilized[cur].frame_number == frame_number:
                # Draw the original rectangle
                assert(cur < len(expanded))
                assert(expanded[cur].frame_number == frame_number)
                d = expanded[cur]
                cv2.rectangle(im, (d.x1, d.y1), (d.x2, d.y2), colour, 1)
                # Draw the stabilized rectangle
                d = stabilized[cur]
                cv2.rectangle(im, (d.x1, d.y1), (d.x2, d.y2), (0, 0, 0), 4)
                cv2.rectangle(im, (d.x1, d.y1), (d.x2, d.y2), colour, 2)
                cursors[i] += 1
            
        frame_number += 1
        out.write(im)
    out.release()

def crop_faces(cap, crop_base, stabilized_tracks, args, output):
    # Remove the empty tracks
    stabilized_tracks = [s for s in stabilized_tracks if s is not None]

    # Open a writer for each track
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    writers = [cv2.VideoWriter(crop_base + "_" + str(i) + ".mp4",
                               fourcc, 25.0, (args.target_width, args.target_height))
               for i in range(len(stabilized_tracks))]

    # Make cursors for each track
    cursors = [0 for _ in range(len(stabilized_tracks))]

    frame_number = 0
    while(cap.isOpened()):
        ret, im = cap.read()
        if not ret:
            break

        # First create the crop frames
        for i in range(len(stabilized_tracks)):
            cur = cursors[i]
            t = stabilized_tracks[i]
            w = writers[i]
            if cur < len(t) and t[cur].frame_number == frame_number:
                w.write(get_crop(im, t[cur]))
                cursors[i] += 1
        frame_number += 1

    # close the writers
    for w in writers:
        w.release()

# gif_id = "CvnAPu8fAQgJq"
# f = "/home/sandro/Documents/ECE496/gif-gan/data_collection/gifs/" + gif_id + ".mp4"
# viz_file = "/home/sandro/Documents/ECE496/gif-gan/data_collection/clean_tracks_v1_interpolated/" + gif_id + ".mp4"
# crop_base = "/home/sandro/Documents/ECE496/gif-gan/data_collection/clean_crops/" + gif_id
# args = parser.parse_args([]) # Get a default set of arguments
# output = Output()

def better_process(f, args, output):
    # First, just get face detections and initial face tracks
    cap = cv2.VideoCapture(f)
    (initial_tracks, frame_size) = get_initial_tracks(cap, args, output)
    
    # Then discard those that don't meet length or other requirements
    (valid_tracks, untracked_detections) = discard_invalid_tracks(
        initial_tracks, args, output)

    # Now interpolate any missing bounding boxes
    interpolated_tracks = interpolate_missing_frames(valid_tracks, args, output)

    # Now we want to adjust all of the rectangles so that they have the correct
    # aspect ratio. We do this by expanding the bounding boxes, never
    # contracting (since we don't want to throw away anything that was inside
    # the original bounding box). This means that some bounding boxes may get so
    # large that they extend outside the original image. We will discard these
    # tracks.
    (expanded_tracks, oversize_tracks) = expand_bounding_boxes(
        interpolated_tracks, frame_size, args, output)
    
    # Now we will perform stabilization. The new stabilized track will have
    # bounding boxes in slightly different locations from the original track. It
    # is possible that the new bounding boxes may go outside of the original
    # image. If this happens, we will discard the track. Stabilization needs to
    # see the frames again, so we reread them from file. TODO: would it be
    # better to keep them all in memory? Probably doesn't make too much of a
    # difference since runtime is not dominated by file access.
    cap = cv2.VideoCapture(f)
    stabilized_tracks = stabilize_tracks(cap, expanded_tracks, frame_size, args, output)

    # Now crop out the faces
    cap = cv2.VideoCapture(f)
    crop_faces(cap, crop_base, stabilized_tracks, args, output)

    # Generate a debug video.
    cap = cv2.VideoCapture(f)
    generate_visualization(cap, viz_file, stabilized_tracks, expanded_tracks,
                           oversize_tracks, untracked_detections, frame_size, args, output)
    
    
# for f in ["iVy6Rgdog5oY", "jetOcz4pWPDck", "JhaOVn64HauaY", "JpA6974tuNRoA", "l41lRpI1ejISAVH0s", "L4vyAauJjxOlq", "lfrhq1753H0LC", "LGSc63wrKtKtG", "ml2Lm6lo5HSVy", "ndWC7pp2wKSWc", "NMH1ANukWHhZK", "ods8tx96CvuBG", "OLqdxkiQ3Q7Cw", "oQ7Kz58ZNpm6c", "oRp8OVyUcDBAI", "otfRWaEBmijv2", "pctqGv7NH8voA", "pruglIqg2Hsyc", "RMj1QZfa4JjZm", "SHFqtiEibgeo8", "TLs8z2Mn0RhOE", "U5MuZ4lELv0Eo", "U5poGkzMYOd7G", "UnpijzhwBafBe", "VqndyRC8rcWnS", "Vui3leSkFpkg8", "W1GXtbO5qAPhS", "WoaluZhDpz3zy", "x20dFskH5nwpW", "Xc4vTdVhgQ4ow", "XG1Iu0NH8VOHS", "XjEKa4BHjn7TW", "xnBhXMpDsQZ6o", "yLZwnMvQWqTkY", "10TeLEbt7fLndC", "11dgjtjk5zchRS", "11PSiVXyLMe1X2", "1241korwKdGMBa", "12zkbg2qEZb3Nu", "1403eCPKl5rrA4", "1CthgbtIOu0Du", "28z8pk38RfSY8", "3o7TKtZqP4MyMG5QC4", "3o85fPE3Irg8Wazl9S", "3rgXBPrh1KX3maLMYg", "5aIPErVMawv8Q", "5utwj4dIKEOk", "60CcjMxxCvq0g", "6iZgSVAGAmsbm", "6VhcRljpIT7A4", "6ZhO6QxQ4yqI0", "7MBQ8YA3Oxt3q", "7pKpsdWxPcAbm", "aVv2exYGNUwc8", "b4O5D4wspbBIc", "blMqtjunYqDm", "Bn3yWoKmd1B7O", "bYLvUDLqHPb7a", "CvnAPu8fAQgJq", "E3QcFMX4BQpQk", "FJE4sp5ezhPr2", "FsfczP3ESd5UA", "gCGnG3BLTwFgI"]:
#     process(f)
args = parser.parse_args([]) # Get a default set of arguments
output = Output()
for gif_id in ["iVy6Rgdog5oY", "jetOcz4pWPDck", "JhaOVn64HauaY", "JpA6974tuNRoA", "l41lRpI1ejISAVH0s", "L4vyAauJjxOlq", "lfrhq1753H0LC", "LGSc63wrKtKtG", "ml2Lm6lo5HSVy", "ndWC7pp2wKSWc", "NMH1ANukWHhZK", "ods8tx96CvuBG", "OLqdxkiQ3Q7Cw", "oQ7Kz58ZNpm6c", "oRp8OVyUcDBAI", "otfRWaEBmijv2", "pctqGv7NH8voA", "pruglIqg2Hsyc", "RMj1QZfa4JjZm", "SHFqtiEibgeo8", "TLs8z2Mn0RhOE", "U5MuZ4lELv0Eo", "U5poGkzMYOd7G", "UnpijzhwBafBe", "VqndyRC8rcWnS", "Vui3leSkFpkg8", "W1GXtbO5qAPhS", "WoaluZhDpz3zy", "x20dFskH5nwpW", "Xc4vTdVhgQ4ow", "XG1Iu0NH8VOHS", "XjEKa4BHjn7TW", "xnBhXMpDsQZ6o", "yLZwnMvQWqTkY", "10TeLEbt7fLndC", "11dgjtjk5zchRS", "11PSiVXyLMe1X2", "1241korwKdGMBa", "12zkbg2qEZb3Nu", "1403eCPKl5rrA4", "1CthgbtIOu0Du", "28z8pk38RfSY8", "3o7TKtZqP4MyMG5QC4", "3o85fPE3Irg8Wazl9S", "3rgXBPrh1KX3maLMYg", "5aIPErVMawv8Q", "5utwj4dIKEOk", "60CcjMxxCvq0g", "6iZgSVAGAmsbm", "6VhcRljpIT7A4", "6ZhO6QxQ4yqI0", "7MBQ8YA3Oxt3q", "7pKpsdWxPcAbm", "aVv2exYGNUwc8", "b4O5D4wspbBIc", "blMqtjunYqDm", "Bn3yWoKmd1B7O", "bYLvUDLqHPb7a", "CvnAPu8fAQgJq", "E3QcFMX4BQpQk", "FJE4sp5ezhPr2", "FsfczP3ESd5UA", "gCGnG3BLTwFgI"]:
    # gif_id = "x20dFskH5nwpW"
    f = "/home/sandro/Documents/ECE496/gif-gan/data_collection/gifs/" + gif_id + ".mp4"
    viz_file = "/home/sandro/Documents/ECE496/gif-gan/data_collection/clean_tracks_v1_interpolated/" + gif_id + ".mp4"
    crop_base = "/home/sandro/Documents/ECE496/gif-gan/data_collection/clean_crops/" + gif_id
    better_process(f, args, output)
    print "processed",f

