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

import numpy as np
import cv2
import math
import sys
import os
from itertools import cycle

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


# Params for algorithm
data_dir = "/home/sandro/opencv-3.1.0/opencv-3.1.0/data/"
v = 'haarcascades/haarcascade_frontalface_alt2.xml'
v = os.path.join(data_dir, v)
cc = cv2.CascadeClassifier(v)
flags = cv2.CASCADE_DO_CANNY_PRUNING
min_jaccard = 0.60
max_skip = 6
min_frame_count = 20
min_total_detections = 10
colours = [
    # in BGR space
    np.array([0,0,255]), # red
    np.array([0,255,0]), # green
    np.array([255,0,0]), # blue
    np.array([0,255,255]), # yellow
    np.array([255,0,255]), # magenta
    np.array([255,255,0]), # cyan
]
# Pretty big for now so we can examine it
target_width = 500
target_height = 500
bounding_box_scaling_factor = 1.0 # Iunno

# Drop counters (in priority order, higher to lower)
cnt_drop_because_low_frame_count = 0
cnt_drop_because_low_total_detections = 0
cnt_drop_because_bb_too_big = 0

# Histograms
hst_frame_count = {}
hst_total_detections = {}
hst_skip = {}
hst_jaccard = {}
hst_jaccard_bin_size = 0.01

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
    res = cv2.resize(crop,(target_width,target_height), interpolation = cv2.INTER_LINEAR)
    #return (np.ones([target_height, target_width, 3]) * c).astype('uint8')
    return res

def process(f):
    global cnt_drop_because_low_total_detections
    global cnt_drop_because_low_frame_count
    global cnt_drop_because_bb_too_big
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
        minlen = int(side / 20)
        maxlen = int(side / 2)

        features = cc.detectMultiScale(im, 1.1, 4, flags, (minlen, minlen), (maxlen, maxlen))
        current_detections = [Detection(rect, frame_number) for rect in features]
        detections_per_frame.append(current_detections)
        current_detections = set(current_detections)

        scored_matches = []
        for current in current_detections:
            for track in tracks:
                previous = track[-1]
                j = jaccard_index(current, previous)
                if j >= min_jaccard:
                    scored_matches.append( (j, (current, track)) )
        scored_matches.sort(reverse=True)
        #print scored_matches
        for (j, (current, track)) in scored_matches:
            if current not in current_detections:
                # We already matched this with someone
                continue
            skip = frame_number - track[-1].frame_number - 1
            #print "track",track,"skip", skip
            if skip <= max_skip:
                #print "assigned",current,"to",track
                track.append(current)
                current_detections.remove(current)
                # Update histograms
                inc(hst_skip, skip)
                jaccard_bin = round(j / hst_jaccard_bin_size) * hst_jaccard_bin_size
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
        if frame_count < min_frame_count:
            #print "dropped track because frame count! Was",frame_count,"not",min_frame_count
            cnt_drop_because_low_frame_count += 1
            continue
        num_detections = len(track)
        if num_detections < min_total_detections:
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
            required_aspect_ratio = float(target_width)/float(target_height)
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
            x1 = int(round(x_scaling * bounding_box_scaling_factor * (d.x1 - centre_x) + centre_x))
            y1 = int(round(y_scaling * bounding_box_scaling_factor * (d.y1 - centre_y) + centre_y))
            x2 = int(round(x_scaling * bounding_box_scaling_factor * (d.x2 - centre_x) + centre_x))
            y2 = int(round(y_scaling * bounding_box_scaling_factor * (d.y2 - centre_y) + centre_y))
            if x1 < 0 or y1 < 0 or x2 >= width or y2 >= height:
                cnt_drop_because_bb_too_big += 1
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
    # # TODO: pretty sure that two things are wrong:
    # # 1) the way that points are fed into estimateRigidTransform ... looks like (y, x), should
    # # probably be (x, y) to make the resulting transformation easier to reason about.
    # # 2) the coordinate system. We are calculating the transform relative to the crop, but then
    # # applying it relative to the entire image.
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
    #                     source[0,c+r*flow.shape[1]] = [r, c]
    #                     target[0,c+r*flow.shape[1]] = [r, c] + flow[r,c]
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
                prev_crop = prev_frame[d.y1:d.y2+1,d.x1:d.x2+1]
                crop = im[d.y1:d.y2+1,d.x1:d.x2+1]
                pnts = cv2.goodFeaturesToTrack(prev_crop, **feature_params)
                #print "before",pnts
                #print "after",pnts
                (pnts2, _, status) = cv2.calcOpticalFlowPyrLK(prev_crop, crop, pnts, None, **lk_params)
                #print "pnts2",pnts2
                pnts = [p for (p, s) in zip(pnts, status) if s]
                pnts2 = [p for (p, s) in zip(pnts2, status) if s]
                pnts += np.array([d.x1,d.y1])
                pnts2 += np.array([d.x1,d.y1])
                transformation = cv2.estimateRigidTransform(pnts,pnts2,fullAffine=False)
                m = transformation[:,:2]
                b = transformation[:,2:3]
                print "b",b
                print "transform",transformation
                new_d = Detection((0,0,0,0), frame_number+1)
                x1y1 = m.dot(np.array([[d.x1],[d.y1]])) + b
                new_d.x1 = x1y1[0,0]
                new_d.y1 = x1y1[1,0]
                x2y2 = m.dot(np.array([[d.x2],[d.y2]])) + b
                new_d.x2 = x2y2[0,0]
                new_d.y2 = x2y2[1,0]
                d = new_d
                print "new_d",d
                if new_d.x1 < 0 or new_d.y1 < 0: break
                detections_per_frame[frame_number].append(d)
                new_track.append(d)
                frame_number += 1
            prev_frame = im
        for d in new_track:
            d.x1 = int(round(d.x1))
            d.y1 = int(round(d.y1))
            d.x2 = int(round(d.x2))
            d.y2 = int(round(d.y2))
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
                               fourcc, 25.0, (target_width, target_height))
               for i in range(len(expanded_tracks))]
    # Make cursors for each track
    cursors = [0 for _ in range(len(expanded_tracks))]

    
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
        for i in range(len(expanded_tracks)):#(w, (t, c), cur) in zip(writers, coloured_tracks, cursors):
            cur = cursors[i]
            t = expanded_tracks[i]
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

for f in ["iVy6Rgdog5oY", "jetOcz4pWPDck", "JhaOVn64HauaY", "JpA6974tuNRoA", "l41lRpI1ejISAVH0s", "L4vyAauJjxOlq", "lfrhq1753H0LC", "LGSc63wrKtKtG", "ml2Lm6lo5HSVy", "ndWC7pp2wKSWc", "NMH1ANukWHhZK", "ods8tx96CvuBG", "OLqdxkiQ3Q7Cw", "oQ7Kz58ZNpm6c", "oRp8OVyUcDBAI", "otfRWaEBmijv2", "pctqGv7NH8voA", "pruglIqg2Hsyc", "RMj1QZfa4JjZm", "SHFqtiEibgeo8", "TLs8z2Mn0RhOE", "U5MuZ4lELv0Eo", "U5poGkzMYOd7G", "UnpijzhwBafBe", "VqndyRC8rcWnS", "Vui3leSkFpkg8", "W1GXtbO5qAPhS", "WoaluZhDpz3zy", "x20dFskH5nwpW", "Xc4vTdVhgQ4ow", "XG1Iu0NH8VOHS", "XjEKa4BHjn7TW", "xnBhXMpDsQZ6o", "yLZwnMvQWqTkY", "10TeLEbt7fLndC", "11dgjtjk5zchRS", "11PSiVXyLMe1X2", "1241korwKdGMBa", "12zkbg2qEZb3Nu", "1403eCPKl5rrA4", "1CthgbtIOu0Du", "28z8pk38RfSY8", "3o7TKtZqP4MyMG5QC4", "3o85fPE3Irg8Wazl9S", "3rgXBPrh1KX3maLMYg", "5aIPErVMawv8Q", "5utwj4dIKEOk", "60CcjMxxCvq0g", "6iZgSVAGAmsbm", "6VhcRljpIT7A4", "6ZhO6QxQ4yqI0", "7MBQ8YA3Oxt3q", "7pKpsdWxPcAbm", "aVv2exYGNUwc8", "b4O5D4wspbBIc", "blMqtjunYqDm", "Bn3yWoKmd1B7O", "bYLvUDLqHPb7a", "CvnAPu8fAQgJq", "E3QcFMX4BQpQk", "FJE4sp5ezhPr2", "FsfczP3ESd5UA", "gCGnG3BLTwFgI"]:
    process(f)

