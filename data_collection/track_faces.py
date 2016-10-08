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
    def __init__(self, rect, frame_number):
        # rect is a tuple (x1, y1, width, height)
        self.x1 = rect[0]
        self.y1 = rect[1]
        self.x2 = rect[0] + rect[2]
        self.y2 = rect[1] + rect[3]
        self.frame_number = frame_number
    @property
    def height(self):
        return self.y2 - self.y1
    @property
    def width(self):
        return self.x2 - self.x1


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

# Drop counters (in priority order, higher to lower)
cnt_drop_because_low_frame_count = 0
cnt_drop_because_low_total_detections = 0

# Histograms
hst_frame_count = {}
hst_total_detections = {}
hst_skip = {}
hst_jaccard = {}
hst_jaccard_bin_size = 0.01

def process(f):
    global cnt_drop_because_low_total_detections
    global cnt_drop_because_low_frame_count
    # Temp
    global coloured_tracks
    global detections_per_frame
    
    cap = cv2.VideoCapture("/home/sandro/Documents/ECE496/gif-gan/data_collection/gifs/" + f + ".mp4")
    
    tracks = []
    detections_per_frame = []
    frame_number = 0
    frame_size = None
    while(cap.isOpened()):
        print
        print
        
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
        print scored_matches
        for (j, (current, track)) in scored_matches:
            if current not in current_detections:
                # We already matched this with someone
                continue
            skip = frame_number - track[-1].frame_number - 1
            print "track",track,"skip", skip
            if skip <= max_skip:
                print "assigned",current,"to",track
                track.append(current)
                current_detections.remove(current)
                # Update histograms
                inc(hst_skip, skip)
                jaccard_bin = round(j / hst_jaccard_bin_size) * hst_jaccard_bin_size
                inc(hst_jaccard, jaccard_bin)

        # Everything that wasn't paired becomes a new track
        for current in current_detections:
            print "new track for",current
            tracks.append([current])

        #out.write(im)
        frame_number += 1
        print "frame",frame_number

    # Do a last pass over the tracks and filter out the bad ones
    valid_tracks = []
    for track in tracks:
        frame_count = track[-1].frame_number - track[0].frame_number + 1
        if frame_count < min_frame_count:
            print "dropped track because frame count! Was",frame_count,"not",min_frame_count
            cnt_drop_because_low_frame_count += 1
            continue
        num_detections = len(track)
        if num_detections < min_total_detections:
            print "dropped track because total detections! Was",num_detections,"not",min_total_detections
            cnt_drop_because_low_total_detections += 1
            continue
        inc(hst_frame_count, frame_count)
        inc(hst_total_detections, num_detections)
        valid_tracks.append(track)

    # Do another run through the video, and colour the detections.
    # White for dropped detections, and other colours for tracks

    # First colour the tracks
    coloured_tracks = zip(valid_tracks, cycle(colours))

    assert(frame_size is not None)
    cap = cv2.VideoCapture("/home/sandro/Documents/ECE496/gif-gan/data_collection/gifs/" + f + ".mp4")
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(
        "/home/sandro/Documents/ECE496/gif-gan/data_collection/tracks_v1/" + f + ".avi",
        fourcc, 25.0, frame_size)

    frame_number = 0
    while(cap.isOpened()):
        ret, im = cap.read()
        if not ret:
            break
        
        detections = detections_per_frame[frame_number]
        print "at frame",frame_number,"got detections",detections
        frame_number += 1

        for d in detections:
            # Make a white rectangle, assuming the detection is spurious
            cv2.rectangle(im, (d.x1, d.y1), (d.x2, d.y2), (0, 0, 0), 4)
            cv2.rectangle(im, (d.x1, d.y1), (d.x2, d.y2), (255, 255, 255), 2)
            for (track, colour) in coloured_tracks:
                if d in track:
                    # This detection is in a track! Colour it appropriately
                    print "found!"
                    cv2.rectangle(im, (d.x1, d.y1), (d.x2, d.y2), colour, 2)
                    break

        out.write(im)
    out.release()
    print "done", f

for f in ["iVy6Rgdog5oY", "jetOcz4pWPDck", "JhaOVn64HauaY", "JpA6974tuNRoA", "l41lRpI1ejISAVH0s", "L4vyAauJjxOlq", "lfrhq1753H0LC", "LGSc63wrKtKtG", "ml2Lm6lo5HSVy", "ndWC7pp2wKSWc", "NMH1ANukWHhZK", "ods8tx96CvuBG", "OLqdxkiQ3Q7Cw", "oQ7Kz58ZNpm6c", "oRp8OVyUcDBAI", "otfRWaEBmijv2", "pctqGv7NH8voA", "pruglIqg2Hsyc", "RMj1QZfa4JjZm", "SHFqtiEibgeo8", "TLs8z2Mn0RhOE", "U5MuZ4lELv0Eo", "U5poGkzMYOd7G", "UnpijzhwBafBe", "VqndyRC8rcWnS", "Vui3leSkFpkg8", "W1GXtbO5qAPhS", "WoaluZhDpz3zy", "x20dFskH5nwpW", "Xc4vTdVhgQ4ow", "XG1Iu0NH8VOHS", "XjEKa4BHjn7TW", "xnBhXMpDsQZ6o", "yLZwnMvQWqTkY", "10TeLEbt7fLndC", "11dgjtjk5zchRS", "11PSiVXyLMe1X2", "1241korwKdGMBa", "12zkbg2qEZb3Nu", "1403eCPKl5rrA4", "1CthgbtIOu0Du", "28z8pk38RfSY8", "3o7TKtZqP4MyMG5QC4", "3o85fPE3Irg8Wazl9S", "3rgXBPrh1KX3maLMYg", "5aIPErVMawv8Q", "5utwj4dIKEOk", "60CcjMxxCvq0g", "6iZgSVAGAmsbm", "6VhcRljpIT7A4", "6ZhO6QxQ4yqI0", "7MBQ8YA3Oxt3q", "7pKpsdWxPcAbm", "aVv2exYGNUwc8", "b4O5D4wspbBIc", "blMqtjunYqDm", "Bn3yWoKmd1B7O", "bYLvUDLqHPb7a", "CvnAPu8fAQgJq", "E3QcFMX4BQpQk", "FJE4sp5ezhPr2", "FsfczP3ESd5UA", "gCGnG3BLTwFgI"]:
    process(f)

