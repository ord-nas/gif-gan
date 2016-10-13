import numpy as np
import cv2
import math
import sys
import os
from itertools import cycle

input_file = "/home/sandro/Documents/ECE496/gif-gan/data_collection/synth.avi"
output_file = "/home/sandro/Documents/ECE496/gif-gan/data_collection/synth_track.avi"

feature_params = dict( maxCorners = 1000,
                       qualityLevel = 0.01,
                       minDistance = 8,
                       blockSize = 10 )
lk_params = dict( winSize  = (19, 19),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

cap = cv2.VideoCapture(input_file)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = None
prev_frame = None
window = ((90,90),(210,210))
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    if not out:
        frame_size = (frame.shape[1], frame.shape[0])
        out = cv2.VideoWriter(output_file, fourcc, 25.0, frame_size)

    if prev_frame is not None:
        prev_im = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        im = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ((x1, y1), (x2, y2)) = window
        (rx1, ry1, rx2, ry2) = (int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2)))
        prev_crop = prev_im[ry1:ry2+1,rx1:rx2+1]
        crop = im[ry1:ry2+1,rx1:rx2+1]
        pnts = cv2.goodFeaturesToTrack(prev_crop, **feature_params)
        (pnts2, _, status) = cv2.calcOpticalFlowPyrLK(prev_crop, crop, pnts, None, **lk_params)
        pnts = [p for (p, s) in zip(pnts, status) if s]
        pnts2 = [p for (p, s) in zip(pnts2, status) if s]
        pnts += np.array([rx1,ry1])
        pnts2 += np.array([rx1,ry1])
        transformation = cv2.estimateRigidTransform(pnts,pnts2,fullAffine=False)
        if transformation is None:
            print "OH NOES!"
            print "pnts", pnts
            print "pnts2", pnts2
            out.release()
            break
        #print "trans", transformation
        m = transformation[:,:2]
        #print "m",m
        b = transformation[:,2:3]
        #print "b",b
        x1y1 = m.dot(np.array([[x1],[y1]])) + b
        #print "pnt", np.array([[x1],[y1]])
        #print "m.pnt", m.dot(np.array([[x1],[y1]]))
        #print "m.pnt+b", m.dot(np.array([[x2],[y2]])) + b
        x2y2 = m.dot(np.array([[x2],[y2]])) + b
        new_window = ((x1y1[0,0], x1y1[1,0]), (x2y2[0,0], x2y2[1,0]))
        x1y2 = m.dot(np.array([[x1],[y2]])) + b
        x2y1 = m.dot(np.array([[x2],[y1]])) + b

        # Okay make some output now!
        
        # First frame contains original window, points with tracks
        f1 = np.copy(prev_frame)
        cv2.rectangle(f1, (rx1, ry1), (rx2, ry2), (255, 255, 255), 2)
        for (p1, p2) in zip(pnts, pnts2):
            p1x, p1y = int(round(p1[0][0])), int(round(p1[0][1]))
            p2x, p2y = int(round(p2[0][0])), int(round(p2[0][1]))
            cv2.circle(f1, (p1x, p1y), 2, (0, 0, 255), -1)
            cv2.circle(f1, (p2x, p2y), 2, (0, 0, 255), -1)
            cv2.line(f1, (p1x, p1y), (p2x, p2y), (0, 0, 255), 1)
        out.write(f1)

        window = new_window
        ((x1, y1), (x2, y2)) = window
        (rx1, ry1, rx2, ry2) = (int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2)))
        
        # Second frame contains new window, new points
        f2 = np.copy(frame)
        cv2.rectangle(f2, (rx1, ry1), (rx2, ry2), (255, 255, 255), 2)
        for p2 in pnts2:
            p2x, p2y = int(round(p2[0][0])), int(round(p2[0][1]))
            cv2.circle(f2, (p2x, p2y), 2, (0, 0, 255), -1)
        #print [x1y1, x2y2, x1y2, x2y1]
        for ((x,),(y,)) in [x1y1, x2y2, x1y2, x2y1]:
            x = int(round(x))
            y = int(round(y))
            cv2.circle(f2, (x, y), 4, (255, 255, 255), -1)
        out.write(f2)
        
    prev_frame = frame

out.release()
