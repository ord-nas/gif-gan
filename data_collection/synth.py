import numpy as np
import cv2
import math
import sys
import os
from itertools import cycle

initial_face_file = "/home/sandro/Documents/ECE496/gif-gan/data_collection/crops/FsfczP3ESd5UA_0.avi"
output_file = "/home/sandro/Documents/ECE496/gif-gan/data_collection/synth.mp4"

def make_path(key_points, num_steps):
    assert(len(key_points) > 0)
    assert(num_steps >= 2)
    path = [key_points[0]]
    for i in range(len(key_points)-1):
        (x1, y1) = key_points[i]
        (x2, y2) = key_points[i+1]
        xs = np.linspace(x1, x2, num_steps)[1:]
        ys = np.linspace(y1, y2, num_steps)[1:]
        path.extend(zip(xs, ys))
    return path

cap = cv2.VideoCapture(initial_face_file)
assert(cap.isOpened())
ret, face = cap.read()
assert(ret)
face = cv2.resize(face, (100,100), interpolation = cv2.INTER_LINEAR)

fourcc = cv2.VideoWriter_fourcc(*'H264')
frame_size = (500, 500) # width, height
out = cv2.VideoWriter(output_file, fourcc, 25.0, frame_size)
key_points = [(100,100), (300,100), (100,300), (300,300)]
num_steps = 25
path = make_path(key_points, num_steps)

for (x, y) in path:
    colour = np.array([255,0,0]).astype('uint8') # blue in BGR
    frame = np.tile(colour, (frame_size[1], frame_size[0], 1))
    (dy, dx, _) = face.shape
    frame[y:y+dy,x:x+dx] = face
    out.write(frame)
out.release()

