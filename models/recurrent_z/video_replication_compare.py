import cv2
import random
import argparse
import os
import numpy as np
import itertools

# Params for algorithm
parser = argparse.ArgumentParser()
parser.add_argument("--real_input_files", required=True, nargs='*', default=[], help="Real input files")
parser.add_argument("--replicated_input_file", required=True, help="Replicated input file")
parser.add_argument("--output_file", required=True, help="Output file")
parser.add_argument("--grid_cell_size", default=64, help="Grid cell size")
parser.add_argument("--output_size", default=256, help="Output size")
parser.add_argument("--frame_rate", type=int, default=25, help="Output frame rate")
parser.add_argument("--vid_length", type=int, default=16, help="Video length")

def main():
    args = parser.parse_args()

    real = []
    for f in args.real_input_files:
        cap = cv2.VideoCapture(f)
        frame = 0
        while cap.isOpened() and frame < args.vid_length:
            ret, im = cap.read()
            if not ret:
                break
            real.append(im)
            frame += 1

    fake = [list() for _ in xrange(len(args.real_input_files))]
    cap = cv2.VideoCapture(args.replicated_input_file)
    while cap.isOpened():
        ret, im = cap.read()
        if not ret:
            break
        for i in xrange(len(args.real_input_files)):
            fake[i].append(im[:args.grid_cell_size,i*args.grid_cell_size:(i+1)*args.grid_cell_size,:])
    fake = list(itertools.chain(*fake))

    frame_size = (args.output_size*2, args.output_size)
    writer = cv2.VideoWriter(args.output_file, 0x20, args.frame_rate, frame_size)
    for (r, f) in zip(real, fake):
        frame = np.zeros((args.output_size, 2*args.output_size, 3), dtype=np.uint8)
        real_small = cv2.resize(r, (args.grid_cell_size, args.grid_cell_size), interpolation=cv2.INTER_LINEAR)
        real_large = cv2.resize(real_small, (args.output_size, args.output_size), interpolation=cv2.INTER_LINEAR)
        fake_large = cv2.resize(f, (args.output_size,args.output_size), interpolation=cv2.INTER_LINEAR)
        frame[:, :args.output_size, :] = fake_large
        frame[:, args.output_size:, :] = real_large
        writer.write(frame)
    writer.release()

if __name__ == "__main__":
    main()
