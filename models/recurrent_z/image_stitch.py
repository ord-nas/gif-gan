import cv2
import random
import argparse
import os
import numpy as np

# Params for algorithm
parser = argparse.ArgumentParser()
parser.add_argument("--input_file", required=True, help="Input file")
parser.add_argument("--output_file", required=True, help="Output file")
parser.add_argument("--image_size", default=64, help="Image size")
parser.add_argument("--video_length", default=16, help="Video length (frames)")
parser.add_argument("--random_seed", default=0, help="Random seed")

def get_frames(files, config):
    frames = []
    for f in files:
        frames.extend(sample_frames.sample_frames_from_video(f, config.samples_per_video))
    assert len(frames) == (len(files) * config.samples_per_video)
    def preprocess(frame):
        frame_size = (config.image_size, config.image_size)
        frame = cv2.resize(frame, frame_size, interpolation=cv2.INTER_LINEAR)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = transform(frame, is_crop=False)
        return frame
    frames = map(preprocess, frames)
    return np.array(frames)

def main():
    args = parser.parse_args()

    size = args.image_size
    im = cv2.imread(args.input_file)
    writer = cv2.VideoWriter(args.output_file, 0x20, 25.0, (size, size))
    rows, cols, _ = im.shape
    rows = rows / size
    cols = cols / size

    choices = random.sample([(r, c) for c in xrange(cols) for r in xrange(rows)],
                            args.video_length)

    for (r, c) in choices:
        writer.write(im[r*size:(r+1)*size,c*size:(c+1)*size,:])
    writer.release()

if __name__ == "__main__":
    main()
