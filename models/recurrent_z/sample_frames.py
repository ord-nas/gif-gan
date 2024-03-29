import cv2
import random
import argparse
import os

# Params for algorithm
parser = argparse.ArgumentParser()
parser.add_argument("--video_list", required=True, nargs='+',
                    help="One or more text files containing names of .mp4 files")
parser.add_argument("--input_directory", required=True,
                    help="Base input directory")
parser.add_argument("--output_directory", required=True,
                    help="Directory to place output")
parser.add_argument("--samples_per_video", type=int, default=1)
parser.add_argument("--random_seed", type=int, default=0)
parser.add_argument("--output_size", type=int, default=64)


def sample_frames_from_video(video, num_samples):
    cap = cv2.VideoCapture(video)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if num_frames <= 0:
        raise Exception("Can't read number of frames from video: %s" % video)
    num_samples = min(num_samples, num_frames)
    choices = random.sample(xrange(num_frames), num_samples)
    frames = []
    frame_number = 0
    while cap.isOpened():
        ret, im = cap.read()
        if not ret:
            break
        if frame_number in choices:
            frames.append(im)
        frame_number += 1
    if frame_number != num_frames:
        raise Exception("Read wrong number of frames from video: %s" % video)
    return frames

def sample_frames(video_list, input_directory, output_directory, samples_per_video, size):
    i = 0
    for lst in video_list:
        with open(lst, 'r') as f:
            for video in f:
                video = video.strip()
                if not video:
                    continue
                video = os.path.join(input_directory, video)
                frames = sample_frames_from_video(
                    video, samples_per_video)
                for frame in frames:
                    output_name = os.path.join(output_directory,
                                               ("%07d.png" % i))
                    frame = cv2.resize(frame, (size, size), interpolation=cv2.INTER_LINEAR)
                    cv2.imwrite(output_name, frame)
                    i += 1
                    if i % 1000 == 0:
                        print "Created %d images sofar ..." % i

def main():
    args = parser.parse_args()

    random.seed(args.random_seed)

    if not os.path.exists(args.output_directory):
        os.makedirs(args.output_directory)
    
    sample_frames(args.video_list,
                  args.input_directory,
                  args.output_directory,
                  args.samples_per_video,
                  args.output_size)

if __name__ == "__main__":
    main()
