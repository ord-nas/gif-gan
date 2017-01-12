import cv2
import random
import argparse
import os
import sample_frames
import tensorflow as tf
import numpy as np
from utils import transform, save_images
from model import DCGAN

# Params for algorithm
parser = argparse.ArgumentParser()
parser.add_argument("--video_list", required=True, nargs='+',
                    help="One or more text files containing names of .mp4 files")
parser.add_argument("--input_directory", required=True, help="Base input directory")
parser.add_argument("--intra_output", default="intra_video_activation_distances.txt")
parser.add_argument("--inter_output", default="inter_video_activation_distances.txt")
parser.add_argument("--random_seed", type=int, default=0)
parser.add_argument("--samples_per_video", type=int, default=8)
parser.add_argument("--videos_per_batch", type=int, default=8)
parser.add_argument("--num_batches", type=int, default=1)
parser.add_argument("--discriminator_mode", required=True, choices=["train", "inference"])
parser.add_argument("--verbosity", type=int, default=0)
parser.add_argument("--save_samples", dest="save_samples", action="store_true")
parser.add_argument("--no_save_samples", dest="save_samples", action="store_false")
parser.set_defaults(save_samples=False)
# DCGAN params
parser.add_argument("--checkpoint_directory", required=True, help="Directory to load checkpoint files from")
parser.add_argument("--image_size", type=int, default=64, help="Size of images used")
parser.add_argument("--output_size", type=int, default=64, help="Size of output images")
parser.add_argument("--c_dim", type=int, default=3, help="Dimension of image colour")

def load_dcgan(sess, args):
    dcgan = DCGAN(sess,
                  image_size=args.image_size,
                  batch_size=(args.samples_per_video * args.videos_per_batch),
                  output_size=args.output_size,
                  c_dim=args.c_dim,
                  dataset_name='',
                  is_crop=False,
                  checkpoint_dir='',
                  sample_dir='',
                  data_dir='',
                  log_dir='',
                  image_glob='',
                  shuffle=False)
    ckpt = tf.train.get_checkpoint_state(args.checkpoint_directory)
    assert ckpt and ckpt.model_checkpoint_path
    ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
    dcgan.saver.restore(sess, os.path.join(args.checkpoint_directory, ckpt_name))
    return dcgan

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
    sess = tf.Session()
    dcgan = load_dcgan(sess, args)

    # Load the list of video files
    random.seed(args.random_seed)
    files = []
    for lst in args.video_list:
        with open(lst, 'r') as f:
            for video in f:
                video = video.strip()
                if not video:
                    continue
                video = os.path.join(args.input_directory, video)
                files.append(video)
    random.shuffle(files)

    # Actually run the batches
    batch_size = args.samples_per_video * args.videos_per_batch
    n = args.videos_per_batch
    inter = []
    intra = []
    for i in xrange(args.num_batches):
        batch_files = files[i*n:(i+1)*n]
        batch_frames = get_frames(batch_files, args)
        train = (args.discriminator_mode == "train")
        activations_tensor = dcgan.D_activations if train else dcgan.D_activations_inf
        activations = sess.run(activations_tensor, feed_dict={
            dcgan.images: np.array(batch_frames),
        })
        assert activations.shape[0] == batch_size
        activations = np.reshape(activations, [args.videos_per_batch, args.samples_per_video, -1])
        samples = np.reshape(batch_frames, [args.videos_per_batch,
                                            args.samples_per_video,
                                            args.image_size,
                                            args.image_size,
                                            args.c_dim])

        if args.save_samples:
            for vid in xrange(args.videos_per_batch):
                save_images(samples[vid], [1, args.samples_per_video],
                            "batch_%d_video_%d.png" % (i, vid))

        # First compute intra_video data
        intra_batch = []
        for v in xrange(args.videos_per_batch):
            intra_video = []
            vid = samples[v]
            a = activations[v]
            for j in xrange(args.samples_per_video):
                # Skip frames that aren't unique
                if j > 0 and np.allclose(vid[j], vid[j-1]):
                    continue
                for k in xrange(j+1, args.samples_per_video):
                    # Skip frames that aren't unique
                    if k > 0 and np.allclose(vid[k], vid[k-1]):
                        continue
                    distance = np.sqrt(np.sum(np.square(a[j] - a[k])))
                    intra_video.append(distance)
            if args.verbosity >= 2:
                print "INTRA_VIDEO %s" % batch_files[v]
                for a in intra_video:
                    print a
            intra_batch.extend(intra_video)
        if args.verbosity >= 1:
            print "INTRA_BATCH %s" % batch_files
            for a in intra_batch:
                print a
        intra.extend(intra_batch)

        inter_batch = []
        for v1 in xrange(args.videos_per_batch):
            vid1 = samples[v1]
            a1 = activations[v1]
            for v2 in xrange(v1+1, args.videos_per_batch):
                vid2 = samples[v2]
                a2 = activations[v2]
                inter_video = []
                for j in xrange(args.samples_per_video):
                    # Skip frames that aren't unique
                    if j > 0 and np.allclose(vid1[j], vid1[j-1]):
                        continue
                    for k in xrange(args.samples_per_video):
                        # Skip frames that aren't unique
                        if k > 0 and np.allclose(vid2[k], vid2[k-1]):
                            continue
                        distance = np.sqrt(np.sum(np.square(a1[j] - a2[k])))
                        inter_video.append(distance)
                if args.verbosity >= 2:
                    print "INTER_VIDEO %s & %s" % (batch_files[v1], batch_files[v2])
                    for a in inter_video:
                        print a
                inter_batch.extend(inter_video)
        if args.verbosity >= 1:
            print "INTER_BATCH %s" % batch_files
            for a in inter_batch:
                print a
        inter.extend(inter_batch)

    # Final output
    with open(args.intra_output, 'w') as f:
        for a in intra:
            f.write("%f\n" % a)
    with open(args.inter_output, 'w') as f:
        for a in inter:
            f.write("%f\n" % a)

if __name__ == "__main__":
    main()
