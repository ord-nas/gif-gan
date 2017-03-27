import cv2
import argparse
import os
import tensorflow as tf
import numpy as np
from utils import transform, save_images, get_images, inverse_transform
from model import DCGAN

# Params for algorithm
parser = argparse.ArgumentParser()
# Basic IO
parser.add_argument("--z_file", required=True, help="Z-file to recreate")
parser.add_argument("--output_filename", required=True, help="Filename to write to")
# DCGAN params
parser.add_argument("--checkpoint_directory", required=True, help="Directory to load checkpoint files from")
parser.add_argument("--image_size", type=int, default=64, help="Size of images used")
parser.add_argument("--output_size", type=int, default=64, help="Size of output images")
parser.add_argument("--c_dim", type=int, default=3, help="Dimension of image colour")

def load_dcgan(sess, args, batch_size):
    z = tf.get_variable('z', [batch_size, 100], initializer=tf.random_uniform_initializer(
        minval=-1.0, maxval=1.0))
    sess.run(tf.initialize_all_variables())
    dcgan = DCGAN(sess,
                  image_size=args.image_size,
                  batch_size=(batch_size),
                  output_size=args.output_size,
                  c_dim=args.c_dim,
                  dataset_name='',
                  is_crop=False,
                  checkpoint_dir='',
                  sample_dir='',
                  data_dir='',
                  log_dir='',
                  image_glob='',
                  shuffle=False,
                  z=z)
    ckpt = tf.train.get_checkpoint_state(args.checkpoint_directory)
    assert ckpt and ckpt.model_checkpoint_path
    ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
    saver = tf.train.Saver([v for v in tf.all_variables() if v is not z])
    saver.restore(sess, os.path.join(args.checkpoint_directory, ckpt_name))
    return dcgan

def prep_for_mp4(im, scaled_size):
    im = inverse_transform(im)
    im = np.around(im * 255).astype('uint8')
    im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
    im = cv2.resize(im, scaled_size, interpolation=cv2.INTER_LINEAR)
    return im

def main():
    # Initialization
    global args
    args = parser.parse_args()
    sess = tf.Session()
    zs = np.load(args.z_file)
    batch_size = zs.shape[0]
    dcgan = load_dcgan(sess, args, batch_size)

    # Find key nodes in graph
    scale_factor = 2
    scaled_size = (scale_factor * args.image_size, scale_factor * args.image_size)
    height, width = scaled_size[0], scaled_size[1]
    w = cv2.VideoWriter(args.output_filename, 0x20, 25.0, (width, height))
    imgs = sess.run(dcgan.sampler, feed_dict={
        dcgan.z: zs,
    })
    for img in imgs:
        img = prep_for_mp4(img, scaled_size)
        w.write(img)
    w.release()

if __name__ == "__main__":
    main()
