import os
from ops import *
import numpy as np
import sys

from model import DCGAN
from utils import pp
from utils import inverse_transform

import tensorflow as tf
import tensorflow.contrib.slim as slim

import math
import glob
import cv2

from z_model_approx_lib import Layers
from z_model_approx_lib import PATH_DCGAN

flags = tf.app.flags
flags.DEFINE_integer("epoch", 25, "Epoch to train [25]")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_integer("vid_batch_size", 64, "The size of batch images [64]")
flags.DEFINE_integer("vid_length", 16, "The length of the videos [16]")
flags.DEFINE_integer("image_size", 64, "The size of images used [64]")
flags.DEFINE_integer("output_size", 64, "The size of the output images to produce [64]")
flags.DEFINE_integer("c_dim", 3, "Dimension of image color. [3]")
flags.DEFINE_string("image_model_dir", "checkpoint", "Directory name to load the image checkpoints [checkpoint]")
flags.DEFINE_string("video_checkpoint_dir", "checkpoint", "Directory name to save the video checkpoints [checkpoint]")
flags.DEFINE_string("video_sample_dir", "samples", "Directory name to save the video samples [samples]")
flags.DEFINE_string("path_data_dir", "./data", "Directory to read dataset from")
# Yeah, yeah, this is pretty ugly, but whatever.
flags._global_parser.add_argument("--path_list", required=False, nargs='*', default=[], help="List(s) of z paths to use")
flags.DEFINE_string("log_dir", "./logs", "Directory to write log files")
flags.DEFINE_boolean("path_shuffle", True, "True to shuffle the dataset, False otherwise [False]")
flags.DEFINE_integer("disc_updates", 1, "Number of discriminator updates per batch [1]")
flags.DEFINE_integer("gen_updates", 1, "Number of generator updates per batch [1]")
flags.DEFINE_float("first_frame_loss_scalar", 0.0, "first_frame_loss_scalar")
# Flags for controlling checkpoint saving behaviour
flags.DEFINE_integer("sample_frequency", 10, "How often to save checkpoints & samples")
flags.DEFINE_integer("max_checkpoints_to_keep", 5, "Max number of checkpoints to keep")
FLAGS = flags.FLAGS


def main(_):
    pp.pprint(flags.FLAGS.__flags)

    with tf.Session() as sess:
        with tf.variable_scope('video_gan'):
            vid_z_dim = 120
            image_z_dim = 100

            # Build the model
            vid_dcgan = PATH_DCGAN(sess,
                                   FLAGS.vid_batch_size,
                                   vid_z_dim,
                                   image_z_dim,
                                   FLAGS.vid_length,
                                   FLAGS.image_size,
                                   FLAGS.output_size,
                                   c_dim=FLAGS.c_dim,
                                   first_frame_loss_scalar=FLAGS.first_frame_loss_scalar)
            print "DONE"

            # Init vars
            sess.run(tf.global_variables_initializer())

            # Load image model weights. This needs to happen *after* init, so
            # that we don't overwrite the weights we load.
            vid_dcgan.load_image_gan(sess, FLAGS.image_model_dir)
            
            vid_dcgan.train(sess, FLAGS)

if __name__ == '__main__':
    tf.app.run()

