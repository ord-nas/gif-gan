import os
from ops import *
#import scipy.misc
import numpy as np
import sys

from model import DCGAN
from utils import pp
from utils import inverse_transform

import tensorflow as tf
import tensorflow.contrib.slim as slim

import math
import glob
import cv2 # For some reason this seems to make the initial variable
           # initialization take a long time

from z_model_lib import Layers
from z_model_lib import VID_DCGAN

flags = tf.app.flags
flags.DEFINE_integer("epoch", 25, "Epoch to train [25]")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
# flags.DEFINE_integer("train_size", np.inf, "The size of train images [np.inf]")
flags.DEFINE_integer("image_batch_size", 64, "The size of batch images [64]")
flags.DEFINE_integer("vid_batch_size", 64, "The size of batch images [64]")
flags.DEFINE_integer("vid_length", 16, "The length of the videos [16]")
flags.DEFINE_integer("image_size", 64, "The size of images used [64]")
flags.DEFINE_integer("output_size", 64, "The size of the output images to produce [64]")
flags.DEFINE_integer("c_dim", 3, "Dimension of image color. [3]")
# flags.DEFINE_string("image_dataset", "celebA", "The name of dataset [celebA, mnist, lsun]")
flags.DEFINE_string("image_model_dir", "checkpoint", "Directory name to load the image checkpoints [checkpoint]")
flags.DEFINE_string("video_checkpoint_dir", "checkpoint", "Directory name to save the video checkpoints [checkpoint]")
flags.DEFINE_string("video_sample_dir", "samples", "Directory name to save the video samples [samples]")
flags.DEFINE_string("video_data_dir", "./data", "Directory to read dataset from")
flags.DEFINE_string("video_dataset", "", "Name of video dataset to use")
# Yeah, yeah, this is pretty ugly, but whatever.
flags._global_parser.add_argument("--video_list", required=False, nargs='*', default=[], help="List(s) of videos to use")
flags.DEFINE_string("log_dir", "./logs", "Directory to write log files")
flags.DEFINE_boolean("is_train", False, "True for training, False for <not implemented yet> [False]")
# flags.DEFINE_boolean("is_crop", False, "True for training, False for testing [False]")
# flags.DEFINE_boolean("visualize", False, "True for visualizing, False for nothing [False]")
flags.DEFINE_boolean("video_shuffle", True, "True to shuffle the dataset, False otherwise [False]")
flags.DEFINE_boolean("train_img_gen", False, "True to make the image generator params trainable [False]")
flags.DEFINE_boolean("train_img_disc", False, "True to make the image discriminator params trainable [False]")
flags.DEFINE_integer("disc_updates", 1, "Number of discriminator updates per batch [1]")
flags.DEFINE_integer("gen_updates", 2, "Number of generator updates per batch [1]")
flags.DEFINE_float("image_noise", 0.0, "Std of noise to add to images")
flags.DEFINE_float("activation_noise", 0.0, "Std of noise to add to D activations")
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
            vid_dcgan = VID_DCGAN(sess,
                                  FLAGS.vid_batch_size,
                                  vid_z_dim,
                                  image_z_dim,
                                  FLAGS.vid_length,
                                  FLAGS.image_size,
                                  FLAGS.output_size,
                                  c_dim=FLAGS.c_dim,
                                  image_noise_std=FLAGS.image_noise,
                                  activation_noise_std=FLAGS.activation_noise
                                  first_frame_loss_scalar=FLAGS.first_frame_loss_scalar)
            print "DONE"

            # Init vars
            sess.run(tf.global_variables_initializer())

            # Load image model weights. This needs to happen *after* init, so
            # that we don't overwrite the weights we load.
            vid_dcgan.load_image_gan(sess, FLAGS.image_model_dir)

            # Generate some z-vectors for one video.
            sample_z = np.random.uniform(-1, 1, size=(FLAGS.vid_batch_size, vid_z_dim))
            out_val = sess.run(vid_dcgan.G, feed_dict={vid_dcgan.z:sample_z})
            print out_val.shape

            # Generate videos from the z-vectors
            imgs = sess.run(vid_dcgan.img_dcgan.sampler, feed_dict={vid_dcgan.img_dcgan.sample_z:out_val})
            vids = np.reshape(imgs, (-1, FLAGS.vid_length, FLAGS.output_size, FLAGS.output_size, FLAGS.c_dim))
            print imgs.shape
            print vids.shape

            # Test the discriminator
            print "FAKE DISC", sess.run(vid_dcgan.d_fake_out, feed_dict={vid_dcgan.img_dcgan.z:out_val})
            print "REAL DISC", sess.run(vid_dcgan.d_real_out, feed_dict={vid_dcgan.img_dcgan.images:imgs})

            # vid_dcgan.dump_sample(sample_z, sess, FLAGS, 4, 20)
            # return
            
            vid_dcgan.train(sess, FLAGS)

            # # Write the videos out to file
            # print "OK OPENCV"
            # for i in xrange(vids.shape[0]):
            #     filename = "/thesis0/yccggrp/youngsan/tmp2/video_%05d.mp4" % i
            #     # filename = "/thesis0/yccggrp/youngsan/tmp/video_%05d" % i
            #     print "Writing to", filename
            #     w = cv2.VideoWriter(filename,
            #                         0x20,
            #                         25.0,
            #                         (FLAGS.output_size, FLAGS.output_size))
            #     for j in xrange(vids.shape[1]):
            #         im = inverse_transform(vids[i][j])
            #         im = np.around(im * 255).astype('uint8')
            #         im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
            #         w.write(im)
            #         #cv2.imwrite(filename + ("_%02d" % j) + ".png", im)
            #     w.release()
            # print "DONE WRITING FILES"

if __name__ == '__main__':
    tf.app.run()

