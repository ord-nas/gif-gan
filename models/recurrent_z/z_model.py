import os
from ops import *
#import scipy.misc
import numpy as np

from model import DCGAN
from utils import pp

import tensorflow as tf
import tensorflow.contrib.slim as slim

import math

flags = tf.app.flags
# flags.DEFINE_integer("epoch", 25, "Epoch to train [25]")
# flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
# flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
# flags.DEFINE_integer("train_size", np.inf, "The size of train images [np.inf]")
flags.DEFINE_integer("image_batch_size", 64, "The size of batch images [64]")
flags.DEFINE_integer("vid_batch_size", 64, "The size of batch images [64]")
flags.DEFINE_integer("vid_length", 16, "The length of the videos [16]")
flags.DEFINE_integer("image_size", 108, "The size of image to use (will be center cropped) [108]")
flags.DEFINE_integer("output_size", 64, "The size of the output images to produce [64]")
flags.DEFINE_integer("c_dim", 3, "Dimension of image color. [3]")
flags.DEFINE_string("image_dataset", "celebA", "The name of dataset [celebA, mnist, lsun]")
flags.DEFINE_string("image_checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
# flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
# flags.DEFINE_string("data_dir", "./data", "Directory to read dataset from")
# flags.DEFINE_string("log_dir", "./logs", "Directory to write log files")
# flags.DEFINE_string("image_glob", "*.jpg", "Glob to use to find images in the dataset directory")
# flags.DEFINE_boolean("is_train", False, "True for training, False for testing [False]")
# flags.DEFINE_boolean("is_crop", False, "True for training, False for testing [False]")
# flags.DEFINE_boolean("visualize", False, "True for visualizing, False for nothing [False]")
# flags.DEFINE_boolean("shuffle", False, "True to shuffle the dataset, False otherwise [False]")
FLAGS = flags.FLAGS

class VID_DCGAN(object):
    def __init__(self, batch_size, output_size, vid_length):
        self.g_bn0 = batch_norm(name='gvideo_bn0')
        self.g_bn1 = batch_norm(name='gvideo_bn1')
        self.g_bn2 = batch_norm(name='gvideo_bn2')
        self.g_bn3 = batch_norm(name='gvideo_bn3')
        self.batch_size = batch_size
        self.output_size = output_size
        self.vid_length = vid_length
    def generator(self, z):
        print "z:", z.get_shape().as_list()
        f = self.output_size
        f, f2, f4, f8, f16 = [int(f*x) for x in np.logspace(math.log10(1),
                                                            math.log10(2),
                                                            5)]#[2,4,8,16]]
        s = self.vid_length
        s_power_2 = 1<<(s-1).bit_length()
        s2, s4, s8, s16 = [int(s_power_2/x) for x in [2,4,8,16]]
        k_w = 3

        self.z_project, self.g0_w, self.g0_b = linear(
            z, s16*f16, 'gvideo_0', with_w=True)
        print "z_proj:", self.z_project.get_shape().as_list()

        self.g0 = tf.reshape(self.z_project, [-1, 1, s16, f16])
        self.r0 = tf.nn.relu(self.g_bn0(self.g0))
        print "r0:", self.r0.get_shape().as_list()

        self.g1, self.g1_w, self.g1_b = deconv2d(
            self.r0, [self.batch_size, 1, s8, f8],
            d_h=1, k_h=1, k_w=k_w, name='gvideo_1', with_w=True)
        self.r1 = tf.nn.relu(self.g_bn1(self.g1))
        print "r1:", self.r1.get_shape().as_list()
        print "filter:", self.g1_w.get_shape().as_list()

        self.g2, self.g2_w, self.g2_b = deconv2d(
            self.r1, [self.batch_size, 1, s4, f4],
            d_h=1, k_h=1, k_w=k_w, name='gvideo_2', with_w=True)
        self.r2 = tf.nn.relu(self.g_bn2(self.g2))
        print "r2:", self.r2.get_shape().as_list()
        print "filter:", self.g2_w.get_shape().as_list()
        
        self.g3, self.g3_w, self.g3_b = deconv2d(
            self.r2, [self.batch_size, 1, s2, f2],
            d_h=1, k_h=1, k_w=k_w, name='gvideo_3', with_w=True)
        self.r3 = tf.nn.relu(self.g_bn3(self.g3))
        print "r3:", self.r3.get_shape().as_list()
        print "filter:", self.g3_w.get_shape().as_list()

        self.g4, self.g4_w, self.g4_b = deconv2d(
            self.r3, [self.batch_size, 1, s, f],
            d_h=1, k_h=1, k_w=k_w, name='gvideo_4', with_w=True)
        self.r4 = tf.nn.tanh(self.g4)
        print "r4:", self.r4.get_shape().as_list()
        print "filter:", self.g4_w.get_shape().as_list()
        
        self.r4_2d = tf.reshape(self.r4, [-1, s, f])
        print "r4_2d:", self.r4_2d.get_shape().as_list()
        self.r4_1d = tf.reshape(self.r4, [-1, f])
        print "r4_1d:", self.r4_1d.get_shape().as_list()

        return self.r4_1d


def main(_):
    pp.pprint(flags.FLAGS.__flags)

    with tf.Session() as sess:
        with tf.variable_scope('image'):
            dcgan = DCGAN(sess, image_size=FLAGS.image_size,
                          batch_size=FLAGS.image_batch_size * FLAGS.vid_length,
                          output_size=FLAGS.output_size, c_dim=FLAGS.c_dim,
                          dataset_name=FLAGS.image_dataset, is_crop=False,
                          checkpoint_dir=FLAGS.image_checkpoint_dir, sample_dir='',
                          data_dir='', log_dir='', image_glob='', shuffle=False)

            model_dir = "%s_%s_%s" % (FLAGS.image_dataset, FLAGS.image_batch_size, FLAGS.output_size)
            checkpoint_dir = os.path.join(FLAGS.image_checkpoint_dir, model_dir)
            print "Loading checkpoints from", checkpoint_dir

            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
                d = {}
                for v in tf.global_variables():
                    n = v.op.name
                    prefix = tf.get_variable_scope().name + "/"
                    assert n.startswith(prefix)
                    d[n[len(prefix):]] = v
                saver = tf.train.Saver(var_list=d)
                saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
                print "Success!"
            else:
                print "FAIL!"

        # print "DVARS", dcgan.d_vars
        # print "GVARS", dcgan.g_vars

        with tf.variable_scope('video'):
            vid_z_dim = 120
            image_z_dim = 100

            vid_dcgan = VID_DCGAN(FLAGS.vid_batch_size,
                                  image_z_dim,
                                  FLAGS.vid_length)

            z = tf.placeholder(tf.float32, [FLAGS.vid_batch_size, vid_z_dim], name='vid_z')
            out = vid_dcgan.generator(z)

            print "DONE"


            print sess.run(tf.report_uninitialized_variables())
            un_init = [v for v in tf.global_variables() if not sess.run(tf.is_variable_initialized(v))]
            sess.run(tf.variables_initializer(un_init))
            print sess.run(tf.report_uninitialized_variables())

            sample_z = np.random.uniform(-1, 1, size=(FLAGS.vid_batch_size, vid_z_dim))
            out_val = sess.run(out, feed_dict={z:sample_z})
            print out_val.shape
            
            import cv2

            print "OK OPENCV"

if __name__ == '__main__':
    tf.app.run()

