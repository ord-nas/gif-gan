import os
from ops import *
#import scipy.misc
import numpy as np

from model import DCGAN
from utils import pp
from utils import inverse_transform

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
        self.d_bn0 = batch_norm(name='dvideo_bn0')
        self.d_bn1 = batch_norm(name='dvideo_bn1')
        self.d_bn2 = batch_norm(name='dvideo_bn2')
        self.d_bn3 = batch_norm(name='dvideo_bn3')
        self.d_bn4 = batch_norm(name='dvideo_bn4')
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
        self.gr0 = tf.nn.relu(self.g_bn0(self.g0))
        print "gr0:", self.gr0.get_shape().as_list()

        self.g1, self.g1_w, self.g1_b = deconv2d(
            self.gr0, [self.batch_size, 1, s8, f8],
            d_h=1, k_h=1, k_w=k_w, name='gvideo_1', with_w=True)
        self.gr1 = tf.nn.relu(self.g_bn1(self.g1))
        print "gr1:", self.gr1.get_shape().as_list()
        print "filter:", self.g1_w.get_shape().as_list()

        self.g2, self.g2_w, self.g2_b = deconv2d(
            self.gr1, [self.batch_size, 1, s4, f4],
            d_h=1, k_h=1, k_w=k_w, name='gvideo_2', with_w=True)
        self.gr2 = tf.nn.relu(self.g_bn2(self.g2))
        print "gr2:", self.gr2.get_shape().as_list()
        print "filter:", self.g2_w.get_shape().as_list()
        
        self.g3, self.g3_w, self.g3_b = deconv2d(
            self.gr2, [self.batch_size, 1, s2, f2],
            d_h=1, k_h=1, k_w=k_w, name='gvideo_3', with_w=True)
        self.gr3 = tf.nn.relu(self.g_bn3(self.g3))
        print "gr3:", self.gr3.get_shape().as_list()
        print "filter:", self.g3_w.get_shape().as_list()

        self.g4, self.g4_w, self.g4_b = deconv2d(
            self.gr3, [self.batch_size, 1, s, f],
            d_h=1, k_h=1, k_w=k_w, name='gvideo_4', with_w=True)
        self.gr4 = tf.nn.tanh(self.g4)
        print "gr4:", self.gr4.get_shape().as_list()
        print "filter:", self.g4_w.get_shape().as_list()
        
        self.gr4_2d = tf.reshape(self.gr4, [-1, s, f])
        print "gr4_2d:", self.gr4_2d.get_shape().as_list()
        self.gr4_1d = tf.reshape(self.gr4, [-1, f])
        print "gr4_1d:", self.gr4_1d.get_shape().as_list()

        return self.gr4_1d
    def discriminator(self, vid, reuse=False):
        # First we wanna just project into a reasonable shape.
        # Squeeze all of the per-frame stuff togther
        print "vid:", vid.get_shape().as_list()
        vid = tf.reshape(vid, [self.batch_size * self.vid_length, -1])
        print "vid (reshaped):", vid.get_shape().as_list()

        f = self.output_size # No reason for it to be this number other than symmetry
        f, f2, f4, f8, f16 = [int(f*x) for x in np.logspace(math.log10(1),
                                                            math.log10(2),
                                                            5)]
        k_w = 3
        
        self.vid_project, self.d0_w, self.d0_b = linear(
            vid, f, 'dvideo_0', with_w=True)
        print "vid_project:", self.vid_project.get_shape().as_list()

        self.d0 = tf.reshape(self.vid_project, [self.batch_size, 1, self.vid_length, -1])
        self.dr0 = tf.nn.relu(self.d_bn0(self.d0))
        print "dr0:", self.dr0.get_shape().as_list()

        self.dr1 = lrelu(self.d_bn1(conv2d(self.dr0, f2, k_h=1, k_w=k_w, d_h=1, name='dvideo_h1')))
        print "dr1:", self.dr1.get_shape().as_list()
        self.dr2 = lrelu(self.d_bn2(conv2d(self.dr1, f4, k_h=1, k_w=k_w, d_h=1, name='dvideo_h2')))
        print "dr2:", self.dr2.get_shape().as_list()
        self.dr3 = lrelu(self.d_bn3(conv2d(self.dr2, f8, k_h=1, k_w=k_w, d_h=1, name='dvideo_h3')))
        print "dr3:", self.dr3.get_shape().as_list()
        self.dr4 = lrelu(self.d_bn4(conv2d(self.dr3, f16, k_h=1, k_w=k_w, d_h=1, name='dvideo_h4')))
        print "dr4:", self.dr4.get_shape().as_list()
        self.d5 = linear(tf.reshape(self.dr4, [self.batch_size, -1]), 1, 'dvideo_h5')
        print "d5:", self.d5.get_shape().as_list()

        return tf.nn.sigmoid(self.d5), self.d5


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

        # for n in tf.get_default_graph().as_graph_def().node:
        #     include = False
        #     for i in [0,1,2,3]:
        #         if ("image/d_h%d_conv/Conv2D" % i) in n.name:
        #             include = True
        #     if include:
        #         print n.name, tf.get_default_graph().get_tensor_by_name(n.name + ":0").get_shape().as_list()
        #pp.pprint([n.name for n in tf.get_default_graph().as_graph_def().node])
        #return

        # print "DVARS", dcgan.d_vars
        # print "GVARS", dcgan.g_vars
        import cv2
        print "OK OPENCV"

        with tf.variable_scope('video'):
            vid_z_dim = 120
            image_z_dim = 100

            vid_dcgan = VID_DCGAN(FLAGS.vid_batch_size,
                                  image_z_dim,
                                  FLAGS.vid_length)

            z = tf.placeholder(tf.float32, [FLAGS.vid_batch_size, vid_z_dim], name='vid_z')
            out = vid_dcgan.generator(z)

            d_penultimate_layer_name = "image/d_h3_conv/Conv2D:0"
            d_tensor = tf.get_default_graph().get_tensor_by_name(d_penultimate_layer_name)
            # print d_tensor.get_shape().as_list()
            # vid_d_input = tf.reshape(d_tensor, [FLAGS.vid_batch_size, FLAGS.vid_length, -1])
            # print vid_d_input.get_shape().as_list()

            d_out = vid_dcgan.discriminator(d_tensor, reuse=False)
        
            print "DONE"
            return

            print sess.run(tf.report_uninitialized_variables())
            un_init = [v for v in tf.global_variables() if not sess.run(tf.is_variable_initialized(v))]
            sess.run(tf.variables_initializer(un_init))
            print sess.run(tf.report_uninitialized_variables())

            sample_z = np.random.uniform(-1, 1, size=(FLAGS.vid_batch_size, vid_z_dim))
            out_val = sess.run(out, feed_dict={z:sample_z})
            print out_val.shape

            imgs = sess.run(dcgan.sampler, feed_dict={dcgan.z:out_val})
            vids = np.reshape(imgs, (-1, FLAGS.vid_length, FLAGS.output_size, FLAGS.output_size, FLAGS.c_dim))
            print imgs.shape
            print vids.shape
            
            # for i in xrange(vids.shape[0]):
            #     filename = "/thesis0/yccggrp/youngsan/tmp/video_%05d.mp4" % i
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
            #     w.release()
            # print "DONE WRITING FILES"

if __name__ == '__main__':
    tf.app.run()

