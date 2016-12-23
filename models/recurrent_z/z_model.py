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
import glob

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
flags.DEFINE_string("image_dataset", "celebA", "The name of dataset [celebA, mnist, lsun]")
flags.DEFINE_string("image_checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
# flags.DEFINE_string("video_checkpoint_dir", "checkpoint", "Directory name to save the video checkpoints [checkpoint]")
flags.DEFINE_string("video_sample_dir", "samples", "Directory name to save the video samples [samples]")
flags.DEFINE_string("video_data_dir", "./data", "Directory to read dataset from")
flags.DEFINE_string("video_dataset", "", "Name of video dataset to use")
# Yeah, yeah, this is pretty ugly, but whatever.
flags._global_parser.add_argument("--video_list", required=True, nargs='+', help="List(s) of videos to use")
flags.DEFINE_string("log_dir", "./logs", "Directory to write log files")
flags.DEFINE_boolean("is_train", False, "True for training, False for <not implemented yet> [False]")
# flags.DEFINE_boolean("is_crop", False, "True for training, False for testing [False]")
# flags.DEFINE_boolean("visualize", False, "True for visualizing, False for nothing [False]")
flags.DEFINE_boolean("video_shuffle", True, "True to shuffle the dataset, False otherwise [False]")
flags.DEFINE_boolean("train_img_gen", False, "True to make the image generator params trainable [False]")
flags.DEFINE_boolean("train_img_disc", False, "True to make the image discriminator params trainable [False]")
FLAGS = flags.FLAGS

class Layers(object):
    pass

class VID_DCGAN(object):
    def __init__(self, batch_size, z_input_size, z_output_size, vid_length,
                 real_img_discriminator, fake_img_discriminator,
                 sample_rows=8, sample_cols=8):
        # Member vars
        self.batch_size = batch_size
        self.z_input_size = z_input_size
        self.z_output_size = z_output_size
        self.vid_length = vid_length
        self.real_img_discriminator = real_img_discriminator
        self.fake_img_discriminator = fake_img_discriminator
        self.sample_rows = sample_rows
        self.sample_cols = sample_cols

        # Batch norm layers
        self.g_bn0 = batch_norm(name='gvideo_bn0')
        self.g_bn1 = batch_norm(name='gvideo_bn1')
        self.g_bn2 = batch_norm(name='gvideo_bn2')
        self.g_bn3 = batch_norm(name='gvideo_bn3')
        self.d_bn0 = batch_norm(name='dvideo_bn0')
        self.d_bn1 = batch_norm(name='dvideo_bn1')
        self.d_bn2 = batch_norm(name='dvideo_bn2')
        self.d_bn3 = batch_norm(name='dvideo_bn3')
        self.d_bn4 = batch_norm(name='dvideo_bn4')

        # Actually construct the model
        self.build_model()

    def build_model(self):
        # Build generator
        self.z = tf.placeholder(tf.float32, [self.batch_size, self.z_input_size],
                                name='gvideo_z')
        with tf.variable_scope('video_generator'):
            print "Making generator..."
            self.G, self.G_layers = self.generator(self.z, reuse=False, train=True)
            print "Making sampler..."
            self.G_sampler, self.G_sampler_layers = self.generator(self.z, reuse=True, train=False)

        # Build discriminator
        with tf.variable_scope('video_discriminator'):
            print "Scope name:", tf.get_variable_scope().name
            print "Making first discriminator..."
            self.d_real_out, self.d_real_out_logits, self.D_real_layers = self.discriminator(
                self.real_img_discriminator, reuse=False)
            print "Making second discriminator..."
            self.d_fake_out, self.d_fake_out_logits, self.D_fake_layers = self.discriminator(
                self.fake_img_discriminator, reuse=True)

        # Define trainable variables
        t_vars = tf.trainable_variables()
        self.d_vid_vars = [var for var in t_vars if 'dvideo_' in var.name]
        self.g_vid_vars = [var for var in t_vars if 'gvideo_' in var.name]
        self.d_img_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_img_vars = [var for var in t_vars if 'g_' in var.name]

        # Define loss
        self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            self.d_real_out_logits, tf.ones_like(self.d_real_out)))
        self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            self.d_fake_out_logits, tf.zeros_like(self.d_fake_out)))
        self.d_loss = self.d_loss_real + self.d_loss_fake
        self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            self.d_fake_out_logits, tf.ones_like(self.d_fake_out)))

    def train(self, config):
        files = []
        for lst in config.video_list:
            with open(lst, 'r') as f:
                for video in f:
                    video = video.strip()
                    if not video:
                        continue
                    video = os.path.join(config.video_data_dir,
                                         config.video_dataset,
                                         video)
                    files.append(video)

        print "Total video files found:", len(files)
        if config.video_shuffle:
            np.random.shuffle(files)

        # Create optimizers
        d_vars = self.d_vid_vars
        if config.train_img_disc:
            d_vars = d_vars + self.d_img_vars
        d_optim = (tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1)
                   .minimize(self.d_loss, var_list=d_vars))

        g_vars = self.g_vid_vars
        if config.train_img_gen:
            g_vars = g_vars + self.g_img_vars
        for v in g_vars:
            print v.name
        g_optim = (tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1)
                   .minimize(self.g_loss, var_list=g_vars))

        sample_z = np.random.uniform(-1, 1, size=(self.sample_rows * self.sample_cols,
                                                  self.z_input_size))

        batch_size = self.batch_size
        for epoch in xrange(config.epoch):
            for i in xrange(0, len(files) // batch_size):
                batch_files = files[i*batch_size:(i+1)*batch_size]
                batch_data = self.load_videos(batch_files)
                return

    def load_videos(self, files):
        n = len(files)
        videos = np.zeros(n, self.vid_length, self.output_size, self.output_size, self.c_dim)
        for (i, f) in enumerate(files):
            print "Reading", f
            cap = cv2.VideoCapture(f)
            frame = 0
            while(cap.isOpened() and frame < self.vid_length):
                ret, im = cap.read()
                if not ret:
                    break
                assert im.shape == (self.output_size, self.output_size, self.c_dim)
                im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                im = transform(im, is_crop=False)
                videos[i,frame,:,:,:] = im
                frame += 1
            assert frame == self.vid_length
        return videos
        
    def generator(self, z, reuse=False, train=True):
        layers = Layers()
        if reuse:
            tf.get_variable_scope().reuse_variables()

        print "z:", z.get_shape().as_list()
        f = self.z_output_size
        f, f2, f4, f8, f16 = [int(f*x) for x in np.logspace(math.log10(1),
                                                            math.log10(2),
                                                            5)]#[2,4,8,16]]
        s = self.vid_length
        s_power_2 = 1<<(s-1).bit_length()
        s2, s4, s8, s16 = [int(s_power_2/x) for x in [2,4,8,16]]
        k_w = 3

        layers.z_project, layers.g0_w, layers.g0_b = linear(
            z, s16*f16, 'gvideo_0', with_w=True)
        print "z_proj:", layers.z_project.get_shape().as_list()

        layers.g0 = tf.reshape(layers.z_project, [-1, 1, s16, f16])
        layers.gr0 = tf.nn.relu(self.g_bn0(layers.g0, train=train))
        print "gr0:", layers.gr0.get_shape().as_list()

        layers.g1, layers.g1_w, layers.g1_b = deconv2d(
            layers.gr0, [self.batch_size, 1, s8, f8],
            d_h=1, k_h=1, k_w=k_w, name='gvideo_1', with_w=True)
        layers.gr1 = tf.nn.relu(self.g_bn1(layers.g1, train=train))
        print "gr1:", layers.gr1.get_shape().as_list()
        print "filter:", layers.g1_w.get_shape().as_list()

        layers.g2, layers.g2_w, layers.g2_b = deconv2d(
            layers.gr1, [self.batch_size, 1, s4, f4],
            d_h=1, k_h=1, k_w=k_w, name='gvideo_2', with_w=True)
        layers.gr2 = tf.nn.relu(self.g_bn2(layers.g2, train=train))
        print "gr2:", layers.gr2.get_shape().as_list()
        print "filter:", layers.g2_w.get_shape().as_list()
        
        layers.g3, layers.g3_w, layers.g3_b = deconv2d(
            layers.gr2, [self.batch_size, 1, s2, f2],
            d_h=1, k_h=1, k_w=k_w, name='gvideo_3', with_w=True)
        layers.gr3 = tf.nn.relu(self.g_bn3(layers.g3, train=train))
        print "gr3:", layers.gr3.get_shape().as_list()
        print "filter:", layers.g3_w.get_shape().as_list()

        layers.g4, layers.g4_w, layers.g4_b = deconv2d(
            layers.gr3, [self.batch_size, 1, s, f],
            d_h=1, k_h=1, k_w=k_w, name='gvideo_4', with_w=True)
        layers.gr4 = tf.nn.tanh(layers.g4)
        print "gr4:", layers.gr4.get_shape().as_list()
        print "filter:", layers.g4_w.get_shape().as_list()
        
        layers.gr4_2d = tf.reshape(layers.gr4, [-1, s, f])
        print "gr4_2d:", layers.gr4_2d.get_shape().as_list()
        layers.gr4_1d = tf.reshape(layers.gr4, [-1, f])
        print "gr4_1d:", layers.gr4_1d.get_shape().as_list()

        return layers.gr4_1d, layers

    def discriminator(self, vid, reuse=False):
        layers = Layers()
        if reuse:
            tf.get_variable_scope().reuse_variables()

        # First we wanna just project into a reasonable shape.
        # Squeeze all of the per-frame stuff togther
        print "vid:", vid.get_shape().as_list()
        vid = tf.reshape(vid, [self.batch_size * self.vid_length, -1])
        print "vid (reshaped):", vid.get_shape().as_list()

        f = self.z_output_size # No reason for it to be this number other than symmetry
        f, f2, f4, f8, f16 = [int(f*x) for x in np.logspace(math.log10(1),
                                                            math.log10(2),
                                                            5)]
        k_w = 3
        
        layers.vid_project, layers.d0_w, layers.d0_b = linear(
            vid, f, 'dvideo_0', with_w=True)
        print "vid_project:", layers.vid_project.get_shape().as_list()

        layers.d0 = tf.reshape(layers.vid_project, [self.batch_size, 1, self.vid_length, -1])
        layers.dr0 = tf.nn.relu(self.d_bn0(layers.d0))
        print "dr0:", layers.dr0.get_shape().as_list()

        layers.dr1 = lrelu(self.d_bn1(conv2d(layers.dr0, f2, k_h=1, k_w=k_w, d_h=1, name='dvideo_h1')))
        print "dr1:", layers.dr1.get_shape().as_list()
        layers.dr2 = lrelu(self.d_bn2(conv2d(layers.dr1, f4, k_h=1, k_w=k_w, d_h=1, name='dvideo_h2')))
        print "dr2:", layers.dr2.get_shape().as_list()
        layers.dr3 = lrelu(self.d_bn3(conv2d(layers.dr2, f8, k_h=1, k_w=k_w, d_h=1, name='dvideo_h3')))
        print "dr3:", layers.dr3.get_shape().as_list()
        layers.dr4 = lrelu(self.d_bn4(conv2d(layers.dr3, f16, k_h=1, k_w=k_w, d_h=1, name='dvideo_h4')))
        print "dr4:", layers.dr4.get_shape().as_list()
        layers.d5 = linear(tf.reshape(layers.dr4, [self.batch_size, -1]), 1, 'dvideo_h5')
        print "d5:", layers.d5.get_shape().as_list()

        return tf.nn.sigmoid(layers.d5), layers.d5, layers


def main(_):
    pp.pprint(flags.FLAGS.__flags)

    with tf.Session() as sess:
        with tf.variable_scope('image'):
            dcgan = DCGAN(sess, image_size=FLAGS.image_size,
                          batch_size=FLAGS.vid_batch_size * FLAGS.vid_length,
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
        # pp.pprint([n.name for n in tf.get_default_graph().as_graph_def().node])
        # return

        # print "DVARS", dcgan.d_vars
        # print "GVARS", dcgan.g_vars

        with tf.variable_scope('video'):
            vid_z_dim = 120
            image_z_dim = 100

            d_penultimate_layer_name = "image/d_h3_conv/Conv2D:0"
            d_tensor = tf.get_default_graph().get_tensor_by_name(d_penultimate_layer_name)
            d_fake_penultimate_layer_name = "image/d_h3_conv_1/Conv2D:0"
            d_fake_tensor = tf.get_default_graph().get_tensor_by_name(d_fake_penultimate_layer_name)            
            
            vid_dcgan = VID_DCGAN(FLAGS.vid_batch_size,
                                  vid_z_dim,
                                  image_z_dim,
                                  FLAGS.vid_length,
                                  d_tensor,
                                  d_fake_tensor)
        
            print "DONE"

            print sess.run(tf.report_uninitialized_variables())
            un_init = [v for v in tf.global_variables() if not sess.run(tf.is_variable_initialized(v))]
            sess.run(tf.variables_initializer(un_init))
            print sess.run(tf.report_uninitialized_variables())

            sample_z = np.random.uniform(-1, 1, size=(FLAGS.vid_batch_size, vid_z_dim))
            out_val = sess.run(vid_dcgan.G, feed_dict={vid_dcgan.z:sample_z})
            print out_val.shape

            imgs = sess.run(dcgan.sampler, feed_dict={dcgan.z:out_val})
            vids = np.reshape(imgs, (-1, FLAGS.vid_length, FLAGS.output_size, FLAGS.output_size, FLAGS.c_dim))
            print imgs.shape
            print vids.shape

            print "FAKE DISC", sess.run(vid_dcgan.d_fake_out, feed_dict={dcgan.z:out_val})
            print "REAL DISC", sess.run(vid_dcgan.d_real_out, feed_dict={dcgan.images:imgs})

            vid_dcgan.train(FLAGS)
            
            # import cv2
            # print "OK OPENCV"
            # for i in xrange(vids.shape[0]):
            #     # filename = "/thesis0/yccggrp/youngsan/tmp/video_%05d.mp4" % i
            #     filename = "/thesis0/yccggrp/youngsan/tmp/video_%05d" % i
            #     print "Writing to", filename
            #     # w = cv2.VideoWriter(filename,
            #     #                     0x20,
            #     #                     25.0,
            #     #                     (FLAGS.output_size, FLAGS.output_size))
            #     for j in xrange(vids.shape[1]):
            #         im = inverse_transform(vids[i][j])
            #         im = np.around(im * 255).astype('uint8')
            #         im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
            #         # w.write(im)
            #         cv2.imwrite(filename + ("_%02d" % j) + ".png", im)
            #     # w.release()
            # print "DONE WRITING FILES"

if __name__ == '__main__':
    tf.app.run()

