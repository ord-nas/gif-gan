import os
import sys
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
import cv2 # For some reason this seems to make the initial variable
           # initialization take a long time

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
flags._global_parser.add_argument("--target_video", required=False, nargs='*', default=[], help="Video to try to recreate")
FLAGS = flags.FLAGS

class Layers(object):
    pass

class VID_DCGAN(object):
    def __init__(self, sess, batch_size, z_input_size, z_output_size, vid_length,
                 input_image_size, output_image_size, c_dim,
                 sample_rows=8, sample_cols=8):
        # Member vars
        self.batch_size = batch_size
        self.z_input_size = z_input_size
        self.z_output_size = z_output_size
        self.vid_length = vid_length
        self.input_image_size = input_image_size
        self.output_image_size = output_image_size
        self.c_dim = c_dim
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
        self.build_model(sess)

    def build_model(self, sess):
        # Build generator
        self.z = tf.get_variable('gvideo_z', [self.batch_size, self.z_input_size],
                                 initializer=tf.random_uniform_initializer(minval=-1.0, maxval=1.0))
        with tf.variable_scope('video_generator'):
            print "Making generator..."
            self.G, self.G_layers = self.generator(self.z, reuse=False, train=True)
            print "Making sampler..."
            self.G_sampler, self.G_sampler_layers = self.generator(self.z, reuse=True, train=False)
            self.is_training = tf.placeholder(tf.bool, [], name='gvideo_train')
            self.G_out = tf.cond(self.is_training, lambda: self.G, lambda: self.G_sampler)

        # Build the inner image gan
        with tf.variable_scope('image_gan'):
            self.img_dcgan = DCGAN(sess, image_size=self.input_image_size,
                                   batch_size=self.batch_size * self.vid_length,
                                   output_size=self.output_image_size, c_dim=self.c_dim,
                                   dataset_name='', is_crop=False,
                                   checkpoint_dir='', sample_dir='',
                                   data_dir='', log_dir='', image_glob='', shuffle=False,
                                   z=self.G_out)
            self.image_gan_scope_name = tf.get_variable_scope().name + "/"

        # Build discriminator
        with tf.variable_scope('video_discriminator'):
            print "Scope name:", tf.get_variable_scope().name
            print "Making first discriminator..."
            #self.d_loss = self.discriminator()

        # Define trainable variables
        t_vars = tf.trainable_variables()
        self.d_vid_vars = [var for var in t_vars if 'dvideo_' in var.name]
        self.g_vid_vars = [var for var in t_vars if 'gvideo_' in var.name]
        self.d_img_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_img_vars = [var for var in t_vars if 'g_' in var.name]

        # Define loss
        self.target_activations_tensor = tf.placeholder(
            tf.float32, self.img_dcgan.D_activations_inf.get_shape().as_list(), 'target_activations')
        activations_tensor = self.img_dcgan.D_activations_inf_
        print "ACTIVATIONS TENSOR:", activations_tensor.get_shape().as_list()
        print "TARGET ACTIVATIONS:", self.target_activations_tensor.get_shape().as_list()
        distance = tf.sqrt(tf.reduce_sum(tf.square(activations_tensor - self.target_activations_tensor),
                                         reduction_indices=[1,2,3]))
        print "DISTANCE:", distance.get_shape().as_list()
        self.loss = tf.reduce_mean(distance)
        print "LOSS:", self.loss.get_shape().as_list()
        self.per_video_loss = tf.reduce_mean(tf.reshape(distance,
                                                        [self.sample_rows * self.sample_cols,
                                                         self.vid_length]),
                                             reduction_indices=[1])
        self.sanity_check_loss = tf.reduce_mean(tf.reshape(distance,
                                                           [self.sample_rows * self.sample_cols,
                                                            self.vid_length]),
                                                reduction_indices=[1])

    def load_image_gan(self, sess, checkpoint_dir):
        print "Loading checkpoints from", checkpoint_dir

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            d = {}
            for v in tf.global_variables():
                n = v.op.name
                prefix = self.image_gan_scope_name
                if not n.startswith(prefix):
                    continue
                d[n[len(prefix):]] = v
            saver = tf.train.Saver(var_list=d)
            saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
            print "Success!"
        else:
            print "FAIL!"
        
    def train(self, sess, config):
        # Load the target video
        videos = np.zeros(shape=(self.sample_rows * self.sample_cols * self.vid_length, self.input_image_size, self.input_image_size, self.c_dim))
        assert (self.sample_rows * self.sample_cols) == len(config.target_video)
        for (i, filename) in enumerate(config.target_video):
            cap = cv2.VideoCapture(filename)
            frame = 0
            while(cap.isOpened() and frame < self.vid_length):
                ret, im = cap.read()
                if not ret:
                    break
                im = cv2.resize(im, (self.input_image_size, self.input_image_size),
                                interpolation=cv2.INTER_LINEAR)
                assert im.shape == (self.input_image_size, self.input_image_size, self.c_dim)
                im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                im = transform(im, is_crop=False)
                videos[i*self.vid_length+frame,:,:,:] = im
                frame += 1
            assert frame == self.vid_length
        video_batch = videos

        # Trying to write out what we read in just to avoid stupid mistakes
        self.output_video_batch(video_batch, "OUTPUT_VIDEO_BATCH.mp4")

        # Compute the discriminator activations for the target
        target_activations = sess.run(self.img_dcgan.D_activations_inf, feed_dict={
            self.img_dcgan.images: video_batch,
        })

        # Create optimizers
        with tf.variable_scope("optimizers"):
            print "Creating generator optimizers..."
            g_vars = self.g_vid_vars
            g_optim = (tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1)
                       .minimize(self.loss, var_list=g_vars))

            # Initialize variables created by optimizers
            current_scope_name = tf.get_variable_scope().name + "/"
            scope_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                           scope=current_scope_name)
            sess.run(tf.variables_initializer(scope_vars))

        # Initialize a save
        saver = tf.train.Saver()
        writer = tf.summary.FileWriter(config.log_dir, sess.graph)

        # Main train loop
        for epoch in xrange(config.epoch):
            _, loss_value, pv_loss, sanity_loss =sess.run(
                [g_optim, self.loss, self.per_video_loss, self.sanity_check_loss],
                feed_dict={
                    self.target_activations_tensor: target_activations,
                    self.is_training: True,
                }
            )
            print "Step %d/%d: loss %f (%s)" % (epoch, config.epoch, loss_value, pv_loss)
            if epoch % 10 == 0:
                self.dump_sample(sess, config, epoch, 0, is_training=True)
                self.dump_sample(sess, config, epoch, 0, is_training=False)
                print "Sanity check loss:"
                print sanity_loss
                saver.save(sess,
                           os.path.join(config.video_checkpoint_dir,
                                        "VID_DCGAN.model"),
                           global_step=epoch)

    def dump_sample(self, sess, config, epoch, idx, is_training=False):
        sz = self.output_image_size
        samples = sess.run([self.img_dcgan.sampler], feed_dict={
            self.is_training: is_training,
        })
        videos = np.reshape(samples, [self.sample_rows,
                                      self.sample_cols,
                                      self.vid_length,
                                      sz,
                                      sz,
                                      self.c_dim])
        
        folder = "train" if is_training else "inference"
        folder = os.path.join(config.video_sample_dir, folder)
        if not os.path.exists(folder):
            # No recursive os.makedirs
            os.mkdir(folder)

        filename = '{}/train_{:02d}_{:04d}.mp4'.format(folder, epoch, idx)
        print "Writing samples to", filename
        w = cv2.VideoWriter(filename,
                            0x20,
                            25.0,
                            (self.sample_cols * sz, self.sample_rows * sz))
        for t in xrange(self.vid_length):
            frame = np.zeros(shape=[self.sample_rows * sz,
                                    self.sample_cols * sz,
                                    self.c_dim],
                             dtype=np.uint8)
            for r in xrange(self.sample_rows):
                for c in xrange(self.sample_cols):
                    im = inverse_transform(videos[r,c,t,:,:,:])
                    im = np.around(im * 255).astype('uint8')
                    im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
                    frame[r*sz:(r+1)*sz, c*sz:(c+1)*sz, :] = im
            w.write(frame)
        w.release()
                
    def load_videos(self, files):
        n = len(files)
        videos = np.zeros(shape=(n, self.vid_length, self.input_image_size, self.input_image_size, self.c_dim))
        for (i, f) in enumerate(files):
            #print "Reading", f
            cap = cv2.VideoCapture(f)
            frame = 0
            while(cap.isOpened() and frame < self.vid_length):
                ret, im = cap.read()
                if not ret:
                    break
                im = cv2.resize(im, (self.input_image_size, self.input_image_size),
                                interpolation=cv2.INTER_LINEAR)
                assert im.shape == (self.input_image_size, self.input_image_size, self.c_dim)
                im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                im = transform(im, is_crop=False)
                videos[i,frame,:,:,:] = im
                frame += 1
            assert frame == self.vid_length
        return np.reshape(videos, [n*self.vid_length, self.input_image_size, self.input_image_size, self.c_dim])
        
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

    def output_video_batch(self, video, filename):
        w = cv2.VideoWriter(filename,
                            0x20,
                            25.0,
                            (self.output_image_size, self.output_image_size))
        for frame in video:
            im = inverse_transform(frame)
            im = np.around(im * 255).astype('uint8')
            im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
            w.write(im)
        w.release()

    def discriminator(self):
        pass


def main(_):
    pp.pprint(flags.FLAGS.__flags)
    np.set_printoptions(threshold=np.nan)

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
                                  c_dim=FLAGS.c_dim)
        
            print "DONE"

            # Init vars
            sess.run(tf.global_variables_initializer())

            # Load image model weights. This needs to happen *after* init, so
            # that we don't overwrite the weights we load.
            vid_dcgan.load_image_gan(sess, FLAGS.image_model_dir)

            # # Generate some z-vectors for one video.
            # sample_z = np.random.uniform(-1, 1, size=(FLAGS.vid_batch_size, vid_z_dim))
            # out_val = sess.run(vid_dcgan.G, feed_dict={vid_dcgan.z:sample_z})
            # print out_val.shape

            # # Generate videos from the z-vectors
            # imgs = sess.run(vid_dcgan.img_dcgan.sampler, feed_dict={vid_dcgan.img_dcgan.sample_z:out_val})
            # vids = np.reshape(imgs, (-1, FLAGS.vid_length, FLAGS.output_size, FLAGS.output_size, FLAGS.c_dim))
            # print imgs.shape
            # print vids.shape

            # # Test the discriminator
            # print "FAKE DISC", sess.run(vid_dcgan.d_fake_out, feed_dict={vid_dcgan.img_dcgan.z:out_val})
            # print "REAL DISC", sess.run(vid_dcgan.d_real_out, feed_dict={vid_dcgan.img_dcgan.images:imgs})

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

