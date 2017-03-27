import os
from ops import *
import numpy as np

from model import DCGAN
from utils import inverse_transform

import tensorflow as tf

import cv2

class Layers(object):
    pass

class VID_DCGAN(object):
    def __init__(self, sess, batch_size, z_input_size, z_output_size, vid_length,
                 input_image_size, output_image_size, c_dim,
                 sample_cols=8, image_noise_std=0.0, activation_noise_std=0.0,
                 use_lstm_generator=False, first_frame_loss_scalar=0.0,
                 all_frames_loss_scalar=0.0):
        # Member vars
        self.batch_size = batch_size
        self.z_input_size = z_input_size
        self.z_output_size = z_output_size
        self.vid_length = vid_length
        self.input_image_size = input_image_size
        self.output_image_size = output_image_size
        self.c_dim = c_dim
        assert batch_size % sample_cols == 0
        self.sample_rows = batch_size / sample_cols
        self.sample_cols = sample_cols
        self.image_noise_std = image_noise_std
        self.activation_noise_std = activation_noise_std
        self.use_lstm_generator = use_lstm_generator
        self.first_frame_loss_scalar = first_frame_loss_scalar
        self.all_frames_loss_scalar = all_frames_loss_scalar

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
        self.z = tf.placeholder(tf.float32, [self.batch_size, self.z_input_size],
                                name='gvideo_z')
        self.z_first_frame_component = self.z[:, :self.z_output_size]
        with tf.variable_scope('video_generator'):
            generator = self.lstm_generator if self.use_lstm_generator else self.generator
            print "Making generator..."
            self.G, self.G_layers = generator(self.z, reuse=False, train=True)
            print "Making sampler..."
            self.G_sampler, self.G_sampler_layers = generator(self.z, reuse=True, train=False)
            self.is_training = tf.placeholder(tf.bool, [], name='gvideo_train')
            self.G_out = tf.cond(self.is_training, lambda: self.G, lambda: self.G_sampler)
            print "G_out shape:", self.G_out.get_shape().as_list()
            self.first_frames = self.G_out[::self.vid_length,:]
            print "first_frames shape:", self.first_frames.get_shape().as_list()
            
        # Build the inner image gan
        with tf.variable_scope('image_gan'):
            self.img_dcgan = DCGAN(sess, image_size=self.input_image_size,
                                   batch_size=self.batch_size * self.vid_length,
                                   output_size=self.output_image_size,
                                   z_dim=self.z_output_size, c_dim=self.c_dim,
                                   dataset_name='', is_crop=False,
                                   checkpoint_dir='', sample_dir='',
                                   data_dir='', log_dir='', image_glob='', shuffle=False,
                                   z=self.G_out, noise_std=self.image_noise_std)
            self.image_gan_scope_name = tf.get_variable_scope().name + "/"

        # Build discriminator
        with tf.variable_scope('video_discriminator'):
            print "Scope name:", tf.get_variable_scope().name
            print "Making first discriminator..."
            self.noisy_D_activations_inf = add_noise(self.img_dcgan.D_activations_inf, self.activation_noise_std)
            self.D_activations_inf_std = get_std(self.img_dcgan.D_activations_inf)
            # self.d_real_out, self.d_real_out_logits, self.D_real_layers = self.discriminator(
            #     self.noisy_D_activations_inf, reuse=False)
            self.d_real_out, self.d_real_out_logits, self.D_real_layers = self.discriminator(
                self.img_dcgan.noisy_images, reuse=False)
            print "Making second discriminator..."
            self.noisy_D_activations_inf_ = add_noise(self.img_dcgan.D_activations_inf_, self.activation_noise_std)
            self.D_activations_inf_std_ = get_std(self.img_dcgan.D_activations_inf_)
            # self.d_fake_out, self.d_fake_out_logits, self.D_fake_layers = self.discriminator(
            #     self.noisy_D_activations_inf_, reuse=True)
            self.d_fake_out, self.d_fake_out_logits, self.D_fake_layers = self.discriminator(
                self.img_dcgan.noisy_sampler, reuse=True)

        # Define trainable variables
        t_vars = tf.trainable_variables()
        self.d_vid_vars = [var for var in t_vars if 'dvideo_' in var.name]
        self.g_vid_vars = [var for var in t_vars if 'gvideo_' in var.name]
        self.d_img_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_img_vars = [var for var in t_vars if 'g_' in var.name]

        # Define loss
        real_targets = tf.random_uniform(self.d_real_out.get_shape(), minval=0, maxval=1)
        real_targets = real_targets - 0.2
        real_targets = tf.floor(real_targets)
        real_targets = real_targets + 1.0
        self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            self.d_real_out_logits, tf.ones_like(self.d_real_out)))
        fake_targets = tf.random_uniform(self.d_fake_out.get_shape(), minval=0, maxval=1)
        fake_targets = fake_targets - 0.8
        fake_targets = tf.floor(fake_targets)
        fake_targets = fake_targets + 1.0
        self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            self.d_fake_out_logits, tf.zeros_like(self.d_fake_out)))
        self.d_loss = self.d_loss_real + self.d_loss_fake
        self.g_loss_realism = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
             self.d_fake_out_logits, tf.ones_like(self.d_fake_out)))
        self.g_loss_first_frame = self.first_frame_loss_scalar * tf.reduce_mean(tf.square(tf.subtract(
            self.first_frames, self.z_first_frame_component)))
        self.g_loss_all_frames = self.all_frames_loss_scalar * tf.reduce_mean(tf.square(tf.subtract(
            tf.reshape(self.G_out, [self.vid_length, self.batch_size, -1]), self.z_first_frame_component)))
        print "First frame loss: average{{%s - %s}^2} -> %s" % (
            self.first_frames.get_shape().as_list(),
            self.z_first_frame_component.get_shape().as_list(),
            self.g_loss_first_frame.get_shape().as_list())
        self.g_loss = self.g_loss_realism + self.g_loss_first_frame + self.g_loss_all_frames

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

    def load_checkpoint(self, sess, checkpoint_dir):
        print "Loading checkpoints from", checkpoint_dir

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            saver = tf.train.Saver()
            saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
            print "Success!"
        else:
            print "FAIL!"
        
    def train(self, sess, config):
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
        with tf.variable_scope("optimizers"):
            print "Creating discriminator optimizers..."
            d_vars = self.d_vid_vars
            if config.train_img_disc:
                d_vars = d_vars + self.d_img_vars
            d_optim = (tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1)
                       .minimize(self.d_loss, var_list=d_vars))

            print "Creating generator optimizers..."
            g_vars = self.g_vid_vars
            if config.train_img_gen:
                g_vars = g_vars + self.g_img_vars
            g_optim = (tf.train.AdamOptimizer(config.learning_rate * 1, beta1=config.beta1)
                       .minimize(self.g_loss, var_list=g_vars))

            # Initialize variables created by optimizers
            current_scope_name = tf.get_variable_scope().name + "/"
            scope_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                           scope=current_scope_name)
            sess.run(tf.variables_initializer(scope_vars))

        sample_z = np.random.uniform(-1, 1, size=(self.sample_rows * self.sample_cols,
                                                  self.z_input_size))
        face_z = np.random.uniform(-1, 1, size=(self.sample_rows, 1, self.z_output_size))
        print "FACE:", face_z.shape
        face_z = np.repeat(face_z, self.sample_cols, axis=1)
        print "FACE:", face_z.shape
        expression_z = np.random.uniform(-1, 1, size=(1, self.sample_cols,
                                                       self.z_input_size - self.z_output_size))
        print "EXPRESSION:", expression_z.shape
        expression_z = np.repeat(expression_z, self.sample_rows, axis=0)
        print "EXPRESSION:", expression_z.shape
        face_expression_z = np.concatenate((face_z, expression_z), axis=2)
        print "TOTAL:", face_expression_z.shape
        cross_sample_z = np.reshape(face_expression_z, sample_z.shape)
        
        # Initialize a saver
        saver = tf.train.Saver(max_to_keep=config.max_checkpoints_to_keep)
        writer = tf.summary.FileWriter(config.log_dir, sess.graph)

        # Main train loop
        counter = 0
        batch_size = self.batch_size
        for epoch in xrange(config.epoch):
            for i in xrange(0, len(files) // batch_size):
                batch_files = files[i*batch_size:(i+1)*batch_size]
                batch_images = self.load_videos(batch_files)
                batch_z = np.random.uniform(-1, 1, size=(
                    self.batch_size, self.z_input_size)).astype(np.float32)

                # Update D
                d_losses = []
                for _ in xrange(config.disc_updates):
                    _, d_loss_value, images_std, sampler_std, real_D_std, fake_D_std = sess.run(
                        [d_optim, self.d_loss, self.img_dcgan.images_std, self.img_dcgan.sampler_std,
                         self.D_activations_inf_std, self.D_activations_inf_std_],
                        feed_dict={
                            self.img_dcgan.images: batch_images,
                            self.z: batch_z,
                            self.is_training: True,
                        }
                    )
                    d_losses.append(d_loss_value)

                # Update G
                g_losses = []
                for _ in xrange(config.gen_updates):
                    tensor_list = [g_optim, self.g_loss_realism, self.g_loss_first_frame, self.g_loss_all_frames]
                    _, g_loss_realism_value, g_loss_first_frame_value, g_loss_all_frames_value = sess.run(tensor_list, feed_dict={
                        self.z: batch_z,
                        self.is_training: True,
                    })
                    g_losses.append(g_loss_realism_value)

                # errD_fake = 0#self.d_loss_fake.eval({self.z: batch_z})
                # errD_real = 0#self.d_loss_real.eval({self.img_dcgan.images: batch_images})
                # errG = 0#self.g_loss.eval({self.z: batch_z})
                counter += 1
                print("Epoch: [%2d] [%4d/%4d] d_loss: %s, g_loss: %s, first_frame_loss: %s all_frames_loss: %s" \
                      % (epoch, i+1, len(files) // batch_size,
                         d_losses, g_losses, g_loss_first_frame_value, g_loss_all_frames_value))
                print "Images std: %0.3f, sampler std: %0.3f | Real D std: %0.3f, fake D std: %0.3f" % (
                    images_std, sampler_std, real_D_std, fake_D_std)

                if counter % config.sample_frequency == 0:
                    #self.dump_sample(sample_z, sess, config, epoch, i, is_training=True)
                    self.dump_sample(sample_z, sess, config, epoch, i, is_training=False)
                    self.dump_sample(cross_sample_z, sess, config, epoch, i, is_training=False, prefix="cross_sample_")
                    saver.save(sess,
                               os.path.join(config.video_checkpoint_dir,
                                            "VID_DCGAN.model"),
                               global_step=counter)

    def dump_sample(self, sample_z, sess, config, epoch, idx, is_training=False, prefix=""):
        sz = self.output_image_size
        (samples, noisy_samples) = sess.run(
            [self.img_dcgan.sampler, self.img_dcgan.noisy_sampler],
            feed_dict={
                self.z: sample_z,
                self.is_training: is_training,
            }
        )
        videos = np.reshape(samples, [self.sample_rows,
                                      self.sample_cols,
                                      self.vid_length,
                                      sz,
                                      sz,
                                      self.c_dim])
        noisy_videos = np.reshape(noisy_samples, [self.sample_rows,
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

        # Write the clean samples
        filename = '{}/{}train_{:02d}_{:04d}.mp4'.format(folder, prefix, epoch, idx)
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

        # Write the noisy samples, if applicable
        if config.image_noise > 0:
            filename = '{}/{}train_noisy_{:02d}_{:04d}.mp4'.format(folder, prefix, epoch, idx)
            print "Writing noisy samples to", filename
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
                        im = inverse_transform(noisy_videos[r,c,t,:,:,:])
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
        z_copied = tf.pack([z] * self.vid_length, axis=1)
        print "z_copied:", z_copied.get_shape().as_list()

        frame_numbers = np.linspace(-1.0, 1.0, self.vid_length)
        frame_numbers = frame_numbers[np.newaxis, :, np.newaxis]
        frame_numbers = np.tile(frame_numbers, [self.batch_size, 1, 1])
        print "frame_numbers:", frame_numbers.shape

        z_with_numbers = tf.concat(2, [z_copied, frame_numbers])
        print "z_with_numbers:", z_with_numbers.get_shape().as_list()

        z_reshaped = tf.reshape(z_with_numbers, [self.batch_size * self.vid_length, -1])
        print "z_reshaped:", z_reshaped.get_shape().as_list()

        layers.gr0 = lrelu(self.g_bn0(linear(z_reshaped, 512, 'gvideo_0'), train=train))
        print "gr0:", layers.gr0.get_shape().as_list()
        layers.gr1 = lrelu(self.g_bn1(linear(layers.gr0, 512, 'gvideo_1'), train=train))
        print "gr1:", layers.gr1.get_shape().as_list()
        layers.gr2 = lrelu(self.g_bn2(linear(layers.gr1, 512, 'gvideo_2'), train=train))
        print "gr2:", layers.gr2.get_shape().as_list()
        layers.gr3 = tf.nn.tanh(linear(layers.gr2, self.z_output_size, 'gvideo_3'))
        print "gr3:", layers.gr3.get_shape().as_list()

        return layers.gr3, layers

    def lstm_generator(self, z, reuse=False, train=True):
        layers = Layers()
        if reuse:
            tf.get_variable_scope().reuse_variables()

        num_layers = 3
        state_size = 100
        input_size = self.z_output_size
        assert z.get_shape().as_list()[1] == input_size + state_size
        input_component = z[:, :input_size]
        print "Input component:", input_component.get_shape().as_list()
        state_component = z[:, input_size:]
        print "State component:", state_component.get_shape().as_list()

        initial_c_state = input_component
        initial_h_state = state_component
        initial_state = tf.nn.rnn_cell.LSTMStateTuple(initial_c_state, initial_h_state)
        state_tuple = tuple([initial_state] * num_layers)
        print "State tuple: %d x %d x %s" % (len(state_tuple), len(state_tuple[0]),
                                             state_tuple[0][0].get_shape().as_list())

        # RNN cell
        cell = tf.nn.rnn_cell.BasicLSTMCell(state_size, state_is_tuple=True)
        cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers, state_is_tuple=True)
        
        frames = [input_component]
        #frames = []
        with tf.variable_scope('gvideo_recurrent'):
            for _ in xrange(self.vid_length - 1):
            #for _ in xrange(self.vid_length):
                input = frames[-1:]
                #input = frames[-1:] if frames else [tf.zeros_like(input_component)]
                outputs, state_tuple = tf.nn.rnn(cell, input, initial_state=state_tuple)
                frames.extend(outputs)
                tf.get_variable_scope().reuse_variables()
        print "Final frames shape: %d x %s" % (len(frames), frames[0].get_shape().as_list())
        frames = tf.pack(frames, axis=1)
        print "Frames tensor shape:", frames.get_shape().as_list()
        frames = tf.reshape(frames, [self.batch_size * self.vid_length, -1])
        print "Frames reshaped:", frames.get_shape().as_list()

        return frames, layers

    # def discriminator(self, vid, reuse=False):
    #     layers = Layers()
    #     if reuse:
    #         tf.get_variable_scope().reuse_variables()

    #     print "vid:", vid.get_shape().as_list()
    #     vid = tf.reshape(vid, [self.batch_size, self.vid_length, 8, 8, -1])
    #     num_filters = vid.get_shape().as_list()[-1]
    #     print "vid (reshaped):", vid.get_shape().as_list()

    #     # f = self.z_output_size # No reason for it to be this number other than symmetry
    #     # f, f2, f4, f8, f16 = [int(f*x) for x in np.logspace(math.log10(1),
    #     #                                                     math.log10(2),
    #     #                                                     5)]
    #     k_w = 3
        
    #     # layers.vid_project, layers.d0_w, layers.d0_b = linear(
    #     #     vid, f, 'dvideo_0', with_w=True)
    #     # print "vid_project:", layers.vid_project.get_shape().as_list()

    #     # layers.d0 = tf.reshape(layers.vid_project, [self.batch_size, 1, self.vid_length, -1])
    #     # layers.dr0 = tf.nn.relu(self.d_bn0(layers.d0))
    #     # print "dr0:", layers.dr0.get_shape().as_list()

    #     layers.dr0 = vid
    #     layers.dr1 = tf.nn.avg_pool3d(vid, [1, 2, 2, 2, 1], [1, 2, 2, 2, 1], padding='SAME')#lrelu(conv3d(layers.dr0, 256, name='dvideo_h1'))
    #     print "dr1:", layers.dr1.get_shape().as_list()
    #     layers.dr2 = lrelu(conv3d(layers.dr1, 256, name='dvideo_h2'))
    #     print "dr2:", layers.dr2.get_shape().as_list()
    #     layers.dr3 = lrelu(conv3d(layers.dr2, 256, name='dvideo_h3'))
    #     print "dr3:", layers.dr3.get_shape().as_list()
    #     layers.d4 = linear(tf.reshape(layers.dr3, [self.batch_size, -1]), 1, 'dvideo_h4')
    #     print "d4:", layers.d4.get_shape().as_list()

    #     return tf.nn.sigmoid(layers.d4), layers.d4, layers

    # def discriminator(self, vid, reuse=False):
    #     layers = Layers()
    #     if reuse:
    #         tf.get_variable_scope().reuse_variables()

    #     print "vid:", vid.get_shape().as_list()
    #     vid = tf.reshape(vid, [self.batch_size, self.vid_length, 8, 8, -1])
    #     num_filters = vid.get_shape().as_list()[-1]
    #     print "vid (reshaped):", vid.get_shape().as_list()

    #     # f = self.z_output_size # No reason for it to be this number other than symmetry
    #     # f, f2, f4, f8, f16 = [int(f*x) for x in np.logspace(math.log10(1),
    #     #                                                     math.log10(2),
    #     #                                                     5)]
    #     k_w = 3
        
    #     # layers.vid_project, layers.d0_w, layers.d0_b = linear(
    #     #     vid, f, 'dvideo_0', with_w=True)
    #     # print "vid_project:", layers.vid_project.get_shape().as_list()

    #     # layers.d0 = tf.reshape(layers.vid_project, [self.batch_size, 1, self.vid_length, -1])
    #     # layers.dr0 = tf.nn.relu(self.d_bn0(layers.d0))
    #     # print "dr0:", layers.dr0.get_shape().as_list()

    #     layers.dr0 = vid
    #     layers.dr1 = lrelu(conv3d(layers.dr0, 256, name='dvideo_h1'))
    #     print "dr1:", layers.dr1.get_shape().as_list()
    #     layers.dr2 = lrelu(conv3d(layers.dr1, 256, name='dvideo_h2'))
    #     print "dr2:", layers.dr2.get_shape().as_list()
    #     layers.dr3 = lrelu(conv3d(layers.dr2, 256, name='dvideo_h3'))
    #     print "dr3:", layers.dr3.get_shape().as_list()
    #     layers.d4 = linear(tf.reshape(layers.dr3, [self.batch_size, -1]), 1, 'dvideo_h4')
    #     print "d4:", layers.d4.get_shape().as_list()

    #     return tf.nn.sigmoid(layers.d4), layers.d4, layers


    def discriminator(self, vid, reuse=False):
        layers = Layers()
        if reuse:
            tf.get_variable_scope().reuse_variables()

        shape = vid.get_shape().as_list()
        print "VID:", shape
        vid = tf.reshape(vid, [self.batch_size, self.vid_length] + shape[1:])
        print "VID (rehsaped):", vid.get_shape().as_list()
        df_dim = 64
        h0 = lrelu(conv3d(vid, df_dim, name='d_h0_conv'))
        print "H0:", h0.get_shape().as_list()
        h1 = lrelu(self.d_bn1(conv3d(h0, df_dim*2, name='d_h1_conv')))
        print "H1:", h1.get_shape().as_list()
        h2 = lrelu(self.d_bn2(conv3d(h1, df_dim*4, name='d_h2_conv')))
        print "H2:", h2.get_shape().as_list()
        h3 = lrelu(self.d_bn3(conv3d(h2, df_dim*8, name='d_h3_conv')))
        print "H3:", h3.get_shape().as_list()
        h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h3_lin')
        print "H4:", h4.get_shape().as_list()

        return tf.nn.sigmoid(h4), h4, layers

    # def discriminator(self, vid, reuse=False):
    #     layers = Layers()
    #     if reuse:
    #         tf.get_variable_scope().reuse_variables()

    #     # First we wanna just project into a reasonable shape.
    #     # Squeeze all of the per-frame stuff togther
    #     print "vid:", vid.get_shape().as_list()
    #     vid = tf.reshape(vid, [self.batch_size, 1, self.vid_length, -1])
    #     num_filters = vid.get_shape().as_list()[-1]
    #     print "vid (reshaped):", vid.get_shape().as_list()

    #     # f = self.z_output_size # No reason for it to be this number other than symmetry
    #     # f, f2, f4, f8, f16 = [int(f*x) for x in np.logspace(math.log10(1),
    #     #                                                     math.log10(2),
    #     #                                                     5)]
    #     k_w = 3
        
    #     # layers.vid_project, layers.d0_w, layers.d0_b = linear(
    #     #     vid, f, 'dvideo_0', with_w=True)
    #     # print "vid_project:", layers.vid_project.get_shape().as_list()

    #     # layers.d0 = tf.reshape(layers.vid_project, [self.batch_size, 1, self.vid_length, -1])
    #     # layers.dr0 = tf.nn.relu(self.d_bn0(layers.d0))
    #     # print "dr0:", layers.dr0.get_shape().as_list()

    #     layers.dr0 = vid
    #     layers.dr1 = lrelu(conv2d(layers.dr0, 128, k_h=1, k_w=k_w, d_h=1, name='dvideo_h1'))
    #     print "dr1:", layers.dr1.get_shape().as_list()
    #     layers.dr2 = lrelu(self.d_bn2(conv2d(layers.dr1, 128, k_h=1, k_w=k_w, d_h=1, name='dvideo_h2')))
    #     print "dr2:", layers.dr2.get_shape().as_list()
    #     layers.dr3 = lrelu(self.d_bn3(conv2d(layers.dr2, 128, k_h=1, k_w=k_w, d_h=1, name='dvideo_h3')))
    #     print "dr3:", layers.dr3.get_shape().as_list()
    #     layers.dr4 = lrelu(self.d_bn4(conv2d(layers.dr3, 128, k_h=1, k_w=k_w, d_h=1, name='dvideo_h4')))
    #     print "dr4:", layers.dr4.get_shape().as_list()
    #     layers.d5 = linear(tf.reshape(layers.dr4, [self.batch_size, -1]), 1, 'dvideo_h5')
    #     print "d5:", layers.d5.get_shape().as_list()

    #     return tf.nn.sigmoid(layers.d5), layers.d5, layers


    # def discriminator(self, vid, reuse=False):
    #     layers = Layers()
    #     if reuse:
    #         tf.get_variable_scope().reuse_variables()

    #     # First we wanna just project into a reasonable shape.
    #     # Squeeze all of the per-frame stuff togther
    #     print "vid:", vid.get_shape().as_list()
    #     vid = tf.reshape(vid, [1, self.batch_size, self.vid_length, -1])
    #     num_filters = vid.get_shape().as_list()[-1]
    #     print "vid (reshaped):", vid.get_shape().as_list()

    #     # f = self.z_output_size # No reason for it to be this number other than symmetry
    #     # f, f2, f4, f8, f16 = [int(f*x) for x in np.logspace(math.log10(1),
    #     #                                                     math.log10(2),
    #     #                                                     5)]
    #     k_w = 3
        
    #     # layers.vid_project, layers.d0_w, layers.d0_b = linear(
    #     #     vid, f, 'dvideo_0', with_w=True)
    #     # print "vid_project:", layers.vid_project.get_shape().as_list()

    #     # layers.d0 = tf.reshape(layers.vid_project, [self.batch_size, 1, self.vid_length, -1])
    #     # layers.dr0 = tf.nn.relu(self.d_bn0(layers.d0))
    #     # print "dr0:", layers.dr0.get_shape().as_list()

    #     layers.dr0 = vid
    #     layers.dr1 = lrelu(conv2d(layers.dr0, 128, k_h=3, k_w=3, name='dvideo_h1'))
    #     print "dr1:", layers.dr1.get_shape().as_list()
    #     layers.dr2 = lrelu(self.d_bn2(conv2d(layers.dr1, 128, k_h=3, k_w=3, name='dvideo_h2')))
    #     print "dr2:", layers.dr2.get_shape().as_list()
    #     layers.dr3 = lrelu(self.d_bn3(conv2d(layers.dr2, 128, k_h=3, k_w=3, name='dvideo_h3')))
    #     print "dr3:", layers.dr3.get_shape().as_list()
    #     layers.dr4 = lrelu(self.d_bn4(conv2d(layers.dr3, 128, k_h=3, k_w=3, name='dvideo_h4')))
    #     print "dr4:", layers.dr4.get_shape().as_list()
    #     layers.d5 = linear(tf.reshape(layers.dr4, [1, -1]), 1, 'dvideo_h5')
    #     print "d5:", layers.d5.get_shape().as_list()

    #     return tf.nn.sigmoid(layers.d5), layers.d5, layers
