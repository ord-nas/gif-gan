from __future__ import print_function, division
import numpy as np
import tensorflow as tf
# import matplotlib.pyplot as plt
from glob import glob
import os
import sys
import cv2
import random

# input_path = "/media/charles/850EVO/ubuntu_documents/ece496/gif-gan/data_collection/data/processed/"
input_path = "/thesis0/yccggrp/dataset/face_mp4s/"
saved_sample_path = "io_tests/test_output"
checkpoint_path = "model_checkpoints"
load = False
quick_test = False

num_epochs = 100
batch_size = 40
video_length = 16

num_layers = 3

output_channels = 3
image_dimension = 64
image_size = image_dimension * image_dimension

filter_dimension = 5
stride = 2

layer_shapes = [[batch_size, 4, 4, 512],
                [batch_size, 8, 8, 256],
                [batch_size, 16, 16, 128],
                [batch_size, 32, 32, 64],
                [batch_size, 64, 64, 3]]

fc_size = layer_shapes[0][1] * layer_shapes[0][2] * layer_shapes[0][3]

state_size = 100
stddev = 0.02
model_dir = "%s_%s_multi-layer" % (batch_size, image_dimension)

def make_gif(images, fname, duration=2):
  import moviepy.editor as mpy

  def make_frame(t):
    try:
      x = images[int(len(images)/duration*t)]
    except:
      x = images[-1]

    return x

  clip = mpy.VideoClip(make_frame, duration=duration)
  clip.write_videofile(fname, fps = len(images)/duration)

def load_batch():
    global samples_path_list

    if (len(samples_path_list) < batch_size):
        sys.exit("Not enough videos for one batch. Exit...")

    x = np.array(np.zeros((batch_size, video_length + 1, image_dimension, image_dimension, output_channels), np.float32))
    for img_idx in range(batch_size):
        # print (samples_path_list[img_idx])
        cap = cv2.VideoCapture(samples_path_list[img_idx])
        frame_num = 0
        while(cap.isOpened() and frame_num < video_length + 1):
            ret, im = cap.read()
            if not ret:
                break
            # im = cv2.cvtColor(cv2.resize(im, (image_dimension, image_dimension)), 6)
            im = cv2.resize(im, (image_dimension, image_dimension))
            # x[img_idx][frame_num] = im.reshape((image_size * output_channels))
            x[img_idx][frame_num] = im
            frame_num += 1

        cap.release()
    samples_path_list = samples_path_list[batch_size:]
    return x

def save_sample(sample, epoch_idx, batchX):
    for im_num in range(batch_size):
        samples = []
        for frame_num in range(1):
            im = (np.array(sample)[frame_num][im_num].reshape((image_dimension, image_dimension, output_channels)) * 256).astype(np.uint8)
            output_path = os.path.join(saved_sample_path, str(epoch_idx) + "_" + str(im_num) + "_" + str(frame_num) + ".jpg")
            if frame_num == 0:
                cv2.imwrite(output_path, im)
            # samples.append(im)

        # make_gif(samples, "./io_tests/test_output/" + str(epoch_idx) + "_" + str(im_num) + ".mp4", duration=16)

        # real_images = []
        # for frame_num in range(1):
        #     im = (batchX[im_num][frame_num]).astype(np.uint8)
        #     output_path = os.path.join(saved_sample_path, str(epoch_idx) + "_" + str(im_num) + "_" + str(frame_num) + "_real.jpg")
        #     if frame_num == 0:
        #         cv2.imwrite(output_path, im)
            # real_images.append(im)

        # make_gif(real_images, "./io_tests/test_output/" + str(epoch_idx) + "_" + str(im_num) + "_real.mp4", duration=16)



# def plot_loss(g_loss_list, d_loss_list, real_score_list, fake_score_list):
#     plt.subplot(2,1,1)
#     plt.cla()
#     plt.plot(g_loss_list, label="G_Loss", linewidth=2)
#     plt.plot(d_loss_list, label="D_Loss", linewidth=2)
#     plt.legend()

#     plt.subplot(2,1,2)
#     plt.cla()
#     plt.plot(real_score_list, label="Real_Score", linewidth=2)
#     plt.plot(fake_score_list, label="Fake_Score", linewidth=2)
#     plt.legend()
    
#     plt.draw()
#     plt.pause(0.0001)

def save_checkpoint(checkpoint_dir):
    global saver
    global first_time

    model_name = "recurrent.model"
    checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

    if os.path.exists(checkpoint_dir):
        if first_time and (not load):
            sys.exit("Model checkpoint already exists. Exit...")
    else:
        os.makedirs(checkpoint_dir)

    first_time = False
    saver.save(sess, os.path.join(checkpoint_dir, model_name))

def load_checkpoint(checkpoint_dir):
    global saver

    print("Reading checkpoints...")
    checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
        return True
    else:
        return False

def lrelu(x, leak=0.2):
    return tf.maximum(x, leak*x)

################################################ main ################################################

# Input
batchINPUT_placeholder = tf.placeholder(tf.int32, [batch_size, video_length + 1, image_dimension, image_dimension, output_channels])
batchX_placeholder = tf.cast(tf.slice(batchINPUT_placeholder, [0,0,0,0,0], [-1,video_length,-1,-1,-1]), tf.float32)
batchY_placeholder = tf.cast(tf.slice(batchINPUT_placeholder, [0,1,0,0,0], [-1,-1,-1,-1,-1]), tf.float32)

# Unpack columns
inputs_series_raw = tf.unpack(batchX_placeholder/256, axis=1)
labels_series = tf.unpack(batchY_placeholder/256, axis=1)

# Generator
# Takes inputs_series_raw and labels_series (training only)
# Produces generator_outputs_series
with tf.variable_scope("generator") as vs_g:
    # RNN states
    init_state = tf.placeholder(tf.float32, [num_layers, 2, batch_size, state_size])

    state_per_layer_list = tf.unpack(init_state, axis=0)
    rnn_tuple_state = tuple(
        [tf.nn.rnn_cell.LSTMStateTuple(state_per_layer_list[idx][0], state_per_layer_list[idx][1])
         for idx in range(num_layers)]
    )

    # Filters for input-to-RNN
    conv_f1 = tf.Variable(np.random.normal(scale=stddev, size=(filter_dimension, filter_dimension, layer_shapes[4][3], layer_shapes[3][3])), dtype=tf.float32)
    conv_f2 = tf.Variable(np.random.normal(scale=stddev, size=(filter_dimension, filter_dimension, layer_shapes[3][3], layer_shapes[2][3])), dtype=tf.float32)
    conv_f3 = tf.Variable(np.random.normal(scale=stddev, size=(filter_dimension, filter_dimension, layer_shapes[2][3], layer_shapes[1][3])), dtype=tf.float32)
    conv_f4 = tf.Variable(np.random.normal(scale=stddev, size=(filter_dimension, filter_dimension, layer_shapes[1][3], layer_shapes[0][3])), dtype=tf.float32)
    conv_f_list = [conv_f1, conv_f2, conv_f3, conv_f4]

    # Input-to-RNN
    inputs_series = []
    for input_layer in inputs_series_raw:
        data_in = input_layer
        # conv, norm, relu layers
        for i in range(4):
            input_conv = tf.nn.conv2d(data_in, conv_f_list[i], [1,stride,stride,1], "SAME")
            mean, variance = tf.nn.moments(input_conv, axes = [0, 1, 2])
            batch_norm = tf.nn.batch_normalization(input_conv, mean, variance, None, None, 1e-5)
            # batch_norm = tf.contrib.layers.batch_norm(input_conv, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, is_training=True)
            data_in = tf.nn.relu(batch_norm)

        # skip fc layer (already in RNN cell)
        inputs_series.append(tf.reshape(data_in, [batch_size, fc_size]))

    # RNN cell
    cell = tf.nn.rnn_cell.BasicLSTMCell(state_size, state_is_tuple=True)
    cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers, state_is_tuple=True)
    states_series, current_state = tf.nn.rnn(cell, inputs_series, initial_state=rnn_tuple_state)

    # Filters for states-to-output
    output_fc_w = tf.Variable(np.random.normal(scale=stddev, size=(state_size, fc_size)), dtype=tf.float32)
    output_fc_bias = tf.Variable(np.zeros((1, fc_size)),dtype=tf.float32)
    deconv_f1 = tf.Variable(np.random.normal(scale=stddev, size=(filter_dimension, filter_dimension, layer_shapes[1][3], layer_shapes[0][3])), dtype=tf.float32)
    deconv_f2 = tf.Variable(np.random.normal(scale=stddev, size=(filter_dimension, filter_dimension, layer_shapes[2][3], layer_shapes[1][3])), dtype=tf.float32)
    deconv_f3 = tf.Variable(np.random.normal(scale=stddev, size=(filter_dimension, filter_dimension, layer_shapes[3][3], layer_shapes[2][3])), dtype=tf.float32)
    deconv_f4 = tf.Variable(np.random.normal(scale=stddev, size=(filter_dimension, filter_dimension, layer_shapes[4][3], layer_shapes[3][3])), dtype=tf.float32)
    deconv_f_list = [deconv_f1, deconv_f2, deconv_f3, deconv_f4]

    # States-to-Output
    generator_outputs_series = []
    for state in states_series:
        # fc layer
        data_in = tf.reshape(tf.matmul(state, output_fc_w) + output_fc_bias, 
                            [batch_size, layer_shapes[0][1], layer_shapes[0][2], layer_shapes[0][3]])
        # deconv, norm, relu layers
        for i in range(4):
            mean, variance = tf.nn.moments(data_in, axes = [0, 1, 2])
            batch_norm = tf.nn.batch_normalization(data_in, mean, variance, None, None, 1e-5)
            # batch_norm = tf.contrib.layers.batch_norm(data_in, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, is_training=True)
            relu = tf.nn.relu(batch_norm)
            data_in = tf.nn.conv2d_transpose(relu, deconv_f_list[i], layer_shapes[i+1], [1,stride,stride,1], "SAME")
            
        generator_outputs_series.append((tf.tanh(data_in) + 1) / 2)

    generator_variables = [v for v in tf.all_variables() if v.name.startswith(vs_g.name)]
    print("G variables:")
    print(generator_variables)



# Discriminator
# Takes generator_outputs_series and labels_series
# Produces d_score_fake_input_logits and d_score_real_input_logits
with tf.variable_scope("discriminator") as vs_d:

    # Filters for conv and fc layers
    d_conv_f1 = tf.Variable(np.random.normal(scale=stddev, size=(filter_dimension, filter_dimension, layer_shapes[4][3], layer_shapes[3][3])), dtype=tf.float32)
    d_conv_f2 = tf.Variable(np.random.normal(scale=stddev, size=(filter_dimension, filter_dimension, layer_shapes[3][3], layer_shapes[2][3])), dtype=tf.float32)
    d_conv_f3 = tf.Variable(np.random.normal(scale=stddev, size=(filter_dimension, filter_dimension, layer_shapes[2][3], layer_shapes[1][3])), dtype=tf.float32)
    d_conv_f4 = tf.Variable(np.random.normal(scale=stddev, size=(filter_dimension, filter_dimension, layer_shapes[1][3], layer_shapes[0][3])), dtype=tf.float32)
    d_conv_f_list = [d_conv_f1, d_conv_f2, d_conv_f3, d_conv_f4]
    d_fc_w = tf.Variable(np.random.normal(scale=stddev, size=(fc_size, state_size)), dtype=tf.float32)
    d_fc_bias = tf.Variable(np.zeros((1, state_size)),dtype=tf.float32)
    d_final_fc_w = tf.Variable(np.random.normal(scale=stddev, size=(state_size * video_length, 1)), dtype=tf.float32)
    d_final_fc_bias = tf.Variable(np.zeros((1, 1)),dtype=tf.float32)

    # Process generator outputs (fake input)
    discriminator_outputs_series = []
    for generator_output in generator_outputs_series:
        data_in = generator_output
        # conv, norm, relu layers
        for i in range(4):
            conv = tf.nn.conv2d(data_in, d_conv_f_list[i], [1,stride,stride,1], "SAME")
            mean, variance = tf.nn.moments(conv, axes = [0, 1, 2])
            batch_norm = tf.nn.batch_normalization(conv, mean, variance, None, None, 1e-5)
            # batch_norm = tf.contrib.layers.batch_norm(conv, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, is_training=True)
            data_in = lrelu(batch_norm)
        
        # fc layer
        discriminator_outputs_series.append(tf.matmul(tf.reshape(data_in, [batch_size, fc_size]), d_fc_w) + d_fc_bias)

    # final fc layer
    concatenated_outputs = tf.concat(1, discriminator_outputs_series)
    d_score_fake_input_logits = tf.matmul(concatenated_outputs, d_final_fc_w) + d_final_fc_bias
    d_score_fake_input = tf.reduce_mean(tf.sigmoid(d_score_fake_input_logits))

    # Process training labels (real input)
    discriminator_outputs_series = []
    for label in labels_series:
        data_in = label
        # conv, norm, relu layers
        for i in range(4):
            conv = tf.nn.conv2d(data_in, d_conv_f_list[i], [1,stride,stride,1], "SAME")
            mean, variance = tf.nn.moments(conv, axes = [0, 1, 2])
            batch_norm = tf.nn.batch_normalization(conv, mean, variance, None, None, 1e-5)
            # batch_norm = tf.contrib.layers.batch_norm(conv, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, is_training=True)
            data_in = lrelu(batch_norm)
        
        # fc layer
        discriminator_outputs_series.append(tf.matmul(tf.reshape(data_in, [batch_size, fc_size]), d_fc_w) + d_fc_bias)

    # final fc layer
    concatenated_outputs = tf.concat(1, discriminator_outputs_series)
    d_score_real_input_logits = tf.matmul(concatenated_outputs, d_final_fc_w) + d_final_fc_bias
    d_score_real_input = tf.reduce_mean(tf.sigmoid(d_score_real_input_logits))

    discriminator_variables = [v for v in tf.all_variables() if v.name.startswith(vs_d.name)]
    print("D variables:")
    print(discriminator_variables)


# Loss Functions
g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(d_score_fake_input_logits, tf.ones_like(d_score_fake_input_logits)))
d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(d_score_fake_input_logits, tf.zeros_like(d_score_fake_input_logits)))
d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(d_score_real_input_logits, tf.ones_like(d_score_real_input_logits)))
d_loss = d_loss_fake + d_loss_real

# model saver
saver = tf.train.Saver()



# Train
g_optim = tf.train.AdamOptimizer(0.0002, beta1=0.5).minimize(g_loss, var_list=generator_variables) 
d_optim = tf.train.AdamOptimizer(0.0002, beta1=0.5).minimize(d_loss, var_list=discriminator_variables)

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    # plt.ion()
    # plt.figure()
    # plt.show()
    g_loss_list = []
    d_loss_list = []
    real_score_list = []
    fake_score_list = []

    # data preparation
    samples_path_list_full = []

    for i in range(0, 48):
        samples_path_list_full += glob(os.path.join(input_path, str(i), "*.mp4"))
        # print (len(samples_path_list))
    
    if quick_test:
        samples_path_list_full = samples_path_list_full[:512]

    print ("Number of images = " + str(len(samples_path_list_full)))
    num_batches = int(len(samples_path_list_full) / batch_size)
    print ("Number of batches = " + str(num_batches))

    if load:
        if load_checkpoint(checkpoint_path):
            print ("Checkpoint loaded...")
        else:
            sys.exit("Checkpoint loading failed. Exit...")

    step_counter = 1
    first_time = True

    for epoch_idx in range(num_epochs):
        random.shuffle(samples_path_list_full)
        samples_path_list = samples_path_list_full[:]
        _current_state = np.zeros((num_layers, 2, batch_size, state_size))

        print("New epoch", epoch_idx)
        for batch_idx in range(num_batches):
            batchX = load_batch().astype(np.float32)

            # Update Discriminator
            _d_optim = sess.run([d_optim],
                feed_dict={
                    batchINPUT_placeholder: batchX,
                    init_state: _current_state
                })

            # Update Generator w.r.t. g_loss
            _g_optim = sess.run([g_optim],
                feed_dict={
                    batchINPUT_placeholder: batchX,
                    init_state: _current_state
                })

            # Update Generator w.r.t. g_loss again and collect some data
            _d_loss, _g_loss, _g_optim, _generator_outputs_series, _d_score_real_input, _d_score_fake_input = sess.run(
                [d_loss, g_loss, g_optim, generator_outputs_series, d_score_real_input, d_score_fake_input],
                feed_dict={
                    batchINPUT_placeholder: batchX,
                    init_state: _current_state
                })

            g_loss_list.append(_g_loss)
            d_loss_list.append(_d_loss)
            real_score_list.append(_d_score_real_input)
            fake_score_list.append(_d_score_fake_input)
            # plot_loss(g_loss_list, d_loss_list, real_score_list, fake_score_list)

            # print("Epoch", epoch_idx, "Batch", batch_idx, "D_Loss", _d_loss, "G_Loss", _g_loss, "D_Score_R", _d_score_real_input, "D_Score_F", _d_score_fake_input)
            print("Epoch %d Batch %d: D_Loss %.4f, G_Loss %.4f, Real_Score %.4f, Fake_Score %.4f" % \
                  (epoch_idx, batch_idx, _d_loss, _g_loss, _d_score_real_input, _d_score_fake_input))

            if (batch_idx % 20) == 0:
                save_sample(_generator_outputs_series, epoch_idx, batchX)

            if (not quick_test) and (step_counter % 300 == 0):
                print ("Saving checkpoint to %s" % checkpoint_path)
                save_checkpoint(checkpoint_path)

            step_counter += 1

# plt.ioff()
# plt.show()

############################################## end main ##############################################