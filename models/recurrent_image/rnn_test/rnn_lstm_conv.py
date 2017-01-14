from __future__ import print_function, division
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from glob import glob
import os
import sys
import cv2

input_path = "/media/charles/850EVO/ubuntu_documents/ece496/gif-gan/data_collection/data/processed/"
saved_sample_path = "io_tests/test_output"

use_real_images = True
video_length = 16
randomize_pixels = True
enable_relu = True

num_epochs = 1000
batch_size = 128
total_series_length = 150000
truncated_backprop_length = video_length
num_classes = 2

output_channels = 1
image_dimension = 32
image_size = image_dimension * image_dimension

filter_dimension = [8, 4]
stride = [2,2]

state_channels = 4
state_dimension = int(image_dimension / stride[0] / stride [1])
state_size = state_channels * state_dimension * state_dimension
# state_size = image_size * output_channels

output_shape = [[batch_size, state_dimension * stride[0], state_dimension * stride[0], 2],
                [batch_size, image_dimension, image_dimension, output_channels]]

echo_step = 3

def generateData():
    if randomize_pixels:
        # x = np.array(np.random.choice(2, total_series_length * image_size * output_channels, p=[0.5, 0.5]))
        x = np.array(np.random.uniform(0, 1, (total_series_length * image_size * output_channels)))
        y = np.roll(x, echo_step * image_size * output_channels)
        y[0:echo_step * image_size * output_channels] = 0
        # y = np.array(np.random.uniform(0, 256, (total_series_length * image_size * output_channels)))
    else:
        # x = np.array(np.random.choice(2, total_series_length, p=[0.5, 0.5]))
        x = np.array(np.random.uniform(0, 1, (total_series_length)))
        y = np.roll(x, echo_step)
        y[0:echo_step] = 0
        x = np.repeat(x, image_size * output_channels, axis = 0)
        y = np.repeat(y, image_size * output_channels, axis = 0)

    x = x.reshape((batch_size, -1, image_size * output_channels))  # The first index changing slowest, subseries as rows
    y = y.reshape((batch_size, -1, image_size * output_channels))

    return (x,y)

def load_batch():
    global samples_path_list

    if (len(samples_path_list) < batch_size):
        sys.exit("Not enough videos for one batch. Exit...")

    x = np.array(np.zeros((batch_size, video_length + 1, image_size * output_channels), np.int32))
    for img_idx in range(batch_size):
        # print (samples_path_list[img_idx])
        cap = cv2.VideoCapture(samples_path_list[img_idx])
        frame_num = 0
        while(cap.isOpened() and frame_num < video_length + 1):
            ret, im = cap.read()
            if not ret:
                break
            im = cv2.cvtColor(cv2.resize(im, (image_dimension, image_dimension)), 6)
            x[img_idx][frame_num] = im.reshape((image_size * output_channels))
            frame_num += 1

        cap.release()
    samples_path_list = samples_path_list[batch_size:]
    # return (x[:,:-1,:],x[:,1:,:])
    return x

def save_sample(sample, epoch_idx):
    for im_num in range(20):
        for frame_num in range(10):
            im = (np.array(sample)[frame_num][im_num].reshape((image_dimension, image_dimension)) * 256.0).astype(int)
            output_path = os.path.join(saved_sample_path, str(epoch_idx) + "_" + str(im_num) + "_" + str(frame_num) + ".jpg")
            cv2.imwrite(output_path, im)

def plot_loss(loss_list):
    plt.cla()
    plt.plot(loss_list)
    plt.draw()
    plt.pause(0.0001)

################################################ main ################################################

batchX_placeholder = tf.placeholder(tf.float32, [batch_size, truncated_backprop_length, image_size * output_channels])
batchY_placeholder = tf.placeholder(tf.float32, [batch_size, truncated_backprop_length, image_size * output_channels])

cell_state = tf.placeholder(tf.float32, [batch_size, state_size])
hidden_state = tf.placeholder(tf.float32, [batch_size, state_size])
init_state = tf.nn.rnn_cell.LSTMStateTuple(cell_state, hidden_state)

# Filter for deconvolutions
f1 = tf.Variable(np.random.rand(filter_dimension[0], filter_dimension[0], output_shape[0][3], state_channels),dtype=tf.float32)
f2 = tf.Variable(np.random.rand(filter_dimension[1], filter_dimension[1], output_shape[1][3], output_shape[0][3]),dtype=tf.float32)

# Unpack columns
inputs_series = tf.unpack(batchX_placeholder/256.0, axis=1)
labels_series = tf.unpack(batchY_placeholder/256.0, axis=1)

# Forward passes
cell = tf.nn.rnn_cell.BasicLSTMCell(state_size, state_is_tuple=True)
states_series, current_state = tf.nn.rnn(cell, inputs_series, init_state)

logits_series = []

for state in states_series:
    input_layer = tf.reshape(state, [batch_size, state_dimension, state_dimension, state_channels])
    
    if enable_relu:
        relu_1 = tf.nn.relu(input_layer)
    else:
        relu_1 = input_layer

    deconv_1 = tf.nn.conv2d_transpose(relu_1, f1, output_shape[0], [1,stride[0],stride[0],1], "SAME")

    if enable_relu:
        relu_2 = tf.nn.relu(deconv_1)
    else:
        relu_2 = deconv_1

    deconv_2 = tf.nn.conv2d_transpose(relu_2, f2, output_shape[1], [1,stride[1],stride[1],1], "SAME")
    logits_series.append(tf.reshape(tf.sigmoid(deconv_2), [batch_size, image_size * output_channels]))

# losses = [tf.nn.sigmoid_cross_entropy_with_logits(logits, labels) for logits, labels in zip(logits_series,labels_series)]
losses = [tf.nn.l2_loss(logits - labels) for logits, labels in zip(logits_series,labels_series)]
total_loss = tf.reduce_mean(losses) / (batch_size * image_size) 

train_step = tf.train.AdagradOptimizer(0.3).minimize(total_loss)
# train_step = tf.train.AdamOptimizer(0.003).minimize(total_loss)

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    plt.ion()
    plt.figure()
    plt.show()
    loss_list = []

    # data preparation
    if use_real_images:
        samples_path_list = []
        batch_list = []

        for i in range(36, 48):
            samples_path_list += glob(os.path.join(input_path, str(i), "*.mp4"))
            # print (len(samples_path_list))
        
        print ("Number of images = " + str(len(samples_path_list)))
        num_batches = int(len(samples_path_list) / batch_size)
        print ("Number of batches = " + str(num_batches))
        
        for batch_idx in range(num_batches):
            batch_list.append(load_batch().astype(np.float32))
    else:
        num_batches = total_series_length//batch_size//truncated_backprop_length

    

    for epoch_idx in range(num_epochs):
        # if use_real_images:
        #     data = glob(os.path.join(input_path, "*.mp4"))
        # else:
        #     x,y = generateData()
        _current_cell_state = np.zeros((batch_size, state_size))
        _current_hidden_state = np.zeros((batch_size, state_size))

        print("New epoch", epoch_idx)

        for batch_idx in range(num_batches):
            if use_real_images:
                batchX = batch_list[batch_idx][:,:-1,:]
                batchY = batch_list[batch_idx][:,1:,:]
            else:
                start_idx = batch_idx * truncated_backprop_length
                end_idx = start_idx + truncated_backprop_length

                batchX = x[:,start_idx:end_idx]
                batchY = y[:,start_idx:end_idx]

            _total_loss, _train_step, _current_state, _logits_series = sess.run(
                [total_loss, train_step, current_state, logits_series],
                feed_dict={
                    batchX_placeholder: batchX,
                    batchY_placeholder: batchY,
                    cell_state: _current_cell_state,
                    hidden_state: _current_hidden_state
                })

            _current_cell_state, _current_hidden_state = _current_state

            if epoch_idx != 0 and batch_idx % 30 == 0:
                loss_list.append(_total_loss)

            if batch_idx % 30 == 0:
                print("Step",batch_idx, "Loss", _total_loss)
        
        if epoch_idx % 10 == 0:
            save_sample(_logits_series, epoch_idx)

        if epoch_idx != 0:
            plot_loss(loss_list)
            print ("Average Loss = %.8f" % (float(sum(loss_list[(0-num_batches):]))/num_batches))

plt.ioff()
plt.show()

############################################## end main ##############################################




############################################## test main #############################################

# x,y = generateData()
# print (x.shape)
# print (x)
# print (y)

############################################ end test main ###########################################