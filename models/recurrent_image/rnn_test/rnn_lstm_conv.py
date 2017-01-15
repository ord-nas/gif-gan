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

video_length = 16
enable_relu = True
enable_bn = True

num_epochs = 1000
batch_size = 64
num_classes = 2

output_channels = 3
image_dimension = 32
image_size = image_dimension * image_dimension

filter_dimension = [5, 5, 5, 5]
stride = [2, 2, 2, 2]

state_channels = 4
state_dimension = int(image_dimension / 16)
state_size = state_channels * state_dimension * state_dimension
# state_size = image_size * output_channels

output_shape = [[batch_size, state_dimension * stride[0], state_dimension * stride[0], 8],
                [batch_size, state_dimension * stride[0] * stride[1], state_dimension * stride[0]* stride[1], 8],
                [batch_size, state_dimension * stride[0] * stride[1] * stride[2], state_dimension * stride[0]* stride[1] * stride[2], 4],
                [batch_size, image_dimension, image_dimension, output_channels]]

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
            # im = cv2.cvtColor(cv2.resize(im, (image_dimension, image_dimension)), 6)
            im = cv2.resize(im, (image_dimension, image_dimension))
            x[img_idx][frame_num] = im.reshape((image_size * output_channels))
            frame_num += 1

        cap.release()
    samples_path_list = samples_path_list[batch_size:]
    # return (x[:,:-1,:],x[:,1:,:])
    return x

def save_sample(sample, epoch_idx, batchX):
    for im_num in range(10):
        for frame_num in range(video_length):
            im = (np.array(sample)[frame_num][im_num].reshape((image_dimension, image_dimension, output_channels)) * 256).astype(int)
            output_path = os.path.join(saved_sample_path, str(epoch_idx) + "_" + str(im_num) + "_" + str(frame_num) + ".jpg")
            cv2.imwrite(output_path, im)

def plot_loss(loss_list):
    plt.cla()
    plt.plot(loss_list)
    plt.draw()
    plt.pause(0.0001)

################################################ main ################################################

batchX_placeholder = tf.placeholder(tf.float32, [batch_size, video_length + 1, image_size * output_channels])

# Unpack columns
inputs_series = tf.unpack(batchX_placeholder[:,:-1,:]/256, axis=1)
labels_series = tf.unpack(batchX_placeholder[:,1:,:]/256, axis=1)

cell_state = tf.placeholder(tf.float32, [batch_size, state_size])
hidden_state = tf.placeholder(tf.float32, [batch_size, state_size])
init_state = tf.nn.rnn_cell.LSTMStateTuple(cell_state, hidden_state)

# Forward passes
cell = tf.nn.rnn_cell.BasicLSTMCell(state_size, state_is_tuple=True)
states_series, current_state = tf.nn.rnn(cell, inputs_series, init_state)

# Filter for deconvolutions
f1 = tf.Variable(np.random.rand(filter_dimension[0], filter_dimension[0], output_shape[0][3], state_channels),dtype=tf.float32)
f2 = tf.Variable(np.random.rand(filter_dimension[1], filter_dimension[1], output_shape[1][3], output_shape[0][3]),dtype=tf.float32)
f3 = tf.Variable(np.random.rand(filter_dimension[2], filter_dimension[2], output_shape[2][3], output_shape[1][3]),dtype=tf.float32)
f4 = tf.Variable(np.random.rand(filter_dimension[3], filter_dimension[3], output_shape[3][3], output_shape[2][3]),dtype=tf.float32)

logits_series = []

for state in states_series:
    input_layer = tf.reshape(state, [batch_size, state_dimension, state_dimension, state_channels])
    
    if enable_bn:
        mean_1, variance_1 = tf.nn.moments(input_layer, axes = [0, 1, 2])
        bn_1 = tf.nn.batch_normalization(input_layer, mean_1, variance_1, None, None, 1e-3)
    else:
        bn_1 = input_layer
    if enable_relu:
        relu_1 = tf.nn.relu(bn_1)
    else:
        relu_1 = bn_1
    deconv_1 = tf.nn.conv2d_transpose(relu_1, f1, output_shape[0], [1,stride[0],stride[0],1], "SAME")

    if enable_bn:
        mean_2, variance_2 = tf.nn.moments(deconv_1, axes = [0, 1, 2])
        bn_2 = tf.nn.batch_normalization(deconv_1, mean_2, variance_2, None, None, 1e-3)
    else:
        bn_2 = deconv_1
    if enable_relu:
        relu_2 = tf.nn.relu(bn_2)
    else:
        relu_2 = bn_2
    deconv_2 = tf.nn.conv2d_transpose(relu_2, f2, output_shape[1], [1,stride[1],stride[1],1], "SAME")

    if enable_bn:
        mean_3, variance_3 = tf.nn.moments(deconv_2, axes = [0, 1, 2])
        bn_3 = tf.nn.batch_normalization(deconv_2, mean_3, variance_3, None, None, 1e-3)
    else:
        bn_3 = deconv_2
    if enable_relu:
        relu_3 = tf.nn.relu(bn_3)
    else:
        relu_3 = bn_3
    deconv_3 = tf.nn.conv2d_transpose(relu_3, f3, output_shape[2], [1,stride[2],stride[2],1], "SAME")

    if enable_bn:
        mean_4, variance_4 = tf.nn.moments(deconv_3, axes = [0, 1, 2])
        bn_4 = tf.nn.batch_normalization(deconv_3, mean_4, variance_4, None, None, 1e-3)
    else:
        bn_4 = deconv_3
    if enable_relu:
        relu_4 = tf.nn.relu(bn_4)
    else:
        relu_4 = bn_4
    deconv_4 = tf.nn.conv2d_transpose(relu_4, f4, output_shape[3], [1,stride[3],stride[3],1], "SAME")

    logits_series.append(tf.reshape((tf.tanh(deconv_4) + 1) / 2, [batch_size, image_size * output_channels]))
    # logits_series.append(tf.reshape(tf.sigmoid(deconv_4), [batch_size, image_size * output_channels]))

# losses = [tf.nn.sigmoid_cross_entropy_with_logits(logits, labels) for logits, labels in zip(logits_series,labels_series)]
losses = [tf.nn.l2_loss(logits - labels) for logits, labels in zip(logits_series,labels_series)]
total_loss = tf.reduce_mean(losses) / (batch_size * image_size * output_channels)

# train_step = tf.train.GradientDescentOptimizer(1).minimize(total_loss)
# train_step = tf.train.AdagradOptimizer(0.3).minimize(total_loss)
# train_step = tf.train.AdamOptimizer(0.0003).minimize(total_loss)
train_step = tf.train.AdadeltaOptimizer().minimize(total_loss)

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    plt.ion()
    plt.figure()
    plt.show()
    loss_list = []

    # data preparation
    samples_path_list = []
    batch_list = []

    for i in range(42, 48):
        samples_path_list += glob(os.path.join(input_path, str(i), "*.mp4"))
        # print (len(samples_path_list))
    
    samples_path_list = samples_path_list[:512] # quick test
    print ("Number of images = " + str(len(samples_path_list)))
    num_batches = int(len(samples_path_list) / batch_size)
    print ("Number of batches = " + str(num_batches))
    
    for batch_idx in range(num_batches):
        batch_list.append(load_batch().astype(np.float32))

    for epoch_idx in range(num_epochs):
        # samples_path_list = samples_path_list_full[:]
        _current_cell_state = np.zeros((batch_size, state_size))
        _current_hidden_state = np.zeros((batch_size, state_size))

        print("New epoch", epoch_idx)
        batch_loss = 0
        for batch_idx in range(num_batches):
            batchX = batch_list[batch_idx]
            # batchX = load_batch().astype(np.float32)

            _total_loss, _train_step, _current_state, _logits_series, _f1= sess.run(
                [total_loss, train_step, current_state, logits_series, f1],
                feed_dict={
                    batchX_placeholder: batchX,
                    cell_state: _current_cell_state,
                    hidden_state: _current_hidden_state
                })

            _current_cell_state, _current_hidden_state = _current_state

            if epoch_idx != 0 and batch_idx % 10 == 0:
                loss_list.append(_total_loss)
            batch_loss += _total_loss

            if batch_idx % 10 == 0:
                print("Step",batch_idx, "Loss", _total_loss)
                # print("f1: %.8f" % _f1[0][0][0][0])
        
        if epoch_idx % 10 == 0:
            save_sample(_logits_series, epoch_idx, batch_list[num_batches-1])

        if epoch_idx != 0:
            plot_loss(loss_list)
        
        print ("Average Loss = %.8f" % (batch_loss / num_batches))

plt.ioff()
plt.show()

############################################## end main ##############################################