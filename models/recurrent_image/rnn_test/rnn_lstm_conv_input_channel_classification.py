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

num_epochs = 100
batch_size = 64
num_classes = 4

output_channels = 3
image_dimension = 64
image_size = image_dimension * image_dimension

filter_dimension = 5
stride = 2

layer_shapes = [[batch_size, 4, 4, 256],
                [batch_size, 8, 8, 128],
                [batch_size, 16, 16, 64],
                [batch_size, 32, 32, 32],
                [batch_size, 64, 64, 3]]

final_output_shape = [batch_size, 64, 64, output_channels * num_classes * 4]

fc_size = layer_shapes[0][1] * layer_shapes[0][2] * layer_shapes[0][3]

state_size = 100

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
    for im_num in range(5):
        for frame_num in range(video_length):
            im = (np.array(sample)[frame_num][im_num].reshape((image_dimension, image_dimension, output_channels))).astype(int)
            output_path = os.path.join(saved_sample_path, str(epoch_idx) + "_" + str(im_num) + "_" + str(frame_num) + ".jpg")
            cv2.imwrite(output_path, im)

        for frame_num in range(video_length):
            im = (batchX[im_num][frame_num]).astype(int)
            output_path = os.path.join(saved_sample_path, str(epoch_idx) + "_" + str(im_num) + "_" + str(frame_num) + "_real.jpg")
            cv2.imwrite(output_path, im)

def plot_loss(loss_list):
    plt.cla()
    plt.plot(loss_list)
    plt.draw()
    plt.pause(0.0001)

################################################ main ################################################
# input
batchINPUT_placeholder = tf.placeholder(tf.int32, [batch_size, video_length + 1, image_dimension, image_dimension, output_channels])
batchX_placeholder = tf.cast(tf.slice(batchINPUT_placeholder, [0,0,0,0,0], [-1,video_length,-1,-1,-1]), tf.float32)
batchY_placeholder = tf.slice(batchINPUT_placeholder, [0,1,0,0,0], [-1,-1,-1,-1,-1])

# split channels for labels to reduce number of classes per channel
batchY_placeholder_1st_channel = tf.cast(batchY_placeholder / 64, tf.int32)
batchY_placeholder_2nd_channel = tf.cast((batchY_placeholder % 64) / 16, tf.int32)
batchY_placeholder_3rd_channel = tf.cast((batchY_placeholder % 16) / 4, tf.int32)
batchY_placeholder_4th_channel = tf.cast((batchY_placeholder % 4), tf.int32)
batchY_placeholder_splitted = tf.concat(4, [batchY_placeholder_1st_channel, 
                                            batchY_placeholder_2nd_channel,
                                            batchY_placeholder_3rd_channel,
                                            batchY_placeholder_4th_channel])

# Unpack columns
inputs_series_raw = tf.unpack(batchX_placeholder, axis=1)
labels_series = tf.unpack(batchY_placeholder_splitted, axis=1)

# RNN states
cell_state = tf.placeholder(tf.float32, [batch_size, state_size])
hidden_state = tf.placeholder(tf.float32, [batch_size, state_size])
init_state = tf.nn.rnn_cell.LSTMStateTuple(cell_state, hidden_state)

# Filters for input-to-RNN
conv_f1 = tf.Variable(np.random.rand(filter_dimension, filter_dimension, layer_shapes[4][3], layer_shapes[3][3]),dtype=tf.float32)
conv_f2 = tf.Variable(np.random.rand(filter_dimension, filter_dimension, layer_shapes[3][3], layer_shapes[2][3]),dtype=tf.float32)
conv_f3 = tf.Variable(np.random.rand(filter_dimension, filter_dimension, layer_shapes[2][3], layer_shapes[1][3]),dtype=tf.float32)
conv_f4 = tf.Variable(np.random.rand(filter_dimension, filter_dimension, layer_shapes[1][3], layer_shapes[0][3]),dtype=tf.float32)
conv_f_list = [conv_f1, conv_f2, conv_f3, conv_f4]
# input_fc_w = tf.Variable(np.random.rand(fc_size, state_size),dtype=tf.float32)
# input_fc_bias = tf.Variable(np.random.rand(1, state_size),dtype=tf.float32)

# Input-to-RNN
inputs_series = []
for input_layer in inputs_series_raw:
    data_in = input_layer
    # conv, norm, relu layers
    for i in range(4):
        input_conv = tf.nn.conv2d(data_in, conv_f_list[i], [1,stride,stride,1], "SAME")
        mean, variance = tf.nn.moments(input_conv, axes = [0, 1, 2])
        batch_norm = tf.nn.batch_normalization(input_conv, mean, variance, None, None, 1e-5)
        data_in = tf.nn.relu(batch_norm)

    # fc layers
    # inputs_series.append(tf.matmul(tf.reshape(data_in, [batch_size, fc_size]), input_fc_w) + input_fc_bias)

    # skip fc layer (already in RNN cell)
    inputs_series.append(tf.reshape(data_in, [batch_size, fc_size]))

# RNN cell
cell = tf.nn.rnn_cell.BasicLSTMCell(state_size, state_is_tuple=True)
states_series, current_state = tf.nn.rnn(cell, inputs_series, init_state)

# Filters for states-to-output
output_fc_w = tf.Variable(np.random.rand(state_size, fc_size),dtype=tf.float32)
output_fc_bias = tf.Variable(np.random.rand(1, fc_size),dtype=tf.float32)
deconv_f1 = tf.Variable(np.random.rand(filter_dimension, filter_dimension, layer_shapes[1][3], layer_shapes[0][3]),dtype=tf.float32)
deconv_f2 = tf.Variable(np.random.rand(filter_dimension, filter_dimension, layer_shapes[2][3], layer_shapes[1][3]),dtype=tf.float32)
deconv_f3 = tf.Variable(np.random.rand(filter_dimension, filter_dimension, layer_shapes[3][3], layer_shapes[2][3]),dtype=tf.float32)
deconv_f4 = tf.Variable(np.random.rand(filter_dimension, filter_dimension, final_output_shape[3], layer_shapes[3][3]),dtype=tf.float32)
deconv_f_list = [deconv_f1, deconv_f2, deconv_f3, deconv_f4]

# States-to-Output
logits_series = []
for state in states_series:
    # fc layer
    data_in = tf.reshape(tf.matmul(state, output_fc_w) + output_fc_bias, 
                        [batch_size, layer_shapes[0][1], layer_shapes[0][2], layer_shapes[0][3]])
    # deconv, norm, relu layers
    for i in range(4):
        mean, variance = tf.nn.moments(data_in, axes = [0, 1, 2])
        batch_norm = tf.nn.batch_normalization(data_in, mean, variance, None, None, 1e-5)
        relu = tf.nn.relu(batch_norm)
        data_in = tf.nn.conv2d_transpose(relu, deconv_f_list[i], layer_shapes[i+1] if i != 3 else final_output_shape, [1,stride,stride,1], "SAME")
        
    # logits_series.append((tf.tanh(data_in) + 1) / 2)
    logits_series.append(tf.reshape(data_in, [batch_size, image_dimension, image_dimension, output_channels * 4, num_classes]))

# Loss Functions
# losses = [tf.nn.l2_loss(logits - labels) for logits, labels in zip(logits_series,labels_series)]
# total_loss = tf.reduce_mean(losses) / (batch_size * image_size * output_channels)

# predictions
predictions_series_splitted = [tf.argmax(tf.nn.softmax(logits), 4) for logits in logits_series]
# re-combine channels
predictions_series = [prediction[:,:,:,0:3] * 64 + \
                      prediction[:,:,:,3:6] * 16 + \
                      prediction[:,:,:,6:9] * 4 + \
                      prediction[:,:,:,9:12] \
                      for prediction in predictions_series_splitted]

# losses = [tf.nn.sigmoid_cross_entropy_with_logits(logits, labels) for logits, labels in zip(logits_series,labels_series)]
losses = [tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels) for logits, labels in zip(logits_series,labels_series)]
total_loss = tf.reduce_mean(losses)

# Train
train_step = tf.train.AdamOptimizer(0.001).minimize(total_loss)

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    plt.ion()
    plt.figure()
    plt.show()
    loss_list = []

    # data preparation
    samples_path_list_full = []
    batch_list = []

    for i in range(36, 48):
        samples_path_list_full += glob(os.path.join(input_path, str(i), "*.mp4"))
        # print (len(samples_path_list))
    
    samples_path_list_full = samples_path_list_full[:512] # quick test
    print ("Number of images = " + str(len(samples_path_list_full)))
    num_batches = int(len(samples_path_list_full) / batch_size)
    print ("Number of batches = " + str(num_batches))
    
    # for batch_idx in range(num_batches):
    #     batch_list.append(load_batch().astype(np.float32))

    for epoch_idx in range(num_epochs):
        samples_path_list = samples_path_list_full[:]
        _current_cell_state = np.zeros((batch_size, state_size))
        _current_hidden_state = np.zeros((batch_size, state_size))

        print("New epoch", epoch_idx)
        batch_loss = 0
        for batch_idx in range(num_batches):
            # batchX = batch_list[batch_idx]
            batchX = load_batch().astype(np.float32)

            _total_loss, _train_step, _current_state, _predictions_series= sess.run(
                [total_loss, train_step, current_state, predictions_series],
                feed_dict={
                    batchINPUT_placeholder: batchX,
                    cell_state: _current_cell_state,
                    hidden_state: _current_hidden_state
                })

            _current_cell_state, _current_hidden_state = _current_state

            loss_list.append(_total_loss)

            batch_loss += _total_loss

            if batch_idx % 5 == 0:
                print("Step",batch_idx, "Loss", _total_loss)

            if batch_idx % 20 == 0:
                plot_loss(loss_list)
                save_sample(_predictions_series, epoch_idx, batchX)
        
        print ("Average Loss = %.8f" % (batch_loss / num_batches))

plt.ioff()
plt.show()

############################################## end main ##############################################