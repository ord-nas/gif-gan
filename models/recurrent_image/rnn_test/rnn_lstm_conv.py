from __future__ import print_function, division
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

num_epochs = 100
total_series_length = 50000
truncated_backprop_length = 4
num_classes = 2
output_channels = 8
image_dimension = 16
image_size = image_dimension * image_dimension
filter_dimension = image_dimension / 2
state_channels = 2
state_size = state_channels * image_size
# state_size = image_size * output_channels
echo_step = 3
batch_size = 5
num_batches = total_series_length//batch_size//truncated_backprop_length

def generateData():
    x = np.array(np.random.choice(2, total_series_length * image_size * output_channels, p=[0.5, 0.5]))
    y = np.roll(x, echo_step * image_size * output_channels)
    y[0:echo_step * image_size * output_channels] = 0
    # y = np.array(np.random.choice(2, total_series_length * image_size, p=[0.5, 0.5]))

    # x = np.repeat(x, image_size, axis = 0)
    # y = np.repeat(y, image_size, axis = 0)

    x = x.reshape((batch_size, -1, image_size * output_channels))  # The first index changing slowest, subseries as rows
    y = y.reshape((batch_size, -1, image_size * output_channels))

    return (x, y)

def plot(loss_list, predictions_series, batchX, batchY):
    plt.subplot(2, 3, 1)
    plt.cla()
    plt.plot(loss_list)

    for batch_series_idx in range(5):
        one_hot_output_series = np.array(predictions_series)[:, batch_series_idx, :]
        single_output_series = np.array([(1 if out[0] < 0.5 else 0) for out in one_hot_output_series])

        plt.subplot(2, 3, batch_series_idx + 2)
        plt.cla()
        plt.axis([0, truncated_backprop_length, 0, 2])
        left_offset = range(truncated_backprop_length)
        plt.bar(left_offset, batchX[batch_series_idx, :], width=1, color="blue")
        plt.bar(left_offset, batchY[batch_series_idx, :] * 0.5, width=1, color="red")
        plt.bar(left_offset, single_output_series * 0.3, width=1, color="green")

    plt.draw()
    plt.pause(0.0001)

def plot_loss(loss_list):
    plt.cla()
    plt.plot(loss_list)
    plt.draw()
    plt.pause(0.0001)

################################################ main ################################################

batchX_placeholder = tf.placeholder(tf.float32, [batch_size, truncated_backprop_length, image_size * output_channels])
batchY_placeholder = tf.placeholder(tf.int32, [batch_size, truncated_backprop_length, image_size * output_channels])

cell_state = tf.placeholder(tf.float32, [batch_size, state_size])
hidden_state = tf.placeholder(tf.float32, [batch_size, state_size])
init_state = tf.nn.rnn_cell.LSTMStateTuple(cell_state, hidden_state)

# Weights and bias for fully connected implementation
# W2 = tf.Variable(np.random.rand(state_size, image_size * output_channels),dtype=tf.float32)
# b2 = tf.Variable(np.zeros((1, image_size * output_channels)), dtype=tf.float32)

# Filter for convolution implementation
f2 = tf.Variable(np.random.rand(filter_dimension, filter_dimension, state_channels, output_channels * num_classes),dtype=tf.float32)

# Unpack columns
inputs_series = tf.unpack(batchX_placeholder, axis=1)
labels_series = tf.unpack(batchY_placeholder, axis=1)

# print ("Input Shape:")
# print (inputs_series[0].get_shape())
# print ("")

# Forward passes
cell = tf.nn.rnn_cell.BasicLSTMCell(state_size, state_is_tuple=True)
states_series, current_state = tf.nn.rnn(cell, inputs_series, init_state)

# print ("State Shape:")
# print (states_series[0].get_shape())
# print ("")

# logits_series = [tf.reshape(tf.matmul(state, W2) + b2, [batch_size, image_size, num_classes]) for state in states_series] #Broadcasted addition
# logits_series = [tf.reshape(state, [batch_size, image_size, num_classes]) for state in states_series]
logits_series = [tf.reshape(tf.nn.conv2d(tf.reshape(state, [batch_size, image_dimension, image_dimension, state_channels]),
                                f2, [1,1,1,1], "SAME"), [batch_size, image_size * output_channels, num_classes]) for state in states_series]

print ("Logit Shape:")
print (logits_series[0].get_shape())
print ("")

predictions_series = [tf.nn.softmax(logits) for logits in logits_series]

print ("Prediction Shape:")
print (logits_series[0].get_shape())
print ("")

print ("Label Shape:")
print (labels_series[0].get_shape())
print ("")

losses = [tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels) for logits, labels in zip(logits_series,labels_series)]
total_loss = tf.reduce_mean(losses)

train_step = tf.train.AdagradOptimizer(0.3).minimize(total_loss)

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    plt.ion()
    plt.figure()
    plt.show()
    loss_list = []

    for epoch_idx in range(num_epochs):
        x,y = generateData()
        _current_cell_state = np.zeros((batch_size, state_size))
        _current_hidden_state = np.zeros((batch_size, state_size))

        print("New data, epoch", epoch_idx)

        for batch_idx in range(num_batches):
            start_idx = batch_idx * truncated_backprop_length
            end_idx = start_idx + truncated_backprop_length

            batchX = x[:,start_idx:end_idx]
            batchY = y[:,start_idx:end_idx]

            _total_loss, _train_step, _current_state, _predictions_series = sess.run(
                [total_loss, train_step, current_state, predictions_series],
                feed_dict={
                    batchX_placeholder: batchX,
                    batchY_placeholder: batchY,
                    cell_state: _current_cell_state,
                    hidden_state: _current_hidden_state

                })

            _current_cell_state, _current_hidden_state = _current_state

            if epoch_idx > 0:
                loss_list.append(_total_loss)

            if batch_idx%100 == 0:
                print("Step",batch_idx, "Loss", _total_loss)
                plot_loss(loss_list)
                # plot(loss_list, _predictions_series, batchX, batchY)

plt.ioff()
plt.show()

############################################## end main ##############################################




############################################## test main #############################################

# x,y = generateData()
# print (x.shape)
# print (x)
# print (y)

############################################ end test main ###########################################