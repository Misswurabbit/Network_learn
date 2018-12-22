import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.ensemble import RandomForestClassifier
# 导入数据
#############################################################
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# hyperparameters
#############################################################
lr = 0.001  # learning rate
training_iters = 100000  # 迭代训练次数
batch_size = 128  # 一批数据的数量
n_inputs = 28  # MNIST data input(img shape:28*28)
n_steps = 28  # time steps
n_hidden_units = 128  # neurons in hidden layer
n_classes = 10  # MNIST classes(0-9 digits)

# placeholder
#############################################################
# shape(128, 28, 28)
xs = tf.placeholder(dtype=tf.float32, shape=[None, n_steps, n_inputs], name='x_input')
# shape(128,10)
ys = tf.placeholder(dtype=tf.float32, shape=[None, n_classes], name='y_input')

# weights and biases
#############################################################
weights = {
    # shape (28,128)
    'in': tf.Variable(tf.random_normal(shape=[n_inputs, n_hidden_units])),
    # shape (128,10)
    'out': tf.Variable(tf.random_normal(shape=[n_hidden_units, n_classes]))
}
biases = {
    # shape (128) ,you can take it as (,128) when it used next
    'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_units])),
    # shape (10) ,you can take it as (,10) when it used next
    'out': tf.Variable(tf.constant(0.1, shape=[n_classes]))
}


# define the rnn layer
#############################################################
def RNN(inputs, weight, biase):
    # reshape matrix to (128*28,28) for matmul operation
    inputs = tf.reshape(inputs, [-1, n_inputs])
    # compute the weights and the biases,shape changes to (128*28,128)
    inputs = tf.matmul(inputs, weight['in']) + biase['in']
    # change the shape to (128, 28, 128)
    inputs = tf.reshape(inputs, [-1, n_steps, n_hidden_units])
    # init the cell
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True)
    init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)
    outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, inputs, initial_state=init_state, time_major=False)
    return tf.matmul(final_state[1], weight['out']) + biase['out']


# define the flow
#############################################################
prediction = RNN(xs, weights, biases)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=ys))
train = tf.train.AdamOptimizer(lr).minimize(loss)

# run
#############################################################
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    step = 0
    while step * batch_size < training_iters:
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        batch_xs = batch_xs.reshape([batch_size, n_steps, n_inputs])
        sess.run(train, feed_dict={xs: batch_xs, ys: batch_ys})
        if step % 10 == 0:
            print(sess.run(loss, feed_dict={xs: batch_xs, ys: batch_ys}))
        step += 1
print('Over')
