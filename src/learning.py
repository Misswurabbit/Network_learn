import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


def layer(inputs, in_size, out_size, n_layer, activate_function=None):
    layer_name = 'Layer%s' % n_layer
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            weights = tf.Variable(tf.random_normal([in_size, out_size]), name='weights')
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.random_normal([1, out_size]), name='biases')
        wx_plus_b = tf.matmul(inputs, weights) + biases
        if activate_function is None:
            outputs = wx_plus_b
        else:
            outputs = activate_function(wx_plus_b)
        return outputs


def compute_accuracy(test_xs, test_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: test_xs})
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(test_ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: test_xs, ys: test_ys})
    return result


# x_data = np.linspace(-1, 1, 300, dtype=np.float32)[:, np.newaxis]
# noise = np.random.normal(0, 0.05, x_data.shape).astype(np.float32)
# y_data = np.square(x_data) + noise - 0.5


with tf.name_scope('input'):
    xs = tf.placeholder(dtype=tf.float32, shape=[None, 784], name='x_input')
    ys = tf.placeholder(dtype=tf.float32, shape=[None, 10], name='y_input')

prediction = layer(xs, 784, 10, 1, activate_function=tf.nn.softmax)

with tf.name_scope('loss'):
    # loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), axis=1), name='loss')
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), axis=1))

with tf.name_scope('train'):
    train = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    # writer = tf.summary.FileWriter('', sess.graph)
    for num in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train, feed_dict={xs: batch_xs, ys: batch_ys})
        if num % 50 == 0:
            print(compute_accuracy(mnist.test.images, mnist.test.labels))
