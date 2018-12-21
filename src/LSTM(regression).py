import tensorflow as tf
import numpy as np

# define parameters
##################################################################
BATCH_START = 0  # 建立 batch data 时候的 index
TIME_STEPS = 20  # backpropagation through time 的 time_steps
BATCH_SIZE = 50
INPUT_SIZE = 1  # sin 数据输入 size
OUTPUT_SIZE = 1  # cos 数据输出 size
CELL_SIZE = 10  # RNN 的 hidden unit size
LR = 0.006  # learning rate


# create the data
##################################################################
def get_batch():
    global BATCH_START, TIME_STEPS
    # xs shape (50batch, 20steps)
    xs = np.arange(BATCH_START, BATCH_START + TIME_STEPS * BATCH_SIZE).reshape((BATCH_SIZE, TIME_STEPS)) / (10 * np.pi)
    seq = np.sin(xs)
    res = np.cos(xs)
    BATCH_START += TIME_STEPS
    # returned seq, res and xs: shape (batch, step, input)
    return [seq[:, :, np.newaxis], res[:, :, np.newaxis], xs]


# define LSTM
##################################################################
class LSTM(object):
    def __init__(self, in_size, out_size, cell_size, batch_size, step_size):
        self.in_size = in_size
        self.out_size = out_size
        self.cell_size = cell_size
        self.batch_size = batch_size
        self.step_size = step_size
        self.xs = tf.placeholder(dtype=tf.float32, shape=[None, self.step_size, self.in_size])
        self.ys = tf.placeholder(dtype=tf.float32, shape=[None, self.out_size])
        self.weight = self._weight_get()
        self.biases = self._biases_get()

    def _add_input_layer(self):
        # reshape to 2D (-1,in_size)
        inputs = tf.reshape(self.xs, shape=[-1, self.in_size])
        # compute weights*inputs+biases,the shape changes to (-1,cell_size)
        wx_plus_b = tf.matmul(inputs, self.weight['in']) + self.biases['in']
        # reshaspe the data to 3D
        self.input_layer_data = tf.reshape(wx_plus_b, [-1, self.step_size, self.cell_size])

    def _weight_get(self):
        # get the init weight
        init = {'in': tf.Variable(tf.random_normal(shape=[self.in_size, self.cell_size])),
                'out': tf.Variable(tf.random_normal(shape=[self.cell_size, self.out_size]))
                }
        return init

    def _biases_get(self):
        # get the init weight
        init = {'in': tf.Variable(tf.random_normal(shape=[self.cell_size])),
                'out': tf.Variable(tf.random_normal(shape=[self.out_size]))
                }
        return init

    print('Over')
