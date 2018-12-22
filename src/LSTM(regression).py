import tensorflow as tf
import numpy as np
from tensorflow.contrib.rnn import BasicLSTMCell

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
        self.ys = tf.placeholder(dtype=tf.float32, shape=[None, self.step_size, self.out_size])
        self._weight_get()
        self._biases_get()
        self._add_input_layer()
        self._add_cell()
        self._add_output_layer()
        self._compute_loss()
        self.train_op = self.train()

    def _add_input_layer(self):
        # reshape to 2D (-1,in_size)
        inputs = tf.reshape(self.xs, shape=[-1, self.in_size])
        # compute weights*inputs+biases,the shape changes to (-1,cell_size)
        wx_plus_b = tf.matmul(inputs, self.weight['in']) + self.biases['in']
        # reshaspe the data to 3D
        self.input_layer_data = tf.reshape(wx_plus_b, [-1, self.step_size, self.cell_size])

    def _add_cell(self):
        # init the lstm cell
        lstm = BasicLSTMCell(num_units=self.cell_size, forget_bias=1.0, state_is_tuple=True)
        self.init_state = lstm.zero_state(self.batch_size, dtype=tf.float32)
        self.cell_outputs, self.cell_final_state = tf.nn.dynamic_rnn(cell=lstm, inputs=self.input_layer_data,
                                                                     initial_state=self.init_state, time_major=False)

    def _add_output_layer(self):
        # define the outlayer
        # reshape the output shape (-1,cell_size)
        outputs = tf.reshape(self.cell_outputs, shape=[-1, self.cell_size])
        self.outputs = tf.matmul(outputs, self.weight['out']) + self.biases['out']

    def _compute_loss(self):
        # compute the loss
        losses = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
            [tf.reshape(self.outputs, [-1], name='reshape_pred')],
            [tf.reshape(self.ys, [-1], name='reshape_target')],
            [tf.ones([self.batch_size * self.step_size], dtype=tf.float32)],
            average_across_timesteps=True,
            softmax_loss_function=self.ms_error,
            name='losses'
        )
        self.loss = tf.div(tf.reduce_sum(losses), self.batch_size)

    @staticmethod
    def ms_error(labels, logits):
        return tf.square(tf.subtract(labels, logits))

    def train(self):
        return tf.train.AdamOptimizer(LR).minimize(self.loss)

    def _weight_get(self):
        # get the init weight
        self.weight = {'in': tf.Variable(tf.random_normal(shape=[self.in_size, self.cell_size])),
                       'out': tf.Variable(tf.random_normal(shape=[self.cell_size, self.out_size]))
                       }

    def _biases_get(self):
        # get the init weight
        self.biases = {'in': tf.Variable(tf.random_normal(shape=[self.cell_size])),
                       'out': tf.Variable(tf.random_normal(shape=[self.out_size]))
                       }


# main entrance
##################################################################
if __name__ == '__main__':
    with tf.Session() as sess:

        model = LSTM(step_size=TIME_STEPS, in_size=INPUT_SIZE, out_size=OUTPUT_SIZE, cell_size=CELL_SIZE,
                     batch_size=BATCH_SIZE)
        sess.run(tf.global_variables_initializer())
        for num in range(1000):
            seq, res, xs = get_batch()  # 提取 batch data
            if num == 0:
                # 初始化 data
                feed_dict = {
                    model.xs: seq,
                    model.ys: res,
                }
            else:
                feed_dict = {
                    model.xs: seq,
                    model.ys: res,
                    model.init_state: state  # 保持 state 的连续性
                }

            # 训练
            _, cost, state, pred = sess.run(
                [model.train_op, model.loss, model.cell_final_state, model.outputs],
                feed_dict=feed_dict)

            # 打印 cost 结果
            if num % 20 == 0:
                print('cost: ', round(cost, 4))

    print('Over')
