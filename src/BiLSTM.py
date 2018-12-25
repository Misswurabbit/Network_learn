import tensorflow as tf
from tensorflow.contrib.rnn import BasicLSTMCell


class BiLSTM(object):
    def __init__(self, batch_size, step_size, in_size, out_size, hidden_layer, learning_rate):
        self.batch_size = batch_size
        self.step_size = step_size
        self.in_size = in_size
        self.out_size = out_size
        self.hidden_layer = hidden_layer
        self.learning_rate =learning_rate
        # init the input placeholder, its shape is (batch_size,step_size,in_size)
        self.xs = tf.placeholder(dtype=tf.float32, shape=[None, self.step_size, self.in_size])
        # init the output placeholder, its shape is (batch_size,step_size,in_size)
        self.ys = tf.placeholder(dtype=tf.float32, shape=[None, self.out_size])
        self._get_weight()
        self._get_biases()
        self._add_input_layer()
        self._add_bilstm_cell()
        self._add_output_layer()
        self._compute_loss()
        self.train = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

    def _add_input_layer(self):
        # reshape the input data to 2D, shape is (-1,in_size)
        inputs = tf.reshape(self.xs, shape=[-1, self.in_size])
        # calculate the wx_plus_b, shape is (-1, hidden_layer)
        wx_plus_b = tf.matmul(inputs, self.weight['in']) + self.biases['in']
        # reshape the result to the 3D,shape like (-1, step_size, hidden_layer)
        self.input_layer_data = tf.reshape(wx_plus_b, shape=[-1, self.step_size, self.hidden_layer])

    def _add_bilstm_cell(self):
        # init the lstm cells. one for fwlstm, another for bwlstm.
        fw_lstm = BasicLSTMCell(num_units=self.hidden_layer, forget_bias=1.0, state_is_tuple=True)
        bw_lstm = BasicLSTMCell(num_units=self.hidden_layer, forget_bias=1.0, state_is_tuple=True)
        # define the init state for bwlstm and fwlstm
        self.fw_init_state = fw_lstm.zero_state(self.batch_size, dtype=tf.float32)
        self.bw_init_state = bw_lstm.zero_state(self.batch_size, dtype=tf.float32)
        output, final_state = tf.nn.bidirectional_dynamic_rnn(cell_fw=fw_lstm, cell_bw=bw_lstm,
                                                              sequence_length=self.batch_size,
                                                              inputs=self.input_layer_data,
                                                              initial_state_bw=self.bw_init_state,
                                                              initial_state_fw=self.fw_init_state)
        self.bilstm_output = tf.concat(output, 2)
        self.fw_final_state = final_state[0]
        self.bw_final_state = final_state[1]

    def _add_output_layer(self):
        # reshape the shape of out_put data,shape like(-1, hidden layer)
        output = tf.reshape(self.bilstm_output, shape=[-1, self.hidden_layer])
        self.output = tf.matmul(self.weight['out'], output) + self.biases['out']

    def _compute_loss(self):
        # compute the loss of the bilstm
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.output, labels=self.ys))

    def _get_weight(self):
        self.weight = {'in': tf.Variable(tf.random_normal(shape=[self.in_size, self.hidden_layer])),
                       'out': tf.Variable(tf.random_normal(shape=[self.hidden_layer, self.out_size]))
                       }

    def _get_biases(self):
        self.biases = {'in': tf.Variable(tf.random_normal(shape=[self.hidden_layer])),
                       'out': tf.Variable(tf.random_normal(shape=[self.out_size]))
                       }
