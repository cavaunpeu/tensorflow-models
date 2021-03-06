import tensorflow as tf

from helpers.dataset import TensorFlowDataset
from helpers.model import TensorFlowBaseModel, graph_node


class RNN(TensorFlowBaseModel):

    RNN_CELL_SCOPE = 'rnn_cell'
    SOFTMAX_LAYER_SCOPE = 'softmax_layer'

    def __init__(self, dataset, n_classes, embedding_layer_size, hidden_state_size, learning_rate):
        self.dataset = dataset
        self._n_classes = n_classes
        self._embedding_layer_size = embedding_layer_size
        self._hidden_state_size = hidden_state_size
        self._learning_rate = learning_rate
        self.unrolled_dataset = None
        super().__init__()

    @property
    def _graph_nodes(self):
        return [self._initialize_rnn_cell_parameters, self._compute_logits, self.optimize, self.compute_loss]

    @graph_node
    def _unroll_dataset(self):
        embeddings = tf.Variable(initial_value=tf.random_uniform(shape=[self._n_classes + 1, self._embedding_layer_size], minval=-1., maxval=1.))
        data = tf.nn.embedding_lookup(params=embeddings, ids=self.dataset.data)
        labels = tf.one_hot(self.dataset.labels, self._n_classes)

        unrolled_data = tf.unpack(data, axis=1)
        unrolled_labels = tf.unpack(labels, axis=1)
        unrolled_labels = [tf.argmax(rnn_label, dimension=1) for rnn_label in unrolled_labels]

        return TensorFlowDataset(data=unrolled_data, labels=unrolled_labels)

    @graph_node
    def _initialize_rnn_cell_parameters(self):
        with tf.variable_scope(name_or_scope=self.RNN_CELL_SCOPE):
            W_shape = [self._embedding_layer_size + self._hidden_state_size, self._hidden_state_size]
            W = tf.get_variable('W', initializer=tf.truncated_normal(shape=W_shape, mean=-.1, stddev=.1))
            b = tf.get_variable('b', shape=[self._hidden_state_size], initializer=tf.constant_initializer(0.))

    @graph_node
    def _compute_logits(self):
        hidden_states = self.feed_forward()
        with tf.variable_scope(name_or_scope=self.SOFTMAX_LAYER_SCOPE):
            W_shape = [self._hidden_state_size, self._n_classes]
            W = tf.get_variable('W', initializer=tf.truncated_normal(shape=W_shape, mean=-.1, stddev=.1))
            b = tf.get_variable('b', shape=[self._n_classes])
            return [tf.matmul(hidden_state, W) + b for hidden_state in hidden_states[1:]]

    @graph_node
    def feed_forward(self, initial_hidden_state=None):
        self.unrolled_dataset = self._unroll_dataset()
        self._initialize_rnn_cell_parameters()
        batch_size = self.dataset.data.get_shape()[0]

        initial_hidden_state = initial_hidden_state or tf.zeros(dtype=tf.float32, shape=[batch_size, self._hidden_state_size])
        hidden_states = [initial_hidden_state]

        for rnn_input in self.unrolled_dataset.data:
            hidden_state = self._rnn_cell(rnn_input=rnn_input, hidden_state=hidden_states[-1])
            hidden_states.append(hidden_state)
        return hidden_states

    @graph_node
    def optimize(self):
        loss = self.compute_loss()
        return tf.train.AdagradOptimizer(self._learning_rate).minimize(loss)

    @graph_node
    def compute_loss(self):
        logits = self._compute_logits()
        losses = [tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels) for \
                  logits, labels in zip(logits, self.unrolled_dataset.labels)]
        return tf.reduce_mean(losses)

    def _rnn_cell(self, rnn_input, hidden_state):
        with tf.variable_scope(name_or_scope=self.RNN_CELL_SCOPE, reuse=True):
            W = tf.get_variable('W')
            b = tf.get_variable('b')
            rnn_input = tf.concat(concat_dim=1, values=[rnn_input, hidden_state])
            return tf.tanh( tf.matmul(rnn_input, W) + b )


class LSTM(RNN):

    @graph_node
    def _initialize_rnn_cell_parameters(self):
        with tf.variable_scope(name_or_scope=self.RNN_CELL_SCOPE):
            W_shape = [4*(self._embedding_layer_size + self._hidden_state_size), 4*self._hidden_state_size]
            W = tf.get_variable('W', initializer=tf.truncated_normal(shape=W_shape, mean=-.1, stddev=.1))
            b = tf.get_variable('b', shape=[4*self._hidden_state_size], initializer=tf.constant_initializer(0.))

    @graph_node
    def feed_forward(self, initial_hidden_state=None):
        self.unrolled_dataset = self._unroll_dataset()
        self._initialize_rnn_cell_parameters()
        batch_size = tf.shape(self.dataset.data)[0]

        initial_hidden_state = initial_hidden_state or tf.zeros(dtype=tf.float32, shape=[batch_size, self._hidden_state_size])
        initial_memory_cell = tf.zeros(dtype=tf.float32, shape=[batch_size, self._hidden_state_size])
        hidden_states = [initial_hidden_state]
        memory_cells = [initial_memory_cell]

        for rnn_input in self.unrolled_dataset.data:
            hidden_state, memory_cell = self._rnn_cell(rnn_input=rnn_input, hidden_state=hidden_states[-1], memory_cell=memory_cells[-1])
            hidden_states.append(hidden_state)
            memory_cells.append(memory_cell)
        return hidden_states

    def _rnn_cell(self, rnn_input, hidden_state, memory_cell):
        with tf.variable_scope(name_or_scope=self.RNN_CELL_SCOPE, reuse=True):
            W = tf.get_variable('W')
            b = tf.get_variable('b')
            rnn_input = tf.concat(concat_dim=1, values=4*[rnn_input, hidden_state])

            Z = tf.matmul(rnn_input, W) + b
            f_i_o = tf.sigmoid( Z[:, :3*self._hidden_state_size] )
            candidate_memory_cell = tf.tanh( Z[:, 3*self._hidden_state_size:] )

            f, i, o = tf.split(split_dim=1, num_split=3, value=f_i_o)
            memory_cell = f * memory_cell + i * candidate_memory_cell
            hidden_state = o * tf.tanh(memory_cell)

            return hidden_state, memory_cell
