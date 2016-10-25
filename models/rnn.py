import tensorflow as tf

from helpers.dataset import TensorFlowDataset
from helpers.model import TensorFlowBaseModel, graph_node


class RNN(TensorFlowBaseModel):

    RNN_CELL_SCOPE = 'rnn_cell'
    SOFTMAX_LAYER_SCOPE = 'softmax_layer'

    def __init__(self, dataset, n_classes, hidden_state_size, learning_rate):
        self.dataset = dataset
        self._n_classes = n_classes
        self._hidden_state_size = hidden_state_size
        self._learning_rate = learning_rate
        self._unrolled_dataset = None
        super().__init__()

    @property
    def _graph_nodes(self):
        return [self._initialize_rnn_cell_parameters, self._compute_logits, self.optimize, self.compute_loss]

    @graph_node
    def _unroll_dataset(self):
        data = tf.one_hot(self.dataset.data, self._n_classes)
        labels = tf.one_hot(self.dataset.labels, self._n_classes)
        rnn_inputs = tf.unpack(data, axis=1)
        rnn_labels = tf.unpack(labels, axis=1)
        rnn_labels = [tf.argmax(rnn_label, dimension=1) for rnn_label in rnn_labels]
        return TensorFlowDataset(data=rnn_inputs, labels=rnn_labels)

    @graph_node
    def _initialize_rnn_cell_parameters(self):
        with tf.variable_scope(name_or_scope=self.RNN_CELL_SCOPE):
            # add in a tf.truncated_normal() initializer here later
            W = tf.get_variable('W', shape=[self._n_classes + self._hidden_state_size, self._hidden_state_size])
            b = tf.get_variable('b', shape=[self._hidden_state_size], initializer=tf.constant_initializer(0.))

    @graph_node
    def _compute_logits(self):
        self._unrolled_dataset = self._unroll_dataset()
        self._initialize_rnn_cell_parameters()
        batch_size = self.dataset.data.get_shape()[0]

        initial_hidden_state = tf.zeros(dtype=tf.float32, shape=[batch_size, self._hidden_state_size])
        hidden_states = [initial_hidden_state]

        for rnn_input in self._unrolled_dataset.data:
            hidden_state = self._rnn_cell(rnn_input=rnn_input, hidden_state=hidden_states[-1])
            hidden_states.append(hidden_state)

        with tf.variable_scope(name_or_scope=self.SOFTMAX_LAYER_SCOPE):
            W = tf.get_variable('W', shape=[self._hidden_state_size, self._n_classes])
            b = tf.get_variable('b', shape=[self._n_classes])
            return [tf.matmul(hidden_state, W) + b for hidden_state in hidden_states[1:]]

    @graph_node
    def _compute_predictions(self):
        logits = self._compute_logits()
        return [tf.nn.softmax(logit) for logit in logits]

    @graph_node
    def optimize(self):
        loss = self.compute_loss()
        return tf.train.AdagradOptimizer(self._learning_rate).minimize(loss)

    @graph_node
    def compute_loss(self):
        logits = self._compute_logits()
        losses = [tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels) for \
                  logits, labels in zip(logits, self._unrolled_dataset.labels)]
        return tf.reduce_mean(losses)

    @graph_node
    def compute_accuracy(self):
        predictions = self._compute_predictions()
        labels = tf.pack(self._unrolled_dataset.labels)
        mistakes = tf.not_equal(labels, tf.argmax(predictions, dimension=2))
        return 1 - tf.reduce_mean(tf.cast(mistakes, tf.float32))

    def _rnn_cell(self, rnn_input, hidden_state):
        with tf.variable_scope(name_or_scope=self.RNN_CELL_SCOPE, reuse=True):
            W = tf.get_variable('W')
            b = tf.get_variable('b')
            rnn_input = tf.concat(concat_dim=1, values=[rnn_input, hidden_state])
            return tf.tanh( tf.matmul(rnn_input, W) + b )
