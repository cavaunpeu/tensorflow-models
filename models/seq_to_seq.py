import tensorflow as tf

from models.rnn import LSTM

from helpers.dataset import TensorFlowDataset
from helpers.model import TensorFlowBaseModel, graph_node


class SequenceToSequenceLSTM(TensorFlowBaseModel):

    def __init__(self, dataset, n_classes, embedding_layer_size, hidden_state_size, learning_rate):
        self.dataset = dataset
        self._n_classes = n_classes
        self._embedding_layer_size = embedding_layer_size
        self._hidden_state_size = hidden_state_size
        self._learning_rate = learning_rate
        super().__init__()

    @property
    def _graph_nodes(self):
        return [self._initialize_encoder_and_decoder_networks, self._compute_logits, self.optimize, self.compute_loss]

    @graph_node
    def _initialize_encoder_and_decoder_networks(self):
        with tf.variable_scope(name_or_scope='encoder'):
            # encoder dataset.labels is set to a dummy tensor of zeros
            self._encoder = LSTM(
                dataset=TensorFlowDataset(
                    data=self.dataset.data,
                    labels=tf.zeros_like(self.dataset.data)
                ),
                n_classes=self._n_classes,
                embedding_layer_size=self._embedding_layer_size,
                hidden_state_size=self._hidden_state_size,
                learning_rate=self._learning_rate,
            )
        with tf.variable_scope(name_or_scope='decoder'):
            self._decoder = LSTM(
                dataset=TensorFlowDataset(
                    data=self.dataset.labels[:, :self.dataset.labels.get_shape()[1]],
                    labels=self.dataset.labels[:, 1:]
                ),
                n_classes=self._n_classes,
                embedding_layer_size=self._embedding_layer_size,
                hidden_state_size=self._hidden_state_size,
                learning_rate=self._learning_rate
            )

    @graph_node
    def _compute_logits(self):
        encoder_hidden_states = self._encoder.feed_forward()
        decoder_hidden_states = self._decoder.feed_forward(initial_hidden_state=encoder_hidden_states[-1])
        with tf.variable_scope(name_or_scope='softmax_layer'):
            W_shape = [self._hidden_state_size, self._n_classes]
            W = tf.get_variable('W', initializer=tf.truncated_normal(shape=W_shape, mean=-.1, stddev=.1))
            b = tf.get_variable('b', shape=[self._n_classes])
            return [tf.matmul(hidden_state, W) + b for hidden_state in decoder_hidden_states[1:]]

    @graph_node
    def predict_next_token(self):
        logits = self._compute_logits()
        return [tf.nn.softmax(logit) for logit in logits]

    @graph_node
    def compute_loss(self):
        logits = self._compute_logits()
        losses = [tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels) for \
                  logits, labels in zip(logits, self._decoder.unrolled_dataset.labels)]
        return tf.reduce_mean(losses)

    @graph_node
    def optimize(self):
        loss = self.compute_loss()
        return tf.train.AdagradOptimizer(self._learning_rate).minimize(loss)
