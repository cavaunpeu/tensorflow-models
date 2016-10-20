import tensorflow as tf

from willwolf.tensorflow_helpers.model import TensorFlowBaseModel, graph_node


class LogisticRegression(TensorFlowBaseModel):

    def __init__(self, dataset, learning_rate, beta=None):
        self.dataset = dataset
        self._learning_rate = learning_rate
        self._beta = beta
        super().__init__()

    @property
    def _graph_nodes(self):
        return [self._compute_logits, self.optimize, self.compute_loss, self.compute_accuracy]

    @graph_node
    def _compute_logits(self):
        num_inputs = int(self.dataset.data.get_shape()[1])
        num_labels = int(self.dataset.labels.get_shape()[1])
        self._weights = tf.Variable(
            initial_value=tf.truncated_normal(shape=(num_inputs, num_labels)),
            name='weights'
        )
        self._biases = tf.Variable(
            initial_value=tf.zeros(shape=(num_labels,)),
            name='biases'
        )
        return tf.matmul(self.dataset.data, self._weights) + self._biases

    @graph_node
    def _compute_predictions(self):
        logits = self._compute_logits()
        return tf.nn.softmax(logits)

    @graph_node
    def optimize(self):
        loss = self.compute_loss()
        return tf.train.GradientDescentOptimizer(learning_rate=self._learning_rate).minimize(loss)

    @graph_node
    def compute_loss(self):
        logits = self._compute_logits()
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.dataset.labels)
        if self._beta:
            loss += self._beta * tf.nn.l2_loss(self._weights) + \
                    self._beta * tf.nn.l2_loss(self._biases)
        return tf.reduce_mean(loss)

    @graph_node
    def compute_accuracy(self):
        predictions = self._compute_predictions()
        mistakes = tf.not_equal(
            tf.argmax(self.dataset.labels, 1), tf.argmax(predictions, 1)
        )
        return 1 - tf.reduce_mean(tf.cast(mistakes, tf.float32))
