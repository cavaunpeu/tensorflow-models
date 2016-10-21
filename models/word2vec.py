import numpy as np
import tensorflow as tf

from helpers.model import TensorFlowBaseModel, graph_node


class Word2Vec(TensorFlowBaseModel):

    def __init__(self, dataset, vocabulary_size, embedding_vector_size,
        negative_examples_to_sample_in_softmax, learning_rate=1.):
        self.dataset = dataset
        self._vocabulary_size = vocabulary_size
        self._embedding_vector_size = embedding_vector_size
        self._negative_examples_to_sample_in_softmax = negative_examples_to_sample_in_softmax
        self._learning_rate = learning_rate
        super().__init__()

    @property
    def _graph_nodes(self):
        return [self._define_network_parameters, self.compute_loss, self.optimize, self.compute_normalized_embeddings]

    @graph_node
    def _define_network_parameters(self):
        self._embeddings = tf.Variable(
            initial_value=tf.random_uniform([self._vocabulary_size, self._embedding_vector_size], -1., 1.)
        )
        self._second_layer_weights = tf.Variable(
            initial_value=tf.truncated_normal([self._vocabulary_size, self._embedding_vector_size],
                                             stddev=1./np.sqrt(self._embedding_vector_size))
        )
        self._second_layer_biases = tf.Variable(tf.zeros([self._vocabulary_size]))

    @graph_node
    def compute_loss(self):
        embedding = tf.nn.embedding_lookup(params=self._embeddings, ids=self.dataset.data)
        total_sampled_softmax_loss = tf.nn.sampled_softmax_loss(
            weights=self._second_layer_weights,
            biases=self._second_layer_biases,
            inputs=embedding,
            labels=self.dataset.labels,
            num_sampled=self._negative_examples_to_sample_in_softmax,
            num_classes=self._vocabulary_size
        )
        return tf.reduce_mean(total_sampled_softmax_loss)

    @graph_node
    def optimize(self):
        loss = self.compute_loss()
        return tf.train.AdagradOptimizer(self._learning_rate).minimize(loss)

    @graph_node
    def compute_normalized_embeddings(self):
        embedding_norms = tf.sqrt(
            tf.reduce_sum(tf.square(self._embeddings), reduction_indices=1, keep_dims=True)
        )
        return self._embeddings / embedding_norms
