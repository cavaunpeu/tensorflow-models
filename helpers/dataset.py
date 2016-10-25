import random
import zipfile

import tensorflow as tf


class TensorFlowDataset:

    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def sample(self, size):
        indices = random.sample(population=range(len(self)), k=size)
        return self.data[indices], self.labels[indices]

    def __len__(self):
        return len(self.data)


def import_zip_file(path, n_characters=None):
    with zipfile.ZipFile(path) as file:
        filename = file.namelist()[0]
        corpus = file.read(filename)
        return tf.compat.as_str(corpus)[:n_characters]
