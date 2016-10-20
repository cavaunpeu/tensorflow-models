import random


class TensorFlowDataset:

    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def sample(self, size):
        indices = random.sample(population=range(len(self)), k=size)
        return self.data[indices], self.labels[indices]

    def __len__(self):
        return len(self.data)
