import unittest


from vanilla_neural_nets.recurrent_neural_network.training_data import WordLevelRNNTrainingDataBuilder, _RNNTrainingData
from helpers.prepare import Word2VecTrainingDataBuilder


class TestWord2VecDataPreparation(unittest.TestCase):

    CORPUS = 'the dog went to the store. it then chased me home.'

    VOCABULARY_SIZE_WHEREBY_NO_WORDS_REMOVED = SOME_LARGE_NUMBER = 100
    EXPECTED_X_TRAIN_WHEN_NO_WORDS_REMOVED = [[10, 9, 7, 10, 9, 7, 10, 9, 7, 10, 9, 7], [1, 5, 1, 5, 1, 5, 1, 5]]
    EXPECTED_Y_TRAIN_WHEN_NO_WORDS_REMOVED = [[7, 2, 10, 2, 10, 9, 9, 7, 6, 7, 6, 0], [4, 8, 8, 1, 5, 3, 3, 0]]

    VOCABULARY_SIZE_WHEREBY_SOME_WORDS_REMOVED = 4
    EXPECTED_X_TRAIN_WHEN_SOME_WORDS_REMOVED = [[1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2], [1, 1, 1, 1, 1, 1, 1, 1]]
    EXPECTED_Y_TRAIN_WHEN_SOME_WORDS_REMOVED = [[2, 1, 1, 1, 1, 1, 1, 2, 1, 2, 1, 0], [1, 1, 1, 1, 1, 1, 1, 0]]

    def test_X_train_correctly_encoded_as_indices_when_no_words_removed(self):
        training_data = Word2VecTrainingDataBuilder.build(
            corpus=self.CORPUS,
            vocabulary_size=self.VOCABULARY_SIZE_WHEREBY_NO_WORDS_REMOVED
        )

        self.assertEqual(training_data.X_train,
            self.EXPECTED_X_TRAIN_WHEN_NO_WORDS_REMOVED)

    def test_y_train_correctly_encoded_as_indices_when_no_words_removed(self):
        training_data = Word2VecTrainingDataBuilder.build(
            corpus=self.CORPUS,
            vocabulary_size=self.VOCABULARY_SIZE_WHEREBY_NO_WORDS_REMOVED
        )

        self.assertEqual(training_data.y_train,
            self.EXPECTED_Y_TRAIN_WHEN_NO_WORDS_REMOVED)

    def test_X_train_correctly_encoded_as_indices_when_some_words_removed(self):
        training_data = Word2VecTrainingDataBuilder.build(
            corpus=self.CORPUS,
            vocabulary_size=self.VOCABULARY_SIZE_WHEREBY_SOME_WORDS_REMOVED
        )

        self.assertEqual(training_data.X_train,
            self.EXPECTED_X_TRAIN_WHEN_SOME_WORDS_REMOVED)

    def test_y_train_correctly_encoded_as_indices_when_some_words_removed(self):
        training_data = Word2VecTrainingDataBuilder.build(
            corpus=self.CORPUS,
            vocabulary_size=self.VOCABULARY_SIZE_WHEREBY_SOME_WORDS_REMOVED
        )

        self.assertEqual(training_data.y_train,
            self.EXPECTED_Y_TRAIN_WHEN_SOME_WORDS_REMOVED)
