from vanilla_neural_nets.recurrent_neural_network.training_data import WordLevelRNNTrainingDataBuilder, _RNNTrainingData


class Word2VecTrainingDataBuilder(WordLevelRNNTrainingDataBuilder):

    @classmethod
    def build(cls, corpus, vocabulary_size):
        tokenized_corpus = cls._tokenize_corpus_into_list_of_tokenized_sentences(corpus)
        tokenized_corpus = cls._remove_uncommon_words(tokenized_corpus, vocabulary_size)
        return _Word2VecTrainingData(tokenized_corpus=tokenized_corpus)


class _Word2VecTrainingData(_RNNTrainingData):

    WORD_PADDING = 2
    TRAINING_INSTANCES_PER_WORD = WORD_PADDING*2

    def _compose_X_train(self):
        return [self.TRAINING_INSTANCES_PER_WORD*training_instance[self.WORD_PADDING:-self.WORD_PADDING] \
                for training_instance in self.training_data_as_indices]

    def _compose_y_train(self):
        return [
            training_instance[:-2*self.WORD_PADDING] + \
            training_instance[1:-2*self.WORD_PADDING+1] + \
            training_instance[self.WORD_PADDING+1:-self.WORD_PADDING+1] + \
            training_instance[2*self.WORD_PADDING:] \
                for training_instance in self.training_data_as_indices
        ]
