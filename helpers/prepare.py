from vanilla_neural_nets.recurrent_neural_network.training_data import WordLevelRNNTrainingDataBuilder, _RNNTrainingData


class Word2VecSkipGramTrainingDataBuilder(WordLevelRNNTrainingDataBuilder):

    @classmethod
    def build(cls, corpus, vocabulary_size):
        tokenized_corpus = cls._tokenize_corpus_into_list_of_tokenized_sentences(corpus)
        tokenized_corpus = cls._remove_uncommon_words(tokenized_corpus, vocabulary_size)
        return _Word2VecSkipGramTrainingData(tokenized_corpus=tokenized_corpus)

class Word2VecCBOWTrainingDataBuilder(WordLevelRNNTrainingDataBuilder):

    @classmethod
    def build(cls, corpus, vocabulary_size):
        tokenized_corpus = cls._tokenize_corpus_into_list_of_tokenized_sentences(corpus)
        tokenized_corpus = cls._remove_uncommon_words(tokenized_corpus, vocabulary_size)
        return _Word2VecCBOWTrainingData(tokenized_corpus=tokenized_corpus)


class _Word2VecSkipGramTrainingData(_RNNTrainingData):

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


class _Word2VecCBOWTrainingData(_RNNTrainingData):

    WORD_PADDING = 2
    FULL_WINDOW_SIZE = 2 * WORD_PADDING + 1

    def __init__(self, tokenized_corpus):
        self.training_data_as_tokens = tokenized_corpus
        self.token_to_index_lookup = self._compose_token_to_index_lookup(tokenized_corpus)
        self.index_to_token_lookup = self._compose_index_to_token_lookup()
        self.training_data_as_indices = self._indexify_tokenized_corpus()
        self.X_train, self.y_train = self._compose_X_train_and_y_train()

    def _compose_X_train_and_y_train(self):
        X_train = []
        y_train = []
        for training_instance in self.training_data_as_indices:
            # for each sentence in our list of sentences
            x = []
            y = []
            for cbow in self._generate_cbows(training_instance):
                # for cbow = ['a', 'b', 'c', 'd', 'e'], this will:
                # x.append( 'c' )
                # y.append( ['a', 'b', 'd', 'e'] )
                x.append( cbow[:self.WORD_PADDING] + cbow[-self.WORD_PADDING:] )
                y.append( cbow[self.WORD_PADDING] )
            X_train.append(x)
            y_train.append(y)
        return X_train, y_train

    def _generate_cbows(self, list_of_tokens):
        for index in range( len(list_of_tokens) ):
            candidate_cbow = list_of_tokens[index:index+self.FULL_WINDOW_SIZE]
            if len(candidate_cbow) == self.FULL_WINDOW_SIZE:
                yield candidate_cbow
