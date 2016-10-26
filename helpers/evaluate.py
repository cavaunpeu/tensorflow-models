
class TensorFlowModelEvaluator:

    def __init__(self, model, session, validation_dataset, test_dataset):
        self._model = model
        self._session = session
        self._validation_feed_dict = self._compose_feed_dict(validation_dataset)
        self._test_feed_dict = self._compose_feed_dict(test_dataset)
        self._training_feed_dict = None

    def optimize(self, training_dataset):
        self._training_feed_dict = self._compose_feed_dict(training_dataset)
        _ = self._session.run(self._model.optimize(), feed_dict=self._training_feed_dict)

    def _compose_feed_dict(self, dataset):
        return {self._model.dataset.data: dataset.data, self._model.dataset.labels: dataset.labels}

    @property
    def training_accuracy(self):
        return self._session.run(self._model.compute_accuracy(), feed_dict=self._training_feed_dict)

    @property
    def validation_accuracy(self):
        return self._session.run(self._model.compute_accuracy(), feed_dict=self._validation_feed_dict)

    @property
    def test_accuracy(self):
        return self._session.run(self._model.compute_accuracy(), feed_dict=self._test_feed_dict)

    @property
    def training_loss(self):
        return self._session.run(self._model.compute_loss(), feed_dict=self._training_feed_dict)

    @property
    def validation_loss(self):
        return self._session.run(self._model.compute_loss(), feed_dict=self._validation_feed_dict)

    @property
    def test_loss(self):
        return self._session.run(self._model.compute_loss(), feed_dict=self._test_feed_dict)
