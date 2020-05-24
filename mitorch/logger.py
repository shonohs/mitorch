import datetime
import uuid
import pymongo
from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.utilities import rank_zero_only
import torch


class StdoutLogger(LightningLoggerBase):
    @rank_zero_only
    def log_metrics(self, metrics, step):
        if metrics:
            print(f"{datetime.datetime.now()}: {step}: {metrics}")

    @rank_zero_only
    def log_hyperparams(self, params):
        print(str(params))

    @property
    def experiment(self):
        return self

    @rank_zero_only
    def log_epoch_metrics(self, metrics, epoch):
        metrics = {key: value.tolist() if isinstance(value, torch.Tensor) else value for key, value in metrics.items()}
        print(f"{datetime.datetime.now()}: Epoch {epoch}: {metrics}")

    @property
    def name(self):
        return "experiment"

    @property
    def version(self):
        return 0


class MongoDBLogger(LightningLoggerBase):
    def __init__(self, db_uri, training_id):
        super().__init__()
        self._initialize(db_uri, training_id)

    def _initialize(self, db_uri, training_id):
        assert isinstance(training_id, uuid.UUID)
        self._db_uri = db_uri
        self.training_id = training_id

        # w=0: Disable write achknowledgement.
        self.client = pymongo.MongoClient(db_uri, uuidRepresentation='standard', w=0)
        self.log_collection = self.client.mitorch.training_metrics

    @rank_zero_only
    def log_metrics(self, metrics, step):
        pass

    @rank_zero_only
    def log_hyperparams(self, params):
        if params and 'model_versions' in params:
            model_versions = params['model_versions']
            self.client.mitorch.trainings.update_one({'_id': self.training_id}, {'$set': {'model_versions': model_versions}})

    @property
    def experiment(self):
        return self

    @rank_zero_only
    def log_epoch_metrics(self, metrics, epoch):
        m = {key: value.tolist() if isinstance(value, torch.Tensor) else value for key, value in metrics.items()}
        self.log_collection.insert_one({'tid': self.training_id, 'e': epoch, 'm': m})

    @property
    def name(self):
        return "experiment"

    @property
    def version(self):
        return 0

    def __getstate__(self):
        return {'db_uri': self._db_uri, 'training_id': self.training_id}

    def __setstate__(self, state):
        self._initialize(state['db_uri'], state['training_id'])
