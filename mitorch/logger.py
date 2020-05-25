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


class SerializableMongoClient:
    def __init__(self, url):
        self._url = url
        # w=0: Disable write achknowledgement.
        self._client = pymongo.MongoClient(url, uuidRepresentation='standard', w=0)

    def __getattr__(self, name):
        return getattr(self._client, name)

    def __getstate__(self):
        return {'url': self._url}

    def __setstate__(self, state):
        self._url = state['url']
        self._client = pymongo.MongoClient(self._url, uuidRepresentation='standard', w=0)


class MongoDBLogger(LightningLoggerBase):
    def __init__(self, db_url, training_id):
        super().__init__()
        assert isinstance(training_id, uuid.UUID)
        self._db_url = db_url
        self.training_id = training_id
        self.client = SerializableMongoClient(db_url)

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
        self.client.mitorch.training_metrics.insert_one({'tid': self.training_id, 'e': epoch, 'm': m})

    @property
    def name(self):
        return "experiment"

    @property
    def version(self):
        return 0
