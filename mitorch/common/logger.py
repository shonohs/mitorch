import datetime
import json
import jsons
import logging
import uuid
import pymongo
from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.utilities import rank_zero_only
import torch


logger = logging.getLogger(__name__)


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


class LoggerBase(LightningLoggerBase):
    @property
    def experiment(self):
        return self

    @property
    def name(self):
        return 'experiment'

    @property
    def version(self):
        return 0

    @rank_zero_only
    def log_hyperparams(self, params):
        pass


class StdoutLogger(LoggerBase):
    @rank_zero_only
    def log_metrics(self, metrics, step):
        print(f"{datetime.datetime.now()}: {step}: {metrics}")

    @rank_zero_only
    def log_hyperparams(self, params):
        print(f"hyperparams: {jsons.dumps(params)}")


class StandardLogger(LoggerBase):
    @rank_zero_only
    def log_metrics(self, metrics, step):
        logger.info(f"Step {step}: {json.dumps(metrics)}")

    @rank_zero_only
    def log_hyperparams(self, params):
        logger.info(f"hyperparams: {jsons.dumps(params)}")


class MongoDBLogger(LoggerBase):
    def __init__(self, db_url, job_id):
        super().__init__()
        assert isinstance(job_id, uuid.UUID)
        self._db_url = db_url
        self._job_id = job_id
        self._client = SerializableMongoClient(db_url)

    @rank_zero_only
    def log_metrics(self, metrics, step):
        m = {key: value.tolist() if torch.is_tensor(value) else value for key, value in metrics.items() if not key.endswith('_step') and key != 'epoch'}
        if m and 'epoch' in metrics:
            epoch = metrics['epoch']
            self._client.mitorch.training_metrics.insert_one({'job_id': self._job_id, 'e': epoch, 'm': m})
