import uuid
import pymongo
from pytorch_lightning.loggers import LightningLoggerBase, rank_zero_only
import torch


class StdoutLogger(LightningLoggerBase):
    class ExperimentLogger:
        def __init__(self, rank):
            self.rank = rank

        @rank_zero_only
        def log_epoch_metrics(self, metrics, epoch):
            print(f"{epoch}: {metrics}")

    @rank_zero_only
    def log_metrics(self, metrics, step):
        print(f"{step}: {metrics}")

    @rank_zero_only
    def log_hyperparams(self, params):
        print(str(params))

    @property
    def experiment(self):
        return StdoutLogger.ExperimentLogger(self.rank)

    @property
    def name(self):
        return "experiment"

    @property
    def version(self):
        return 0


class MongoDBLogger(LightningLoggerBase):
    class ExperimentLogger:
        def __init__(self, rank, log_collection, training_id):
            self.rank = rank
            self.log_collection = log_collection
            self.training_id = training_id

        @rank_zero_only
        def log_epoch_metrics(self, metrics, epoch):
            m = {key: value.tolist() if isinstance(value, torch.Tensor) else value for key, value in metrics.items()}
            self.log_collection.insert_one({'tid': self.training_id, 'e': epoch, 'm': m})

    def __init__(self, db_uri, training_id):
        super().__init__()
        assert isinstance(training_id, uuid.UUID)
        # w=0: Disable write achknowledgement.
        self.client = pymongo.MongoClient(db_uri, uuidRepresentation='standard', w=0)
        self.training_id = training_id
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
        return MongoDBLogger.ExperimentLogger(self.rank, self.log_collection, self.training_id)

    @property
    def name(self):
        return "experiment"

    @property
    def version(self):
        return 0
