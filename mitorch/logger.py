import json
import uuid
import pymongo
from pytorch_lightning.loggers import LightningLoggerBase, rank_zero_only


class StdoutLogger(LightningLoggerBase):
    @rank_zero_only
    def log_metrics(self, metrics, step):
        print(f"{step}: {metrics}")

    @rank_zero_only
    def log_test_result(self, results):
        print(f"results: {json.dumps(results)}")

    @rank_zero_only
    def log_hyperparams(self, params):
        print(str(params))


class MongoDBLogger(LightningLoggerBase):
    def __init__(self, db_uri, training_id, evaluation_filepath=None):
        super(MongoDBLogger, self).__init__()
        assert isinstance(training_id, uuid.UUID)
        # w=0: Disable write achknowledgement.
        self.client = pymongo.MongoClient(db_uri, uuidRepresentation='standard', w=0)
        self.training_id = training_id
        self.log_collection = self.client.mitorch.training_metrics
        self.evaluation_filepath = evaluation_filepath

    @rank_zero_only
    def log_metrics(self, metrics, step):
        log_entry = {'tid': self.training_id,
                     's': step,
                     'm': metrics}
        self.log_collection.insert_one(log_entry)

    @rank_zero_only
    def log_test_result(self, results):
        if self.evaluation_filepath:
            with open(self.evaluation_filepath, 'w') as f:
                json.dump(results, f)

    @rank_zero_only
    def log_hyperparams(self, params):
        if 'model_versions' in params:
            model_versions = params['model_versions']
            print(f"model_versions: {model_versions}")
            self.client.mitorch.trainings.update_one({'_id': self.training_id}, {'$set': {'model_versions': model_versions}})
