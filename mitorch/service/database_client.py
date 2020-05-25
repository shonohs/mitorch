import dataclasses
import datetime
import uuid
import pymongo
from ..settings import Settings


class DatabaseClient:
    def __init__(self, mongodb_url):
        client = pymongo.MongoClient(mongodb_url, uuidRepresentation='standard')
        self.db = client.mitorch

    def add_job(self, config, priority):
        record = {'config': config}
        record['_id'] = uuid.uuid4()
        record['created_at'] = datetime.datetime.utcnow()
        record['priority'] = priority
        record['status'] = 'new'
        self.db.jobs.insert_one(record)

        return record['_id']

    def add_training(self, config, priority):
        record = {'config': config}
        record['_id'] = uuid.uuid4()
        record['created_at'] = datetime.datetime.utcnow()
        record['priority'] = priority
        record['status'] = 'new'
        self.db.trainings.insert_one(record)

        return record['_id']

    def find_job_by_id(self, job_id):
        record = self.db.trainings.find_one({'_id': job_id})
        if not record:
            record = self.db.jobs.find_one({'_id': job_id})
        return record

    def get_running_jobs(self):
        return self.db.jobs.find({'status': 'running'})

    def find_training_by_id(self, training_id):
        return self.db.trainings.find_one({'_id': training_id})

    def get_new_trainings(self, max_num=100):
        return self.db.trainings.find({'status': 'new'}).sort('priority').limit(max_num)

    def get_running_trainings(self):
        return self.db.trainings.find({'status': 'running'})

    def get_queued_trainings(self):
        return self.db.trainings.find({'status': 'queued'})

    def update_training(self, training_id, set_data):
        assert isinstance(set_data, dict)
        result = self.db.trainings.update_one({'_id': training_id}, {'$set': set_data})
        return result.modified_count == 1

    def start_training(self, training_id, num_gpus):
        result = self.db.trainings.update_one({'_id': training_id}, {'$set': {'status': 'running',
                                                                              'started_at': datetime.datetime.utcnow(),
                                                                              'machine': {'num_gpus': num_gpus}}})
        return result.modified_count == 1

    def complete_training(self, training_id):
        # Get the test metrics
        result = self.db.training_metrics.find_one({'tid': training_id, 'm.test_loss': {'$exists': True}})
        metrics = result['m']

        result = self.db.trainings.update_one({'_id': training_id}, {'$set': {'status': 'completed',
                                                                              'completed_at': datetime.datetime.utcnow(),
                                                                              'evaluation': metrics}})
        return result.modified_count == 1

    def find_dataset_by_name(self, dataset_name, version=None):
        # TODO: Get the latest version
        return self.db.datasets.find_one({'name': dataset_name})

    def add_dataset(self, dataset):
        return self.db.datasets.insert_one(dataset)

    # Common Settings
    def get_settings(self):
        record = self.db.settings.find_one({'key': 'settings'})
        return record and Settings(**record['value'])

    def put_settings(self, settings):
        settings = dataclasses.asdict(settings)
        if self.get_settings():
            result = self.db.settings.update_one({'key': 'settings'}, {'$set': {'value': settings}})
            assert result.modified_count == 1
        else:
            self.db.settings.insert_one({'key': 'settings', 'value': settings})
