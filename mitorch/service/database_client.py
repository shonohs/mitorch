import copy
import dataclasses
import datetime
import uuid
import pymongo
from ..settings import Settings


class DatabaseClient:
    def __init__(self, mongodb_url):
        client = pymongo.MongoClient(mongodb_url, uuidRepresentation='standard')
        self.db = client.mitorch

    def add_training(self, config, priority=100):
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

    def find_training_by_id(self, training_id):
        return self.db.trainings.find_one({'_id': training_id})

    def find_training_by_config(self, config):
        return self.db.trainings.find_one({'config': config})

    def get_new_trainings(self, max_num=100):
        return self.db.trainings.find({'status': 'new'}).sort('priority').limit(max_num)

    def get_running_trainings(self):
        return self.db.trainings.find({'status': 'running'})

    def get_queued_trainings(self):
        return self.db.trainings.find({'status': 'queued'})

    def get_failed_trainings(self):
        return self.db.trainings.find({'status': 'failed'})

    def delete_training(self, training_id):
        result = self.db.trainings.delete_one({'_id': training_id})
        return result.deleted_count == 1

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

    def fail_training(self, training_id):
        result = self.db.trainings.update_one({'_id': training_id}, {'$set': {'status': 'failed',
                                                                              'completed_at': datetime.datetime.utcnow()}})
        return result.modified_count == 1

    # Datasets
    def find_dataset_by_name(self, dataset_name, version=None):
        # TODO: Get the latest version
        return self.db.datasets.find_one({'name': dataset_name})

    def add_dataset(self, dataset):
        return self.db.datasets.insert_one(dataset)

    # Common Settings
    def get_settings(self):
        record = self.db.settings.find_one({'key': 'settings'})
        return record and Settings.from_dict(record['value'])

    def put_settings(self, settings):
        settings = dataclasses.asdict(settings)
        if self.get_settings():
            result = self.db.settings.update_one({'key': 'settings'}, {'$set': {'value': settings}})
            assert result.modified_count == 1
        else:
            self.db.settings.insert_one({'key': 'settings', 'value': settings})

    # Tasks
    def add_task(self, task):
        assert 'config' in task
        assert 'max_trainings' in task
        record = copy.deepcopy(task)
        record['_id'] = uuid.uuid4()
        record['created_at'] = datetime.datetime.utcnow()
        record['status'] = 'active'
        self.db.tasks.insert_one(record)
        return record['_id']

    def get_tasks(self):
        return self.db.tasks.find()

    def get_task_by_id(self, task_id):
        return self.db.tasks.find_one({'_id': task_id})

    def get_active_tasks(self):
        return self.db.tasks.find({'status': 'active'})

    def update_task(self, task_description):
        assert task_description['_id']
        self.db.tasks.update_one({'_id': task_description['_id']}, {'$set': task_description})

    def cancel_task(self, task_id):
        result = self.db.tasks.update_one({'_id': task_id}, {'$set': {'status': 'cancelled'}})
        return result.modified_count == 1

    def delete_task(self, task_id):
        result = self.db.tasks.delete_one({'_id': task_id})
        return result.deleted_count == 1
