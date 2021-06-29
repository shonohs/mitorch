"""Repository for training configs. Used by mitorch-agent."""
import dataclasses
import datetime
from typing import Optional
import uuid
import jsons
import pymongo
import tenacity
from mitorch.common.training_config import TrainingConfig


@dataclasses.dataclass(frozen=True)
class JobRecord:
    job_id: uuid.UUID
    status: str
    config: TrainingConfig
    base_job_id: Optional[uuid.UUID] = None
    priority: int = 2
    created_at: datetime.datetime = None
    updated_at: datetime.datetime = None


class JobRepository:
    def __init__(self, mongodb_url):
        client = pymongo.MongoClient(mongodb_url, uuidRepresentation='standard')
        db = client.mitorch
        self._job_collection = db.jobs

    @tenacity.retry(retry=tenacity.retry_if_exception_type(pymongo.errors.PyMongoError), stop=tenacity.stop_after_attempt(2), reraise=True)
    def get_next_job(self, num_processes=-1):
        assert isinstance(num_processes, int)
        raw_data = self._job_collection.find_one({'status': 'queued', 'config.num_processes': {'$in': [-1, num_processes]}},
                                                 sort=[('priority', pymongo.ASCENDING), ('created_at', pymongo.ASCENDING)])
        if not raw_data:
            return None
        return self._job_document_to_record(raw_data)

    @tenacity.retry(retry=tenacity.retry_if_exception_type(pymongo.errors.PyMongoError), stop=tenacity.stop_after_attempt(2), reraise=True)
    def update_job_status(self, job_id: uuid.UUID, new_status):
        assert isinstance(job_id, uuid.UUID)
        assert new_status in ['queued', 'running', 'failed', 'cancelled', 'completed']

        # TODO: Transaction
        result = self._job_collection.update_one({'_id': job_id}, {'$set': {'status': new_status,
                                                                            'updated_at': datetime.datetime.utcnow()}})
        if result.modified_count == 0:
            raise RuntimeError(f"Job not found: {job_id}")

    @tenacity.retry(retry=tenacity.retry_if_exception_type(pymongo.errors.PyMongoError), stop=tenacity.stop_after_attempt(2), reraise=True)
    def add_new_job(self, training_config: TrainingConfig, priority=2, base_job_id=None):
        assert base_job_id is None or isinstance(base_job_id, uuid.UUID)
        job_id = uuid.uuid4()
        job_dict = {'_id': job_id,
                    'status': 'queued',
                    'config': dataclasses.asdict(training_config),
                    'priority': priority,
                    'base_job_id': base_job_id,
                    'created_at': datetime.datetime.utcnow(),
                    'updated_at': datetime.datetime.utcnow()}

        self._job_collection.insert_one(job_dict)
        return job_id

    def query_jobs(self):
        jobs = self._job_collection.find(sort=[('updated_at', pymongo.DESCENDING)])
        return [self._job_document_to_record(d) for d in jobs]

    def get_job(self, job_id):
        assert isinstance(job_id, uuid.UUID)
        raw_data = self._job_collection.find_one({'_id': job_id})
        return self._job_document_to_record(raw_data)

    @staticmethod
    def _job_document_to_record(raw_data):
        raw_data['job_id'] = raw_data['_id']
        raw_data['config'] = jsons.load(raw_data['config'], TrainingConfig)
        del raw_data['_id']
        return JobRecord(**raw_data)
