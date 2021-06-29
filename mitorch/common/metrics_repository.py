import dataclasses
import typing
import uuid
import pymongo
import tenacity


@dataclasses.dataclass(frozen=True)
class Metrics:
    job_id: uuid.UUID
    epoch: int
    metrics: typing.Any


class MetricsRepository:
    def __init__(self, mongodb_url):
        client = pymongo.MongoClient(mongodb_url, uuidRepresentation='standard')
        db = client.mitorch
        self._metrics_collection = db.training_metrics

    @tenacity.retry(retry=tenacity.retry_if_exception_type(pymongo.errors.PyMongoError), stop=tenacity.stop_after_attempt(2), reraise=True)
    def get_metrics(self, job_id):
        results = self._metrics_collection.find({'job_id': job_id})
        return [self._to_metrics(r) for r in results]

    @staticmethod
    def _to_metrics(raw_data):
        return Metrics(job_id=raw_data['job_id'], epoch=raw_data['e'], metrics=raw_data['m'])
