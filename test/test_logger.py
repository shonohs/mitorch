import pickle
import unittest
import uuid
from mitorch.common.logger import MongoDBLogger


class TestLogger(unittest.TestCase):
    def test_pickle_mongodb_logger(self):
        url = 'mongodb://localhost/'
        training_id = uuid.uuid4()
        logger = MongoDBLogger(url, training_id)
        new_logger = pickle.loads(pickle.dumps(logger))
        self.assertEqual(new_logger._job_id, training_id)
        self.assertIsNotNone(new_logger._client)


if __name__ == '__main__':
    unittest.main()
