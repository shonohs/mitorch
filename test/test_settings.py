import dataclasses
import unittest
from mitorch.settings import Settings


class TestSettings(unittest.TestCase):
    def test_from_dict(self):
        obj = {'storage_url': 'storage_url',
               'readonly_storage_url': None,
               'dataset_url': {'region': 'dataset_url'},
               'azureml_settings': [{'subscription_id': 'subscription_id',
                                     'workspace_name': 'workspace_name',
                                     'cluster_name': 'cluster',
                                     'region_name': 'region'}]}

        settings = Settings(**obj)
        self.assertEqual(dataclasses.asdict(settings), obj)


if __name__ == '__main__':
    unittest.main()
