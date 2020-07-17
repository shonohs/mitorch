import unittest
from mitorch.service.random_search_task import RandomSearchTask


class TestRandomSearchTask(unittest.TestCase):
    BASE_CONFIG = {'name': 'random_search',
                   'status': 'active',
                   'num_trainings': 0,
                   'max_trainings': 100,
                   'config': {'test0': 'value0',
                              'test1': {'_choice': [1, 2, 3]},
                              'test2': {'_choice': [100, 200, 300]}}}

    def test_get_random_same_choice(self):
        task = RandomSearchTask(self.BASE_CONFIG)
        results = [task.fetch_next() for i in range(50)]
        results_set = set([str(r) for r in results])
        self.assertEqual(len(results_set), 1)

    def test_update_random_choice(self):
        task = RandomSearchTask(self.BASE_CONFIG)
        results = []
        for i in range(100):
            results.append(task.fetch_next())
            task.update_training_status({'status': 'completed'})
        results_set = set([str(r) for r in results])
        self.assertEqual(len(results_set), 9)  # Most likely it covers all combinations.

    def test_complete_task(self):
        task = RandomSearchTask(self.BASE_CONFIG)
        for i in range(100):
            task.fetch_next()
            task.update_training_status({'status': 'completed'})
        self.assertEqual(task.status, 'completed')
        self.assertIsNone(task.fetch_next())


if __name__ == '__main__':
    unittest.main()
