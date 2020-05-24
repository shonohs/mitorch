import pickle
import unittest
from unittest.mock import MagicMock
import torch
from mitorch.builders.lr_scheduler_builder import LinearDecreasingLR


class TestLrScheduler(unittest.TestCase):
    def test_pickle_linear_decreasing_lr(self):
        optimizer = torch.optim.SGD(torch.nn.Conv2d(1, 1, 1).parameters(), lr=1)
        scheduler = LinearDecreasingLR(optimizer, 100)

        optimizer.step()
        result = scheduler.step(10)
        new_scheduler = pickle.loads(pickle.dumps(scheduler))
        new_result = new_scheduler.step(10)
        self.assertEqual(result, new_result)


if __name__ == '__main__':
    unittest.main()
