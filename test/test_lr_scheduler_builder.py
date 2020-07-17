import pickle
import unittest
import torch
from mitorch.builders.lr_scheduler_builder import LinearDecreasingLR, WarmupLR


class TestLrScheduler(unittest.TestCase):
    def test_pickle_linear_decreasing_lr(self):
        optimizer = torch.optim.SGD(torch.nn.Conv2d(1, 1, 1).parameters(), lr=1)
        scheduler = LinearDecreasingLR(optimizer, 100)

        optimizer.step()
        result = scheduler.step(10)
        new_scheduler = pickle.loads(pickle.dumps(scheduler))
        new_result = new_scheduler.step(10)
        self.assertEqual(result, new_result)

    def test_warmup_lr(self):
        parameters = torch.nn.Conv2d(1, 1, 1).parameters()
        optimizer = torch.optim.SGD(parameters, lr=1)
        scheduler = LinearDecreasingLR(optimizer, 100)
        warmup_scheduler = WarmupLR(scheduler, 5, 0.01)

        lrs = self.step_and_get_lr(warmup_scheduler, optimizer, 10)
        expected_lrs = [0.01, 0.01, 0.01, 0.01, 0.01, 0.94, 0.93, 0.92, 0.91, 0.90]
        for lr, expected in zip(lrs, expected_lrs):
            self.assertAlmostEqual(lr, expected)

    @staticmethod
    def step_and_get_lr(scheduler, optimizer, num_steps):
        lrs = []
        for i in range(num_steps):
            optimizer.step()
            lrs.append(optimizer.param_groups[0]['lr'])
            scheduler.step()
        return lrs


if __name__ == '__main__':
    unittest.main()
