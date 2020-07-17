import unittest
import torch
from mitorch.builders.lr_scheduler_builder import LinearDecreasingLR, WarmupLR


class TestLrScheduler(unittest.TestCase):
    def test_load_linear_decreasing_lr(self):
        optimizer = torch.optim.SGD(torch.nn.Conv2d(1, 1, 1).parameters(), lr=1)
        scheduler = LinearDecreasingLR(optimizer, 100)

        optimizer.step()
        scheduler.step()
        state_dict = scheduler.state_dict()
        optimizer.step()
        scheduler.step()
        lrs = self.step_and_get_lr(scheduler, optimizer, 10)

        scheduler.load_state_dict(state_dict)
        optimizer.step()
        scheduler.step()
        lrs2 = self.step_and_get_lr(scheduler, optimizer, 10)
        self.assert_almost_equal_lists(lrs, lrs2)

    def test_warmup_lr(self):
        parameters = torch.nn.Conv2d(1, 1, 1).parameters()
        optimizer = torch.optim.SGD([{'params': parameters, 'initial_lr': 0.3}], lr=0.3)
        scheduler = LinearDecreasingLR(optimizer, 100)
        warmup_scheduler = WarmupLR(scheduler, 5, 0.01)

        lrs = self.step_and_get_lr(warmup_scheduler, optimizer, 10)
        expected_lrs = [0.003, 0.003, 0.003, 0.003, 0.003, 0.3*0.94, 0.3*0.93, 0.3*0.92, 0.3*0.91, 0.3*0.90]
        self.assert_almost_equal_lists(lrs, expected_lrs)

    def assert_almost_equal_lists(self, list0, list1):
        self.assertEqual(len(list0), len(list1))
        for value0, value1 in zip(list0, list1):
            self.assertAlmostEqual(value0, value1, msg=f"{list0} vs {list1}")

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
