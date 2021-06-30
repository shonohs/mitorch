import json
import pathlib
import unittest
import jsons
import torch
from mitorch.builders import OptimizerBuilder
from mitorch.common import TrainingConfig

CONFIGS = [{'name': 'sgd', 'momentum': 0.9, 'weight_decay': 1e-5},
           {'name': 'adam'},
           {'name': 'rmsprop', 'weight_decay': 0}]


class TestOptimizerBuilder(unittest.TestCase):
    def test_build(self):
        sample_filepath = pathlib.Path(__file__).parent / 'configs' / 'simple_train.json'
        base_config = json.loads(sample_filepath.read_text())
        for optimizer_config in CONFIGS:
            base_config['optimizer'] = optimizer_config
            config = jsons.load(base_config, TrainingConfig)
            model = torch.nn.Conv2d(3, 3, 3)
            optimizer = OptimizerBuilder(config).build(model)
            self.assertIsInstance(optimizer, torch.optim.Optimizer)


if __name__ == '__main__':
    unittest.main()
