import unittest
from unittest.mock import patch
from mitorch.builders import ModelBuilder
from mitorch.common.training_config import ModelConfig, TrainingConfig


class TestModelBuilder(unittest.TestCase):
    def test_build(self):
        model_config = ModelConfig(name='MobileNetV2', input_size=224)
        config = TrainingConfig(task_type='multiclass_classification', model=model_config, batch_size=4, max_epochs=10)
        builder = ModelBuilder(config)
        with patch('mitorch.builders.model_builder.ModelFactory') as mock_factory:
            model = builder.build(3)
            self.assertIsNotNone(model)
            mock_factory.create.assert_called_once_with('MobileNetV2', 3, [])

    def test_build_multilabel(self):
        model_config = ModelConfig(name='MobileNetV2', input_size=224)
        config = TrainingConfig(task_type='multilabel_classification', model=model_config, batch_size=4, max_epochs=10)
        builder = ModelBuilder(config)
        with patch('mitorch.builders.model_builder.ModelFactory') as mock_factory:
            model = builder.build(3)
            self.assertIsNotNone(model)
            mock_factory.create.assert_called_once_with('MobileNetV2', 3, ['multilabel'])


if __name__ == '__main__':
    unittest.main()
