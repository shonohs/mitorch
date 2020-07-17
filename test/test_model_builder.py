import unittest
from unittest.mock import patch
from mitorch.builders import ModelBuilder


class TestModelBuilder(unittest.TestCase):
    def test_build(self):
        config = {'task_type': 'multiclass_classification',
                  'model': {'name': 'MobileNetV2',
                            'options': []}}
        builder = ModelBuilder(config)
        with patch('mitorch.builders.model_builder.ModelFactory') as mock_factory:
            model = builder.build(3)
            self.assertIsNotNone(model)
            mock_factory.create.assert_called_once_with('MobileNetV2', 3, [])

    def test_build_multilabel(self):
        config = {'task_type': 'multilabel_classification',
                  'model': {'name': 'MobileNetV2',
                            'options': []}}
        builder = ModelBuilder(config)
        with patch('mitorch.builders.model_builder.ModelFactory') as mock_factory:
            model = builder.build(3)
            self.assertIsNotNone(model)
            mock_factory.create.assert_called_once_with('MobileNetV2', 3, ['multilabel'])


if __name__ == '__main__':
    unittest.main()
