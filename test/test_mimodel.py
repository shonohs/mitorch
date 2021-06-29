import unittest
from unittest.mock import MagicMock, patch
from mitorch.common.mimodel import MiModel


class TestMiModel(unittest.TestCase):

    def test_init(self):
        mock_model_builder = MagicMock()
        mock_model_builder.build.return_value = [None, None, None]
        config = MagicMock()
        config.task_type = 'multiclass_classification'
        with patch('mitorch.common.mimodel.ModelBuilder', return_value=mock_model_builder):
            MiModel(config, None)


if __name__ == '__main__':
    unittest.main()
