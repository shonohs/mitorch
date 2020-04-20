import unittest
from unittest.mock import MagicMock, patch
from mitorch.mimodel import MiModel


class TestMiModel(unittest.TestCase):

    def test_version(self):
        mock_builder = MagicMock()
        mock_builder.build.return_value = [MagicMock(), [None]]
        mock_model_builder = MagicMock()
        mock_model_builder.build.return_value = [None, None, None]
        mock_optimizer_builder = MagicMock()
        mock_lrscheduler_builder = MagicMock()
        MiModel._get_evaluator = MagicMock()
        with patch('mitorch.mimodel.DataLoaderBuilder', return_value=mock_builder):
            with patch('mitorch.mimodel.ModelBuilder', return_value=mock_model_builder):
                with patch('mitorch.mimodel.OptimizerBuilder', return_value=mock_optimizer_builder):
                    with patch('mitorch.mimodel.LrSchedulerBuilder', return_value=mock_lrscheduler_builder):
                        model = MiModel(MagicMock(), None, None, None)

        model.model = MagicMock()
        model.model.version = 42
        self.assertEqual(model.version, 42)


if __name__ == '__main__':
    unittest.main()
