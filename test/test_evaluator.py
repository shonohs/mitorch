import unittest
import torch
from mitorch.evaluators import MulticlassClassificationEvaluator


class TestMulticlassClassificationEvaluator(unittest.TestCase):
    def test_empty(self):
        evaluator = MulticlassClassificationEvaluator()
        self.assertEqual(evaluator.get_report(), {'top1_accuracy': 0, 'top5_accuracy': 0, 'average_precision': 0})

    def test_perfect(self):
        evaluator = MulticlassClassificationEvaluator()
        evaluator.add_predictions(torch.tensor([[1, 0, 0], [0, 1, 0], [0.5, 0, 1]]), torch.tensor([0, 1, 2]))
        evaluator.add_predictions(torch.tensor([[1, 0, 0], [0, 1, 0], [0.5, 0, 1]]), torch.tensor([0, 1, 2]))
        evaluator.add_predictions(torch.tensor([[1, 0, 0], [0, 1, 0], [0.5, 0, 1]]), torch.tensor([0, 1, 2]))
        self.assertEqual(evaluator.get_report()['top1_accuracy'], 1)
        evaluator.add_predictions(torch.tensor([[1, 0, 0], [0, 1, 0], [0.9, 0, 0.3]]), torch.tensor([0, 1, 0]))
        self.assertEqual(evaluator.get_report()['top1_accuracy'], 1)

        evaluator.reset()
        self.assertEqual(evaluator.get_report(), {'top1_accuracy': 0, 'top5_accuracy': 0, 'average_precision': 0})

    def test_half(self):
        evaluator = MulticlassClassificationEvaluator()
        evaluator.add_predictions(torch.tensor([[1, 0, 0], [0, 1, 0]]), torch.tensor([0, 2]))
        self.assertEqual(evaluator.get_report()['top1_accuracy'], 0.5)
        evaluator.add_predictions(torch.tensor([[1, 0, 0], [0, 1, 0], [0.5, 0, 1]]), torch.tensor([0, 1, 2]))
        evaluator.add_predictions(torch.tensor([[1, 0, 0], [0, 1, 0], [0.5, 0, 1]]), torch.tensor([1, 2, 0]))
        self.assertEqual(evaluator.get_report()['top1_accuracy'], 0.5)
        self.assertEqual(evaluator.get_report()['top5_accuracy'], 1)


if __name__ == '__main__':
    unittest.main()
