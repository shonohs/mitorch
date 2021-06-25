import unittest
import torch
from mitorch.builders.dataloader_builder import _default_collate


class TestDataloaderBuilder(unittest.TestCase):
    def test_collate_object_detection(self):
        batch = ((torch.zeros(3, 224, 224, dtype=torch.float32), [[0, 0, 0, 224, 224]]),
                 (torch.zeros(3, 224, 224, dtype=torch.float32), [[0, 0, 0, 224, 224]]))

        image, target = _default_collate('object_detection', batch)
        self.assertEqual(image.shape, (2, 3, 224, 224))
        self.assertIsInstance(target, (list, tuple))
        self.assertEqual(len(target), 2)
        self.assertEqual(target[0], [[0, 0, 0, 224, 224]])
        self.assertEqual(target[1], [[0, 0, 0, 224, 224]])


if __name__ == '__main__':
    unittest.main()
