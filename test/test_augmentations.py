import unittest
import numpy as np
import PIL.Image
import torch
from mitorch.datasets import ResizeTransform, ResizeFlipTransform, RandomResizedCropTransform


TRANSFORMS = [ResizeTransform, ResizeFlipTransform, RandomResizedCropTransform]

class TestAugmentations(unittest.TestCase):
    def test_classification_augmentations_valid_tensor(self):
        image = PIL.Image.new('RGB', (500, 800))
        target = [1, 3, 5]
        for augmentation_class in TRANSFORMS:
            augmentation = augmentation_class(224)
            tensor, new_target = augmentation(image, target)
            self.assertIsInstance(tensor, torch.Tensor)
            self.assertLess(torch.max(torch.abs(tensor)), 1)
            self.assertEqual(tensor.shape, (3, 224, 224))

    def test_detection_augmentations_valid_tensor(self):
        image = PIL.Image.new('RGB', (500, 800))
        target = [[1, 0.1, 0.1, 0.9, 0.9], [3, 0.5, 0.5, 0.9, 0.9]]
        for augmentation_class in TRANSFORMS:
            augmentation = augmentation_class(224, is_object_detection=True)
            tensor, new_target = augmentation(image, target)
            self.assertIsInstance(tensor, torch.Tensor)
            self.assertLess(torch.max(torch.abs(tensor)), 1)
            self.assertEqual(tensor.shape, (3, 224, 224))


if __name__ == '__main__':
    unittest.main()
