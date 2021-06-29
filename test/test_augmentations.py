import unittest
import PIL.Image
import torch
from mitorch.datasets.albumentations_transforms import ResizeTransform, ResizeFlipTransform, RandomResizedCropTransform


TRANSFORMS = [ResizeTransform, ResizeFlipTransform, RandomResizedCropTransform]


class TestAugmentations(unittest.TestCase):
    def test_classification_augmentations_valid_tensor(self):
        image = PIL.Image.new('RGB', (500, 800))
        target = [1, 3, 5]
        for augmentation_class in TRANSFORMS:
            augmentation = augmentation_class(224, is_object_detection=False)
            tensor, new_target = augmentation(image, target)
            self.assertIsInstance(tensor, torch.Tensor)
            self.assertLess(torch.max(torch.abs(tensor)), 1)
            self.assertEqual(tensor.shape, (3, 224, 224))

    def test_detection_augmentations_valid_tensor(self):
        image = PIL.Image.new('RGB', (500, 800))
        target = [[1, 100, 100, 400, 400], [3, 250, 250, 500, 800]]
        for augmentation_class in TRANSFORMS:
            augmentation = augmentation_class(224, is_object_detection=True)
            tensor, new_target = augmentation(image, target)
            self.assertIsInstance(tensor, torch.Tensor)
            self.assertLess(torch.max(torch.abs(tensor)), 1)
            self.assertEqual(tensor.shape, (3, 224, 224))

            self.assertIsInstance(new_target, list)


if __name__ == '__main__':
    unittest.main()
