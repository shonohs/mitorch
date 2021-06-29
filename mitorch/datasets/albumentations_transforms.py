import albumentations
import albumentations.pytorch
import cv2
import numpy as np


class AlbumentationsTransform:
    def __init__(self, input_size, is_object_detection):
        bbox_params = albumentations.BboxParams(format='albumentations', label_fields=['category_id'], min_area=16, min_visibility=0.1) if is_object_detection else None
        normalize = albumentations.Normalize(mean=[0.5, 0.5, 0.5], std=[1, 1, 1])
        to_tensor = albumentations.pytorch.ToTensorV2()
        transforms = self.get_transforms(input_size)
        self.transforms = albumentations.Compose(transforms + [normalize, to_tensor], bbox_params=bbox_params)
        self.is_object_detection = is_object_detection

    def __call__(self, image, target):
        w, h = image.width, image.height
        image = np.array(image)

        if self.is_object_detection:
            bboxes = [[t[1] / w, t[2] / h, t[3] / w, t[4] / h] for t in target]
            category_id = [t[0] for t in target]
            augmented = self.transforms(image=image, bboxes=bboxes, category_id=category_id)
            target = [[label, *bbox] for label, bbox in zip(augmented['category_id'], augmented['bboxes'])]
        else:
            augmented = self.transforms(image=image)

        return augmented['image'], target

    def get_transforms(self, input_size):
        raise NotImplementedError


class SurveillanceCameraTransform(AlbumentationsTransform):
    def get_transforms(self, input_size):
        return [albumentations.RandomSizedBBoxSafeCrop(input_size, input_size, interpolation=cv2.INTER_CUBIC),
                albumentations.HorizontalFlip(),
                albumentations.ImageCompression(quality_lower=20, quality_upper=100),
                albumentations.RandomBrightnessContrast(),
                albumentations.ToGray(p=0.1)]


class RandomResizedCropTransform(AlbumentationsTransform):
    def get_transforms(self, input_size):
        return [albumentations.RandomResizedCrop(input_size, input_size, interpolation=cv2.INTER_CUBIC),
                albumentations.HorizontalFlip(),
                albumentations.RandomBrightnessContrast()]


class ResizeTransform(AlbumentationsTransform):
    def get_transforms(self, input_size):
        return [albumentations.Resize(input_size, input_size)]


class ResizeFlipTransform(AlbumentationsTransform):
    def get_transforms(self, input_size):
        return [albumentations.Resize(input_size, input_size),
                albumentations.Flip()]


class RandomSizedBBoxSafeCropTransform(AlbumentationsTransform):
    def get_transforms(self, input_size):
        return [albumentations.RandomSizedBBoxSafeCrop(input_size, input_size, erosion_rate=0.2),
                albumentations.Flip(),
                albumentations.RandomBrightnessContrast()]


class CenterCropTransform(AlbumentationsTransform):
    """This method was found in pytorch's imagenet training example."""
    def get_transforms(self, input_size):
        return [albumentations.SmallestMaxSize(int(input_size / 224 * 256), interpolation=cv2.INTER_CUBIC),
                albumentations.CenterCrop(input_size, input_size)]
