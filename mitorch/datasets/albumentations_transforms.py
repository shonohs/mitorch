import albumentations
import albumentations.pytorch
import cv2
import numpy as np


class AlbumentationsTransform:
    def __init__(self, components, is_object_detection):
        bbox_params = albumentations.BboxParams(format='albumentations', label_fields=['category_id'], min_area=16, min_visibility=0.1) if is_object_detection else None
        normalize = albumentations.Normalize(mean=[0.5, 0.5, 0.5], std=[1, 1, 1])
        to_tensor = albumentations.pytorch.ToTensorV2()
        self._transforms = albumentations.Compose(components + [normalize, to_tensor], bbox_params=bbox_params)
        self._is_object_detection = is_object_detection

    def __call__(self, image, target):
        w, h = image.width, image.height
        image = np.array(image)

        if self._is_object_detection:
            bboxes = [[t[1] / w, t[2] / h, t[3] / w, t[4] / h] for t in target]
            category_id = [t[0] for t in target]
            augmented = self._transforms(image=image, bboxes=bboxes, category_id=category_id)
            target = [[label, *bbox] for label, bbox in zip(augmented['category_id'], augmented['bboxes'])]
        else:
            augmented = self._transforms(image=image)

        return augmented['image'], target


class AlbumentationsTransform2(AlbumentationsTransform):
    def __init__(self, input_size, is_object_detection):
        components = self.get_transforms(input_size)
        bbox_params = albumentations.BboxParams(format='albumentations', label_fields=['category_id'], min_area=16, min_visibility=0.1) if is_object_detection else None
        normalize = albumentations.Normalize(mean=[0.5, 0.5, 0.5], std=[1, 1, 1])
        to_tensor = albumentations.pytorch.ToTensorV2()
        self._transforms = albumentations.Compose(components + [normalize, to_tensor], bbox_params=bbox_params)
        self._is_object_detection = is_object_detection


class SurveillanceCameraTransform(AlbumentationsTransform2):
    def get_transforms(self, input_size):
        return [albumentations.RandomSizedBBoxSafeCrop(input_size, input_size, interpolation=cv2.INTER_CUBIC),
                albumentations.HorizontalFlip(),
                albumentations.ImageCompression(quality_lower=20, quality_upper=100),
                albumentations.RandomBrightnessContrast(),
                albumentations.ToGray(p=0.1)]
