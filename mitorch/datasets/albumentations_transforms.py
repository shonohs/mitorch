import albumentations
import numpy as np
import torch
import torchvision

MIN_BOX_SIZE = 0.001


class AlbumentationsTransform:
    def __init__(self, transforms, is_object_detection):
        bbox_params = albumentations.BboxParams(format='pascal_voc', label_fields=['category_id']) if is_object_detection else None
        self.is_object_detection = is_object_detection
        self.transforms = albumentations.Compose(transforms, bbox_params=bbox_params)
        self.to_tensor = torchvision.transforms.ToTensor()
        self.mean_value = torch.tensor([0.5, 0.5, 0.5]).reshape(3, 1, 1)

    def __call__(self, image, target):
        image = np.array(image)

        if self.is_object_detection:
            bboxes = [t[1:] for t in target]
            category_id = [t[0] for t in target]
            augmented = self.transforms(image=np.array(image), bboxes=bboxes, category_id=category_id)
            image = augmented['image']
            w, h = image.shape[0:2]
            target = [[label, bbox[0] / w, bbox[1] / h, bbox[2] / w, bbox[3] / h] for label, bbox in zip(augmented['category_id'], augmented['bboxes'])
                      if bbox[0] + MIN_BOX_SIZE < bbox[2] and bbox[1] + MIN_BOX_SIZE < bbox[3]]
        else:
            image = self.transforms(image=image)['image']

        image = self.to_tensor(image) - self.mean_value
        return image, target


class RandomResizedCropTransformV2(AlbumentationsTransform):
    def __init__(self, input_size, is_object_detection):
        transforms = [albumentations.augmentations.transforms.RandomResizedCrop(input_size, input_size),
                      albumentations.augmentations.transforms.HorizontalFlip()]
        super().__init__(transforms, is_object_detection)


class RandomResizedCropTransform(AlbumentationsTransform):
    def __init__(self, input_size, is_object_detection):
        transforms = [albumentations.augmentations.transforms.RandomResizedCrop(input_size, input_size),
                      albumentations.augmentations.transforms.Flip(),
                      albumentations.augmentations.transforms.RandomBrightnessContrast()]
        super().__init__(transforms, is_object_detection)


class RandomResizedCropTransformV3(AlbumentationsTransform):
    def __init__(self, input_size, is_object_detection):
        transforms = [albumentations.augmentations.transforms.RandomResizedCrop(input_size, input_size),
                      albumentations.augmentations.transforms.HorizontalFlip(),
                      albumentations.augmentations.transforms.RandomBrightnessContrast()]
        super().__init__(transforms, is_object_detection)


class ResizeTransform(AlbumentationsTransform):
    def __init__(self, input_size, is_object_detection):
        transforms = [albumentations.augmentations.transforms.Resize(input_size, input_size)]
        super().__init__(transforms, is_object_detection)


class ResizeFlipTransform(AlbumentationsTransform):
    def __init__(self, input_size, is_object_detection):
        transforms = [albumentations.augmentations.transforms.Resize(input_size, input_size),
                      albumentations.augmentations.transforms.Flip()]
        super().__init__(transforms, is_object_detection)


class RandomSizedBBoxSafeCropTransform(AlbumentationsTransform):
    def __init__(self, input_size, is_object_detection):
        transforms = [albumentations.augmentations.transforms.RandomSizedBBoxSafeCrop(input_size, input_size, erosion_rate=0.2),
                      albumentations.augmentations.transforms.Flip(),
                      albumentations.augmentations.transforms.RandomBrightnessContrast()]
        super().__init__(transforms, is_object_detection)


class CenterCropTransform(AlbumentationsTransform):
    def __init__(self, input_size, is_object_detection):
        transforms = [albumentations.augmentations.transforms.SmallestMaxSize(input_size),
                      albumentations.augmentations.transforms.CenterCrop(input_size, input_size)]
        super().__init__(transforms, is_object_detection)
