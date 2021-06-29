from mitorch.datasets.albumentations_transforms import (CenterCropTransform, ResizeTransform, ResizeFlipTransform, RandomResizedCropTransform,
                                                        RandomSizedBBoxSafeCropTransform, SurveillanceCameraTransform)
from mitorch.datasets.transforms import InceptionTransform


class TransformFactory:
    def __init__(self, is_object_detection):
        self._is_object_detection = is_object_detection

    def create(self, name, input_size):
        augmentation_class = {'center_crop': CenterCropTransform,
                              'inception': InceptionTransform,
                              'resize': ResizeTransform,
                              'resize_flip': ResizeFlipTransform,
                              'random_resize': RandomResizedCropTransform,
                              'random_resize_bbox': RandomSizedBBoxSafeCropTransform,
                              'surveillance': SurveillanceCameraTransform
                              }.get(name)

        if not augmentation_class:
            return None

        return augmentation_class(input_size, self._is_object_detection)
