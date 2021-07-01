import albumentations
import cv2
from mitorch.datasets.albumentations_transforms import SurveillanceCameraTransform, AlbumentationsTransform
from mitorch.datasets.transforms import AutoAugmentTransform, InceptionTransform

COMPONENT_BUILDERS = {
    'center_crop': lambda input_size: albumentations.CenterCrop(input_size, input_size),
    'flip': lambda input_size: albumentations.Flip(),
    'horizontal_flip': lambda input_size: albumentations.HorizontalFlip(),
    'image_compression': lambda input_size: albumentations.ImageCompression(quality_lower=20, quality_upper=100),
    'random_brightness_contrast': lambda input_size: albumentations.RandomBrightnessContrast(),
    'random_sized_bbox_safe_crop': lambda input_size: albumentations.RandomSizedBBoxSafeCrop(input_size, input_size, interpolation=cv2.INTER_CUBIC),
    'random_resized_crop': lambda input_size: albumentations.RandomResizedCrop(input_size, input_size, interpolation=cv2.INTER_CUBIC),
    'resize': lambda input_size: albumentations.Resize(input_size, input_size),
    'smallest_max_size': lambda input_size: albumentations.SmallestMaxSize(int(input_size / 224 * 256), interpolation=cv2.INTER_CUBIC),
    'to_gray': lambda input_size: albumentations.ToGray(p=0.1),
}

# For backward compatibility.
# If the augmentation description is a simple string, return the following transforms.
SPECIAL_TRANSFORM = {
    'auto_augment': AutoAugmentTransform,
    'inception': InceptionTransform,
    'surveillance': SurveillanceCameraTransform
}


class TransformFactory:
    def __init__(self, is_object_detection, input_size):
        self._is_object_detection = is_object_detection
        self._input_size = input_size

    def create(self, augmentations):
        if len(augmentations) == 1 and augmentations[0] in SPECIAL_TRANSFORM:
            return SPECIAL_TRANSFORM[augmentations[0]](self._input_size, self._is_object_detection)

        components = [self._build_component(n) for n in augmentations]
        transform = AlbumentationsTransform(components, self._is_object_detection)
        return transform

    def _build_component(self, component_description):
        builder = COMPONENT_BUILDERS[component_description]
        return builder(self._input_size)
