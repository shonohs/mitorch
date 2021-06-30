import torchvision


INPUT_MEAN = [0.5, 0.5, 0.5]


class Transform:
    def __init__(self, input_size, is_object_detection=False):
        assert not is_object_detection
        self._transform = torchvision.transforms.Compose(self.get_transforms(input_size))

    def __call__(self, image, target):
        image = self._transform(image)
        return image, target


class InceptionTransform(Transform):
    def get_transforms(self, input_size):
        return [torchvision.transforms.RandomResizedCrop(input_size),
                torchvision.transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(INPUT_MEAN, [1, 1, 1], inplace=True)]


class AutoAugmentTransform(Transform):
    def get_transforms(self, input_size):
        return [torchvision.transforms.RandomResizedCrop(input_size),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.AutoAugment(interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(INPUT_MEAN, [1, 1, 1], inplace=True)]
