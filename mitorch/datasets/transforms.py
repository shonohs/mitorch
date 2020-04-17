import inspect
import random
import PIL.ImageOps
import torchvision


INPUT_MEAN = [0.5, 0.5, 0.5]


class Transform:
    def __init__(self, transforms):
        self.transforms = transforms
        self.num_params = [len(inspect.signature(t).parameters) for t in transforms]

    def __call__(self, image, target):
        assert image is not None
        assert target is not None

        for t, n in zip(self.transforms, self.num_params):
            if n == 1:
                image = t(image)
            elif n == 2:
                image, target = t(image, target)
            else:
                raise NotImplementedError(f"Non supported number of params: {n}")
        assert image is not None
        assert target is not None
        return image, target


class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, target):
        if random.randon() < self.p:
            image = PIL.ImageOps.mirror(image)
            if target and isinstance(target[0], list):
                target = [[t[0], 1 - t[3], t[2], 1 - t[1], t[4]] for t in target]
        return image, target


class ResizeTransform(Transform):
    def __init__(self, input_size):
        transforms = [torchvision.transforms.Resize((input_size, input_size)),
                      torchvision.transforms.ToTensor(),
                      torchvision.transforms.Normalize(INPUT_MEAN, [1, 1, 1], inplace=True)]
        super().__init__(transforms)


class InceptionTransform(Transform):
    def __init__(self, input_size):
        transforms = [torchvision.transforms.RandomResizedCrop(input_size),
                      torchvision.transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                      torchvision.transforms.ToTensor(),
                      torchvision.transforms.Normalize(INPUT_MEAN, [1, 1, 1], inplace=True)]
        super().__init__(transforms)


class ResizeFlipTransform(Transform):
    def __init__(self, input_size):
        transforms = [torchvision.transforms.Resize((input_size, input_size)),
                      RandomHorizontalFlip(),
                      torchvision.transforms.ToTensor(),
                      torchvision.transforms.Normalize(INPUT_MEAN, [1, 1, 1], inplace=True)]
        super().__init__(transforms)
