import inspect
import torchvision


INPUT_MEAN = [0.5, 0.5, 0.5]


class Transform:
    def __init__(self, transforms):
        self.transforms = transforms
        self.num_params = [len(inspect.signature(t).parameters) for t in transforms]

    def __call__(self, image, target):
        for t, n in zip(self.transforms, self.num_params):
            if n == 1:
                image = t(image)
            elif n == 2:
                image, target = t(image, target)
            else:
                raise NotImplementedError(f"Non supported number of params: {n}")
        return image, target


class ResizeTransform(Transform):
    def __init__(self, input_size):
        transforms = [torchvision.transforms.Resize((input_size, input_size)),
                      torchvision.transforms.ToTensor(),
                      torchvision.transforms.Normalize(INPUT_MEAN, [1, 1, 1], inplace=True)]
        super(ResizeTransform, self).__init__(transforms)


class InceptionTransform(Transform):
    def __init__(self, input_size):
        transforms = [torchvision.transforms.RandomResizedCrop(input_size),
                      torchvision.transforms.ColorJitter(brightness=0.1, contract=0.1, saturation=0.1, hue=0.1),
                      torchvision.transforms.ToTensor(),
                      torchvision.transforms.Normalize(INPUT_MEAN, [1, 1, 1], inplace=True)]
        super(ResizeTransform, self).__init__(transforms)


class ResizeFlipTransform(Transform):
    def __init__(self, input_size):
        transforms = [torchvision.transforms.Resize((input_size, input_size)),
                      torchvision.transforms.RandomHorizontalFlip(),
                      torchvision.transforms.ToTensor(),
                      torchvision.transforms.Normalize(INPUT_MEAN, [1, 1, 1], inplace=True)]
        super(ResizeTransform, self).__init__(transforms)
