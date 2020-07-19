import inspect
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


class InceptionTransform(Transform):
    def __init__(self, input_size, is_object_detection=False):
        transforms = [torchvision.transforms.RandomResizedCrop(input_size),
                      torchvision.transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                      torchvision.transforms.ToTensor(),
                      torchvision.transforms.Normalize(INPUT_MEAN, [1, 1, 1], inplace=True)]
        super().__init__(transforms)


class DevTransform(Transform):
    """Temporarl transforms for development."""
    def __init__(self, input_size, is_object_detection=False):
        transforms = [torchvision.transforms.RandomResizedCrop(input_size),
                      torchvision.transforms.RandomHorizontalFlip(),
                      torchvision.transforms.ToTensor(),
                      torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        super().__init__(transforms)


class Dev2Transform(Transform):
    """Temporarl transforms for development."""
    def __init__(self, input_size, is_object_detection=False):
        transforms = [torchvision.transforms.Resize(256),
                      torchvision.transforms.CenterCrop(224),
                      torchvision.transforms.ToTensor(),
                      torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        super().__init__(transforms)
