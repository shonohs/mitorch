import torch
from ..datasets import ImageDataset, ResizeTransform, ResizeFlipTransform, RandomResizedCropTransform


def _default_collate(batch):
    image, target = zip(*batch)
    image = torch.stack(image, 0)
    if isinstance(target[0], (list, tuple)) and not isinstance(target[0][0], (list, tuple)):  # Multilabel classification
        raise NotImplementedError()

    return image, target


class DataLoaderBuilder:
    def __init__(self, config):
        self.augmentation_config = config['augmentation']
        self.config = config

    def build(self, train_dataset_filepath, val_dataset_filepath):
        is_object_detection = self.config['task_type'] == 'object_detection'
        train_augmentation = self.build_augmentation(self.augmentation_config['train'], self.config['input_size'], is_object_detection)
        train_dataset = ImageDataset.from_file(train_dataset_filepath, train_augmentation)

        val_augmentation = self.build_augmentation(self.augmentation_config['val'], self.config['input_size'], is_object_detection)
        val_dataset = ImageDataset.from_file(val_dataset_filepath, val_augmentation)

        batch_size = self.config['batch_size']
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=True, num_workers=8, pin_memory=True, collate_fn=_default_collate)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size, shuffle=False, num_workers=8, pin_memory=True, collate_fn=_default_collate)

        return train_dataloader, val_dataloader

    @staticmethod
    def build_augmentation(name, input_size, is_object_detection):
        augmentation_class = {'resize': ResizeTransform,
                              'resize_flip': ResizeFlipTransform,
                              'random_resize': RandomResizedCropTransform}.get(name)

        if not augmentation_class:
            raise NotImplementedError(f"Non supported augmentation: {name}")

        return augmentation_class(input_size, is_object_detection)
