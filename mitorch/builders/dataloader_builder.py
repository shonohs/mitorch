import torch
from ..datasets import ImageDataset, ResizeTransform


class DataLoaderBuilder:
    def __init__(self, config):
        self.augmentation_config = config['augmentation']
        self.config = config

    def build(self, train_dataset_filepath, val_dataset_filepath):
        train_augmentation = self._build_augmentation(self.augmentation_config['train'], self.config['input_size'])
        train_dataset = ImageDataset.from_file(train_dataset_filepath, train_augmentation)

        val_augmentation = self._build_augmentation(self.augmentation_config['val'], self.config['input_size'])
        val_dataset = ImageDataset.from_file(val_dataset_filepath, val_augmentation)

        batch_size = self.config['batch_size']
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=True, num_workers=4, pin_memory=True)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size, shuffle=False, num_workers=4, pin_memory=True)

        return train_dataloader, val_dataloader

    def _build_augmentation(self, name, input_size):
        # TODO: Add more augmentation
        if name == 'resize':
            return ResizeTransform(input_size)
        else:
            raise NotImplementedError(f"Non supported augmentation: {name}")
