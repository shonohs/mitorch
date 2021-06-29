import functools
import logging
import torch
from mitorch.datasets import ImageDataset, TransformFactory

NUM_WORKERS = 4


def _default_collate(task_type, batch):
    image, target = zip(*batch)
    image = torch.stack(image, 0)
    if task_type == 'multiclass_classification':
        target = torch.tensor(target)
    elif task_type == 'multilabel_classification':
        raise NotImplementedError  # TODO: Support multilabel datasets
    return image, target


class DataLoaderBuilder:
    def __init__(self, config):
        self.augmentation_config = config.augmentation
        self.task_type = config.task_type
        self.input_size = config.model.input_size
        self.batch_size = config.batch_size

    def build(self, train_dataset_filepath, val_dataset_filepath):
        logging.info(f"Building a data_loader. train: {train_dataset_filepath}, val: {val_dataset_filepath}, augmentation: {self.augmentation_config}, "
                     f"task: {self.task_type}, input_size: {self.input_size}, batch_size: {self.batch_size}")

        is_object_detection = self.task_type == 'object_detection'
        collate_fn = functools.partial(_default_collate, self.task_type)

        train_augmentation = self.build_augmentation(self.augmentation_config.train, self.input_size, is_object_detection)
        train_dataset = ImageDataset.from_file(train_dataset_filepath, train_augmentation)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, self.batch_size, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True, collate_fn=collate_fn)

        if val_dataset_filepath:
            val_augmentation = self.build_augmentation(self.augmentation_config.val, self.input_size, is_object_detection)
            val_dataset = ImageDataset.from_file(val_dataset_filepath, val_augmentation)
            val_dataloader = torch.utils.data.DataLoader(val_dataset, self.batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True, collate_fn=collate_fn)
        else:
            val_dataloader = None

        return train_dataloader, val_dataloader

    @staticmethod
    def build_augmentation(name, input_size, is_object_detection):
        transform = TransformFactory(is_object_detection).create(name, input_size)
        if not transform:
            raise NotImplementedError(f"Non supported augmentation: {name}")
        return transform
