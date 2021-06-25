import functools
import logging
import torch
from ..datasets import (ImageDataset, CenterCropTransform, CenterCropTransformV2, CenterCropTransformV3, ResizeTransform, ResizeFlipTransform,
                        RandomResizedCropTransform, RandomResizedCropTransformV2, RandomResizedCropTransformV3, RandomResizedCropTransformV4,
                        RandomSizedBBoxSafeCropTransform, InceptionTransform, DevTransform, Dev2Transform)


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
        self.augmentation_config = config['augmentation']
        self.task_type = config['task_type']
        self.input_size = config['input_size']
        self.batch_size = config['batch_size']

    def build(self, train_dataset_filepath, val_dataset_filepath):
        logging.info(f"Building a data_loader. train: {train_dataset_filepath}, val: {val_dataset_filepath}, augmentation: {self.augmentation_config}, "
                     f"task: {self.task_type}, input_size: {self.input_size}, batch_size: {self.batch_size}")

        is_object_detection = self.task_type == 'object_detection'
        train_augmentation = self.build_augmentation(self.augmentation_config['train'], self.input_size, is_object_detection)
        train_dataset = ImageDataset.from_file(train_dataset_filepath, train_augmentation)

        val_augmentation = self.build_augmentation(self.augmentation_config['val'], self.input_size, is_object_detection)
        val_dataset = ImageDataset.from_file(val_dataset_filepath, val_augmentation)

        collate_fn = functools.partial(_default_collate, self.task_type)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, self.batch_size, shuffle=True, num_workers=8, pin_memory=True, collate_fn=collate_fn)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, self.batch_size, shuffle=False, num_workers=8, pin_memory=True, collate_fn=collate_fn)

        return train_dataloader, val_dataloader

    @staticmethod
    def build_augmentation(name, input_size, is_object_detection):
        augmentation_class = {'center_crop': CenterCropTransform,
                              'center_crop_v2': CenterCropTransformV2,
                              'center_crop_v3': CenterCropTransformV3,
                              'dev': DevTransform,
                              'dev2': Dev2Transform,
                              'inception': InceptionTransform,
                              'resize': ResizeTransform,
                              'resize_flip': ResizeFlipTransform,
                              'random_resize': RandomResizedCropTransform,
                              'random_resize_v2': RandomResizedCropTransformV2,
                              'random_resize_v3': RandomResizedCropTransformV3,
                              'random_resize_v4': RandomResizedCropTransformV4,
                              'random_resize_bbox': RandomSizedBBoxSafeCropTransform}.get(name)

        if not augmentation_class:
            raise NotImplementedError(f"Non supported augmentation: {name}")

        return augmentation_class(input_size, is_object_detection)
