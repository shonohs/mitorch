import torch
from mitorch.models import ModelFactory


class ModelBuilder:
    def __init__(self, config):
        self.config = config['model']

    def build(self, dataloader):
        num_classes = len(dataloader.dataset.labels)
        dataset_type = dataloader.dataset.dataset_type

        model = ModelFactory.create(self.config['name'], num_classes)
        if dataset_type == 'multiclass_classification':
            criterion = torch.nn.CrossEntropyLoss()
            predictor = torch.nn.Softmax(dim=1)
        elif dataset_type == 'multilabel_classification':
            criterion = torch.nn.BCEWithLogitsLoss()
            predictor = torch.nn.Sigmoid()
        elif dataset_type == 'object_detection':
            criterion = model.loss
            predictor = model.predictor
        else:
            raise NotImplementedError(f"Non supported dataset type: {dataset_type}")

        return model, criterion, predictor
