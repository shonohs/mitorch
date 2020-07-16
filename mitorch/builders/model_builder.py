import hashlib
import logging
import pickle
import torch
from mitorch.models import ModelFactory


class ModelBuilder:
    def __init__(self, config):
        self.config = config['model']

    def build(self, dataloader, weights_filepath=None):
        logging.info(f"Building a model. weights: {weights_filepath}, config: {self.config}")
        num_classes = len(dataloader.dataset.labels)
        dataset_type = dataloader.dataset.dataset_type

        model = ModelFactory.create(self.config['name'], num_classes, self.config['options'])

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

        if weights_filepath:
            self._load_weights(model, weights_filepath)

        self._dump_model_hash(model)
        return model, criterion, predictor

    def _load_weights(self, model, weights_filepath):
        def _get_depth(param_names):
            max_depth = 0
            for name in param_names:
                depth = 0
                while name.startswith('base_model.'):
                    depth += 1
                    name = name[11:]  # Remove 'base_model.'
                max_depth = max(depth, max_depth)
            return max_depth

        weights = torch.load(weights_filepath, map_location=torch.device('cpu'))
        src_depth = _get_depth(weights.keys())
        dst_depth = _get_depth(model.state_dict().keys())
        logging.debug(f"Source depth: {src_depth}, Destination depth: {dst_depth}")

        if dst_depth > src_depth:
            for i in range(dst_depth - src_depth):
                model = model.base_model
        elif src_depth > dst_depth:
            raise NotImplementedError

        try:
            model.load_state_dict(weights, strict=False)
        except RuntimeError as e:
            logging.warning(f"Ignored load erros: {e}")

    @staticmethod
    def _dump_model_hash(model):
        serialized = pickle.dumps({k: v.numpy() for k, v in model.state_dict().items()})
        model_hash = hashlib.sha1(serialized).hexdigest()
        logging.info(f"Model hash: {model_hash}")
