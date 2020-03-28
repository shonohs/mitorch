import pytorch_lightning as pl
import torch
from .builders import DataLoaderBuilder, LrSchedulerBuilder, ModelBuilder, OptimizerBuilder
from .evaluators import MulticlassClassificationEvaluator, MultilabelClassificationEvaluator, ObjectDetectionEvaluator


class MiModel(pl.LightningModule):
    def __init__(self, config, train_dataset_filepath, val_dataset_filepath, weights_filepath):
        super(MiModel, self).__init__()
        self._train_dataloader, self._val_dataloader = DataLoaderBuilder(config).build(train_dataset_filepath, val_dataset_filepath)
        self.model, self.criterion, self.predictor = ModelBuilder(config).build(self._train_dataloader, weights_filepath)
        self.optimizer = OptimizerBuilder(config).build(self.model)
        max_iters = len(self._train_dataloader) * config['max_epochs']
        self.lr_scheduler = LrSchedulerBuilder(config).build(self.optimizer, max_iters)
        self.evaluator = self._get_evaluator(self._train_dataloader.dataset.dataset_type)
        self.logger.log_hyperparams({'model_versions': self.model.version})

    @staticmethod
    def _get_evaluator(dataset_type):
        if dataset_type == 'multiclass_classification':
            return MulticlassClassificationEvaluator()
        elif dataset_type == 'multilabel_classification':
            return MultilabelClassificationEvaluator()
        elif dataset_type == 'object_detection':
            return ObjectDetectionEvaluator()
        raise NotImplementedError

    def configure_optimizers(self):
        # lr_scheduler.step() is called after every training steps.
        return self.optimizer, {'scheduler': self.lr_scheduler, 'interval': 'step'}

    def train_dataloader(self):
        return self._train_dataloader

    def val_dataloader(self):
        return self._val_dataloader

    def test_dataloader(self):
        return self.val_dataloader()

    def training_step(self, batch, batch_index):
        image, target = batch
        output = self.forward(image)
        loss = self.criterion(output, target)
        return {'loss': loss}

    def validation_step(self, batch, batch_index):
        image, target = batch
        output = self.forward(image)
        loss = self.criterion(output, target)
        predictions = self.predictor(output)
        self.evaluator.add_predictions(predictions, target)

        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        results = self.evaluator.get_reports()
        self.evaluator.reset()
        results = {key: torch.tensor(value) for key, value in results.items()}
        results['val_loss'] = torch.tensor([o['val_loss'] for o in outputs]).mean()
        return {'log': results}

    def test_step(self, batch, batch_index):
        image, target = batch
        output = self.forward(image)
        loss = self.criterion(output, target)
        predictions = self.predictor(output)
        self.evaluator.add_predictions(predictions, target)

        return {'test_loss': loss}

    def test_epoch_end(self, outputs):
        results = self.evaluator.get_reports()
        self.evaluator.reset()
        results = {key: torch.tensor(value) for key, value in results.items()}
        results['test_loss'] = torch.tensor([o['test_loss'] for o in outputs]).mean()
        self.logger.log_test_result(results)
        return {'log': results}

    def forward(self, x):
        return self.model(x)

    def save(self, filepath):
        state_dict = self.model.state_dict()
        torch.save(state_dict, filepath)
