import hashlib
import io
import logging
import os
import pytorch_lightning as pl
import torch
from .builders import DataLoaderBuilder, LrSchedulerBuilder, ModelBuilder, OptimizerBuilder
from .evaluators import MulticlassClassificationEvaluator, MultilabelClassificationEvaluator, ObjectDetectionEvaluator


class MiModel(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        # Save arguments so that checkpoint can load it. Remove after updating the pytorch lightning
        self.hparams = hparams
        config = hparams['config']
        train_dataset_filepath = hparams['train_dataset_filepath']
        val_dataset_filepath = hparams['val_dataset_filepath']
        weights_filepath = hparams['weights_filepath']

        self._train_dataloader, self._val_dataloader = DataLoaderBuilder(config).build(train_dataset_filepath, val_dataset_filepath)
        num_classes = len(self._train_dataloader.dataset.labels)
        self.model = ModelBuilder(config).build(num_classes, weights_filepath)
        self.optimizer = OptimizerBuilder(config).build(self.model)
        self.lr_scheduler = LrSchedulerBuilder(config).build(self.optimizer, len(self._train_dataloader))
        self.evaluator = self._get_evaluator(config['task_type'])
        self.train_epoch = 0

    @property
    def model_version(self):
        return self.model.version

    @staticmethod
    def _get_evaluator(task_type):
        mappings = {'multiclass_classification': MulticlassClassificationEvaluator,
                    'multilabel_classification': MultilabelClassificationEvaluator,
                    'object_detection': ObjectDetectionEvaluator}
        assert task_type in mappings
        return mappings[task_type]()

    def configure_optimizers(self):
        # lr_scheduler.step() is called after every training steps.
        return {'optimizer': self.optimizer, 'lr_scheduler': {'scheduler': self.lr_scheduler, 'interval': 'step'}}

    def train_dataloader(self):
        return self._train_dataloader

    def val_dataloader(self):
        return self._val_dataloader

    def test_dataloader(self):
        return self.val_dataloader()

    def training_step(self, batch, batch_index):
        image, target = batch
        output = self.forward(image)
        loss = self.model.loss(output, target)
        self._get_model_hash()
        return {'loss': loss, 'log': {'train_loss': float(loss)}}

    def training_epoch_end(self, outputs):
        train_loss = torch.cat([o['loss'] if o['loss'].shape else o['loss'].unsqueeze(0) for o in outputs], dim=0).mean()
        self._log_epoch_metrics({'train_loss': train_loss}, self.current_epoch)
        return {}

    def validation_step(self, batch, batch_index):
        image, target = batch
        output = self.forward(image)
        loss = self.model.loss(output, target)
        predictions = self.model.predictor(output)
        self.evaluator.add_predictions(predictions, target)
        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        results = self.evaluator.get_report()
        self.evaluator.reset()
        results = {key: torch.tensor(value).to(self.device) for key, value in results.items()}
        results['val_loss'] = torch.cat([o['val_loss'] if o['val_loss'].shape else o['val_loss'].unsqueeze(0) for o in outputs], dim=0).to(self.device).mean()
        self._log_epoch_metrics(results, self.current_epoch)
        return {'log': results}

    def test_step(self, batch, batch_index):
        val_loss = self.validation_step(batch, batch_index)
        return {'test_loss': val_loss['val_loss']}

    def test_epoch_end(self, outputs):
        results = self.evaluator.get_report()
        self.evaluator.reset()
        results = {key: torch.tensor(value).to(self.device) for key, value in results.items()}
        results['test_loss'] = torch.cat([o['test_loss'] if o['test_loss'].shape else o['test_loss'].unsqueeze(0) for o in outputs], dim=0).to(self.device).mean()
        self._log_epoch_metrics(results, self.current_epoch)
        return {'log': results}

    def forward(self, x):
        return self.model(x)

    def save(self, filepath):
        print(f"save: NODE{os.getenv('LOCAL_RANK')} Model hash: {self._get_model_hash()}")
        if int(os.getenv('LOCAL_RANK', 0)) == 0:
            logging.info(f"Saving a model to {filepath}")
            state_dict = self.model.state_dict()
            torch.save(state_dict, filepath)

    def _get_model_hash(self):
        state_dict = self.model.state_dict()
        values = {}
        attention = state_dict['base_model.features.conv0.conv.weight'].cpu()
        bytesio = io.BytesIO()
        torch.save(attention, bytesio)
        bytesio.seek(0)
        attention_hash = hashlib.sha1(bytesio.getvalue()).hexdigest()

        logging.info(f"NODE{os.getenv('LOCAL_RANK')} base_model.features.conv0.conv.weight: mean: {torch.mean(attention)}, max: {torch.mean(attention)} hash: {attention_hash}")
        return values


    def _log_epoch_metrics(self, metrics, epoch):
        loggers = self.logger.experiment if isinstance(self.logger.experiment, list) else [self.logger.experiment]
        for l in loggers:
            l.log_epoch_metrics(metrics, epoch)
