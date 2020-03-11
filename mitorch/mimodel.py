from .builders import DataLoaderBuilder, LrSchedulerBuilder, ModelBuilder, OptimizerBuilder
from .evaluators import MulticlassClassificationEvaluator, MultilabelClassificationEvaluator, ObjectDetectionEvaluator
import pytorch_lightning as pl

# TODO: Support multigpu
class MiModel(pl.LightningModule):
    def __init__(self, config, train_dataset_filepath, val_dataset_filepath):
        super(MiModel, self).__init__()
        self._train_dataloader, self._val_dataloader = DataLoaderBuilder(config).build(train_dataset_filepath, val_dataset_filepath)
        self.model, self.criterion, self.predictor = ModelBuilder(config).build(self._train_dataloader)
        self.optimizer = OptimizerBuilder(config).build(self.model)
        max_iters = len(self._train_dataloader) * config['max_epochs']
        self.lr_scheduler = LrSchedulerBuilder(config).build(self.optimizer, max_iters)
        self.evaluator = self._get_evaluator(self._train_dataloader.dataset.dataset_type)

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
        for key in results:
            results[key] = torch.tensor(results[key])
        results['val_loss'] = torch.tensor([o['val_loss'] for o in outputs]).mean()
        return {'log': results}

    def forward(self, x):
        return self.model(x)
