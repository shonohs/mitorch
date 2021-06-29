"""Lightning Module class for all trainings in mitorch."""
import logging
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities import rank_zero_only
import torch
from mitorch.builders import EvaluatorBuilder, LrSchedulerBuilder, ModelBuilder, OptimizerBuilder


class MiModel(LightningModule):
    def __init__(self, config, num_classes, weights_filepath=None):
        super().__init__()
        self.save_hyperparameters('config')
        self.model = ModelBuilder(config).build(num_classes, weights_filepath)
        # TODO: Leverage torchmetrics
        self.evaluator = EvaluatorBuilder(config).build()

    def configure_optimizers(self):
        # lr_scheduler.step() is called after every training steps.
        optimizer = OptimizerBuilder(self.hparams['config']).build(self.model)
        num_samples = len(self.train_dataloader.dataloader.dataset)
        lr_scheduler = LrSchedulerBuilder(self.hparams['config']).build(optimizer, num_samples)
        return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': lr_scheduler, 'interval': 'step'}}

    def training_step(self, batch, batch_index):
        image, target = batch
        output = self.forward(image)
        loss = self.model.loss(output, target)
        self.log('train_loss', loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_index):
        image, target = batch
        output = self.forward(image)
        loss = self.model.loss(output, target)
        predictions = self.model.predictor(output)
        self.evaluator.add_predictions(predictions, target)
        self.log('val_loss', loss, sync_dist=True)

    def validation_epoch_end(self, outputs):
        results = self.evaluator.get_report()
        self.evaluator.reset()
        results = {'val_' + key: torch.tensor(value).to(self.device) for key, value in results.items()}
        self.log_dict(results, sync_dist=True)

    def test_step(self, batch, batch_index):
        image, target = batch
        output = self.forward(image)
        loss = self.model.loss(output, target)
        predictions = self.model.predictor(output)
        self.evaluator.add_predictions(predictions, target)
        self.log('test_loss', loss, sync_dist=True)

    def test_epoch_end(self, outputs):
        results = self.evaluator.get_report()
        self.evaluator.reset()
        results = {'test_' + key: torch.tensor(value).to(self.device) for key, value in results.items()}
        self.log_dict(results, sync_dist=True)

    def forward(self, x):
        return self.model(x)

    @rank_zero_only
    def save(self, filepath):
        logging.info(f"Saving a model to {filepath}")
        state_dict = self.model.state_dict()
        torch.save(state_dict, filepath)
