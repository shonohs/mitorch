"""Train a model based on the given config."""
import argparse
import importlib.metadata
import logging
import pathlib
import uuid
import jsons
import pytorch_lightning as pl
import torch
from mitorch.builders import DataLoaderBuilder
from mitorch.common import MiModel, TrainingConfig, StandardLogger, MongoDBLogger
from mitorch.commands.common import init_logging

_logger = logging.getLogger(__name__)


def train(config, train_dataset_filepath, val_dataset_filepath, weights_filepath, output_filepath, job_id, db_url, tensorboard_log_dir, fast_dev_run=False):
    try:
        _logger.info(f"Training started. mitorch version is {importlib.metadata.version('mitorch')}. model version is {importlib.metadata.version('mitorch-models')}")
    except Exception:
        _logger.info("Training started.")

    logger = [StandardLogger()]
    if job_id and db_url:
        logger.append(MongoDBLogger(db_url, job_id))
    if tensorboard_log_dir:
        logger.append(pl.loggers.TensorBoardLogger(str(tensorboard_log_dir)))

    pl.seed_everything(0)

    _logger.debug(f"Loaded config: {config}")

    if torch.cuda.is_available():
        precision = 16 if config.use_fp16 else 32
        gpus = config.num_processes
    else:
        precision = 32
        gpus = None
        if config.num_processes > 1:
            _logger.warning("Multiple processes are requested, but only 1 CPU is available on this node.")
        if config.use_fp16:
            _logger.warning("AMP is requested, but GPU is not available.")

    callbacks = []
    if config.use_swa:
        callbacks.append(pl.callbacks.StochasticWeightAveraging(swa_epoch_start=config.swa_epoch_start))

    train_dataloader, val_dataloader = DataLoaderBuilder(config).build(train_dataset_filepath, val_dataset_filepath)
    num_classes = len(train_dataloader.dataset.labels) if train_dataloader else len(val_dataloader.dataset.labels)

    trainer = pl.Trainer(max_epochs=config.max_epochs, fast_dev_run=fast_dev_run, gpus=gpus, distributed_backend='ddp', terminate_on_nan=True,
                         logger=logger, progress_bar_refresh_rate=0, check_val_every_n_epoch=10, num_sanity_val_steps=0, deterministic=False,
                         accumulate_grad_batches=config.accumulate_grad_batches, checkpoint_callback=False, precision=precision, callbacks=callbacks, sync_batchnorm=True)

    model = MiModel(config, num_classes, weights_filepath)
    trainer.fit(model, train_dataloader, val_dataloader)
    _logger.info("Training completed.")

    # Save the weights before validation since there is a risk of failure.
    if output_filepath:
        model.save(output_filepath)

    trainer.validate(model, val_dataloader)
    _logger.info("Validation completed.")


def main():
    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument('config_filepath', type=pathlib.Path)
    parser.add_argument('train_dataset_filepath', type=pathlib.Path)
    parser.add_argument('val_dataset_filepath', nargs='?', type=pathlib.Path)
    parser.add_argument('--weights_filepath', '-w', type=pathlib.Path)
    parser.add_argument('--output_filepath', '-o', type=pathlib.Path)
    parser.add_argument('--fast_dev_run', '-d', action='store_true')
    parser.add_argument('--job_id', type=uuid.UUID)
    parser.add_argument('--db_url')
    parser.add_argument('--tensorboard_log', type=pathlib.Path)
    parser.add_argument('--log_file', type=pathlib.Path)

    args = parser.parse_args()
    init_logging(args.log_file)

    config = jsons.loads(args.config_filepath.read_text(), TrainingConfig)

    try:
        train(config, args.train_dataset_filepath, args.val_dataset_filepath, args.weights_filepath,
              args.output_filepath, args.job_id, args.db_url, args.tensorboard_log, args.fast_dev_run)
    except Exception:
        _logger.exception("Training failed.")
        raise


if __name__ == '__main__':
    main()
