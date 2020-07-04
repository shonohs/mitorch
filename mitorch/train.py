"""Train a model based on the given config."""
import argparse
import json
import logging
import pathlib
import uuid
import pytorch_lightning as pl
import torch
from .logger import StdoutLogger, MongoDBLogger
from .mimodel import MiModel

_logger = logging.getLogger(__name__)


def train(config_filepath, train_dataset_filepath, val_dataset_filepath, weights_filepath, output_filepath, fast_dev_run, job_id, db_url):
    _logger.info("started")

    logger = [StdoutLogger()]
    if job_id and db_url:
        logger.append(MongoDBLogger(db_url, job_id))

    pl.seed_everything(0)

    config = json.loads(config_filepath.read_text())

    if fast_dev_run:
        config['batch_size'] = 2

    num_processes = config.get('num_processes', -1)
    gpus = num_processes if torch.cuda.is_available() else None
    if num_processes > 1 and not gpus:
        _logger.warning(f"Multiple processes are requested, but only 1 CPU is available on this node.")

    hparams = {'config': config, 'train_dataset_filepath': train_dataset_filepath, 'val_dataset_filepath': val_dataset_filepath, 'weights_filepath': weights_filepath}
    model = MiModel(hparams)
    for lo in logger:
        lo.log_hyperparams({'model_versions': model.model_version})

    trainer = pl.Trainer(max_epochs=config['max_epochs'], fast_dev_run=fast_dev_run, gpus=gpus, distributed_backend='ddp',
                         logger=logger, progress_bar_refresh_rate=0, check_val_every_n_epoch=10, num_sanity_val_steps=0, deterministic=True)

    trainer.fit(model)
    if output_filepath:
        model.save(output_filepath)


def main():
    logging.getLogger().setLevel(logging.INFO)
    logging.getLogger('mitorch').setLevel(logging.DEBUG)

    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument('config_filepath', type=pathlib.Path)
    parser.add_argument('train_dataset_filepath', type=pathlib.Path)
    parser.add_argument('val_dataset_filepath', type=pathlib.Path)
    parser.add_argument('--weights_filepath', '-w', type=pathlib.Path)
    parser.add_argument('--output_filepath', '-o', type=pathlib.Path)
    parser.add_argument('--fast_dev_run', '-d', action='store_true')
    parser.add_argument('--job_id', type=uuid.UUID)
    parser.add_argument('--db_url')

    args = parser.parse_args()
    train(args.config_filepath, args.train_dataset_filepath, args.val_dataset_filepath, args.weights_filepath,
          args.output_filepath, args.fast_dev_run, args.job_id, args.db_url)


if __name__ == '__main__':
    main()
