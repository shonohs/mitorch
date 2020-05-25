"""Train a model based on the given config."""
import argparse
import json
import random
import numpy
import pytorch_lightning as pl
import torch
from .logger import StdoutLogger
from .mimodel import MiModel


def set_random_seed(seed):
    torch.manual_seed(seed)
    numpy.random.seed(seed)
    random.seed(seed)


def train(config, train_dataset_filepath, val_dataset_filepath, weights_filepath, output_filepath, fast_dev_run, logger=None):
    if not logger:
        logger = StdoutLogger()

    set_random_seed(0)

    if isinstance(config, str):
        with open(config) as f:
            config = json.load(f)

    gpus = -1 if torch.cuda.is_available() else None
    distributed_backend = 'ddp' if torch.cuda.is_available() else 'ddp_cpu'
    model = MiModel(config, train_dataset_filepath, val_dataset_filepath, weights_filepath)
    for l in logger if isinstance(logger, list) else [logger]:
        l.log_hyperparams({'model_versions': model.model_version})

    trainer = pl.Trainer(max_epochs=config['max_epochs'], fast_dev_run=fast_dev_run, gpus=gpus, distributed_backend=distributed_backend,
                         logger=logger, progress_bar_refresh_rate=0, check_val_every_n_epoch=10, num_sanity_val_steps=0, checkpoint_callback=False)

    trainer.fit(model)
    trainer.test(model)
    if output_filepath:
        model.save(output_filepath)


def main():
    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument('config_filepath', help="Filepath to config.json")
    parser.add_argument('train_dataset_filepath', help="Filepath to training dataset")
    parser.add_argument('val_dataset_filepath', help="Filepath to validation dataset")
    parser.add_argument('--weights_filepath', '-w', help="Filepath to the pretrained weights")
    parser.add_argument('--output_filepath', '-o', help="Filepath to the output model weights")
    parser.add_argument('--fast_dev_run', '-d', action='store_true')

    args = parser.parse_args()
    train(args.config_filepath, args.train_dataset_filepath, args.val_dataset_filepath, args.weights_filepath,
          args.output_filepath, args.fast_dev_run)


if __name__ == '__main__':
    main()
