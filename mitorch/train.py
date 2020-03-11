import argparse
import json
import pytorch_lightning as pl
import torch
from .mimodel import MiModel


def train(config_filepath, train_dataset_filepath, val_dataset_filepath, fast_dev_run):
    with open(config_filepath) as f:
        config = json.load(f)

    # TODO: Look into logger.
    gpus = -1 if torch.cuda.is_available() else None
    distributed_backend = 'dp' if torch.cuda.is_available() else None
    trainer = pl.Trainer(fast_dev_run=fast_dev_run, gpus=gpus, distributed_backend=distributed_backend, checkpoint_callback=False)
    model = MiModel(config, train_dataset_filepath, val_dataset_filepath)
    trainer.fit(model)


def main():
    parser = argparse.ArgumentParser("Train a model")
    parser.add_argument('config_filepath', type=str, help="Filepath to config.json")
    parser.add_argument('train_dataset_filepath', type=str, help="Filepath to training dataset")
    parser.add_argument('val_dataset_filepath', type=str, help="Filepath to validation dataset")
    parser.add_argument('--fast_dev_run', action='store_true', default=False)

    args = parser.parse_args()
    train(args.config_filepath, args.train_dataset_filepath, args.val_dataset_filepath, args.fast_dev_run)


if __name__ == '__main__':
    main()
