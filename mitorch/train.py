"""Train a model based on the given config."""
import argparse
import json
import pytorch_lightning as pl
import torch
from .mimodel import MiModel


def train(config, train_dataset_filepath, val_dataset_filepath, weights_filepath, output_filepath, fast_dev_run, logger=None):

    if isinstance(config, str):
        with open(config) as f:
            config = json.load(f)

    gpus = -1 if torch.cuda.is_available() else None
    distributed_backend = 'dp' if torch.cuda.is_available() else None
    trainer = pl.Trainer(fast_dev_run=fast_dev_run, gpus=gpus, distributed_backend=distributed_backend, checkpoint_callback=False, logger=logger)
    model = MiModel(config, train_dataset_filepath, val_dataset_filepath, weights_filepath)
    trainer.fit(model)
    trainer.test(model)
    model.save(output_filepath)


def main():
    parser = argparse.ArgumentParser("Train a model")
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
