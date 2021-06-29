# MiTorch
A simple training platform for PyTorch.

The models are implemented in [mitorch-models](https://github.com/shonohs/mitorch_models).

# Install
Use python 3.8+.
```bash
pip install mitorch
```

# Usage

## Training Command
```bash
mitrain <config_filepath> <train_filepath> <val_filepath> [-w <pth_filepath>] [-o <output_filepath>] [-d]
```
- config_filepath
  - Json-serialized training config.
- train_filepath / val_filepath
  - Filepath to the training / validation dataset
- weights_filepath
  - Optional base model weights
- output_filepath
  - Filepath to where the trained weight is saved
- fast_dev_run [-d]
  - If set, run 1 iteration of training and validation to test a pipeline.

The validatoin results will be printed on stdout.

## Training config
The definition of training config is in mitorch/common/training_config.py.

Here is an example configuration.
```
{
    "model": {
        "input_size": 224,
        "name": "MobileNetV3"
    },
    "optimizer": {
        "name": "sgd",
        "momentum": 0.9,
        "weight_decay": 0.0001
    },
    "lr_scheduler": {
        "name": "cosine_annealing",
        "base_lr": 0.01
    },
    "augmentation": {  # The names are defined in mitorch/datasets/factory.py.
        "train": "random_resize",
        "val": "center_crop"
    },
    "dataset": {  # This setting is used by mitorch-agent.
        "train": "mnist/train_images.txt",
        "val": "mnist/test_images.txt"
    },
    "batch_size": 2,
    "max_epochs": 5,
    "task_type": "multiclass_classification",
    "num_processes": 1  # For mitorch-agent. Specify the number of GPU/CPU for the training.
}
```

## Dataset format
See [simpledataset](https://github.com/shonohs/simpledataset).

# Advanced usage: experiment management
You can manage experiments on remote machines using this framework. 

## Setup
First, please set up an Azure Blob Storage and a mongo DB account.

- Blob Storage URL with SAS token
- MongoDB URL with the access token

Set those information to the following environment variables.
```bash
export MITORCH_STORAGE_URL=<storage sas url>
export MITORCH_DATABASE_URL=<MongoDB endpoint>
```

## Queue a job
```bash
misubmit <config_filepath>
```
This command will send a config file to the Mongo DB. 

## Run an agent
On a powerful machine you want to use, run the follwing command.
```bash
miagent --data <dataset_directory>
```
It will get a job from the Mongo DB, train it, and save the results to the MongoDB and the Blob storage.

## Commands
```bash
# Queue a new training
misubmit <config_filepath> [--priority <priority>]

# Get status of a training. If a job_id is not provided, it shows a list of jobs.
miquery [--job_id JOB_ID]
```
