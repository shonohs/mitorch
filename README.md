# MiTorch
A simple training platform for PyTorch.

The models are implemented in [mitorch-models](https://github.com/shonohs/mitorch_models).

There are two ways to use this platform. Use as a simple training command on local machine, or use as a service.

# Usage
To install,
```bash
pip install mitorch
```

## Training Command
```bash
mitrain <config_filepath> <train_filepath> <val_filepath> [-w <pth_filepath>] [-o <output_filepath>] [-d]
```
- config_filepath
  - Json-serialized training config. For the detail, please see the sample configs in samples/ directory.
- train_filepath / val_filepath
  - Filepath to the training / validation dataset
- weights_filepath
  - Optional base model weights
- output_filepath
  - Filepath to where the trained weight is saved
- fast_dev_run [-d]
  - If set, run 1 iteration of training and validation to test a pipeline.

The validatoin results will be printed on stdout.

## Validation Command
```bash
mitest <config_filepath> <train_filepath> <val_filepath> -w <weights_filepath> [-d]
```

# Usage as a service
This library can be used as a service using AzureML and MongoDB. Queue training jobs to MongoDB, and the trainings will be run on AzureML instances. The training results will be stored on MongoDB after the trainings.

## Setup
First, you need to createa AzureML and MongoDB resource on Azure portal. For the detail of this step, please read the Azure official documents. Once you set up the resources, collect the following informations.
- Subscription ID
- AzureML workspace name
- AzureML compute cluster name
- Username/password to access the AzureML resource. (Service Principal is recommended)
- MongoDB Endpoing with the access token

Set those information to the following environment variables.
```bash
export MITORCH_AZURE_SUBSCRIPTION_ID=<subscription id>
export MITORCH_AML_WORKSPACE=<workspace name>
export MITORCH_AML_COMPUTE=<compute cluster name>
export MITORCH_AML_AUTH=<username for AML>:<password for AML>
export MITORCH_DB_URI=<MongoDB endpoint>
```

Second, upload the datasets to Azure Blob Storage. Our dataset format is described in the later section. Register the dataset infomation to the MongoDB using midataset command.
```bash
midataset register <datasets_definition_json>
```
The format of the dataset definition is:
```javascript
[{"name": "dataset_name",
  "version": 0,
  "train": {"path": "path", "support_files": ["path"]}, // "path" is a Azure Blob storage URL with a sas token
  "val": {"path": "path", "support_files": ["path"]}}]
```

Third, run micontrol every 5 minutes. You can use any method to achive this step as long as the environment varialbes are correctly provided. You can manually execute them every 5 minutes, you can set up a cron job, or deploy to Azure Functions (recommended).

That's it. Now you are ready to use the service.

## Commands
```bash
# Queue a new training
misubmit <config_filepath> [--priority <priority>]

# Queue a hyper-parameter search job
misubmit <job_config_filepath> [--priority <priority>]

# Get status of a training
miquery <training_id or job_id>

# Launch a web UI for managing the service
miviewer
```

## Data structures
### Training job config
```javascript
{"_id": "<guid>",
 "status": "<status>", // "new", "running", "failed", or "completed".
 "prority": 100, // lower has more priority.
 "created_at": "<datetime in utc>",
 "dataset": "<dataset name>",
 "config": {} // Training configs.
 }
```
### Training config
```javascript
{"base": "<guid>", // Existing training id
 "augmentation": {},
 "lr_scheduler": {},
 "model": {},
 "optimizer": {}
}
```
### Job config structure
```javascript
{"job_type": "search", // Only "search" job is supported now.
}
```
### Dataset format
TBD

### MongoDB database structure
This library will create one database on the given MongoDB endpoint. The database name is "mitorch" by default.

The database has the following collections.
- trainings
  - Each record represents one training. New record will be added when a new training job is queued. The record will track the status of the job. Final evaluation results will be added to this record.
- training_results
  - Training loss/validation loss for each training epochs. Those records will be updated real-time during the trainings.
- jobs
  - Hyper-parameter search jobs will be stored in this collection. One job can create multiple trainings.
- datasets
  - Information of registered datasets. This collection needs to be created manually before the trainings.