"""Runner script on AzureML instance

The job of this script is:
- Install required packages if AML hasn't
- Download a target config from MongoDB.
  - ID will be given as a commandline argument.
- Download target datasets from Azure Storage.
  - There is a json file to describe existing datasets.
- Download a base weights from Azure Storage if it is specified.
- Run Train command
- Upload the standard outputs to Azure Storage
- Upload the trained model to Azure Storage
"""
import argparse
import os
import shutil
import tempfile
import time
import urllib
import uuid
import requests
import torch
from mitorch.logger import MongoDBLogger, StdoutLogger
from mitorch.service import DatabaseClient
from mitorch.train import train


class AzureMLRunner:
    def __init__(self, db_url, job_id):
        assert isinstance(job_id, uuid.UUID)
        self.db_url = db_url
        self.job_id = job_id
        self.client = DatabaseClient(self.db_url)
        self.dataset_base_uri = self.client.get_dataset_uri()
        self.blob_storage_url = self.client.get_storage_uri()

    def run(self):
        """First method to be run on AzureML instance"""

        # Get the job description from the database.
        job = self.client.find_training_by_id(self.job_id)
        if not job:
            raise RuntimeError(f"Unknown job id {self.job_id}.")

        config = job['config']
        dataset_name = config['dataset']

        print(config)

        # Record machine setup.
        num_gpus = torch.cuda.device_count()
        self.client.start_training(self.job_id, num_gpus)

        with tempfile.TemporaryDirectory() as work_dir:
            os.mkdir(os.path.join(work_dir, 'outputs'))
            output_filepath = os.path.join(work_dir, 'outputs', 'model.pth')
            train_filepath, val_filepath = self.download_dataset(dataset_name, work_dir)
            weights_filepath = self.download_weights(config['base'], work_dir) if 'base' in config else None
            logger = [MongoDBLogger(self.db_url, self.job_id), StdoutLogger()]
            print("Starting the training.")
            train(config, train_filepath, val_filepath, weights_filepath, output_filepath, False, logger)
            print("Training completed.")

            self.upload_files([output_filepath])
            self.client.complete_training(self.job_id)

    def download_dataset(self, dataset_name, directory):
        dataset = self.client.find_dataset_by_name(dataset_name)
        train_filepath = self._download_blob_file(self.dataset_base_uri, dataset['train']['path'], directory)
        val_filepath = self._download_blob_file(self.dataset_base_uri, dataset['val']['path'], directory)

        files = set(dataset['train']['support_files'] + dataset['val']['support_files'])
        for uri in files:
            self._download_blob_file(self.dataset_base_uri, uri, directory)

        return train_filepath, val_filepath

    def download_weights(self, base_job_id, directory):
        blob_path = os.path.join(base_job_id.hex, 'model.pth')
        return self._download_blob_file(self.blob_storage_url, blob_path, directory)

    def upload_files(self, files):
        for filepath in files:
            blob_path = os.path.join(self.job_id.hex, os.path.basename(filepath))
            self._upload_blob_file(self.blob_storage_url, filepath, blob_path)

    @staticmethod
    def _upload_blob_file(base_blob_uri, local_filepath, blob_path):
        parts = urllib.parse.urlparse(base_blob_uri)
        path = os.path.join(parts[2], blob_path)
        url = urllib.parse.urlunparse((parts[0], parts[1], path, parts[3], parts[4], parts[5]))
        print(f"Uploading {local_filepath} to {url}")
        with open(local_filepath, 'rb') as f:
            data = f.read()
        requests.put(url=url, data=data, headers={'Content-Type': 'application/octet-stream', 'x-ms-blob-type': 'BlockBlob'})

    @staticmethod
    def _download_blob_file(base_blob_uri, blob_path, directory):
        parts = urllib.parse.urlparse(base_blob_uri)
        path = os.path.join(parts[2], blob_path)
        url = urllib.parse.urlunparse((parts[0], parts[1], path, parts[3], parts[4], parts[5]))
        return AzureMLRunner._download_file(url, directory)

    @staticmethod
    def _download_file(url, directory):
        filename = os.path.basename(urllib.parse.urlparse(url).path)
        filepath = os.path.join(directory, filename)
        print(f"Downloading {url} to {filepath}")
        start = time.time()
        with requests.get(url, stream=True, allow_redirects=True) as r:
            with open(filepath, 'wb') as f:
                shutil.copyfileobj(r.raw, f, length=4194304)  # 4MB
        print(f"Downloaded. {time.time() - start}s.")
        return filepath


def main():
    parser = argparse.ArgumentParser("Run training on AzureML")
    parser.add_argument('job_id', help="Guid for the target run")
    parser.add_argument('db_url', help="MongoDB URI for training management")

    args = parser.parse_args()

    runner = AzureMLRunner(args.db_url, uuid.UUID(args.job_id))
    runner.run()


if __name__ == '__main__':
    main()
