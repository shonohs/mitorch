"""Runner script on AzureML instance

The job of this script is:
- Download a training config from MongoDB.
  - ID will be given as a commandline argument.
- Download training datasets from Azure Storage.
- Download a base weights from Azure Storage if it is specified.
- Run Train command
- Upload the trained model to Azure Storage
- Run Test command
- Upload the standard outputs to Azure Storage

"""
import argparse
import json
import logging
import os
import pathlib
import shutil
import subprocess
import tempfile
import time
import urllib
import uuid
import requests
import tenacity
import torch
from mitorch.service import DatabaseClient

_logger = logging.getLogger(__name__)


class AzureMLRunner:
    def __init__(self, db_url, job_id):
        assert isinstance(job_id, uuid.UUID)
        self.db_url = db_url
        self.job_id = job_id
        self.client = DatabaseClient(self.db_url)

    def run(self):
        """First method to be run on AzureML instance"""
        # Get the job description from the database.
        job = self.client.find_training_by_id(self.job_id)
        if not job:
            raise RuntimeError(f"Unknown job id {self.job_id}.")

        config = job['config']
        dataset_name = config['dataset']
        region = job['region']
        _logger.info(f"Training config: {config}")

        settings = self.client.get_settings()
        self.dataset_base_url = settings.dataset_url[region]
        self.blob_storage_url = settings.storage_url

        # Record machine setup.
        num_gpus = torch.cuda.device_count()
        self.client.start_training(self.job_id, num_gpus)

        with tempfile.TemporaryDirectory() as work_dir:
            work_dir = pathlib.Path(work_dir)
            os.mkdir(work_dir / 'outputs')
            output_filepath = work_dir / 'outputs' / 'model.pth'
            config_filepath = work_dir / 'config.json'
            config_filepath.write_text(json.dumps(config))
            train_filepath, val_filepath = self.download_dataset(dataset_name, work_dir)
            weights_filepath = self.download_weights(uuid.UUID(config['base']), work_dir) if 'base' in config else None

            command = ['mitrain', str(config_filepath), str(train_filepath), str(val_filepath), '--output_filepath', str(output_filepath), '--job_id', str(self.job_id), '--db_url', self.db_url]
            if weights_filepath:
                command.extend(['--weights_filepath', str(weights_filepath)])

            _logger.info(f"Starting the training. command: {command}")
            proc = subprocess.run(command)
            _logger.info(f"Training completed. returncode: {proc.returncode}")

            if proc.returncode != 0 or not output_filepath.exists():
                self.client.fail_training(self.job_id)
                return

            self.upload_files([output_filepath])

            command = ['mitest', str(config_filepath), str(train_filepath), str(val_filepath), '--weights_filepath', str(output_filepath),
                       '--job_id', str(self.job_id), '--db_url', self.db_url]
            _logger.info(f"Starting test. command: {command}")
            proc = subprocess.run(command)
            _logger.info(f"Test completed. returncode: {proc.returncode}")

            if proc.returncode == 0:
                self.client.complete_training(self.job_id)
            else:
                self.client.fail_training(self.job_id)

        _logger.info("All completed.")

    def download_dataset(self, dataset_name, directory):
        dataset = self.client.find_dataset_by_name(dataset_name)
        train_filepath = self._download_blob_file(self.dataset_base_url, dataset['train']['path'], directory)
        val_filepath = self._download_blob_file(self.dataset_base_url, dataset['val']['path'], directory)

        files = set(dataset['train']['support_files'] + dataset['val']['support_files'])
        for uri in files:
            self._download_blob_file(self.dataset_base_url, uri, directory)

        return train_filepath, val_filepath

    def download_weights(self, base_job_id, directory):
        blob_path = os.path.join(base_job_id.hex, 'model.pth')
        return self._download_blob_file(self.blob_storage_url, blob_path, directory)

    def upload_files(self, files):
        for filepath in files:
            blob_path = os.path.join(self.job_id.hex, filepath.name)
            self._upload_blob_file(self.blob_storage_url, filepath, blob_path)

    @staticmethod
    def _upload_blob_file(base_blob_uri, local_filepath, blob_path):
        parts = urllib.parse.urlparse(base_blob_uri)
        path = os.path.join(parts[2], blob_path)
        url = urllib.parse.urlunparse((parts[0], parts[1], path, parts[3], parts[4], parts[5]))
        _logger.info(f"Uploading {local_filepath} to {url}")
        requests.put(url=url, data=local_filepath.read_bytes(), headers={'Content-Type': 'application/octet-stream', 'x-ms-blob-type': 'BlockBlob'})

    @staticmethod
    def _download_blob_file(base_blob_uri, blob_path, directory):
        parts = urllib.parse.urlparse(base_blob_uri)
        path = os.path.join(parts[2], blob_path)
        url = urllib.parse.urlunparse((parts[0], parts[1], path, parts[3], parts[4], parts[5]))
        return AzureMLRunner._download_file(url, directory)

    @staticmethod
    @tenacity.retry(tenacity.stop_after_attempt(3))
    def _download_file(url, directory):
        filename = os.path.basename(urllib.parse.urlparse(url).path)
        filepath = directory / filename
        _logger.info(f"Downloading {url} to {filepath}")
        start = time.time()
        with requests.get(url, stream=True, allow_redirects=True) as r:
            with open(filepath, 'wb') as f:
                shutil.copyfileobj(r.raw, f, length=4194304)  # 4MB
        _logger.debug(f"Downloaded. {time.time() - start}s.")
        return filepath


def main():
    logging.getLogger().setLevel(logging.INFO)
    logging.getLogger('mitorch').setLevel(logging.DEBUG)

    parser = argparse.ArgumentParser("Run training on AzureML")
    parser.add_argument('job_id', type=uuid.UUID, help="Guid for the target run")
    parser.add_argument('db_url', help="MongoDB URI for training management")

    args = parser.parse_args()

    runner = AzureMLRunner(args.db_url, args.job_id)
    runner.run()


if __name__ == '__main__':
    main()
