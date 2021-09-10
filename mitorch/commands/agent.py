"""Get training configs from database and run mitrain with it."""
import argparse
import dataclasses
import json
import logging
import pathlib
import subprocess
import tempfile
import time
import torch
from mitorch.common import Environment, JobRepository, ModelRepository
from mitorch.commands.common import init_logging

logger = logging.getLogger(__name__)


def process_one_job(job, db_url, model_repository, data_dir):
    # Get the next training config.
    train_dataset_filepath = data_dir / job.config.dataset.train
    val_dataset_filepath = data_dir / job.config.dataset.val

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = pathlib.Path(temp_dir)
        config_filepath = temp_dir / 'config.json'
        output_filepath = temp_dir / 'trained_weights.pth'

        config = dataclasses.asdict(job.config)
        config_filepath.write_text(json.dumps(config, indent=4))

        if job.base_job_id:
            pretrained_weights_filepath = temp_dir / 'pretrained_weights.pth'
            model_repository.download_weights(job.base_job_id, pretrained_weights_filepath)
        else:
            pretrained_weights_filepath = None

        log_filepath = temp_dir / 'training.log'
        tb_log_dir = temp_dir / 'tensorboard/'

        command = ['mitrain', str(config_filepath), str(train_dataset_filepath), str(val_dataset_filepath),
                   '--output_filepath', str(output_filepath),
                   '--job_id', str(job.job_id),
                   '--db_url', db_url,
                   '--tensorboard_log', str(tb_log_dir),
                   '--log_file', str(log_filepath)]

        if pretrained_weights_filepath:
            command.extend(['--weights_filepath', str(pretrained_weights_filepath)])

        logger.info(f"Starting the training. command: {command}")
        proc = subprocess.run(command)
        if proc.returncode != 0:
            logger.warning(f"Training return code is {proc.returncode}")

        # Optional file uploads.
        try:
            model_repository.upload_file(job.job_id, config_filepath)
        except Exception:
            logger.exception("Failed to upload a job config.")

        try:
            model_repository.upload_file(job.job_id, log_filepath)
        except Exception:
            logger.exception("Failed to upload a log file.")

        try:
            model_repository.upload_dir(job.job_id, tb_log_dir)
        except Exception:
            logger.exception("Failed to upload a tensorboard log.")

        model_repository.upload_weights(job.job_id, output_filepath)


def run_agent(db_url, storage_url, data_dir, num_runs):
    logger.info("Starting an agent.")

    job_repository = JobRepository(db_url)
    model_repository = ModelRepository(storage_url)

    num_processes = torch.cuda.device_count() if torch.cuda.is_available() else 1

    for _ in range(num_runs):
        job = job_repository.get_next_job(num_processes=num_processes)
        if not job:
            logger.info("The job queue is empty. Will retry in 10 minutes.")
            time.sleep(600)
            continue

        logger.info(f"Got a new job! {job.job_id}")

        try:
            job_repository.update_job_status(job.job_id, 'running')
            process_one_job(job, db_url, model_repository, data_dir)
            job_repository.update_job_status(job.job_id, 'completed')
        except Exception:
            logger.exception(f"Failed to process a job {job.job_id}.")
            job_repository.update_job_status(job.job_id, 'failed')

        logger.info(f"Job {job.job_id} completed!")

    logger.info("All done!")


def main():
    init_logging()
    env = Environment()
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, type=pathlib.Path)
    parser.add_argument('--num_runs', '-n', type=int, default=10000)
    parser.add_argument('--db_url', default=env.db_url, help="URL for a mongo db that stores training configs.")
    parser.add_argument('--storage_url', default=env.storage_url, help="Blob container URL with SAS token. Trained weights will be stored here.")

    args = parser.parse_args()

    if not args.db_url:
        parser.error("A database url must be specified.")
    if not args.storage_url:
        parser.error("A storage url must be specified.")

    run_agent(args.db_url, args.storage_url, args.data, args.num_runs)


if __name__ == '__main__':
    main()
