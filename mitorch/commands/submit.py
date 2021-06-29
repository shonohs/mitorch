"""A command to submit a new job."""
import argparse
import pathlib
import uuid
import jsons
from mitorch.common import Environment, TrainingConfig, JobRepository
from mitorch.commands.common import init_logging


def submit_job(training_config, priority, base_job_id, job_repository):
    print(f"Adding {training_config}")
    job_id = job_repository.add_new_job(training_config, priority, base_job_id)
    print(f"Successfully added. Job id is {job_id}")


def main():
    init_logging()

    env = Environment()
    parser = argparse.ArgumentParser()
    parser.add_argument('config_filepath', type=pathlib.Path)
    parser.add_argument('--db_url', default=env.db_url)
    parser.add_argument('--priority', type=int, default=2, help="Lower value has higher priority.")
    parser.add_argument('--base_job_id', type=uuid.UUID)

    args = parser.parse_args()

    training_config = jsons.loads(args.config_filepath.read_text(), TrainingConfig)
    job_repository = JobRepository(args.db_url)

    submit_job(training_config, args.priority, args.base_job_id, job_repository)


if __name__ == '__main__':
    main()
