import argparse
import json

from .database_client import DatabaseClient
from ..environment import Environment


def submit(config_filepath, priority, db_url):
    with open(config_filepath) as f:
        config = json.load(f)

    client = DatabaseClient(db_url)
    if 'job_type' in config:
        job_id = client.add_job(config, priority)
    else:
        job_id = client.add_training(config, priority)

    print(f"Queued successfully. id={job_id}, priority={priority}")


def main():
    parser = argparse.ArgumentParser("Submit a job to mitorch service")
    parser.add_argument('config_filepath', help="Filepath to the config file to be submitted")
    parser.add_argument('--priority', type=int, default=100, help="Priority of the job. Lower value means more priority.")

    args = parser.parse_args()
    env = Environment()

    submit(args.config_filepath, args.priority, env.db_url)


if __name__ == '__main__':
    main()
