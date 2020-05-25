import argparse
import uuid

from ..environment import Environment
from .database_client import DatabaseClient


def query(db_url, job_id):
    client = DatabaseClient(db_url)
    record = client.find_job_by_id(job_id)
    if record:
        print(record)
    else:
        print(f"Job {job_id} is not found")


def query_all(db_url):
    client = DatabaseClient(db_url)
    print("===== Active jobs =====")
    _print_jobs(client.get_running_jobs())

    print("===== Running trainings =====")
    _print_jobs(client.get_running_trainings())

    print("===== Trainings in queue =====")
    _print_jobs(client.get_new_trainings())


def _print_jobs(jobs):
    for job in jobs:
        print(job)


def main():
    parser = argparse.ArgumentParser("Query a mitorch job")
    parser.add_argument('job_id', nargs='?', help="Job ID to query")

    args = parser.parse_args()
    env = Environment()
    if args.job_id:
        query(env.db_url, uuid.UUID(args.job_id))
    else:
        query_all(env.db_url)


if __name__ == '__main__':
    main()
