"""A command to delete a job."""
import argparse
import uuid
from mitorch.common import Environment, JobRepository
from mitorch.commands.common import init_logging


def change_priority(db_url, job_id, new_priority):
    job_repository = JobRepository(db_url)
    job = job_repository.get_job(job_id)

    if not job:
        print(f"Job {job_id} not found.")
        return

    if job.status != 'queued':
        print(f"WARNING: the job is not in queued state.")

    print(job)

    response = input(f"Update the priority to {new_priority}? [y/n]")

    if response == 'y':
        job_repository.update_job_priority(job_id, new_priority)
        print("Updated the job.")


def main():
    init_logging()

    env = Environment()
    parser = argparse.ArgumentParser()
    parser.add_argument('--db_url', default=env.db_url)
    parser.add_argument('job_id', type=uuid.UUID)
    parser.add_argument('new_priority', type=int)

    args = parser.parse_args()
    if not args.db_url:
        parser.error("A database url must be specified via commandline argument or environment variable.")

    change_priority(args.db_url, args.job_id, args.new_priority)


if __name__ == '__main__':
    main()
