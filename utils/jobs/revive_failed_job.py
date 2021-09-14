"""A command to delete a job."""
import argparse
import uuid
from mitorch.common import Environment, JobRepository, MetricsRepository
from mitorch.commands.common import init_logging


def revive_failed_job(db_url, job_id):
    job_repository = JobRepository(db_url)
    metrics_repository = MetricsRepository(db_url)
    job = job_repository.get_job(job_id)
    if not job:
        print(f"Job {job_id} not found.")
        return

    if job.status != 'failed':
        print(f"Job {job_id} is not in failed state.")

    metrics = metrics_repository.get_metrics(job_id)
    print(job)
    for m in metrics:
        print(m)

    response = input("Revive? [y/n]")
    if response == 'y':
        if metrics:
            metrics_repository.delete_metrics(job_id)
            print("Deleted the metrics.")
        job_repository.update_job_status(job_id, 'queued')
        print("Updated the job.")


def main():
    init_logging()

    env = Environment()
    parser = argparse.ArgumentParser()
    parser.add_argument('--db_url', default=env.db_url)
    parser.add_argument('job_id', type=uuid.UUID)

    args = parser.parse_args()
    if not args.db_url:
        parser.error("A database url must be specified via commandline argument or environment variable.")

    revive_failed_job(args.db_url, args.job_id)


if __name__ == '__main__':
    main()
