"""A command to submit a new job."""
import argparse
import uuid
from mitorch.common import Environment, JobRepository, MetricsRepository
from mitorch.commands.common import init_logging


def query_job(db_url, job_id):
    job_repository = JobRepository(db_url)
    metrics_repository = MetricsRepository(db_url)
    job = job_repository.get_job(job_id)
    if not job:
        print("Not found.")
        return

    metrics = metrics_repository.get_metrics(job_id)

    print(job)
    for m in metrics:
        print(m)


def query_job_list(db_url):
    job_repository = JobRepository(db_url)
    jobs = job_repository.query_jobs()
    if not jobs:
        print("No job found.")

    for job_record in jobs:
        print(job_record)


def main():
    init_logging()

    env = Environment()
    parser = argparse.ArgumentParser()
    parser.add_argument('--db_url', default=env.db_url)
    parser.add_argument('--job_id', type=uuid.UUID)

    args = parser.parse_args()
    if not args.db_url:
        parser.error("A database url must be specified via commandline argument or environment variable.")

    if args.job_id:
        query_job(args.db_url, args.job_id)
    else:
        query_job_list(args.db_url)


if __name__ == '__main__':
    main()
