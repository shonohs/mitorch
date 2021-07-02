"""A command to query jobs."""
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


def query_job_list(db_url, show_short_description):
    job_repository = JobRepository(db_url)
    metrics_repository = MetricsRepository(db_url)
    jobs = job_repository.query_jobs()
    if not jobs:
        print("No job found.")

    for job_record in jobs:
        if show_short_description:
            print(f"[{job_record.status}] {job_record.job_id} ({job_record.config.model.name}) {job_record.config.dataset.train}")
        else:
            print(job_record)

        if job_record.status == 'completed':
            final_metrics = metrics_repository.get_final_metrics(job_record.job_id)
            if final_metrics:
                print(f"        Metrics: {final_metrics.metrics}")


def main():
    init_logging()

    env = Environment()
    parser = argparse.ArgumentParser()
    parser.add_argument('--db_url', default=env.db_url)
    parser.add_argument('--job_id', type=uuid.UUID)
    parser.add_argument('--short', action='store_true')

    args = parser.parse_args()
    if not args.db_url:
        parser.error("A database url must be specified via commandline argument or environment variable.")

    if args.job_id:
        query_job(args.db_url, args.job_id)
    else:
        query_job_list(args.db_url, args.short)


if __name__ == '__main__':
    main()
