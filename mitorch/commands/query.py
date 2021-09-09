"""A command to query jobs."""
import argparse
import collections
import pathlib
import uuid
from mitorch.common import Environment, JobRepository, MetricsRepository, ModelRepository
from mitorch.commands.common import init_logging


def query_job(job_repository, metrics_repository, job_id):
    job = job_repository.get_job(job_id)
    if not job:
        print("Not found.")
        return

    metrics = metrics_repository.get_metrics(job_id)

    print(job)
    for m in metrics:
        print(m)


def query_job_list(job_repository, metrics_repository, show_short_description):
    jobs = job_repository.query_jobs()
    if not jobs:
        print("No job found.")

    status_counter = collections.Counter()
    for job_record in jobs:
        if show_short_description:
            print(f"[{job_record.status}] {job_record.job_id} ({job_record.config.model.name}) {job_record.config.dataset.train}")
        else:
            print(job_record)
        status_counter[job_record.status] += 1

        if job_record.status == 'completed':
            final_metrics = metrics_repository.get_final_metrics(job_record.job_id)
            if final_metrics:
                print(f"        Metrics: {final_metrics.metrics}")
    print(status_counter)


def download_job_files(job_repository, model_repository, job_id, output_dir):
    job = job_repository.get_job(job_id)
    if not job:
        print("Not found.")
        return

    num_files = model_repository.download_all_files(job_id, output_dir)
    if num_files > 0:
        print(f"Downloaded {num_files} files to {output_dir}")
    else:
        print("No file found.")


def main():
    init_logging()

    env = Environment()
    parser = argparse.ArgumentParser()
    parser.add_argument('job_id', nargs='?', type=uuid.UUID)
    parser.add_argument('--db_url', default=env.db_url)
    parser.add_argument('--storage_url', default=env.storage_url)
    parser.add_argument('--short', action='store_true')
    parser.add_argument('--download', '-d', action='store_true')

    args = parser.parse_args()
    if not args.db_url:
        parser.error("A database url must be specified via commandline argument or environment variable.")

    job_repository = JobRepository(args.db_url)
    metrics_repository = MetricsRepository(args.db_url)

    if args.download:
        if not args.job_id:
            parser.error("You must specify job_id to download.")
        if not args.storage_url:
            parser.error("You must specify storage_url to download.")

    if args.job_id:
        if args.download:
            model_repository = ModelRepository(args.storage_url)
            output_dir = pathlib.Path(str(args.job_id))
            download_job_files(job_repository, model_repository, args.job_id, output_dir)
        else:
            query_job(job_repository, metrics_repository, args.job_id)
    else:
        query_job_list(job_repository, metrics_repository, args.short)


if __name__ == '__main__':
    main()
