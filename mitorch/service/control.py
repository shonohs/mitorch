import argparse
import datetime
import time

from ..azureml import AzureMLManager
from ..environment import Environment
from .database_client import DatabaseClient


def control_loop(env):
    while True:
        try:
            control(env)
        except KeyboardInterrupt:
            return
        except Exception as e:
            print(f"Exception happened: {e}")
            # Ignore the exception

        time.sleep(300)  # Sleep for 5 minutes


def control(env):
    client = DatabaseClient(env.db_url)
    aml_manager = AzureMLManager(client.get_settings().azureml_settings)

    # If there is available resource, pick a training job and submit to AzureML.
    process_trainings(client, aml_manager, env.db_url)

    # Check the status of jobs. Queue new trainings if needed.
    # TODO: Implement process_jobs
    # process_jobs(client)

    print(f"{datetime.datetime.now()}: Completed")


def process_trainings(client, aml_manager, db_url):
    # If there are available resources, submit new AML jobs
    num_available_nodes = aml_manager.get_num_available_nodes()
    if num_available_nodes > 0:
        pending_jobs = client.get_new_trainings(num_available_nodes)
        for job in pending_jobs:
            submit_result = aml_manager.submit(db_url, job['_id'])
            if submit_result:
                aml_run_id, region = submit_result
                updated = client.update_training(job['_id'], {'status': 'queued', 'run_id': aml_run_id, 'region': aml_manager.region})
                if not updated:
                    raise RuntimeError(f"Failed to update {job['_id']}")
                print(f"Queued a new AML run: id: {job['_id']}, run_id: {aml_run_id}, region: {region}")

    # Check the status of ongoing tasks. If there is a task which is dead silently, update its record.
    jobs = client.get_running_trainings() + client.get_queued_trainings()
    for job in jobs:
        status = aml_manager.query(job['run_id'], job['region'])
        if status in ['completed', 'failed']:
            print(f"Unexpected run status: id: {job['_id']}, run_id: {job['run_id']}, status: {status}")
            updated = client.update_training(job['_id'], {'status': 'failed'})
            if not updated:
                raise RuntimeError(f"Failed to update {job['_id']}")


def process_jobs(client):
    # Get all active jobs

    # For each job
    # Check the last experiment status
    # If it's done, check the job status. If still active, queue a next training.
    # Update the record
    raise NotImplementedError


def main():
    parser = argparse.ArgumentParser("Control")
    parser.add_argument('--loop', '-l', action='store_true', help="Keep running until interuppted. The jobs will be processed every 5 minutes.")
    args = parser.parse_args()

    env = Environment()
    if args.loop:
        control_loop(env)
    else:
        control(env)


if __name__ == '__main__':
    main()
