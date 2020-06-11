import argparse
import uuid
from mitorch.environment import Environment
from mitorch.service import DatabaseClient


def cancel_task(task_id):
    env = Environment()
    client = DatabaseClient(env.db_url)
    client.cancel_task(task_id)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('task_id', type=uuid.UUID)

    args = parser.parse_args()

    cancel_task(args.task_id)


if __name__ == '__main__':
    main()
