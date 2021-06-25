import argparse
import uuid
from mitorch.environment import Environment
from mitorch.service import DatabaseClient


def delete_task(task_id):
    assert isinstance(task_id, uuid.UUID)
    env = Environment()
    client = DatabaseClient(env.db_url)
    task = client.get_task_by_id(task_id)
    print(task)
    response = input("Delete this task? [y/N]")
    if response == 'y':
        result = client.delete_task(task_id)
        assert result
        print("Deleted")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('task_id', type=uuid.UUID)

    args = parser.parse_args()

    delete_task(args.task_id)


if __name__ == '__main__':
    main()
