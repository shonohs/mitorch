import argparse
import json
import pathlib
from mitorch.environment import Environment
from mitorch.service import DatabaseClient, Task


def submit_task(json_filepath):
    task_dict = json.loads(json_filepath.read_text())
    task = Task.from_dict(task_dict)

    print(task)
    response = input("Add the task? [y/N]")
    if response == 'y':
        env = Environment()
        client = DatabaseClient(env.db_url)
        task_id = client.add_task(task.to_dict())
        print(f"Added task: {task_id}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('json_filepath', type=pathlib.Path)

    args = parser.parse_args()

    submit_task(args.json_filepath)


if __name__ == '__main__':
    main()
