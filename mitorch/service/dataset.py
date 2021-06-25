import argparse
import json
from ..environment import Environment
from .database_client import DatabaseClient


def add_dataset(db_url, json_filepath):
    with open(json_filepath) as f:
        datasets = json.load(f)

    client = DatabaseClient(db_url)
    for dataset in datasets:
        if not client.find_dataset_by_name(dataset['name'], dataset['version']):
            print(f"Adding dataset: {dataset}")
            client.add_dataset(dataset)


def main():
    parser = argparse.ArgumentParser("Manage the regisgered datasets")
    subparsers = parser.add_subparsers(dest='command')
    parser_add = subparsers.add_parser('add')
    parser_add.add_argument('json_filepath', help="JSON file which has the datasets")

    args = parser.parse_args()
    env = Environment()
    if args.command == 'add':
        add_dataset(env.db_url, args.json_filepath)


if __name__ == '__main__':
    main()
