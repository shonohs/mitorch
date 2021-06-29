import argparse
import logging
import pymongo
from mitorch.common import Environment


def init_database(db_url):
    client = pymongo.MongoClient(db_url)
    response = client.mitorch.jobs.create_index([('priority', pymongo.ASCENDING), ('created_at', pymongo.ASCENDING)])
    print(response)


def main():
    logging.basicConfig(level=logging.INFO)

    env = Environment()
    parser = argparse.ArgumentParser()
    parser.add_argument('--db_url', default=env.db_url)

    args = parser.parse_args()
    if not args.db_url:
        parser.error("A database url must be specified via commandline argument or environment variable.")

    init_database(args.db_url)


if __name__ == '__main__':
    main()
