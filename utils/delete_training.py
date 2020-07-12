import argparse
import uuid
from mitorch.environment import Environment
from mitorch.service import DatabaseClient


def remove_training(training_id):
    assert isinstance(training_id, uuid.UUID)
    env = Environment()
    client = DatabaseClient(env.db_url)

    training = client.find_training_by_id(training_id)
    if not training:
        print(f"Training {training_id} not found")
        return

    print(training)

    response = input("Remove? [y/N]: ")
    if response.lower() == 'y':
        result = client.delete_training(training_id)
        assert result
        print("Removed successfully")


def remove_failed_trainings():
    env = Environment()
    client = DatabaseClient(env.db_url)
    trainings = client.get_failed_trainings()
    for t in trainings:
        remove_training(t['_id'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('training_id', nargs='?', type=uuid.UUID)
    parser.add_argument('--delete_failed_trainings', action='store_true')

    args = parser.parse_args()
    if args.delete_failed_trainings and args.training_id:
        parser.error("training_id and --delete_failed_trainings cannot be specified at the same time.")

    if not (args.delete_failed_trainings or args.training_id):
        parser.error("Please specify training_id or --delete_failed_trainings.")

    if args.delete_failed_trainings:
        remove_failed_trainings()
    else:
        remove_training(args.training_id)


if __name__ == '__main__':
    main()
