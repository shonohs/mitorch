import argparse
import json
import pathlib
from mitorch.environment import Environment
from mitorch.service import DatabaseClient
from mitorch.settings import Settings


def update_settings(json_filepath):
    env = Environment()
    client = DatabaseClient(env.db_url)
    current_settings = client.get_settings()
    print(f"Current settings: {current_settings}")

    if json_filepath:
        settings_data = json.loads(json_filepath.read_text())
        settings = Settings(**settings_data)
        print(f"New settings: {settings}")
        response = input("Continue? [y/N]")
        if response == 'y':
            client.put_settings(settings)
            print("Updated")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('settings_filepath', nargs='?', type=pathlib.Path)

    args = parser.parse_args()

    update_settings(args.settings_filepath)


if __name__ == '__main__':
    main()
