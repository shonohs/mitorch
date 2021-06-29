import os


class Environment:
    def __init__(self):
        self._db_url = os.getenv('MITORCH_DATABASE_URL')
        self._storage_url = os.getenv('MITORCH_STORAGE_URL')

    @property
    def db_url(self):
        return self._db_url

    @property
    def storage_url(self):
        return self._storage_url
