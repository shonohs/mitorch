import os


class Environment:
    def __init__(self):
        self._db_url = os.getenv('MITORCH_DATABASE_URL')

    @property
    def db_url(self):
        if not self._db_url:
            raise RuntimeError("MITORCH_DATABASE_URL is not set")
        return self._db_url
