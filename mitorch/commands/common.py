import logging


def init_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(module)s %(process)d: %(message)s')
