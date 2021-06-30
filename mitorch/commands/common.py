import logging


def init_logging(log_filepath=None):
    LOG_FORMAT = '%(asctime)s %(levelname)s %(module)s %(process)d: %(message)s'
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    if log_filepath:
        log_handler = logging.FileHandler(log_filepath)
        formatter = logging.Formatter(LOG_FORMAT)
        log_handler.setFormatter(formatter)
        logging.getLogger().addHandler(log_handler)
