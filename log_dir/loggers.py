import logging
import os.path

from help_funcs import date_MSC_str

formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(funcName)s: %(lineno)d - %(message)s")


def setup_logger(name, log_file, level=logging.INFO):
    """To setup as many loggers as you want"""

    log_file = 'log_dir/' + log_file + '_' + date_MSC_str() + '.log'
    # if name == 'ya':
    #     level=logging.DEBUG
    name = name + '_logger_' + date_MSC_str()
    if os.path.isfile(log_file):
        old_logger = logging.getLogger(name)
        if old_logger.hasHandlers():
            return old_logger
        old_handler = logging.FileHandler(log_file)
        old_handler.setFormatter(formatter)

        old_logger.setLevel(level)
        old_logger.addHandler(old_handler)
        return old_logger
    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger


db_logger = setup_logger('db', 'db')
