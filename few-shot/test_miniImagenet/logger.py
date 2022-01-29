import logging
import sys

from os import path, mkdir, remove


console_logging_format = "%(message)s"

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=console_logging_format)

print(logging.getLogger().handlers)


def create_logger(log_dir, file_name):
    # get logger
    logger = logging.getLogger()

    # remove previous handlers, if they exist
    if bool(logger.handlers):
        logger.handlers.clear()

    # create a log directory, if not exists
    if not path.exists(log_dir):
        mkdir(log_dir)

    log_file_path = path.join(log_dir, file_name)

    # remove old log file (w/ same name)
    if path.exists(log_file_path):
        remove(log_file_path)

    # create a new log file
    f = open(log_file_path, 'w+')
    f.close()

    # create a file handler for output file
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.INFO)

    # configure message to log in file
    file_logging_format = "[%(levelname)s] %(asctime)s: %(message)s"
    formatter = logging.Formatter(file_logging_format)
    file_handler.setFormatter(formatter)

    # create a console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    # configure message to log in console
    console_logging_format = "%(message)s"
    formatter = logging.Formatter(console_logging_format)
    console_handler.setFormatter(formatter)

    # add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
