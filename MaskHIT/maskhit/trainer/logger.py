import os
import logging


def setup_logger(name, log_file, file_mode, to_console=False, override=False):
    """
            https://stackoverflow.com/questions/11232230/logging-to-two-files-with-different-settings
        To setup as many loggers as you want
    """

    formatter = logging.Formatter('%(message)s')
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    if override:
        pass
    elif os.path.isfile(log_file) and file_mode == 'w':
        raise Exception(
            'log file already exists! Use --override-logs to ignore. %s' %
            (log_file, ))
    handler = logging.FileHandler(log_file, mode=file_mode)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)

    if to_console:
        logger.addHandler(logging.StreamHandler())

    return logger


def compose_logging(file_mode, model_name, to_console=False, override=False):
    writer = {}
    writer["meta"] = setup_logger("meta",
                                  os.path.join("logs",
                                               "%s_meta.log" % model_name),
                                  file_mode,
                                  to_console=to_console,
                                  override=override)
    writer["data"] = setup_logger("data",
                                  os.path.join("logs",
                                               "%s_data.csv" % model_name),
                                  file_mode,
                                  to_console=to_console,
                                  override=override)
    return writer