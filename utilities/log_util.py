import logging
import os
import sys


LOG_LEVEL_MAPPING = {
    "LOG_LEVEL_INFO": logging.INFO,
    "LOG_LEVEL_ERROR": logging.ERROR,
    "LOG_LEVEL_DEBUG": logging.DEBUG,
    "LOG_LEVEL_WARN": logging.WARNING,
    "INFO": logging.INFO,
    "ERROR": logging.ERROR,
    "DEBUG": logging.DEBUG,
    "WARN": logging.WARNING,
}

DEFAULT_LOG_LEVEL = logging.INFO


_logger = logging.getLogger(__name__)


def initialize_logging(**kwargs):
    log_level: str = kwargs.get("log_level")
    if not log_level:
        log_level_env = os.environ.get("LOG_LEVEL")
        log_level = LOG_LEVEL_MAPPING.get(log_level_env, DEFAULT_LOG_LEVEL)
    else:
        log_level = LOG_LEVEL_MAPPING.get(log_level.upper(), DEFAULT_LOG_LEVEL)

    logging.basicConfig(format='[%(asctime)s.%(msecs)03d][%(levelname)s] %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=log_level)


def log_error(exc_info, stack_trace, **kwargs):
    logger = kwargs.get("logger", _logger)
    try:
        error_type = exc_info[0]
        error = exc_info[1]
        message = f"ErrorType: {error_type}\n Error: {error}\n Trace: {stack_trace}"
        logger.error(message)
        return message
    except:
        logger.error(exc_info)
        logger.error(sys.exc_info()[1])
    return None
