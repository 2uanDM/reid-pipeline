import functools
import logging
import os
import sys
import traceback
from datetime import datetime
from typing import Any

from pythonjsonlogger import jsonlogger

log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)


class Logger:
    _instances = {}

    def __new__(cls, name: str = "default"):
        if name not in cls._instances:
            cls._instances[name] = super().__new__(cls)
            cls._instances[name]._setup_logger(name)
        return cls._instances[name]

    def _setup_logger(self, name: str):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)

        formatter = jsonlogger.JsonFormatter(
            "%(asctime)s %(levelname)s %(name)s %(message)s"
        )

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        # File handler
        file_handler = logging.FileHandler(
            f"{log_dir}/{datetime.now().strftime('%Y-%m-%d')}.log"
        )
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

    def info(self, message, extra=None):
        self.logger.info(message, extra=extra)

    def error(self, message, extra=None):
        self.logger.error(message, extra=extra)

    def warning(self, message, extra=None):
        self.logger.warning(message, extra=extra)

    def debug(self, message, extra=None):
        self.logger.debug(message, extra=extra)


def auto_log_error(logger_name: str = "default", response_if_error: Any = None):
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            logger = Logger(logger_name)
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                # Get stack trace info
                tb = traceback.extract_tb(e.__traceback__)
                frame = tb[-1]  # Last frame in traceback

                logger.error(
                    "Unexpected error",
                    extra={
                        "error": str(e),
                        "location": f"{func.__qualname__}",
                        "file": frame.filename,
                        "line": frame.lineno,
                        "function": frame.name,
                    },
                )

                if response_if_error:
                    return response_if_error
                else:
                    raise

        return wrapper

    return decorator
