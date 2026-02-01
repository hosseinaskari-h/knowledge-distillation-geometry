"""
Logging setup
"""
import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(log_file: Optional[Path] = None,
                 level: int = logging.INFO,
                 format_string: Optional[str] = None):
    """
    Setup logging configuration

    Args:
        log_file: Optional file path for logging
        level: Logging level
        format_string: Custom format string
    """
    if format_string is None:
        format_string = '%(asctime)s | %(name)s | %(levelname)s | %(message)s'

    handlers = [logging.StreamHandler(sys.stdout)]

    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding='utf-8'))

    # Reset root logger
    root = logging.getLogger()
    root.setLevel(level)

    # Remove existing handlers
    for handler in root.handlers[:]:
        root.removeHandler(handler)

    # Add new handlers
    formatter = logging.Formatter(format_string)
    for handler in handlers:
        handler.setFormatter(formatter)
        root.addHandler(handler)

    # Reduce noise from external libraries
    logging.getLogger('transformers').setLevel(logging.WARNING)
    logging.getLogger('torch').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)


class LoggerMixin:
    """Mixin class to add logging capabilities"""

    @property
    def logger(self) -> logging.Logger:
        if not hasattr(self, '_logger'):
            self._logger = logging.getLogger(self.__class__.__name__)
        return self._logger
