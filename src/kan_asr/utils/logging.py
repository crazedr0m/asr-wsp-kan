"""
Единая настройка логирования для KAN-ASR.
"""

import logging
import os
import sys

_DEFAULT_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DEFAULT_DATEFMT = "%Y-%m-%d %H:%M:%S"


def setup_logging(level: str = None, log_file: str = None) -> logging.Logger:
    """
    Настраивает корневой логгер проекта.

    level: уровень логирования (по умолчанию из переменной окружения KAN_ASR_LOG_LEVEL или INFO).
    log_file: опциональный путь к файлу лога (в дополнение к stderr).
    """
    if level is None:
        level = os.environ.get("KAN_ASR_LOG_LEVEL", "INFO")

    root = logging.getLogger("kan_asr")
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    root.handlers.clear()

    formatter = logging.Formatter(_DEFAULT_FORMAT, datefmt=_DEFAULT_DATEFMT)

    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)

    if log_file:
        os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    return root


def get_logger(name: str) -> logging.Logger:
    """Возвращает дочерний логгер ``kan_asr.<name>``."""
    return logging.getLogger(f"kan_asr.{name}")
