"""Shared logging, reproducibility, and output utilities."""

import json
import logging
import random
from pathlib import Path
from typing import Any

import numpy as np

from src.config import LOG_FORMAT, LOG_LEVEL, LOG_PATH, SEED


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger with console and file handlers."""
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))
    logger.propagate = False

    formatter = logging.Formatter(LOG_FORMAT)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(LOG_PATH, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def set_seed(seed: int = SEED) -> None:
    """Set random seeds for Python, NumPy, and TensorFlow."""
    random.seed(seed)
    np.random.seed(seed)

    try:
        import tensorflow as tf
    except ImportError:
        get_logger(__name__).warning(
            "TensorFlow is unavailable; only Python and NumPy seeds were set."
        )
        return

    tf.random.set_seed(seed)


def save_json(data: dict[str, Any], output_path: Path) -> None:
    """Save a dictionary as UTF-8 JSON, creating parent directories."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")

    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2, default=str)
        file.write("\n")

    temporary_path.replace(output_path)