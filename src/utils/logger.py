"""Centralized logging configuration for FPAI."""

from __future__ import annotations

import logging
from pathlib import Path


def configure_logger(level: int = logging.INFO) -> logging.Logger:
    """Configure root logger with console and file handlers."""
    logs_dir = Path("data/logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = logs_dir / "fpai.log"

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    if root_logger.handlers:
        return root_logger

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    return root_logger


def get_logger(name: str) -> logging.Logger:
    """Return a module-scoped logger."""
    return logging.getLogger(name)
