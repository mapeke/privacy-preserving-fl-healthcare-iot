"""Structured logging configuration.

Every entry point should call :func:`setup_logging` once. The framework writes a single log file
per experiment to ``<output_dir>/run.log`` in addition to the console.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s :: %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(
    *,
    level: str | int = "INFO",
    log_file: str | Path | None = None,
) -> logging.Logger:
    """Configure the root logger with console (and optionally file) handlers.

    Args:
        level: Standard ``logging`` level name or numeric value.
        log_file: If provided, also write to this path. Parent directories are created.

    Returns:
        The configured root logger.
    """
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)

    if isinstance(level, str):
        level = level.upper()
    root.setLevel(level)

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    root.addHandler(console)

    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    # Quiet down noisy third-party loggers.
    for noisy in ("urllib3", "matplotlib", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    return root


def get_logger(name: str) -> logging.Logger:
    """Convenience wrapper for ``logging.getLogger`` with a consistent style."""
    return logging.getLogger(name)
