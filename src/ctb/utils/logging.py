"""Structured logging configuration.

Provides a single :func:`configure_logging` entry point that writes human-readable logs to both
stderr and a timestamped file under ``logs/``. Modules obtain loggers via
``logging.getLogger(__name__)`` and should never use :func:`print`.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

from ctb.config import LOGS_DIR

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%dT%H:%M:%S%z"
_configured = False


def configure_logging(level: int = logging.INFO, log_file: Path | None = None) -> Path:
    """Configure root logging to stderr and a file. Idempotent within a process.

    Parameters
    ----------
    level : int, default ``logging.INFO``
        Root logging level.
    log_file : pathlib.Path or None, optional
        Explicit log-file path. If ``None``, a timestamped file is created under ``logs/``.

    Returns
    -------
    pathlib.Path
        The path to the active log file.
    """
    global _configured
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    if log_file is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        log_file = LOGS_DIR / f"ctb_{stamp}.log"

    root = logging.getLogger()
    if _configured:
        return log_file
    root.setLevel(level)

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    _configured = True
    logging.getLogger(__name__).info("Logging configured; writing to %s", log_file)
    return log_file
