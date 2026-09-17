"""
logging_config.py
==================
Central logging setup. Every module does::

    from .logging_config import get_logger
    logger = get_logger(__name__)

so log records are hierarchically named (``hydrogen_res_opt.model_builder``,
``hydrogen_res_opt.powerfactory_interface``, ...) and can be filtered
independently, while all sharing the same handlers/format configured
once here.
"""

from __future__ import annotations

import logging
import logging.handlers
import os
import sys
from pathlib import Path

_CONFIGURED = False
_DEFAULT_LOG_DIR = Path(os.environ.get("HRO_LOG_DIR", "./logs"))
_DEFAULT_LOG_FILE = _DEFAULT_LOG_DIR / "hydrogen_res_opt.log"

_LOG_FORMAT = (
    "%(asctime)s | %(levelname)-8s | %(name)s | "
    "%(filename)s:%(lineno)d | %(message)s"
)


def configure_logging(
    level: int = logging.INFO,
    log_file: Path = _DEFAULT_LOG_FILE,
    max_bytes: int = 5_000_000,
    backup_count: int = 5,
) -> None:
    """Idempotently configure the root ``hydrogen_res_opt`` logger.

    Safe to call multiple times (e.g. once from the CLI, once from a
    notebook) -- subsequent calls are no-ops so handlers are never
    duplicated.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    log_file.parent.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger("hydrogen_res_opt")
    root.setLevel(level)
    root.propagate = False

    formatter = logging.Formatter(_LOG_FORMAT)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)

    file_handler = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)  # keep full detail on disk

    root.addHandler(console_handler)
    root.addHandler(file_handler)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the ``hydrogen_res_opt`` namespace,
    configuring handlers on first use if the caller forgot to."""
    if not _CONFIGURED:
        configure_logging()
    return logging.getLogger(name)
