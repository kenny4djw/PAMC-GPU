"""Package-level logger for PTMC-GPU.

Usage:
    from ptmc.log import logger
    logger.info("message")
    logger.debug("details")
    logger.warning("caution")
"""
from __future__ import annotations

import logging

logger = logging.getLogger("ptmc")


def configure_logger(level: int = logging.INFO) -> None:
    """Set level and attach a minimal stderr handler if none are present."""
    logger.setLevel(level)
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("[%(name)s] %(levelname)s %(message)s"))
        logger.addHandler(h)
