from __future__ import annotations

import sys

from loguru import logger


def configure_logging(level: str = "INFO") -> None:
    logger.remove()
    logger.add(sys.stderr, level=level, enqueue=False, backtrace=False, diagnose=False)
