"""Project-wide logging helpers built on top of ``loguru``."""

from __future__ import annotations

import os
import sys
from typing import Optional

from loguru import logger as _base_logger


_LOG_LEVEL_ENV = "PEPBENCHMARK_LOGLEVEL"
_CURRENT_LEVEL: Optional[str] = None


class ProjectLogger:
    """Compatibility wrapper around loguru.

    The project historically used the standard ``logging`` API with printf-style
    placeholders (for example ``logger.info("%s", value)``). This wrapper keeps
    that call style working while routing everything through ``loguru``.
    """

    def __init__(self, bound_logger):
        self._logger = bound_logger

    def _format_message(self, message, *args) -> str:
        text = str(message)
        if not args:
            return text

        if "{}" in text:
            try:
                return text.format(*args)
            except Exception:
                pass

        try:
            return text % args
        except Exception:
            return " ".join([text, *[str(arg) for arg in args]])

    def _log(self, level: str, message, *args, **kwargs) -> None:
        self._logger.log(level, self._format_message(message, *args), **kwargs)

    def debug(self, message, *args, **kwargs) -> None:
        self._log("DEBUG", message, *args, **kwargs)

    def info(self, message, *args, **kwargs) -> None:
        self._log("INFO", message, *args, **kwargs)

    def warning(self, message, *args, **kwargs) -> None:
        self._log("WARNING", message, *args, **kwargs)

    def error(self, message, *args, **kwargs) -> None:
        self._log("ERROR", message, *args, **kwargs)

    def critical(self, message, *args, **kwargs) -> None:
        self._log("CRITICAL", message, *args, **kwargs)

    def exception(self, message, *args, **kwargs) -> None:
        self._logger.opt(exception=True).error(
            self._format_message(message, *args), **kwargs
        )

    def bind(self, **kwargs):
        return ProjectLogger(self._logger.bind(**kwargs))

    def __getattr__(self, name):
        return getattr(self._logger, name)


def _log_format() -> str:
    """Return the default loguru format string."""
    return "{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {extra[module_name]} | {message}"


def configure_logging(level: Optional[str] = None) -> None:
    """Configure the shared project logger.

    Args:
        level: Optional log level override. If omitted, the value is read from
            ``PEPBENCHMARK_LOGLEVEL`` and falls back to ``INFO``.
    """
    global _CURRENT_LEVEL

    resolved_level = (level or os.environ.get(_LOG_LEVEL_ENV, "INFO")).upper()
    if _CURRENT_LEVEL == resolved_level:
        return

    _base_logger.remove()
    _base_logger.add(
        sys.stdout,
        level=resolved_level,
        format=_log_format(),
        colorize=False,
        enqueue=False,
    )
    _CURRENT_LEVEL = resolved_level


def get_logger(name: str = "pepbenchmark"):
    """Return a module-aware project logger.

    Args:
        name: Logical module name shown in log output.

    Returns:
        A configured ``loguru`` logger bound with ``module_name``.
    """
    configure_logging()
    return ProjectLogger(_base_logger.bind(module_name=name))


def set_log_level(level: str) -> None:
    """Update the global PepBenchmark log level.

    Args:
        level: Target logging level such as ``INFO`` or ``DEBUG``.
    """
    configure_logging(level)


def enable_logging() -> None:
    """Enable standard informational logging."""
    set_log_level("INFO")


def disable_logging() -> None:
    """Reduce logging output to critical errors only."""
    set_log_level("CRITICAL")


configure_logging()
