"""Project-wide logging setup.

Uses ``rich`` for readable console logs when available, falling back to the
standard formatter otherwise.
"""

from __future__ import annotations

import logging

from backend.app.config import settings

_CONFIGURED = False


def configure_logging(level: str | None = None) -> None:
    """Configure root logging once for the whole process."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    log_level = (level or settings.log_level).upper()

    try:
        from rich.logging import RichHandler

        handler: logging.Handler = RichHandler(rich_tracebacks=True, show_path=False)
        fmt = "%(message)s"
    except ImportError:  # pragma: no cover - rich is optional
        handler = logging.StreamHandler()
        fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

    logging.basicConfig(level=log_level, format=fmt, handlers=[handler])
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger for ``name``."""
    configure_logging()
    return logging.getLogger(name)
