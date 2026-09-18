"""Structured logging with request-scoped correlation IDs.

Every log record carries the ``request_id`` of the request that produced it, so a
single query can be traced across retrieval, reranking, and generation.
"""

from __future__ import annotations

import logging
import sys
import uuid
from contextvars import ContextVar, Token

from pythonjsonlogger.json import JsonFormatter

_request_id: ContextVar[str] = ContextVar("request_id", default="-")

_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s %(request_id)s %(message)s"
_TEXT_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

# Third-party loggers that are noisy at INFO and drown out application logs.
_NOISY_LOGGERS = (
    "httpx",
    "httpcore",
    "urllib3",
    "sentence_transformers",
    "qdrant_client",
    "cohere",
    "huggingface_hub",
)


def get_request_id() -> str:
    """Correlation ID of the request currently being handled."""
    return _request_id.get()


def set_request_id(value: str) -> Token[str]:
    """Bind a correlation ID to the current context, returning a reset token."""
    return _request_id.set(value)


def reset_request_id(token: Token[str]) -> None:
    """Restore the previous correlation ID."""
    _request_id.reset(token)


def new_request_id() -> str:
    return uuid.uuid4().hex[:16]


class RequestIdFilter(logging.Filter):
    """Injects the active request ID into every record."""

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "request_id"):
            record.request_id = _request_id.get()
        return True


def configure_logging(level: str = "INFO", json_output: bool = True) -> None:
    """Install a single stdout handler on the root logger."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter(_LOG_FORMAT) if json_output else logging.Formatter(_TEXT_FORMAT))
    handler.addFilter(RequestIdFilter())

    root = logging.getLogger()
    for existing in list(root.handlers):
        root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(level.upper())

    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
