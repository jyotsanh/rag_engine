"""Loguru setup: JSON (or human) sink, stdlib interception, and context IDs.

`request_id`, `job_id` and `tenant_id` live in contextvars, so every log line
emitted inside a request/task carries them without passing them around.
Bind them with `log_context(...)`.
"""

import inspect
import json
import logging
import sys
import traceback
from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar, Token
from typing import TYPE_CHECKING, Any, TextIO

from loguru import logger

from rag_engine.config.enums import Environment
from rag_engine.config.settings import settings

if TYPE_CHECKING:
    from loguru import Record

CONTEXT_FIELDS = ("request_id", "job_id", "tenant_id")
# set by InterceptHandler so stdlib records keep their own logger name
_STDLIB_NAME_KEY = "stdlib_logger"

_context_vars: dict[str, ContextVar[str | None]] = {
    name: ContextVar(name, default=None) for name in CONTEXT_FIELDS
}

# stdlib loggers that install their own handlers; reset so they propagate to root
_STDLIB_LOGGERS = (
    "uvicorn",
    "uvicorn.error",
    "uvicorn.access",
    "fastapi",
    "celery",
    "celery.task",
    "celery.worker",
    "celery.app.trace",
    "celery.beat",
    "kombu",
    "httpx",
    "pymongo",
)

_HUMAN_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | "
    "<cyan>{name}:{function}:{line}</cyan> | "
    "req={extra[request_id]} job={extra[job_id]} tenant={extra[tenant_id]} - "
    "<level>{message}</level>\n{exception}"
)


@contextmanager
def log_context(**ids: str | None) -> Generator[None]:
    """Bind context IDs (request_id / job_id / tenant_id) for the enclosed block."""
    unknown = set(ids) - set(CONTEXT_FIELDS)
    if unknown:
        raise ValueError(f"unknown log context fields: {sorted(unknown)}")
    tokens: list[tuple[ContextVar[str | None], Token[str | None]]] = [
        (_context_vars[name], _context_vars[name].set(value)) for name, value in ids.items()
    ]
    try:
        yield
    finally:
        for var, token in reversed(tokens):
            var.reset(token)


def get_log_context() -> dict[str, str | None]:
    return {name: var.get() for name, var in _context_vars.items()}


def _patch_record(record: "Record") -> None:
    """Fill context IDs from contextvars unless explicitly bound via `logger.bind`."""
    for name, value in get_log_context().items():
        if record["extra"].get(name) is None:
            record["extra"][name] = value


def _serialize(record: "Record") -> str:
    extra = {
        k: v
        for k, v in record["extra"].items()
        if k not in CONTEXT_FIELDS and k != _STDLIB_NAME_KEY
    }
    payload: dict[str, Any] = {
        "timestamp": record["time"].isoformat(),
        "level": record["level"].name,
        "message": record["message"],
        "logger": record["extra"].get(_STDLIB_NAME_KEY) or record["name"],
        "function": record["function"],
        "line": record["line"],
        **{name: record["extra"].get(name) for name in CONTEXT_FIELDS},
        "extra": extra,
    }
    exc = record["exception"]
    if exc is not None and exc.type is not None:
        payload["exception"] = {
            "type": exc.type.__name__,
            "value": str(exc.value),
            "traceback": "".join(traceback.format_exception(exc.type, exc.value, exc.traceback)),
        }
    return json.dumps(payload, default=str, ensure_ascii=False)


def _json_sink(stream: TextIO):
    def sink(message: Any) -> None:
        stream.write(_serialize(message.record) + "\n")
        stream.flush()

    return sink


class InterceptHandler(logging.Handler):
    """Route stdlib `logging` records into loguru (loguru's documented recipe)."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level: str | int = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        frame, depth = inspect.currentframe(), 0
        while frame and (depth == 0 or frame.f_code.co_filename == logging.__file__):
            frame = frame.f_back
            depth += 1

        logger.bind(**{_STDLIB_NAME_KEY: record.name}).opt(
            depth=depth, exception=record.exc_info
        ).log(level, record.getMessage())


def setup_logging(
    *,
    level: str | None = None,
    json_logs: bool | None = None,
    stream: TextIO | None = None,
) -> None:
    """(Re)configure loguru and capture stdlib logging. Safe to call more than once."""
    level = level or settings.LOG_LEVEL
    json_logs = settings.LOG_JSON if json_logs is None else json_logs
    stream = stream or sys.stdout
    # variable values in tracebacks can leak secrets: only outside prod
    diagnose = settings.ENVIRONMENT is not Environment.PROD

    logger.remove()
    logger.configure(
        extra=dict.fromkeys(CONTEXT_FIELDS),
        patcher=_patch_record,
    )
    if json_logs:
        logger.add(_json_sink(stream), level=level, backtrace=False, diagnose=False)
    else:
        logger.add(stream, level=level, format=_HUMAN_FORMAT, backtrace=True, diagnose=diagnose)

    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)
    for name in _STDLIB_LOGGERS:
        std_logger = logging.getLogger(name)
        std_logger.handlers = []
        std_logger.propagate = True


__all__ = ["InterceptHandler", "get_log_context", "log_context", "logger", "setup_logging"]
