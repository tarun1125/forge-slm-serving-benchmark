"""Structured JSON logging, shared by every FORGE module.

Convention (see AGENTS.md):
  - structlog with JSON output; every module gets a bound logger via get_logger().
  - Every experiment run emits a run_id (UUID4) on its first log line, bound to
    every subsequent record for that run via start_run().
  - Log at boundaries (model load, request in/out, tool call, eval start/finish,
    error) with latency_ms included.
  - Never log raw API keys, full prompts containing user data, or full model
    weights paths that leak local directory structure — hash instead (see
    hash_for_log()).
  - Log level comes from LOG_LEVEL env var, defaulting to INFO. DEBUG must be
    explicitly opted into.
"""

from __future__ import annotations

import hashlib
import logging
import os
import uuid

import structlog


def _log_level() -> int:
    level_name = os.environ.get("LOG_LEVEL", "INFO").upper()
    return (
        logging.getLevelNamesMapping().get(level_name, logging.INFO)
        if hasattr(logging, "getLevelNamesMapping")
        else getattr(logging, level_name, logging.INFO)
    )


def configure_logging() -> None:
    """Idempotent — safe to call at the top of every entrypoint."""
    logging.basicConfig(
        format="%(message)s",
        level=_log_level(),
    )
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(_log_level()),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)


def start_run(logger: structlog.stdlib.BoundLogger, **extra: object) -> str:
    """Mint a run_id, bind it into the contextvars so every subsequent log call
    in this process carries it, log the first line, and return it."""
    run_id = str(uuid.uuid4())
    structlog.contextvars.bind_contextvars(run_id=run_id)
    logger.info("run.start", run_id=run_id, **extra)
    return run_id


def hash_for_log(value: str, length: int = 12) -> str:
    """Stable short hash for correlating sensitive values (prompts, paths,
    keys) in logs without ever writing the raw value."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]
