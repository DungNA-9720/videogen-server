"""structlog setup. Bind project_id, shot_id, provider, job_id via `bind_context`."""

import logging
import sys
from typing import Any

import structlog

CONTEXT_KEYS = ("project_id", "shot_id", "provider", "job_id")


def configure_logging(level: str = "INFO", json_logs: bool = True) -> None:
    logging.basicConfig(format="%(message)s", stream=sys.stdout, level=level.upper(), force=True)
    renderer: Any = (
        structlog.processors.JSONRenderer() if json_logs else structlog.dev.ConsoleRenderer()
    )
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelNamesMapping()[level.upper()]
        ),
        cache_logger_on_first_use=False,
    )


def get_logger(name: str | None = None) -> Any:
    return structlog.get_logger(name)


def bind_context(
    *,
    project_id: str | None = None,
    shot_id: str | None = None,
    provider: str | None = None,
    job_id: str | None = None,
) -> None:
    values = {
        "project_id": project_id,
        "shot_id": shot_id,
        "provider": provider,
        "job_id": job_id,
    }
    structlog.contextvars.bind_contextvars(**{k: v for k, v in values.items() if v is not None})


def clear_context() -> None:
    structlog.contextvars.clear_contextvars()
