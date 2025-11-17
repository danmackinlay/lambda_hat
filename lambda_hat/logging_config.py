# lambda_hat/logging_config.py
from __future__ import annotations

import logging
import logging.config
import os
import queue
from logging.handlers import QueueHandler, QueueListener

try:
    from pythonjsonlogger import jsonlogger  # pip install python-json-logger
except Exception:  # optional dependency
    jsonlogger = None

_QUEUE = None
_LISTENER: QueueListener | None = None


def configure_logging(
    *,
    level: str | None = None,
    format: str | None = None,
    json_mode: bool | None = None,
    capture_warnings: bool = True,
) -> None:
    """
    Call once from entrypoints (CLI, notebooks, servers, scripts).
    Library code MUST NOT call this.
    """
    global _QUEUE, _LISTENER

    # Defaults from env, then sensible fallbacks
    level = (level or os.getenv("LOG_LEVEL") or "INFO").upper()
    json_mode = json_mode if json_mode is not None else os.getenv("LOG_FORMAT") == "json"

    # Base formatter(s)
    if json_mode and jsonlogger:
        fmt_name = "json"
        fmt_def = {
            "format": "%(asctime)s %(levelname)s %(name)s %(message)s",
        }
    else:
        fmt_name = "plain"
        fmt_def = {
            "format": "[%(asctime)s] %(levelname)s %(name)s: %(message)s",
            "datefmt": "%H:%M:%S",
        }

    # Handlers: stream only; containers and CI should log to stdout/stderr.
    handlers = {
        "console": {
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
            "formatter": fmt_name,
        }
    }

    # Optional: swap in JSON formatter
    formatters = {
        "plain": {"format": fmt_def["format"], "datefmt": fmt_def.get("datefmt")},
    }
    if json_mode and jsonlogger:
        formatters["json"] = {
            "()": "pythonjsonlogger.jsonlogger.JsonFormatter",
            "fmt": fmt_def["format"],
        }
    else:
        formatters["json"] = formatters["plain"]  # fallback

    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": formatters,
        "handlers": handlers,
        "root": {
            "level": level,
            "handlers": ["console"],
        },
    }

    logging.config.dictConfig(config)

    if capture_warnings:
        logging.captureWarnings(True)  # route warnings -> logging under 'py.warnings'


def start_mp_logging_queue(base_handlers: list[logging.Handler] | None = None):
    """
    Multiprocessing-safe logging: use QueueHandler in workers, QueueListener in parent.
    Call *in the parent process* once, then call get_worker_queue_handler() in workers.
    """
    global _QUEUE, _LISTENER
    if _QUEUE is None:
        _QUEUE = queue.Queue()
        handlers = base_handlers or list(logging.getLogger().handlers)
        _LISTENER = QueueListener(_QUEUE, *handlers, respect_handler_level=True)
        _LISTENER.start()


def get_worker_queue_handler() -> logging.Handler:
    """Return a QueueHandler to install in child processes."""
    if _QUEUE is None:
        raise RuntimeError("Call start_mp_logging_queue() in the parent first.")
    return QueueHandler(_QUEUE)


def shutdown_logging():
    global _LISTENER
    if _LISTENER:
        _LISTENER.stop()
        _LISTENER = None
