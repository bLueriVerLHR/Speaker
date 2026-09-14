"""Primary development logger (loguru-based).

This is THE logger for all development use — not a training-domain logger.
Sinks: stdout (console) + optional run.log mirror. The stdout sink is installed
at import time (level from LOG_LEVEL, default INFO), so every module that does
``from speaker.log import logger`` logs to stdout with zero boilerplate —
no hand-written print() anywhere in the repo (rich tables/progress in
speaker/terminal.py are the only console exception). Training entries call
``setup_logger(run_dir)`` first thing to add the run.log mirror; ad-hoc tools
just log (stdout only).

Usage:
    from speaker.log import logger, setup_logger, add_jsonl, emit, event

    setup_logger(run_dir=save_dir)   # console + run.log mirror
    add_jsonl(os.path.join(save_dir, "metrics.jsonl"), "metrics")
    logger.debug("gate routing trace")  # hidden at the default INFO level
    logger.info("human one-liner")      # -> stdout (+ run.log)
    logger.warning("suspicious ...") / logger.exception("crashed")
    emit("metrics", step=step, lm=...)  # -> metrics.jsonl only
    event("plateau beat ...")           # human stream, [event] prefix

Separation rule: records carrying ``extra["jsonl"]`` go ONLY to their JSONL
sink; everything else goes to the human stream (stdout + run.log). Levels
filter as usual (DEBUG hidden at the default INFO level).
"""
from __future__ import annotations

import json
import os
import sys

from loguru import logger

__all__ = ["logger", "setup_logger", "add_jsonl", "emit", "event"]

_CONSOLE_FMT = "{time:HH:mm:ss} {level:<5} {message}"
_FILE_FMT = "{time:YYYY-MM-DD HH:mm:ss} {level:<5} {name}:{line} {message}"


def _is_human(record) -> bool:
    return "jsonl" not in record["extra"]


def _session_level() -> str:
    return os.environ.get("LOG_LEVEL", "INFO")


def _jsonl_dumps(payload: dict) -> str:
    # the message itself IS the JSON line (loguru renders format="{message}"
    # verbatim); non-serializable payloads fail loudly here, not in a sink
    return json.dumps(payload, ensure_ascii=False)


# Import-time console sink: every importer logs to stdout immediately, no
# setup call required (setup_logger only re-levels + adds the run.log mirror).
logger.remove()
logger.add(sys.stdout, format=_CONSOLE_FMT, level=_session_level(),
           filter=_is_human)


def setup_logger(run_dir=None, level: str | None = None, console: bool = True):
    """Reset sinks: stdout console + optional run.log mirror in run_dir."""
    if level is None:
        level = _session_level()
    logger.remove()
    if console:
        logger.add(sys.stdout, format=_CONSOLE_FMT, level=level,
                   filter=_is_human)
    if run_dir is not None:
        os.makedirs(run_dir, exist_ok=True)
        logger.add(os.path.join(run_dir, "run.log"), format=_FILE_FMT,
                   level=level, mode="a", filter=_is_human)
    return logger


def add_jsonl(path: str, key: str):
    """Append-only JSONL sink: records emitted via ``emit(key, ...)`` land here."""
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    logger.add(path, format="{message}", level="INFO", mode="a",
               filter=lambda r, k=key: r["extra"].get("jsonl") == k)
    return logger


def emit(key: str, **payload) -> None:
    """Write one JSON object to the JSONL sink registered for ``key``."""
    logger.bind(jsonl=key).info(_jsonl_dumps(payload))


def event(msg: str) -> None:
    """Operational event on the human stream (plateau/save/converge)."""
    logger.info(f"[event] {msg}")
