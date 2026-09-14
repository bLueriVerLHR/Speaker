"""Shared pytest session setup: route the library logger to stderr.

The library speaks through speaker.log (loguru). Tests run without a run_dir,
so this hook points the human stream at stderr with the level taken from
LOG_LEVEL (default INFO): ``LOG_LEVEL=DEBUG pytest tests/`` exposes gate
diagnostics without touching any test file. Tests that need file sinks
(test_logging.py) call setup_logger themselves and restore the session level
afterwards.
"""
import os

from speaker.log import setup_logger


def pytest_configure():
    setup_logger(run_dir=None,
                 level=os.environ.get("LOG_LEVEL", "INFO"),
                 console=True)
