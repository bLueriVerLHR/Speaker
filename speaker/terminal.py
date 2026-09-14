"""Shared terminal output (rich): console, tracebacks, tables, progress.

All entry points (Typer commands) call ``setup_terminal()`` first: readable
tracebacks on crashes, one shared Console for tables/summaries. The log stream
goes through speaker.log (loguru: stdout + run.log); rich is used only for
final summary tables and interactive progress.

``track()`` is tty-aware: a rich progress bar on interactive terminals, a
silent passthrough when piped (so redirected logs get no bar noise).
"""
from __future__ import annotations

import sys

from rich.console import Console
from rich.table import Table

console = Console()
err_console = Console(stderr=True)

_INSTALLED = False


def setup_terminal() -> Console:
    """Install the rich traceback handler (once) and return the console."""
    global _INSTALLED
    if not _INSTALLED:
        from rich.traceback import install
        install(show_locals=False, suppress=[])
        _INSTALLED = True
    return console


def summary_table(title: str, columns: list, rows: list) -> None:
    """Render a small string table (all cells pre-formatted by the caller)."""
    t = Table(title=title, show_header=True, header_style="bold")
    for c in columns:
        t.add_column(str(c), justify="right" if c != columns[0] else "left")
    for r in rows:
        t.add_row(*[str(c) for c in r])
    console.print(t)


def track(iterable, total=None, desc=""):
    """tty-aware progress: rich bar when interactive, passthrough when piped."""
    if sys.stderr.isatty():
        from rich.progress import track as _track
        return _track(iterable, total=total, description=desc)
    return iterable


__all__ = ["console", "err_console", "setup_terminal", "summary_table", "track"]
