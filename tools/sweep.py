"""Cartesian sweep launcher (Typer replacement for hydra --multirun).

A sweep YAML declares one base command plus override axes; this expands the
cartesian product and either prints (``--dry-run``, default) or executes each
run sequentially, tee-ing stdout to ``.logs/<TS>_<name>.log`` per the repo
runtime-log convention. For neu-sbox submissions, dry-run the command lines
and paste them into the submit template (AGENTS.md) — one quoted string each.

Example (configs/moe_kmax.yaml):
  command: ["finetune/train.py", "--model_id ...", "--gate_mode moe"]
  axes:
    kmax: [12, 16, 20]
    ul_coef: [0.3, 0.6]
"""
from __future__ import annotations

import itertools
import subprocess
import sys
import time
from pathlib import Path
from typing import Annotated, Optional

import typer

from speaker.log import logger

app = typer.Typer(add_completion=False)


def load_sweep(path: str) -> dict:
    import yaml
    with open(path, encoding="utf-8") as f:
        d = yaml.safe_load(f)
    if not isinstance(d, dict) or "command" not in d:
        raise ValueError(f"{path}: sweep needs a 'command' list (+ optional 'axes' dict)")
    return d


def expand_runs(spec: dict) -> list:
    """Cartesian expansion -> [(run_name, argv_list)]."""
    base = list(spec.get("command", []))
    axes = spec.get("axes") or {}
    if not axes:
        return [("single", base)]
    keys = sorted(axes)
    runs = []
    for combo in itertools.product(*[axes[k] for k in keys]):
        tag = "_".join(f"{k}{v}" for k, v in zip(keys, combo))
        argv = list(base)
        for k, v in zip(keys, combo):
            argv += [f"--{k}", str(v)]
        runs.append((tag, argv))
    return runs


@app.command()
def run(
    config: Annotated[str, typer.Option("--config", help="sweep YAML path")] = "configs/moe_kmax.yaml",
    dry_run: Annotated[bool, typer.Option("--dry-run/--no-dry-run", help="print commands without executing")] = True,
    log_dir: Annotated[str, typer.Option("--log-dir", help="runtime log dir")] = ".logs",
    python: Annotated[str, typer.Option("--python", help="python binary")] = sys.executable,
) -> None:
    """Expand the sweep and dry-run (default) or execute it."""
    spec = load_sweep(config)
    runs = expand_runs(spec)
    logger.info(f"sweep {config}: {len(runs)} runs")
    for tag, argv in runs:
        cmd = [python, *argv]
        logger.info(f"[{tag}] {' '.join(cmd)}")
        if dry_run:
            continue
        ts = time.strftime("%m%d_%H%M")
        log = Path(log_dir) / f"{ts}_sweep_{tag}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        with open(log, "wb") as f:
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            for chunk in iter(lambda: p.stdout.read1(1 << 16) if p.stdout else b"", b""):
                f.write(chunk)
            p.wait()
        if p.returncode == 0:
            logger.info(f"[{tag}] rc=0 -> {log}")
        else:
            logger.warning(f"[{tag}] rc={p.returncode} -> {log}")
        if p.returncode != 0:
            raise typer.Exit(code=p.returncode)


@app.command()
def show(
    config: Annotated[str, typer.Option("--config", help="sweep YAML path")] = "configs/moe_kmax.yaml",
) -> None:
    """Print the expanded run list without executing."""
    for tag, argv in expand_runs(load_sweep(config)):
        logger.info(f"[{tag}] {' '.join(argv)}")


def main(argv: Optional[list] = None) -> None:
    # programmatic entry (tests): parse like parse_args shims elsewhere.
    sys.argv[1:] = argv if argv is not None else sys.argv[1:]
    app()


if __name__ == "__main__":
    app()


__all__ = ["app", "run", "show", "main", "load_sweep", "expand_runs"]
