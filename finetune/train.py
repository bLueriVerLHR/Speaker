"""Finetune route: validate the Speaker design (shared + gated layers) on an existing
model for quick effect testing.

Thin entry point (P0 structure): CLI lives in finetune/cli.py, the training
loop in finetune/pipeline.py. This module keeps the historical import surface
(``python finetune/train.py ...`` and the UL shim
``from finetune.train import ngram_repeat_trigger, unlikelihood_loss``
used by tests/test_ul.py) working unchanged.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from finetune.cli import app, train  # noqa: E402,F401
from finetune.pipeline import run_finetune  # noqa: E402,F401
from speaker.ul import (  # noqa: E402,F401 (historical shim: UL lived here before P2)
    gt_repeat_rate,
    ngram_repeat_trigger,
    rollout_rep3_probe,
    rollout_unlikelihood,
    unlikelihood_loss,
)

__all__ = [
    "app", "train", "run_finetune",
    "gt_repeat_rate", "ngram_repeat_trigger", "rollout_rep3_probe",
    "rollout_unlikelihood", "unlikelihood_loss",
]


if __name__ == "__main__":
    app()
