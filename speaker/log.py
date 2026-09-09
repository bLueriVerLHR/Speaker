"""Unified training logging (from ed5 on, finetune/pretrain share one RunLogger):
- metrics.jsonl  for analysis: one line with all numeric values per log_interval
                 (machine-readable, source for plots/replays)
- layers.jsonl   for analysis: low-frequency per-layer execution rates (layer-load evolution)
- stdout         system run status: compact one-line main status + events (plateau/save/
                 convergence), for humans
- converge.jsonl model state: still written by each train.py itself (subset evaluation /
                 early-stop basis)
"""
from __future__ import annotations

import json
import os


class RunLogger:
    def __init__(self, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)
        self._m = open(os.path.join(save_dir, "metrics.jsonl"), "a", encoding="utf-8")
        self._l = open(os.path.join(save_dir, "layers.jsonl"), "a", encoding="utf-8")

    def metrics(self, **kw):
        self._m.write(json.dumps(kw, ensure_ascii=False) + "\n")
        self._m.flush()

    def layers(self, step: int, usage: dict, always_on):
        rates = {str(i): round(v[0], 4) for i, v in sorted(usage.items())}
        self._l.write(json.dumps({"step": step, "exec": rates,
                                  "always_on": sorted(always_on)}) + "\n")
        self._l.flush()

    @staticmethod
    def status(msg: str):
        print(msg, flush=True)

    @staticmethod
    def event(msg: str):
        print(f"[event] {msg}", flush=True)

    def close(self):
        self._m.close()
        self._l.close()
