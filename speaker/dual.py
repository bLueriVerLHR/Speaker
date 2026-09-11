"""Dual controller (P0.5 consolidation): the single orchestration point of the
accuracy dual.

Historically the dual's logic was scattered across three files: the EMA state lived
in the training scripts, the warmup gate (`if step > price_warmup_steps`) in the
training scripts, and the adaptation rule in SpeakerModelWrapper.adapt_price. Any
semantic change (warmup bound, cadence, reset-on-resume) had to be synchronized in
finetune/train.py + pretrain/train.py + wrapper.py — the classic cascade.

DualController owns the orchestration; the adaptation *math* stays in
SpeakerModelWrapper.adapt_price (the primitive, also kept for direct callers/tests).
observe() is called once per training step and is behavior-bit-identical to the
historical inline triple:

    ema_acc = ema_update(ema_acc, acc_item)
    if step > cfg.price_warmup_steps:
        mod_model.adapt_price(ema_acc)

Resume semantics unchanged (documented contract): the EMA restarts from None and λ
resets to the CLI value, as before.
"""
from __future__ import annotations

from typing import Optional

import torch

from .evaluate import ema_update


def difficulty_mult(teacher_nll, easy_nll: float, hard_nll: float,
                    easy_mult: float, hard_mult: float):
    """Per-token λ multiplier from frozen-teacher difficulty (ed8, both schemes):
    easy (nll < easy_nll) → easy_mult, hard (nll > hard_nll) → hard_mult,
    mid → 1.0 (boundaries belong to mid). Pure function: the dual still adapts the
    global λ_base against acc_target; this map only shapes it per token
    (level ↔ dual, shape ↔ difficulty)."""
    m = torch.ones_like(teacher_nll)
    m = torch.where(teacher_nll < easy_nll,
                    torch.full_like(m, float(easy_mult)), m)
    m = torch.where(teacher_nll > hard_nll,
                    torch.full_like(m, float(hard_mult)), m)
    return m


class DualController:
    """Accuracy dual: ema_acc + warmup gate + λ adaptation, one call per step."""

    def __init__(self, model_wrapper, cfg):
        self.model = model_wrapper
        self.cfg = cfg
        self.ema_acc: Optional[float] = None

    def observe(self, step: int, acc_item: float) -> Optional[float]:
        """Updates the accuracy EMA and (past warmup) adapts λ toward the target.
        Returns the current ema_acc for logging."""
        self.ema_acc = ema_update(self.ema_acc, acc_item)
        if step > self.cfg.price_warmup_steps:
            self.model.adapt_price(self.ema_acc)
        return self.ema_acc
