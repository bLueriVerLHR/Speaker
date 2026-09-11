"""Accuracy ruler (single source of truth for the accuracy scale, P0 consolidation).

Background (r7 "three rulers" lesson): use_chat / valid_mode / soft-vs-hard were
free per-script knobs, and the accuracy scale silently drifted between the
training-time dual adjustment, the internal held-out eval and the external judge
(eval_compare). One semantic defined in N places is exactly the cascade this module
kills:

- valid_positions(batch, mode): the denominator accounting — "labels"
  (SFT standard: supervised segment labels != -100; numerically identical to
  attention_mask under the plain collate because padding is labeled -100) |
  "attention_mask" (legacy all-token accounting);
- batch_accuracy(logits, batch, mode): the per-batch training accuracy that feeds
  the dual (replaces the three divergent inline copies in the training scripts);
- AccRuler: acc_target policy —
    * "auto" (finetune default): the floor is derived from the dense baseline
      measured on the SAME slice/protocol (target = dense_acc - margin), so the
      sparsity/accuracy tradeoff is anchored to the same starting line whatever the
      base model / data slice / chat protocol — a fine-tuned dense reference can be
      fed in via resolve(dense_ft_acc) for the stricter repair-budget reading;
    * plain float ("0.55"): legacy absolute floor (bit-compatible reruns);
    * "none": dual disabled.
The resolved target is stamped loudly at run start and lands in mod_config.json
(cfg.acc_target) so every ckpt self-documents the line it was trained against.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch

from .metrics import per_token_correct

VALID_MODES = ("labels", "attention_mask")
LEGACY_DEFAULT_TARGET = 0.55  # fallback when auto has no dense reference (kept = legacy default)


def valid_positions(batch: dict, mode: str = "labels") -> torch.Tensor:
    """[B,T] bool mask of positions that count in the accuracy/loss statistics."""
    if mode == "labels":
        return batch["labels"] != -100
    if mode == "attention_mask":
        return batch["attention_mask"].bool()
    raise ValueError(f"valid mode must be one of {VALID_MODES}, got {mode!r}")


def batch_accuracy(logits: torch.Tensor, batch: dict, mode: str = "labels") -> float:
    """Per-batch token accuracy on the ruler's valid positions (training-time signal
    for the dual; same formula as the historical inline blocks)."""
    correct = per_token_correct(logits.float(), batch["labels"])
    valid = valid_positions(batch, mode)
    return correct[valid].float().mean().item() if valid.any() else 0.0


def parse_target(s: str) -> Tuple[str, Optional[float]]:
    """CLI string -> (policy, value): "auto" | "none" | float literal."""
    t = str(s).strip().lower()
    if t == "auto":
        return "auto", None
    if t in ("none", ""):
        return "none", None
    try:
        return "fixed", float(t)
    except ValueError:
        raise ValueError(
            f"acc_target must be 'auto', 'none' or a float, got {s!r}") from None


class AccRuler:
    """Owns the accuracy scale: valid-position mode + acc_target derivation policy."""

    def __init__(self, policy: str = "auto", fixed: Optional[float] = None,
                 margin: float = 0.03, mode: str = "labels"):
        if policy not in ("auto", "fixed", "none"):
            raise ValueError(f"policy must be auto|fixed|none, got {policy!r}")
        if policy == "fixed" and fixed is None:
            raise ValueError("fixed policy needs a value")
        if not 0.0 <= margin <= 1.0:
            raise ValueError(f"margin must be in [0,1], got {margin}")
        if mode not in VALID_MODES:
            raise ValueError(f"mode must be one of {VALID_MODES}, got {mode!r}")
        self.policy = policy
        self.fixed = fixed
        self.margin = margin
        self.mode = mode
        self.resolved: Optional[float] = None  # filled by resolve(), for logging/stamping
        self._dense_acc: Optional[float] = None  # last dense reference seen by resolve()

    @classmethod
    def from_cli(cls, target_str: str, margin: float = 0.03,
                 mode: str = "labels") -> "AccRuler":
        policy, value = parse_target(target_str)
        return cls(policy=policy, fixed=value, margin=margin, mode=mode)

    def resolve(self, dense_acc: Optional[float]) -> Optional[float]:
        """Derives the effective acc floor from the measured dense reference.
        auto without a reference falls back to the legacy default (dual stays active)."""
        self._dense_acc = dense_acc
        if self.policy == "none":
            self.resolved = None
        elif self.policy == "fixed":
            self.resolved = self.fixed
        elif dense_acc is None:
            print(f"WARNING [ruler] acc_target auto has no dense reference "
                  f"(empty eval slice?); falling back to the legacy default "
                  f"{LEGACY_DEFAULT_TARGET}", flush=True)
            self.resolved = LEGACY_DEFAULT_TARGET
        else:
            self.resolved = min(1.0, max(0.0, dense_acc - self.margin))
        return self.resolved

    def batch_acc(self, logits: torch.Tensor, batch: dict) -> float:
        return batch_accuracy(logits, batch, self.mode)

    def describe(self) -> str:
        if self.policy == "auto":
            src = (f"dense {self._dense_acc:.3f} - margin {self.margin}") \
                if self._dense_acc is not None else "dense reference not yet measured"
            return f"acc_target auto -> {self.resolved} ({src}; valid_mode={self.mode})"
        if self.policy == "none":
            return f"acc_target none (dual disabled; valid_mode={self.mode})"
        return (f"acc_target fixed {self.fixed} (valid_mode={self.mode})")
