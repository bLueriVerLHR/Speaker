"""Gating-strategy polymorphism seam (P1 structure).

Historically every consumer branched on ``config.gate_mode`` strings
(``if gate_mode == "moe"`` scattered across wrapper/config/checkpoint/
assemble). This module introduces the Strategy seam so new schemes plug in
without touching call sites:

- ``GatingStrategy``: Protocol both schemes satisfy (``name`` + pure helpers);
- ``ThresholdStrategy`` / ``MoeStrategy``: thin delegates over the existing
  numerics in speaker/gating.py (no math moved, no behavior change — the
  layer/hub keep calling the same functions; strategies are the discoverable
  entry point for tooling/tests and the extension point for future schemes);
- ``get_strategy(gate_mode)``: Registry lookup (canonicalizes the
  speaker/mol CLI aliases the same way SpeakerConfig does).

Numerics guarantee: every method here forwards to the exact function the
training loop already calls, so outputs are bit-identical by construction
(covered by tests/test_strategies.py).
"""
from __future__ import annotations

from typing import Dict, Protocol, runtime_checkable

from .gating import RouteDecision, select_and_weight


@runtime_checkable
class GatingStrategy(Protocol):
    """Interface every gating scheme satisfies."""

    name: str

    def describe(self) -> str:
        ...


class ThresholdStrategy:
    """Legacy per-layer scheme: independent Router(H->1)+tau per layer, sigmoid
    threshold gating + STE, budget = λ·mean(k) + over-kmax penalty."""

    name = "threshold"

    def describe(self) -> str:
        return "threshold: per-layer Router+tau, sigmoid+STE, λ·mean(k)+over-kmax"

    @staticmethod
    def budget_kind() -> str:
        return "ste_sum"

    @staticmethod
    def needs_joint_router() -> bool:
        return False


class MoeStrategy:
    """Mainline joint scheme: one JointRouter(H->G) at the gated-region entry,
    top-p/top-k selection in log-p space, pmax/renorm residual weighting,
    budget = λ·mean(k_soft) via soft inclusion."""

    name = "moe"

    def describe(self) -> str:
        return "moe: JointRouter(H->G) top-p + pmax weights, λ·mean(k_soft)"

    @staticmethod
    def budget_kind() -> str:
        return "soft_mean"

    @staticmethod
    def needs_joint_router() -> bool:
        return True

    @staticmethod
    def route(logits, **kw) -> RouteDecision:
        """Joint routing decision (forwards to gating.select_and_weight)."""
        return select_and_weight(logits, **kw)


_ALIASES = {"speaker": "threshold", "mol": "moe"}
_REGISTRY: Dict[str, object] = {
    "threshold": ThresholdStrategy(),
    "moe": MoeStrategy(),
}


def canonical_gate_mode(gate_mode: str) -> str:
    """speaker->threshold, mol->moe (same normalization as SpeakerConfig)."""
    return _ALIASES.get(gate_mode, gate_mode)


def get_strategy(gate_mode: str):
    """Registry lookup by (possibly aliased) gate mode; raises ValueError on unknown."""
    key = canonical_gate_mode(gate_mode)
    try:
        return _REGISTRY[key]
    except KeyError:
        raise ValueError(
            f"unknown gating strategy {gate_mode!r} "
            f"(expected one of {sorted(set(_REGISTRY) | set(_ALIASES))})") from None


def list_strategies() -> list:
    return sorted(_REGISTRY)


__all__ = [
    "GatingStrategy", "ThresholdStrategy", "MoeStrategy",
    "get_strategy", "list_strategies", "canonical_gate_mode",
]
