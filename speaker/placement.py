"""Hierarchical placement primitives (split from wrapper.py, P0 structure).

Pure functions operating on a SpeakerModelWrapper (duck-typed: needs
``layers``/``mod_config``/``joint_router`` attributes). The hub methods
``set_placement`` / ``resident_gb`` / ``schedule_placement`` delegate here, so
the numerics and device semantics are unchanged — this module only gives the
logic a home outside the 800-line wrapper.

Import path compatibility: existing callers keep using
``model.set_placement(...)`` / ``model.resident_gb(...)`` /
``model.schedule_placement(...)``; these functions are the seam for future
placement policies.
"""
from __future__ import annotations

from typing import List

import torch


def apply_placement(hub, resident_ids, gpu_device="cuda", cpu_device="cpu",
                    force_always_gpu: bool = True) -> List[int]:
    """Hierarchical placement: hot layers (resident + high frequency) on GPU, cold layers on
    CPU. forward moves tensors across device boundaries automatically.
    resident_ids: layer indices kept on GPU; the remaining gated layers move to CPU
    (always_on forced to stay on GPU).
    force_always_gpu=False lifts that pin (dense-wrap: every layer is always_on, pinning
    them all would force the whole stack onto the GPU regardless of the budget).
    moe's JointRouter follows the hf_model's main device; _compute_route aligns devices internally."""
    resident = set(resident_ids)
    if force_always_gpu:
        resident |= set(hub.mod_config.always_on_layers)
    gd = torch.device(gpu_device)
    cd = torch.device(cpu_device)
    for w in hub.layers:
        w.to(gd if w.layer_idx in resident else cd)
    return sorted(resident)


def resident_gb_of(hub, device_type="cuda") -> float:
    """Resident parameter size on the given device (GB, weights only)."""
    n = sum(p.numel() * p.element_size() for p in hub.parameters() if p.device.type == device_type)
    n += sum(b.numel() * b.element_size() for b in hub.buffers() if b.device.type == device_type)
    return n / 1e9


def schedule_placement_for(hub, strategy: str = "lfu", **kwargs):
    """Scheduled placement for CPU-GPU collaborative inference (ed5): fixed layers stay
    resident on the GPU, gated layers are scheduled into the leftover weights budget
    (GPU allowance minus a KV-cache/activation reserve) by a pluggable strategy
    (random | lru | lfu — see speaker/scheduler.py). Plans and applies once, returns
    the LayerScheduler; call sched.reschedule() between generations to re-plan from
    the observed activation counters."""
    from .scheduler import LayerScheduler
    sched = LayerScheduler(hub, strategy, **kwargs)
    sched.reschedule()
    return sched


__all__ = ["apply_placement", "resident_gb_of", "schedule_placement_for"]
