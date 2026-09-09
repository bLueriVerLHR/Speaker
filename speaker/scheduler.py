"""GPU-residency scheduling for CPU-GPU collaborative inference (pluggable strategies).

Complements load_profile.plan_placement (static, measured-load planning): here the gated
layers' GPU residency is decided by a swappable strategy while the fixed (always-on) layers
are force-resident on the GPU. Contract (user design):

1. fixed (always-on) layers go to the GPU first — they execute for every token (mandatory;
   a config that cannot even fit them raises);
2. the remaining gated layers are scheduled into the leftover weights budget by strategy:
   - random: uniformly shuffle the candidates, greedily pack as many as fit (an optional
     k caps the count) — the simplest "random K layers onto the leftover GPU space";
   - lru: rank by most-recently-activated first (windowed granularity: one tick per
     observe()), never-activated layers rank last;
   - lfu (primitive, by design): cumulative per-layer activation token counts, top layers
     stay resident;
   the weights budget = GPU allowance minus an explicit reserve for KV cache and
   activations (the reserve is never spent on weights);
3. everything not on the GPU computes on the CPU (SpeakerLayerWrapper already moves layer
   inputs across the device boundary; moe route slices are realigned per layer).

Statistics come from the wrapper's own hard-activation accumulators via
SpeakerModelWrapper.pop_layer_counts (read-and-reset; whoever reads first wins the window,
so don't interleave get_layer_usage logging with observe() at fine granularity).

Lifecycle: model.schedule_placement(...) plans and applies once; after enough inference,
call scheduler.reschedule() to harvest counters and re-apply the (possibly changed) plan.
Reschedule BETWEEN generations only — per-layer KV caches are not migrated, and a cache
written on one device would mismatch a layer moved to the other.
"""
from __future__ import annotations

import random
from typing import Dict, List, Optional, Sequence

import torch

from .load_profile import estimate_layer_bytes

STRATEGIES = ("random", "lru", "lfu")


def select_gated_resident(strategy: str, gated: Sequence[int], layer_bytes: Dict[int, int],
                          budget: int, freq: Optional[Dict[int, float]] = None,
                          last_used: Optional[Dict[int, int]] = None,
                          rng: Optional[random.Random] = None,
                          k: Optional[int] = None) -> List[int]:
    """Pure decision function: which gated layers may reside on the GPU.

    Ranks the candidates by strategy, then greedily packs them in rank order into
    `budget` bytes (an oversized candidate is skipped — a smaller later one may still
    fit). `k` optionally caps the number of chosen layers. Ties break toward the lower
    layer index (deterministic; with no observations the fallback is index order, so
    repeated reschedules without new data do not thrash).
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"strategy must be one of {STRATEGIES}, got {strategy!r}")
    if k is not None and k < 0:
        raise ValueError(f"k must be >= 0, got {k}")
    order = list(gated)
    if strategy == "random":
        (rng or random).shuffle(order)
    elif strategy == "lfu":
        f = freq or {}
        order.sort(key=lambda i: (-f.get(i, 0.0), i))
    else:  # lru: most recently used first, never-used last
        lu = last_used or {}
        order.sort(key=lambda i: (-lu.get(i, -1), i))
    chosen: List[int] = []
    used = 0
    for i in order:
        if k is not None and len(chosen) >= k:
            break
        b = layer_bytes.get(i, 0)
        if used + b <= budget:
            chosen.append(i)
            used += b
    return sorted(chosen)


class LayerScheduler:
    """Ties a strategy to a wrapped model: observes activation counters, plans the
    gated-layer residency, and applies it through SpeakerModelWrapper.set_placement
    (moves happen only when the plan actually changes).

    Typical use:
        sched = model.schedule_placement("lfu", gpu_total_gb=18, reserve_gb=4)
        ... generate ...
        sched.reschedule()   # between generations (KV caches are not migrated)
    """

    def __init__(self, model, strategy: str = "lfu", *, k: Optional[int] = None,
                 gpu_total_gb: Optional[float] = None,
                 reserve_gb: float = 0.0, reserve_fraction: float = 0.25,
                 gpu_device: str = "cuda", cpu_device: str = "cpu",
                 seed: Optional[int] = None):
        if strategy not in STRATEGIES:
            raise ValueError(f"strategy must be one of {STRATEGIES}, got {strategy!r}")
        self.model = model
        self.strategy = strategy
        self.k = k
        self.gpu_device = gpu_device
        self.cpu_device = cpu_device
        self.rng = random.Random(seed)
        cfg = model.mod_config
        self.always_on = sorted(cfg.always_on_layers)
        self.gated = list(cfg.gated_layers)
        self.layer_bytes = estimate_layer_bytes(model)
        # weights budget = GPU allowance - KV cache / activation reserve (never weights)
        if gpu_total_gb is None:
            gd = torch.device(gpu_device)
            if gd.type != "cuda":
                raise ValueError(
                    "gpu_total_gb is required when gpu_device is not cuda "
                    "(auto-detection needs torch.cuda.mem_get_info)")
            gpu_total = torch.cuda.mem_get_info(gd)[0]  # free bytes, conservative
        else:
            gpu_total = int(gpu_total_gb * 1e9)
        reserve = int(reserve_gb * 1e9) if reserve_gb > 0 else int(gpu_total * reserve_fraction)
        self.weights_budget = max(int(gpu_total) - reserve, 0)
        self.reserve_bytes = reserve
        self.fixed_bytes = sum(self.layer_bytes.get(i, 0) for i in self.always_on)
        if self.fixed_bytes > self.weights_budget:
            raise ValueError(
                f"fixed (always-on) layers need {self.fixed_bytes / 1e9:.2f}GB but the "
                f"weights budget is only {self.weights_budget / 1e9:.2f}GB "
                f"(allowance {gpu_total / 1e9:.2f}GB - reserve {reserve / 1e9:.2f}GB); "
                f"raise the budget or lower the reserve")
        # activation statistics (LFU: cumulative counts; LRU: tick of last activation)
        self.freq: Dict[int, float] = {i: 0.0 for i in self.gated}
        self.last_used: Dict[int, int] = {i: -1 for i in self.gated}
        self.tick = 0
        self.resident: Optional[List[int]] = None  # full GPU layer list (fixed + chosen)

    # ----- Observation -----

    def observe(self) -> Dict[int, int]:
        """Harvests the per-layer hard-activation token counts since the last read
        (read-and-reset on the model) and folds them into freq/last_used. One call =
        one LRU tick; layers activated since the previous harvest count as 'recent'."""
        counts = self.model.pop_layer_counts()
        self.tick += 1
        for i in self.gated:
            c = counts.get(i, 0)
            if c > 0:
                self.freq[i] += c
                self.last_used[i] = self.tick
        return counts

    # ----- Planning / application -----

    def plan(self) -> List[int]:
        """Strategy decision: gated layers allowed on the GPU within the leftover budget."""
        return select_gated_resident(
            self.strategy, self.gated, self.layer_bytes,
            self.weights_budget - self.fixed_bytes,
            freq=self.freq, last_used=self.last_used, rng=self.rng, k=self.k)

    def reschedule(self) -> List[int]:
        """observe -> plan -> apply (moves only when the plan changes). Returns the full
        GPU-resident layer list. Call between generations: KV caches are not migrated."""
        self.observe()
        target = sorted(set(self.always_on) | set(self.plan()))
        if target != self.resident:
            self.model.set_placement(target, gpu_device=self.gpu_device,
                                     cpu_device=self.cpu_device)
            self.resident = target
        return target

    # ----- Diagnostics -----

    def describe(self) -> str:
        res = self.resident if self.resident is not None else []
        gpu_gb = sum(self.layer_bytes.get(i, 0) for i in res) / 1e9
        total_gb = sum(self.layer_bytes.values()) / 1e9
        hot = sorted(self.gated, key=lambda i: -self.freq.get(i, 0.0))[:3]
        return (f"LayerScheduler({self.strategy}): gpu {len(res)} layers {gpu_gb:.2f}GB "
                f"/ total {total_gb:.2f}GB, weights budget {self.weights_budget / 1e9:.2f}GB "
                f"(reserve {self.reserve_bytes / 1e9:.2f}GB), "
                f"top-freq {[(i, int(self.freq[i])) for i in hot]}")
