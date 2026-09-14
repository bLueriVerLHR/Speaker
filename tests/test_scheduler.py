"""Scheduler self-tests (CPU, no real HF weights needed): strategy selection (random /
lru / lfu), budget greedy-packing, activation-counter harvesting, and the end-to-end
schedule_placement -> forward -> reschedule flow on a fake wrapped model.
"""
import random

import torch

from speaker import SpeakerConfig, convert_to_speaker
from speaker.load_profile import estimate_layer_bytes
from speaker.scheduler import STRATEGIES, LayerScheduler, select_gated_resident


from tests._fakes import FakeHF, force_onehot


def test_select_random():
    sizes = {2: 100, 3: 100, 4: 100, 5: 100}
    r1 = select_gated_resident("random", [2, 3, 4, 5], sizes, 250, rng=random.Random(0))
    r2 = select_gated_resident("random", [2, 3, 4, 5], sizes, 250, rng=random.Random(0))
    assert r1 == r2, "seeded random must be reproducible"
    assert len(r1) == 2 and sum(sizes[i] for i in r1) <= 250, r1
    rk = select_gated_resident("random", [2, 3, 4, 5], sizes, 400, rng=random.Random(1), k=1)
    assert len(rk) == 1, "k caps the chosen count"
    rk0 = select_gated_resident("random", [2, 3, 4, 5], sizes, 400, rng=random.Random(1), k=0)
    assert rk0 == [], "k=0 schedules nothing"


def test_select_lfu():
    sizes = {2: 100, 3: 100, 4: 100, 5: 100}
    r = select_gated_resident("lfu", [2, 3, 4, 5], sizes, 200,
                              freq={2: 5.0, 3: 9.0, 4: 0.0, 5: 7.0})
    assert r == [3, 5], f"lfu ranks by activation count, got {r}"
    # no observations: ties break toward the lower index (deterministic, no thrashing)
    r0 = select_gated_resident("lfu", [2, 3, 4, 5], sizes, 200)
    assert r0 == [2, 3], f"no-data fallback must be index order, got {r0}"


def test_select_lru():
    sizes = {2: 100, 3: 100, 4: 100, 5: 100}
    r = select_gated_resident("lru", [2, 3, 4, 5], sizes, 200,
                              last_used={2: 3, 3: -1, 4: 5, 5: 1})
    assert r == [2, 4], f"lru keeps the most recently used (4 then 2), got {r}"


def test_select_greedy_skip():
    # an oversized candidate is skipped; a smaller later one may still fit
    sizes = {2: 100, 3: 60, 4: 20}
    r = select_gated_resident("lfu", [2, 3, 4], sizes, 120,
                              freq={2: 9.0, 3: 5.0, 4: 1.0})
    assert r == [2, 4], f"greedy packing must skip oversized and keep fitting, got {r}"


def test_select_invalid():
    for bad in (dict(strategy="fifo"), dict(strategy="random", k=-1)):
        try:
            select_gated_resident(bad.get("strategy"), [2], {2: 1}, 10, k=bad.get("k"))
            raise AssertionError(f"should reject {bad}")
        except ValueError:
            pass


def _wrapped(n=8, h=32):
    cfg = SpeakerConfig(num_hidden_layers=n, hidden_size=h, gate_mode="moe",
                        kmax=4, gumbel_scale=0.0)
    mod = convert_to_speaker(FakeHF(n=n, h=h), cfg)
    return cfg, mod


def test_scheduler_flow():
    """schedule_placement -> forward (one-hot routing on slot 1 = layer 3) ->
    pop_layer_counts -> reschedule: LFU must move the residency to the hot layer,
    and an unchanged plan must not re-apply (no redundant layer moves)."""
    cfg, mod = _wrapped()
    assert cfg.always_on_layers == [0, 1, 6, 7] and cfg.gated_layers == [2, 3, 4, 5]
    lb = estimate_layer_bytes(mod)
    fixed_b = sum(lb[i] for i in cfg.always_on_layers)
    budget = fixed_b + lb[2]  # exactly one gated layer fits alongside the fixed ones
    applied = []
    orig_set = mod.set_placement

    def spy(resident_ids, **kw):
        applied.append(sorted(resident_ids))
        return orig_set(resident_ids, **kw)

    mod.set_placement = spy
    sched = mod.schedule_placement("lfu", gpu_total_gb=(budget + 1) / 1e9,
                                   reserve_gb=1 / 1e9, gpu_device="cpu",
                                   cpu_device="cpu", seed=0)
    # initial plan: no observations -> index-order fallback picks gated[0] = 2
    assert sched.resident == [0, 1, 2, 6, 7], sched.resident
    assert applied == [[0, 1, 2, 6, 7]]
    # forward with one-hot routing on slot 1 (layer 3): all activations land there;
    # pop_layer_counts is read-and-reset (whoever reads first wins the window — check the
    # direct read on window 1, then let the scheduler harvest window 2)
    force_onehot(mod, slot=1)
    am = torch.ones(2, 4, dtype=torch.long)
    x = torch.randn(2, 4, 32)
    mod(hidden_states=x, attention_mask=am)
    counts = mod.pop_layer_counts()
    assert counts[3] == 8 and all(counts[i] == 0 for i in (2, 4, 5)), counts
    assert mod.pop_layer_counts() == {i: 0 for i in (2, 3, 4, 5)}, "read once, cleared once"
    mod(hidden_states=x, attention_mask=am)  # fresh window for the scheduler
    # reschedule: LFU now ranks layer 3 first -> residency moves 2 -> 3
    sched.reschedule()
    assert sched.freq[3] == 8 and sched.resident == [0, 1, 3, 6, 7], (sched.freq, sched.resident)
    assert applied[-1] == [0, 1, 3, 6, 7]
    # reschedule again with no new forwards: plan unchanged -> no re-apply (no layer moves)
    n_applied = len(applied)
    sched.reschedule()
    assert len(applied) == n_applied, "unchanged plan must not re-apply"


def test_scheduler_fixed_too_big():
    cfg, mod = _wrapped()
    lb = estimate_layer_bytes(mod)
    fixed_b = sum(lb[i] for i in cfg.always_on_layers)
    try:
        LayerScheduler(mod, "lfu", gpu_total_gb=(fixed_b - 1) / 1e9, reserve_gb=0.001 / 1e9,
                       gpu_device="cpu", cpu_device="cpu")
        raise AssertionError("fixed layers not fitting the budget must raise")
    except ValueError:
        pass


def test_scheduler_k_cap():
    cfg, mod = _wrapped()
    lb = estimate_layer_bytes(mod)
    total = sum(lb.values())
    sched = mod.schedule_placement("random", k=0, gpu_total_gb=total * 2 / 1e9,
                                   reserve_gb=total / 1e9, gpu_device="cpu",
                                   cpu_device="cpu", seed=0)
    assert sched.resident == cfg.always_on_layers, "k=0 keeps only the fixed layers"


def test_strategies_registry():
    assert STRATEGIES == ("random", "lru", "lfu")
