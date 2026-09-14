"""Init-time calibration primitives (split from wrapper.py, P0 structure).

Moved verbatim from SpeakerModelWrapper.calibrate_tau /
calibrate_router_temp: threshold's StableSkip-style model-aware tau init and
moe's JointRouter logit-temperature calibration. The hub methods delegate here
so call sites (``mod_model.calibrate_tau(...)`` /
``mod_model.calibrate_router_temp(...)`` in finetune/train.py and the tests)
are unchanged and numerics are bit-identical.
"""
from __future__ import annotations

from typing import Optional

import torch

from .gating import select_and_weight


def calibrate_tau_for(hub, batches, spread: float = 0.5) -> dict:
    """threshold only: StableSkip-style model-aware tau initialization (moe returns {})."""
    if hub.mod_config.gate_mode != "threshold":
        return {}
    sims: dict = {}
    was_training = hub.training
    hub.eval()
    with torch.no_grad():
        for b in batches:
            hub(**b)
            for w in hub.layers:
                if w.is_always_on:
                    continue
                s = getattr(w, "_last_sim", None)
                if s is not None and s.numel():
                    sims.setdefault(w.layer_idx, []).append(float(s.sum() / max((s != 0).sum(), 1)))
    if was_training:
        hub.train()
    if not sims:
        return {}
    keys = sorted(sims)
    vals = [sum(sims[k]) / len(sims[k]) for k in keys]
    mu = sum(vals) / len(vals)
    sd = (sum((v - mu) ** 2 for v in vals) / len(vals)) ** 0.5 + 1e-6
    out = {}
    for k, v in zip(keys, vals):
        w = hub.layers[k]
        w.tau.data.add_(torch.tensor(spread * (v - mu) / sd,
                                     device=w.tau.device, dtype=w.tau.dtype))
        out[k] = float(w.tau.detach())
    hub.get_layer_usage()  # clear usage accumulated during calibration
    return out


def calibrate_router_temp_for(hub, batches, target_k: Optional[int] = None) -> dict:
    """moe only: init-time JointRouter logit-temperature calibration so the initial top-p
    mean k starts near target_k (threshold returns {}). Mirror of calibrate_tau's
    philosophy: enter the budget-ramp window from a near-dense routing distribution.
    The temperature is persisted in SpeakerConfig (mod_config.json), not in state_dict —
    gate.pt key sets stay unchanged; 1.0 (default / legacy ckpts) = identity."""
    if hub.mod_config.gate_mode != "moe" or hub.joint_router is None:
        return {}
    cfg = hub.mod_config
    G = len(cfg.gated_layers)
    lo_k = max(int(cfg.min_layers), 1)
    hi_k = min(int(cfg.kmax), G)
    if target_k is None:
        target_k = max(lo_k, min(hi_k, round(0.6 * G)))
    target_k = int(max(lo_k, min(hi_k, int(target_k))))
    was_training = hub.training
    hub.eval()
    hub._calib_capture = []
    with torch.no_grad():
        for b in batches:
            hub(**b)
        captured = hub._calib_capture
        hub._calib_capture = None
        if not captured:
            if was_training:
                hub.train()
            return {}
        h = torch.cat([x.reshape(-1, x.shape[-1]) for x, _ in captured], dim=0)
        v = torch.cat([w.reshape(-1) for _, w in captured], dim=0) \
            if captured[0][1] is not None else None
        if v is None:
            v = torch.ones(h.shape[0], device=h.device)
        keep = v > 0
        h, v = h[keep], v[keep]
        if h.shape[0] > 8192:  # cap calibration tokens for speed
            idx = torch.randperm(h.shape[0], device=h.device)[:8192]
            h, v = h[idx], v[idx]
        raw = hub.joint_router(h)  # [N,G] fp32, no gumbel (eval)
        Ta0 = max(float(cfg.temp_affinity), 0.05)

        def mean_k(t: float) -> float:
            rd = select_and_weight(raw / (Ta0 * t), select_mode=cfg.select_mode,
                                   top_p=cfg.top_p, top_k=cfg.top_k,
                                   min_layers=cfg.min_layers, kmax=cfg.kmax,
                                   count_temp=cfg.count_temp, valid=v)
            return float((rd.k * v).sum() / v.sum().clamp_min(1.0))

        k_before = mean_k(1.0)
        t_best = 1.0
        if k_before < target_k - 0.25:
            # flatten the distribution (t > 1) until mean k reaches the target;
            # only flatten — a start that is already dense enough stays untouched
            lo, hi = 0.0, 8.0  # bisection on log2(t)
            if mean_k(2.0 ** hi) < target_k - 0.25:
                t_best = 2.0 ** hi  # best effort at the cap
            else:
                for _ in range(40):
                    mid = (lo + hi) / 2
                    if mean_k(2.0 ** mid) < target_k:
                        lo = mid
                    else:
                        hi = mid
                t_best = 2.0 ** ((lo + hi) / 2)
            cfg.router_temp = float(t_best)
        k_after = mean_k(t_best)
    if was_training:
        hub.train()
    hub.get_layer_usage()  # clear usage accumulated during calibration
    return {"router_temp": float(t_best), "k_before": k_before,
            "k_after": k_after, "target_k": target_k}


__all__ = ["calibrate_tau_for", "calibrate_router_temp_for"]
