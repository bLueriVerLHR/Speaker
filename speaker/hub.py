"""Joint-routing hub (split from wrapper.py, P0 structure).

SpeakerModelWrapper: dual-scheme hub — threshold = legacy statistics/budget
(per-layer _last); moe = joint routing hub (JointRouter single-point
registration + route decision). The statistics / budget / placement interfaces
share the same names and semantics.

Placement (set_placement / resident_gb / schedule_placement) delegates to
speaker/placement.py; init-time calibration (calibrate_tau /
calibrate_router_temp) delegates to speaker/calibrate.py. Bodies moved
verbatim — numerics and device semantics are bit-identical.

Import path compatibility: ``speaker.wrapper`` re-exports everything here.
"""
from __future__ import annotations

from dataclasses import replace as _dc_replace
from typing import Optional

import torch
import torch.nn as nn

from .gating import JointRouter, RouteDecision, select_and_weight
from .layer import SpeakerLayerWrapper
from .log import logger


def _to_common(ts, dim=0):
    """torch.stack across per-layer devices (sharded backbones keep each layer's
    gating tensors on its own card): gather on the first tensor's device.
    Single-device: identical tensors in, identical stack out."""
    d0 = ts[0].device
    return torch.stack([t.to(d0) if t.device != d0 else t for t in ts], dim=dim)


def _budget_core(kk, cfg, price):
    """Shared elastic core over valid-token k (1D, differentiable). All forms keep
    ∂L/∂k ≥ 0 so the dual lever (adapt_price) stays connected:
    - mean (legacy): price·mean(k);
    - hinge: price·max(0, mean−T), T = setpoint (parks mean at T; dual tightening
      while parked below T is a no-op, so λ may ratchet to price_max — harmless,
      wall just gets steeper);
    - tail: price·mean + price·tail_coef·P(k>B) with a sigmoid soft counter
      (SLO-style: mean pressure plus tail-violation pressure).
    T/B resolve to kmax when budget_target=0 (auto). Deliberately NOT 1/var-style
    shapes: batch var rewards polarized (bimodal) collapse and disconnects the
    dual (r8 ablation note)."""
    form = cfg.budget_form
    T = float(cfg.budget_target) if cfg.budget_target > 0 else float(cfg.kmax)
    if form == "mean":
        return price * kk.mean()
    if form == "hinge":
        return price * torch.clamp(kk.mean() - T, min=0)
    if form == "tail":
        frac = torch.sigmoid((kk - T) / max(float(cfg.tail_temp), 1e-3))
        return price * (kk.mean() + float(cfg.tail_coef) * frac.mean())
    raise ValueError(f"budget_form must be mean|hinge|tail, got {form!r}")


class SpeakerModelWrapper(nn.Module):
    """Dual-scheme hub: threshold = legacy statistics/budget (per-layer _last); moe = joint
    routing hub (JointRouter single-point registration + route decision). The statistics /
    budget / placement interfaces share the same names and semantics."""

    def __init__(self, hf_model: nn.Module, mod_config):
        super().__init__()
        self.hf_model = hf_model
        self.mod_config = mod_config
        self._ta_start = float(mod_config.temp_affinity)
        self._g_start = float(mod_config.gumbel_scale)
        self.route: Optional[RouteDecision] = None  # moe: joint routing of the current forward (written by the entry layer)
        self._calib_capture = None  # moe router-temp calibration: list collecting (entry hidden, valid) per batch
        self._patch()
        if self.mod_config.gate_mode == "moe":
            n_gated = len(self.mod_config.gated_layers)
            self.joint_router = JointRouter(self.mod_config.hidden_size, n_gated) if n_gated > 0 else None
        else:
            self.joint_router = None

    def _find_layers(self):
        """Returns (ModuleList, parent_module, attr name). Under peft wrapping, chained getattr
        sees through to the real layers. VL/nested-text backbones (Qwen3_5) keep the decoder
        stack at model.language_model.layers — probed before the legacy paths."""
        N = self.mod_config.num_hidden_layers
        for path in (["model", "language_model", "layers"], ["model", "layers"],
                     ["layers"], ["language_model", "layers"], ["transformer", "h"]):
            cur = self.hf_model
            try:
                for a in path:
                    cur = getattr(cur, a)
                if isinstance(cur, nn.ModuleList) and len(cur) == N:
                    parent = self.hf_model
                    for a in path[:-1]:
                        parent = getattr(parent, a)
                    return cur, parent, path[-1]
            except Exception:
                pass
        for name, m in self.hf_model.named_modules():
            if isinstance(m, nn.ModuleList) and len(m) == N and (
                    hasattr(m[0], "self_attn") or hasattr(m[0], "linear_attn")):
                if not name:
                    continue
                parent = self.hf_model
                if "." in name:
                    pname, attr = name.rsplit(".", 1)
                    for a in pname.split("."):
                        parent = getattr(parent, a)
                else:
                    attr = name
                return m, parent, attr
        raise ValueError("cannot find layers")

    def _patch(self):
        layers, parent, attr = self._find_layers()
        moe = self.mod_config.gate_mode == "moe"
        new = nn.ModuleList([SpeakerLayerWrapper(l, i, self.mod_config,
                                                 hub=(self if moe else None))
                             for i, l in enumerate(layers)])
        if moe:
            gated = self.mod_config.gated_layers
            slots = {lid: j for j, lid in enumerate(gated)}
            for w in new:
                if not w.is_always_on:
                    w._slot = slots[w.layer_idx]
                    w._is_entry = (w.layer_idx == gated[0])
        setattr(parent, attr, new)
        self.layers = new
        # Device alignment for sharded backbones (device_map=auto): the fresh gating
        # params are created on CPU while their layers may live on any GPU — move each
        # wrapper onto its layer's device (inner .layer is already there, no-op).
        # Whole-card mode kept working via the historical .to(device) after convert;
        # moe's JointRouter stays on the hub device and _compute_route moves hidden
        # states to it explicitly.
        for w in new:
            if w.is_always_on or w.router is None:
                continue
            dev = w._layer_device()
            if dev.type != "cpu":
                w.to(dev)
        # the layers' parent module (Qwen1_5: model.model / Qwen3_5 VL + peft:
        # wherever the stack lives) — inference tooling reads it to find the
        # sibling final-norm for GPU placement (see tools/edge_bench.py).
        # object.__setattr__: must NOT register as a submodule (state_dict stays clean)
        object.__setattr__(self, "_layers_parent", parent)

    # ----- Joint routing (moe) -----

    def _compute_route(self, hidden_states: torch.Tensor, attention_mask=None):
        """Called by the entry layer: gated-region entry hidden state -> RouteDecision for all
        G gated layers."""
        cfg = self.mod_config
        if self.joint_router is None:
            self.route = None
            return
        rdev = self.joint_router.net.weight.device
        h = hidden_states.to(rdev) if hidden_states.device != rdev else hidden_states
        logits = self.joint_router(h)
        if self.training and float(cfg.gumbel_scale) > 0:
            u = torch.rand_like(logits).clamp_(1e-6, 1.0 - 1e-6)
            logits = logits + (-torch.log(-torch.log(u))) * float(cfg.gumbel_scale)
        Ta = max(float(cfg.temp_affinity), 0.05) * float(getattr(cfg, "router_temp", 1.0))
        valid = None
        if attention_mask is not None and attention_mask.dim() == 2:
            valid = attention_mask.to(torch.float32)
        cap = self._calib_capture
        if cap is not None:
            cap.append((h.detach(), None if valid is None else valid.detach()))
        route = select_and_weight(logits / Ta, select_mode=cfg.select_mode,
                                  top_p=cfg.top_p, top_k=cfg.top_k,
                                  min_layers=cfg.min_layers, kmax=cfg.kmax,
                                  count_temp=cfg.count_temp, valid=valid,
                                  weight_mode=getattr(cfg, "weight_mode", "pmax"))
        if route.logits.device != hidden_states.device:
            route = _rd_to(route, hidden_states.device)
        self.route = route

    def forward(self, *a, **kw):
        if self.mod_config.gate_mode == "moe":
            self.route = None  # reset each forward, entry layer recomputes (prevents stale decisions across forwards)
        return self.hf_model(*a, **kw)

    def generate(self, *a, **kw):
        # generate calls hf_model.forward step by step; the moe entry layer recomputes routing each step
        return self.hf_model.generate(*a, **kw)

    # ----- Statistics -----

    def _gated(self):
        return [w for w in self.layers if not w.is_always_on and w.last_gating_output is not None]

    @staticmethod
    def _cos_term(w):
        """StableSkip-style per-layer term: mean(soft * detached sim on soft's device)."""
        q = w.last_gating_output.soft_mask.squeeze(-1).float()
        return (q * w._last_sim.to(q.device)).mean()

    def get_aux_loss(self):
        if self.mod_config.gate_mode == "moe":
            return self._aux_loss_moe()
        ls = self._gated()
        if not ls:
            return None
        z = [w.last_gating_output.aux_loss for w in ls if w.last_gating_output.aux_loss is not None]
        loss = _to_common(z).mean() if z else 0
        if self.mod_config.balance_loss_coef > 0 and len(ls) > 1:
            means = _to_common([w.last_gating_output.soft_mask.float().mean() for w in ls])
            loss = loss + self.mod_config.balance_loss_coef * means.var(unbiased=False)
        if self.mod_config.cos_reg_coef > 0:
            # StableSkip-style: tokens whose representation barely changes should be skipped
            # (high sim -> low soft). sim is detached; gradients flow only through soft
            # into router/tau.
            raw = [self._cos_term(w) for w in ls
                   if w.last_gating_output is not None and w._last_sim is not None]
            if raw:
                loss = loss + self.mod_config.cos_reg_coef * _to_common(raw).mean()
        return loss

    def _aux_loss_moe(self):
        """moe auxiliary loss: z-loss (squared logits) + inter-layer load balancing (variance of
        the mean routing distribution) + StableSkip-style cos regularization (layers barely
        changing the representation get less routing mass)."""
        route = self.route
        if route is None:
            return None
        cfg = self.mod_config
        loss = None
        valid = route.valid.to(route.logits.device)
        nv = valid.sum().clamp_min(1.0)
        if cfg.z_loss_coef > 0:
            lse = torch.logsumexp(route.logits, dim=-1)  # [B,T]
            z = (lse.pow(2) * valid).sum() / nv
            loss = cfg.z_loss_coef * z if loss is None else loss + cfg.z_loss_coef * z
        if cfg.balance_loss_coef > 0:
            # Inter-layer variance of the mean routing distribution: prevents routing collapse
            # onto a few layers (collapsed layers would be absorbed via profile promotion)
            q = (route.probs * valid.unsqueeze(-1)).sum(dim=(0, 1)) / nv  # [G]
            var = q.var(unbiased=False)
            loss = cfg.balance_loss_coef * var if loss is None else loss + cfg.balance_loss_coef * var
        if cfg.cos_reg_coef > 0:
            # sim is detached (zero-padding masked); gradients flow only through the soft
            # probabilities into the router
            terms = [self._cos_term(w) for w in self._gated() if w._last_sim is not None]
            if terms:
                cos = torch.stack(terms).mean()
                loss = cfg.cos_reg_coef * cos if loss is None else loss + cfg.cos_reg_coef * cos
        return loss

    def _stack_masks(self):
        """Returns ste/weights[B,T,G], soft[B,T,G], hard[B,T,G] (stacked on one device —
        under placement the per-layer slices may live on different devices)."""
        ls = self._gated()
        if not ls:
            return None, None, None

        def _stack(get):
            return _to_common([get(w).squeeze(-1) for w in ls], dim=-1)

        return (_stack(lambda w: w.last_gating_output.mask),
                _stack(lambda w: w.last_gating_output.soft_mask),
                _stack(lambda w: w.last_gating_output.hard_mask))

    def get_active_counts(self, hard: bool = True):
        """Per-token active layer count [B,T] (hard or soft accounting)."""
        ste, soft, hard_m = self._stack_masks()
        m = hard_m if hard else soft
        return m.sum(-1) if m is not None else None

    def get_soft_counts(self):
        return self.get_active_counts(hard=False)

    # ----- Budget -----

    def get_budget_loss(self, attention_mask=None, lambda_map=None):
        """Pure Lagrangian budget loss (differentiable). Price accounting = actual demand per
        inference (per-token average activation memory), peak/residency ignored.
        threshold: loss = core(k) + over-kmax penalty, k via STE;
        moe: loss = core(k_soft), k via soft inclusion (in topk mode k is fixed, returns None).
        core = _budget_core per cfg.budget_form (mean|hinge|tail).
        lambda_map (ed8): optional per-token λ multiplier [B,T] from difficulty shaping
        (dual.difficulty_mult); the loss becomes λ·mean(map·k) — the dual still owns the
        global level, the map only redistributes pressure across tokens. None = legacy path."""
        cfg = self.mod_config
        if cfg.gate_mode == "moe":
            route = self.route
            if route is None or cfg.sparsity_price <= 0 or cfg.select_mode != "topp":
                return None
            valid = route.k_soft > 0 if attention_mask is None \
                else attention_mask.bool().to(route.k_soft.device)
            if not bool(valid.any()):
                return None
            k = route.k_soft[valid]
            kk = k if lambda_map is None else lambda_map.to(k.device).float()[valid] * k
            return _budget_core(kk, cfg, cfg.sparsity_price)
        ste, _, _ = self._stack_masks()
        if ste is None:
            return None
        k = ste.sum(-1)  # [B,T] differentiable (via STE)
        valid = attention_mask.bool() if attention_mask is not None \
            else torch.ones_like(k, dtype=torch.bool)
        loss = 0
        if bool(valid.any()):
            if cfg.over_budget_coef > 0:
                over = torch.clamp(k[valid] - cfg.kmax, min=0)
                loss = loss + cfg.over_budget_coef * over.pow(2).mean()
            if cfg.sparsity_price > 0:
                # elastic budget core: charge per layer, λ presses the form statistic
                kk = k[valid]
                if lambda_map is not None:
                    kk = lambda_map.to(kk.device).float()[valid] * kk
                loss = loss + _budget_core(kk, cfg, cfg.sparsity_price)
        return loss

    def adapt_price(self, ema_acc: float | None):
        """Dual adjustment: relax pricing when accuracy misses the target, push sparsity when
        there is headroom. The caller handles warmup."""
        cfg = self.mod_config
        if not cfg.price_adapt or ema_acc is None or cfg.acc_target is None:
            return
        if ema_acc < cfg.acc_target:
            cfg.sparsity_price *= 1.0 - cfg.adapt_rate
        else:
            cfg.sparsity_price *= 1.0 + cfg.adapt_rate
        cfg.sparsity_price = min(max(cfg.sparsity_price, cfg.price_min), cfg.price_max)

    def get_router_parameters(self):
        """Gating parameters: threshold = per-layer Router+tau+comp; moe = single-point JointRouter."""
        if self.mod_config.gate_mode == "moe":
            if self.joint_router is None:
                return []
            return list(self.joint_router.parameters())
        seen = set()
        ps = []

        def _add(t):
            if id(t) not in seen:
                seen.add(id(t))
                ps.append(t)

        for w in self.layers:
            if w.is_always_on:
                continue
            for p in w.router.parameters():
                _add(p)
            _add(w.tau)
            if w.comp is not None:
                _add(w.comp)
        return ps

    # ----- Diagnostics -----

    def get_layer_sparsity(self):
        d = {}
        for i, w in enumerate(self.layers):
            if w.is_always_on:
                d[i] = 0.0
            else:
                o = w.last_gating_output
                d[i] = 1.0 - o.hard_mask.float().mean().item() if o else float("nan")
        return d

    def get_layer_usage(self, reset: bool = True):
        """Per-layer usage rate over the window {idx:(hard rate, soft rate)}, always_on recorded
        as 1.0. Resets on read. The GPU->CPU sync happens only here (read time), not per step."""
        d = {}
        for w in self.layers:
            if w.is_always_on:
                d[w.layer_idx] = (1.0, 1.0)
            else:
                n = max(w._use_n, 1)
                hard = float(w._use_hard) if torch.is_tensor(w._use_hard) else 0.0
                soft = float(w._use_soft) if torch.is_tensor(w._use_soft) else 0.0
                d[w.layer_idx] = (hard / n, soft / n)
                if reset:
                    w._use_hard = w._use_soft = None
                    w._use_n = 0
        return d

    def get_skip_hits(self, reset: bool = False) -> int:
        """Cumulative decode-time sparse skip count (summed over layers)."""
        n = sum(w._skip_hits for w in self.layers)
        if reset:
            for w in self.layers:
                w._skip_hits = 0
        return n

    def pop_layer_counts(self) -> dict:
        """Raw per-layer hard-activation token counts since the last read (read-and-reset,
        gated layers only). Feeds the placement scheduler's LFU/LRU statistics; resets the
        same accumulators get_layer_usage reads (whoever reads first wins the window)."""
        d = {}
        for w in self.layers:
            if w.is_always_on:
                continue
            d[w.layer_idx] = int(w._use_hard.item()) if torch.is_tensor(w._use_hard) else 0
            w._use_hard = w._use_soft = None
            w._use_n = 0
        return d

    def get_tau_params(self):
        """threshold: current tau parameter values (diagnosing imbalance); moe: empty dict."""
        if self.mod_config.gate_mode != "threshold":
            return {}
        return {w.layer_idx: float(w.tau.detach()) for w in self.layers if not w.is_always_on}

    def calibrate_tau(self, batches, spread: float = 0.5):
        """threshold only: StableSkip-style model-aware tau initialization (moe returns {}).

        Body lives in speaker/calibrate.py; this delegate keeps the call site stable."""
        from .calibrate import calibrate_tau_for
        return calibrate_tau_for(self, batches, spread=spread)

    def calibrate_router_temp(self, batches, target_k: Optional[int] = None):
        """moe only: init-time JointRouter logit-temperature calibration (threshold returns {}).

        Body lives in speaker/calibrate.py; this delegate keeps the call site stable."""
        from .calibrate import calibrate_router_temp_for
        return calibrate_router_temp_for(self, batches, target_k=target_k)

    # ----- Placement (bodies in speaker/placement.py; delegates keep call sites stable) -----

    def set_placement(self, resident_ids, gpu_device="cuda", cpu_device="cpu"):
        """Hierarchical placement: hot layers on GPU, cold layers on CPU (always_on forced GPU)."""
        from .placement import apply_placement
        return apply_placement(self, resident_ids, gpu_device, cpu_device)

    def resident_gb(self, device_type="cuda"):
        """Resident parameter size on the given device (GB, weights only)."""
        from .placement import resident_gb_of
        return resident_gb_of(self, device_type)

    def schedule_placement(self, strategy: str = "lfu", **kwargs):
        """Scheduled placement for CPU-GPU collaborative inference (ed5). Plans and applies
        once, returns the LayerScheduler; call sched.reschedule() between generations."""
        from .placement import schedule_placement_for
        return schedule_placement_for(self, strategy, **kwargs)

    # ----- Inference / annealing controls -----

    def set_skip_mode(self, mode: str):
        self.mod_config.skip_mode = mode

    def anneal(self, step: int, total: int, ta_end: float = 0.3, g_end: float = 0.0):
        p = min(max(step / max(total, 1), 0.0), 1.0)
        self.mod_config.temp_affinity = self._ta_start + (ta_end - self._ta_start) * p
        self.mod_config.gumbel_scale = self._g_start + (g_end - self._g_start) * p


def _rd_to(route: RouteDecision, dev: torch.device) -> RouteDecision:
    return _dc_replace(route, **{
        f.name: getattr(route, f.name).to(dev)
        for f in RouteDecision.__dataclass_fields__.values()
        if isinstance(getattr(route, f.name), torch.Tensor)})


def convert_to_speaker(hf_model, mod_config=None, hf_config=None, **overrides):
    if mod_config is None:
        from .config import SpeakerConfig
        src = hf_config if hf_config is not None else hf_model.config
        mod_config = SpeakerConfig.from_model_config(src, **overrides)
    else:
        for k, v in overrides.items():
            if hasattr(mod_config, k):
                setattr(mod_config, k, v)
    return SpeakerModelWrapper(hf_model, mod_config)


def apply_decode_config(speaker_model, decode) -> dict:
    """Applies a ckpt-shipped decode recipe (ed9/C: SpeakerConfig.decode) onto the
    underlying HF model's generation_config, so the artifact carries its validated
    inference behavior (no reliance on the caller remembering CLI flags).
    None/{} = legacy no-op. Only known GenerationConfig fields are set; unknown
    keys warn loudly. Explicit generate() kwargs still override these defaults.
    Returns the applied {field: value} mapping."""
    if not decode:
        return {}
    target = getattr(speaker_model, "hf_model", speaker_model)
    gc = getattr(target, "generation_config", None)
    if gc is None:
        logger.warning(f"decode recipe {decode} has nowhere to go "
                       f"({type(target).__name__} has no generation_config) — ignored")
        return {}
    applied = {}
    for k, v in decode.items():
        if hasattr(gc, k):
            setattr(gc, k, v)
            applied[k] = v
        else:
            logger.warning(f"decode key {k!r} is not a GenerationConfig field — ignored")
    if applied:
        logger.info(f"[decode] applied {applied} to {type(target).__name__}.generation_config")
    return applied


__all__ = [
    "SpeakerModelWrapper", "convert_to_speaker", "apply_decode_config", "_to_common",
    "_budget_core",
]
