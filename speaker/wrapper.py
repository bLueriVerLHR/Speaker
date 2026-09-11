"""Model wrapping: SpeakerLayerWrapper (fixed-layer passthrough / gated-layer mixing +
sparse KV cache) and SpeakerModelWrapper (routing hub / statistics / budget / placement).

Dual scheme coexists (config.gate_mode switch, the two schemes' checkpoints are not
interchangeable):
- threshold (legacy scheme): per-layer independent Router+tau+comp, sigmoid threshold gating + STE;
- moe (default): the gated-region entry single-point JointRouter does joint routing; the entry
  layer computes log p and the selection for all G gated layers in its own forward, stores them
  in the hub (SpeakerModelWrapper.route), and subsequent gated layers consume their slot slices;
  the selected layers' probabilities are renormalized into weighted residuals.

Deployment forms (mainly the decode stage, consistent across both schemes):
- Sufficient VRAM: hard gating layer-skipping + sparse KV cache, fewer active layers for lower latency;
- Insufficient VRAM: set_placement hierarchical placement, fixed + high-load layers resident on GPU,
  low-load layers on CPU;
- prefill: degenerates to dense like MoE (no skipping, cache fully written) — only slower, never wrong.
Old class names MoDLayerWrapper/MoDModelWrapper remain exported as aliases.
"""
from __future__ import annotations

from dataclasses import replace as _dc_replace
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .gating import GatingOutput, JointRouter, RouteDecision, Router, select_and_weight


class SpeakerLayerWrapper(nn.Module):
    """Gated-layer wrapper (dual scheme):
    - threshold: per-layer Router(H->1)+tau(+comp), logit=(router_logit-tau)/Ta(+Gumbel), STE;
    - moe: no per-layer parameters, consumes the joint-routing slice from the hub (the entry
      layer computes the routing).
    Sparse KV cache: during decode, gate closed / layer not selected -> no execution and no K/V
    written (per-layer cache lengths may diverge).
    """

    def __init__(self, layer: nn.Module, layer_idx: int, config, hub=None):
        super().__init__()
        self.layer = layer
        self.layer_idx = layer_idx
        self.config = config
        self.is_always_on = config.is_always_on(layer_idx)
        # moe: the hub uses object.__setattr__ so nn.Module does not register the model wrapper
        # as a submodule (otherwise state_dict would show duplicate registration paths
        # layers.<i>._hub.*); always None for threshold
        object.__setattr__(self, "_hub", hub)
        self._slot = 0        # moe: index within the gated-layer sequence (routing slice index)
        self._is_entry = False  # moe: first layer of the gated region, computes the joint routing
        self._last: Optional[GatingOutput] = None
        self._last_sim: Optional[torch.Tensor] = None  # detached layer input/output cos similarity [B,T]
        self._returns_tuple = True    # HF decoder layers return a tuple by default; assume True, correct after execution
        self._skip_hits = 0           # decode-time sparse skip counter (diagnostics)
        # Hot-path gates (ed5): usage stats accumulate as GPU-side fp64 tensors — the
        # historical per-step float() forced a GPU->CPU pipeline sync on every gated layer
        # every step; values are synced to float only when get_layer_usage reads them.
        # _need_sim skips the cosine-similarity diagnostics when nothing consumes them
        # (moe with cos_reg_coef=0; threshold always keeps it — calibrate_tau reads
        # _last_sim for its initialization).
        self._need_sim = (not self.is_always_on) and (
            config.gate_mode == "threshold" or config.cos_reg_coef > 0)
        # Load accumulators (hard/soft usage over a window; single-step noise at batch=1 is
        # large, read window means). fp64 tensor sums are bit-identical to the historical
        # Python-float accumulation (fp32 per-step sum widened exactly, then added in fp64).
        self._use_hard = None   # Optional[torch.Tensor], created on first gated forward
        self._use_soft = None   # Optional[torch.Tensor]
        self._use_n = 0
        if self.is_always_on:
            self.router = None
            self.tau = None
            self.comp = None
        elif config.gate_mode == "threshold":
            self.router = Router(config.hidden_size, config.router_hidden_dim)
            self.tau = nn.Parameter(torch.tensor(float(config.tau_init)))
            # Skip compensation (DASH-style): learnable bias added when skipped, zero init = legacy behavior
            self.comp = nn.Parameter(torch.zeros(config.hidden_size))
        else:  # moe
            self.router = None
            self.tau = None
            self.comp = None

    # ---------- Gating (threshold) ----------

    def _compute_gating(self, hidden_states: torch.Tensor,
                        attention_mask: Optional[torch.Tensor]) -> GatingOutput:
        hf = hidden_states.float() if hidden_states.dtype in (torch.bfloat16, torch.float16) else hidden_states
        r_logits = self.router(hf)
        Ta = max(float(self.config.temp_affinity), 0.05)
        base = (r_logits - self.tau) / Ta
        if self.training and float(self.config.gumbel_scale) > 0:
            u = torch.rand_like(base).clamp_(1e-6, 1.0 - 1e-6)
            gumbel = -torch.log(-torch.log(u))
            base = base + gumbel * float(self.config.gumbel_scale)
        soft = torch.sigmoid(base)
        if attention_mask is not None and attention_mask.dim() == 2:
            soft = soft * attention_mask.to(soft.dtype).unsqueeze(-1)  # padding forced to 0, consumes no budget
        hard = (soft >= 0.5).float()
        if self.config.skip_mode == "hard":
            mask = hard
        elif self.config.use_ste:
            mask = soft + (hard - soft).detach()  # forward = hard count, backward through soft
        else:
            mask = soft
        aux = None
        if self.config.z_loss_coef > 0:
            aux = self.config.z_loss_coef * r_logits.pow(2).mean()
        with torch.no_grad():
            # GPU-side fp64 accumulation, no per-step sync (see __init__ note)
            if self._use_hard is None:
                self._use_hard = hard.sum().double()
                self._use_soft = soft.sum().double()
            else:
                self._use_hard += hard.sum()
                self._use_soft += soft.sum()
            self._use_n += hard.numel()
        return GatingOutput(mask=mask, soft_mask=soft, hard_mask=hard,
                            router_logits=r_logits, aux_loss=aux)

    # ---------- Route slicing (moe) ----------

    def _consume_route(self, dev: Optional[torch.device] = None) -> GatingOutput:
        hub = self._hub
        if hub is None:
            raise RuntimeError("gated layers in moe mode must be built via convert_to_speaker (joint routing requires the hub)")
        route = hub.route
        if route is None:
            raise RuntimeError("missing routing decision: gated layers must run inside a full model forward (entry layer first)")
        j = self._slot
        w, s, h, lg = (route.weights[..., j:j + 1], route.probs[..., j:j + 1],
                       route.selected[..., j:j + 1], route.logits[..., j:j + 1])
        if dev is not None and w.device != dev:
            # placement: the route lives on the entry layer's device; realign this
            # layer's slice so mixing/stepping and the usage stats stay on-device.
            # Single-device forwards take the no-move path (bit-identical to history).
            w, s, h, lg = w.to(dev), s.to(dev), h.to(dev), lg.to(dev)
        return GatingOutput(mask=w, soft_mask=s, hard_mask=h, router_logits=lg)

    # ---------- Skip path (shared by both schemes) ----------

    @staticmethod
    def _is_decode_step(past, layer_idx) -> bool:
        """Object-style cache with content for this layer <=> decode step (may skip, no K/V written).
        legacy tuple cache / no cache / probe failure -> False, caller falls back to full execution."""
        get_len = getattr(past, "get_seq_length", None)
        if get_len is None:
            return False
        try:
            return get_len(layer_idx) > 0
        except Exception:
            return False

    def _try_skip(self, hidden_states: torch.Tensor, kwargs: dict):
        """Attempts to skip the whole layer when hard is all zero (gates fully closed / no token
        selected this layer). Returns None meaning full-execution fallback is required
        (only slower, never wrong)."""
        skipped_hidden = hidden_states
        if self.comp is not None:
            skipped_hidden = hidden_states + self.comp.to(hidden_states.device,
                                                           hidden_states.dtype)
        past = kwargs.get("past_key_value", kwargs.get("past_key_values", None))  # newer HF uses the plural key
        if not self.training:
            if self._is_decode_step(past, self.layer_idx):
                # decode step: sparse KV cache — no execution and no K/V written, this layer's
                # cache length lags behind the global one
                self._skip_hits += 1
                return (skipped_hidden,) if self._returns_tuple else skipped_hidden
            if past is None and kwargs.get("use_cache", False) is not True:
                # no cache (training / cacheless prefill / per-sentence recompute): skip directly, nothing to align
                return (skipped_hidden,) if self._returns_tuple else skipped_hidden
            return None  # prefill with cache: execute to guarantee the cache is fully written
        # training-time hard: keep the legacy behavior (no cache; gradient_checkpointing already disables it)
        if self._returns_tuple:
            return (skipped_hidden, past) if past is not None else (skipped_hidden,)
        return skipped_hidden

    def _run_and_mix(self, hidden_states: torch.Tensor, mask: torch.Tensor,
                     *args, attention_mask, **kwargs):
        """Executes the real layer and applies gated mixing: h_out = h_in + m·(F_l(h_in) - h_in)
        (+ (1-m)·comp). Also records the detached input/output cosine similarity (StableSkip-style:
        layers barely changing the representation should be skipped)."""
        layer_out = self.layer(hidden_states, *args, attention_mask=attention_mask, **kwargs)
        self._returns_tuple = isinstance(layer_out, tuple)
        layer_hidden = layer_out[0] if self._returns_tuple else layer_out
        rest = layer_out[1:] if self._returns_tuple else ()
        m = mask.to(layer_hidden.dtype)
        mixed = hidden_states + m * (layer_hidden - hidden_states)
        if self.comp is not None:
            mixed = mixed + (1.0 - m) * self.comp.to(layer_hidden.dtype)
        if self._need_sim:
            with torch.no_grad():
                sim = F.cosine_similarity(hidden_states.float(), layer_hidden.float(), dim=-1)  # [B,T]
                if attention_mask is not None and attention_mask.dim() == 2:
                    sim = sim * attention_mask.to(sim.dtype)
                self._last_sim = sim.detach()
        return (mixed, *rest) if self._returns_tuple else mixed

    def forward(self, hidden_states: torch.Tensor, *args, attention_mask=None, **kwargs):
        # placement level: automatic moves across device boundaries (hot layers GPU / cold layers
        # CPU), output stays on the compute device
        dev = self._layer_device()
        hidden_states, attention_mask, kwargs = self._move_layer_inputs(
            hidden_states, attention_mask, kwargs, dev)
        if self.is_always_on:
            return self.layer(hidden_states, *args, attention_mask=attention_mask, **kwargs)
        if self.config.gate_mode == "moe":
            return self._forward_moe(hidden_states, *args, attention_mask=attention_mask, **kwargs)
        # ---- threshold (legacy scheme) ----
        self._last = self._compute_gating(hidden_states, attention_mask)
        if self.config.skip_mode == "hard" and bool((self._last.hard_mask == 0).all()):
            skipped = self._try_skip(hidden_states, kwargs)
            if skipped is not None:
                return skipped
        return self._run_and_mix(hidden_states, self._last.mask, *args,
                                 attention_mask=attention_mask, **kwargs)

    def _forward_moe(self, hidden_states: torch.Tensor, *args, attention_mask=None, **kwargs):
        """moe: entry layer computes the joint routing -> each layer consumes its own slice ->
        weighted residual / skip."""
        if self._is_entry:
            self._hub._compute_route(hidden_states, attention_mask)
        g = self._consume_route(hidden_states.device)
        self._last = g
        with torch.no_grad():
            # GPU-side fp64 accumulation, no per-step sync (see __init__ note)
            if self._use_hard is None:
                self._use_hard = g.hard_mask.sum().double()
                self._use_soft = g.soft_mask.sum().double()
            else:
                self._use_hard += g.hard_mask.sum()
                self._use_soft += g.soft_mask.sum()
            self._use_n += g.hard_mask.numel()
        if self.config.skip_mode == "hard" and bool((g.hard_mask == 0).all()):
            skipped = self._try_skip(hidden_states, kwargs)
            if skipped is not None:
                return skipped
        return self._run_and_mix(hidden_states, g.mask, *args,
                                 attention_mask=attention_mask, **kwargs)

    @property
    def last_gating_output(self) -> Optional[GatingOutput]:
        return self._last

    # ---------- Cross-device ----------

    def _layer_device(self) -> torch.device:
        try:
            return next(self.layer.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    @staticmethod
    def _to_dev(t, dev):
        if isinstance(t, torch.Tensor):
            return t.to(dev) if t.device != dev else t
        if isinstance(t, (tuple, list)):
            return type(t)(SpeakerLayerWrapper._to_dev(x, dev) for x in t)
        return t

    def _move_layer_inputs(self, hidden_states, attention_mask, kwargs, dev):
        """Placement level: automatic moves across device boundaries (including position_embeddings
        cos/sin tuples)."""
        hidden_states = self._to_dev(hidden_states, dev)
        attention_mask = self._to_dev(attention_mask, dev)
        kwargs = {k: (self._to_dev(v, dev) if k in ("position_ids", "cache_position", "position_embeddings") else v)
                  for k, v in kwargs.items()}
        return hidden_states, attention_mask, kwargs


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
        sees through to the real layers."""
        N = self.mod_config.num_hidden_layers
        for path in (["model", "layers"], ["layers"], ["transformer", "h"]):
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
            if isinstance(m, nn.ModuleList) and len(m) == N and hasattr(m[0], "self_attn"):
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

    def get_aux_loss(self):
        if self.mod_config.gate_mode == "moe":
            return self._aux_loss_moe()
        ls = self._gated()
        if not ls:
            return None
        z = [w.last_gating_output.aux_loss for w in ls if w.last_gating_output.aux_loss is not None]
        loss = torch.stack(z).mean() if z else 0
        if self.mod_config.balance_loss_coef > 0 and len(ls) > 1:
            means = torch.stack([w.last_gating_output.soft_mask.float().mean() for w in ls])
            loss = loss + self.mod_config.balance_loss_coef * means.var(unbiased=False)
        if self.mod_config.cos_reg_coef > 0:
            # StableSkip-style: tokens whose representation barely changes should be skipped
            # (high sim -> low soft). sim is detached; gradients flow only through soft
            # into router/tau.
            terms = [(w.last_gating_output.soft_mask.squeeze(-1).float()
                      * w._last_sim.to(w.last_gating_output.soft_mask.device)).mean()
                     for w in ls
                     if w.last_gating_output is not None and w._last_sim is not None]
            if terms:
                loss = loss + self.mod_config.cos_reg_coef * torch.stack(terms).mean()
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
            terms = []
            for w in self._gated():
                if w._last_sim is None:
                    continue
                q_j = w.last_gating_output.soft_mask.squeeze(-1).float()
                terms.append((q_j * w._last_sim.to(q_j.device)).mean())
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
            ms = [get(w).squeeze(-1) for w in ls]
            d0 = ms[0].device
            if any(m.device != d0 for m in ms):
                ms = [m.to(d0) for m in ms]
            return torch.stack(ms, dim=-1)

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
        threshold: loss = λ·mean(k) + over-kmax penalty, k via STE;
        moe: loss = λ·mean(k_soft), k via soft inclusion (in topk mode k is fixed, returns None).
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
            if lambda_map is None:
                return cfg.sparsity_price * k.mean()
            w = lambda_map.to(k.device).float()[valid]
            return cfg.sparsity_price * (w * k).mean()
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
                # elastic budget core: charge per layer, λ presses mean(k)
                kk = k[valid]
                if lambda_map is not None:
                    kk = lambda_map.to(kk.device).float()[valid] * kk
                loss = loss + cfg.sparsity_price * kk.mean()
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
        """threshold: current τ parameter values (diagnosing imbalance); moe: empty dict."""
        if self.mod_config.gate_mode != "threshold":
            return {}
        return {w.layer_idx: float(w.tau.detach()) for w in self.layers if not w.is_always_on}

    def calibrate_tau(self, batches, spread: float = 0.5):
        """threshold only: StableSkip-style model-aware τ initialization (moe returns {})."""
        if self.mod_config.gate_mode != "threshold":
            return {}
        sims: dict = {}
        was_training = self.training
        self.eval()
        with torch.no_grad():
            for b in batches:
                self(**b)
                for w in self.layers:
                    if w.is_always_on:
                        continue
                    s = getattr(w, "_last_sim", None)
                    if s is not None and s.numel():
                        sims.setdefault(w.layer_idx, []).append(float(s.sum() / max((s != 0).sum(), 1)))
        if was_training:
            self.train()
        if not sims:
            return {}
        keys = sorted(sims)
        vals = [sum(sims[k]) / len(sims[k]) for k in keys]
        mu = sum(vals) / len(vals)
        sd = (sum((v - mu) ** 2 for v in vals) / len(vals)) ** 0.5 + 1e-6
        out = {}
        for k, v in zip(keys, vals):
            w = self.layers[k]
            w.tau.data.add_(torch.tensor(spread * (v - mu) / sd,
                                         device=w.tau.device, dtype=w.tau.dtype))
            out[k] = float(w.tau.detach())
        self.get_layer_usage()  # clear usage accumulated during calibration
        return out

    def calibrate_router_temp(self, batches, target_k: Optional[int] = None):
        """moe only: init-time JointRouter logit-temperature calibration so the initial top-p
        mean k starts near target_k (threshold returns {}). Mirror of calibrate_tau's
        philosophy: enter the budget-ramp window from a near-dense routing distribution.
        The temperature is persisted in SpeakerConfig (mod_config.json), not in state_dict —
        gate.pt key sets stay unchanged; 1.0 (default / legacy ckpts) = identity."""
        if self.mod_config.gate_mode != "moe" or self.joint_router is None:
            return {}
        cfg = self.mod_config
        G = len(cfg.gated_layers)
        lo_k = max(int(cfg.min_layers), 1)
        hi_k = min(int(cfg.kmax), G)
        if target_k is None:
            target_k = max(lo_k, min(hi_k, round(0.6 * G)))
        target_k = int(max(lo_k, min(hi_k, int(target_k))))
        was_training = self.training
        self.eval()
        self._calib_capture = []
        with torch.no_grad():
            for b in batches:
                self(**b)
            captured = self._calib_capture
            self._calib_capture = None
            if not captured:
                if was_training:
                    self.train()
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
            raw = self.joint_router(h)  # [N,G] fp32, no gumbel (eval)
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
            self.train()
        self.get_layer_usage()  # clear usage accumulated during calibration
        return {"router_temp": float(t_best), "k_before": k_before,
                "k_after": k_after, "target_k": target_k}

    # ----- Placement -----

    def set_placement(self, resident_ids, gpu_device="cuda", cpu_device="cpu"):
        """Hierarchical placement: hot layers (resident + high frequency) on GPU, cold layers on
        CPU. forward moves tensors across devices automatically.
        resident_ids: layer indices kept on GPU; the remaining gated layers move to CPU
        (always_on forced to stay on GPU).
        moe's JointRouter follows the hf_model's main device; _compute_route aligns devices internally."""
        resident = set(resident_ids) | set(self.mod_config.always_on_layers)
        gd = torch.device(gpu_device)
        cd = torch.device(cpu_device)
        for w in self.layers:
            w.to(gd if w.layer_idx in resident else cd)
        return sorted(resident)

    def resident_gb(self, device_type="cuda"):
        """Resident parameter size on the given device (GB, weights only)."""
        n = sum(p.numel() * p.element_size() for p in self.parameters() if p.device.type == device_type)
        n += sum(b.numel() * b.element_size() for b in self.buffers() if b.device.type == device_type)
        return n / 1e9

    def schedule_placement(self, strategy: str = "lfu", **kwargs):
        """Scheduled placement for CPU-GPU collaborative inference (ed5): fixed layers stay
        resident on the GPU, gated layers are scheduled into the leftover weights budget
        (GPU allowance minus a KV-cache/activation reserve) by a pluggable strategy
        (random | lru | lfu — see speaker/scheduler.py). Plans and applies once, returns
        the LayerScheduler; call sched.reschedule() between generations to re-plan from
        the observed activation counters."""
        from .scheduler import LayerScheduler
        sched = LayerScheduler(self, strategy, **kwargs)
        sched.reschedule()
        return sched

    # ----- Inference / annealing controls -----

    def set_skip_mode(self, mode: str):
        self.mod_config.skip_mode = mode

    def set_temperatures(self, ta: float | None = None, gumbel: float | None = None):
        if ta is not None:
            self.mod_config.temp_affinity = float(ta)
        if gumbel is not None:
            self.mod_config.gumbel_scale = float(gumbel)

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
        print(f"WARNING: decode recipe {decode} has nowhere to go "
              f"({type(target).__name__} has no generation_config) — ignored", flush=True)
        return {}
    applied = {}
    for k, v in decode.items():
        if hasattr(gc, k):
            setattr(gc, k, v)
            applied[k] = v
        else:
            print(f"WARNING: decode key {k!r} is not a GenerationConfig field — ignored",
                  flush=True)
    if applied:
        print(f"[decode] applied {applied} to {type(target).__name__}.generation_config",
              flush=True)
    return applied
