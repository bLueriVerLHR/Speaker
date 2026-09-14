"""Gated decoder-layer wrapper (split from wrapper.py, P0 structure).

SpeakerLayerWrapper: fixed-layer passthrough / gated-layer mixing + sparse KV
cache. Dual scheme (config.gate_mode switch):
- threshold (legacy): per-layer independent Router+tau+comp, sigmoid threshold
  gating + STE;
- moe (default): no per-layer parameters, consumes the joint-routing slice from
  the hub (the entry layer computes the routing).

Import path compatibility: ``speaker.wrapper`` re-exports this class, and
``speaker/__init__.py`` keeps exporting it, so existing imports are unaffected.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .gating import GatingOutput, Router


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
        self._accum_use(hard, soft)
        return GatingOutput(mask=mask, soft_mask=soft, hard_mask=hard,
                            router_logits=r_logits, aux_loss=aux)

    def _accum_use(self, hard, soft):
        with torch.no_grad():
            # GPU-side fp64 accumulation, no per-step sync (see __init__ note)
            if self._use_hard is None:
                self._use_hard = hard.sum().double()
                self._use_soft = soft.sum().double()
            else:
                self._use_hard += hard.sum()
                self._use_soft += soft.sum()
            self._use_n += hard.numel()

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
        legacy tuple cache / no cache / probe failure -> False, caller falls back to full execution.
        Hybrid caches (linear+full mix, e.g. Qwen3_5): KV-tracking layers answer via
        get_seq_length(layer_idx); linear-attention layers carry recurrent/conv state instead
        (get_seq_length raises) and answer via has_previous_state(layer_idx)."""
        get_len = getattr(past, "get_seq_length", None)
        if get_len is None:
            return False
        try:
            return get_len(layer_idx) > 0
        except Exception:
            pass
        has_prev = getattr(past, "has_previous_state", None)
        if has_prev is not None:
            try:
                return bool(has_prev(layer_idx))
            except Exception:
                return False
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
        self._accum_use(g.hard_mask, g.soft_mask)
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


__all__ = ["SpeakerLayerWrapper"]
