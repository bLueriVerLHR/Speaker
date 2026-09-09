"""Shared mechanism library for the baselines (ed5 convergence): layer patches + statistics + eval adapters for MoD / RT / MoDification.

train_*.py keeps only the CLI and the training loop; eval_compare / eval_gen / probe_* take the mechanism functions from here.
Third-party read-only code: RT's routing math primitives still come from baselines/router-tuning/ (gitignored),
layer forwards rewritten for the newer transformers convention (mathematically equivalent, see the original notes in train_rt).

Eval adapter (valid="labels" protocol, consistent with the three training scripts; equivalent to attention_mask under the plain collate):
k_provider returns the per-token total active layer count (routed + fixed layers), aggregation goes through speaker.evaluate.eval_heldout,
loss/acc values are bit-identical to the original three eval_heldout_* functions (same formula, same denominator).
"""
from __future__ import annotations

import json
import os
import sys
from types import MethodType

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "router-tuning"))
from utils.model.model_patch import (  # noqa: E402
    _apply_routing_mask,
    _compute_routing_state,
    _parse_granularity,
)

from speaker.checkpoint import gate_state_dict, is_base_key, strip_wrapper_prefix  # noqa: E402
from speaker.evaluate import eval_heldout  # noqa: E402

# Baseline gate key markers (only router is attached, no tau/comp; gate_state_dict's superset filter would not hit extra keys here)
GATE_KEYS = ("router",)


# ==================== MoD original (token-choice top-k capacity scheme) ====================

def _select_topk(r: torch.Tensor, valid: torch.Tensor | None, capacity: float) -> torch.Tensor:
    """Per-sample top-k selection mask [B,T] bool. k = clamp(round(cap × number of valid tokens), 1, number of valid tokens)."""
    B, T = r.shape
    scores = r if valid is None else r.masked_fill(~valid, float("-inf"))
    n_valid = valid.sum(-1) if valid is not None else torch.full((B,), T, device=r.device, dtype=torch.long)
    sel = torch.zeros_like(r, dtype=torch.bool)
    for b in range(B):
        nv = int(n_valid[b])
        if nv <= 0:
            continue  # all-padding sample: select no tokens
        kk = int(torch.clamp(torch.round(torch.tensor(float(capacity) * nv)), 1, nv))
        if kk >= T:
            sel[b] = torch.ones(T, dtype=torch.bool, device=r.device) if valid is None else valid[b]
        else:
            th = scores[b].topk(kk).values[-1]
            sel[b] = scores[b] >= th
    return sel


def _mod_layer_forward(self, hidden_states, attention_mask=None, position_ids=None,
                       past_key_values=None, use_cache=False, position_embeddings=None, **kwargs):
    """MoD layer: dense execution + mask-mixing simulation. x' = x + sel·r·(block(x) − x)."""
    self._last_sel = None
    self._last_rlogits = None
    self._last_bce = None
    if not self.is_routed:
        return self._mod_dense_forward(hidden_states, attention_mask=attention_mask,
                                       position_ids=position_ids, past_key_values=past_key_values,
                                       use_cache=use_cache, position_embeddings=position_embeddings,
                                       **kwargs)
    hf = hidden_states.float() if hidden_states.dtype in (torch.bfloat16, torch.float16) else hidden_states
    r = self.router(hf).squeeze(-1)  # [B,T] raw weight (paper: no sigmoid, raw values go straight onto the gradient path)
    valid = attention_mask.bool() if attention_mask is not None and attention_mask.dim() == 2 else None
    sel = _select_topk(r, valid, self.route_capacity)
    out = self._mod_dense_forward(hidden_states, attention_mask=attention_mask,
                                  position_ids=position_ids, past_key_values=past_key_values,
                                  use_cache=use_cache, position_embeddings=position_embeddings,
                                  **kwargs)
    returns_tuple = isinstance(out, tuple)
    layer_hidden = out[0] if returns_tuple else out
    delta = layer_hidden - hidden_states
    w = (r.to(hidden_states.dtype).unsqueeze(-1)
         * sel.unsqueeze(-1).to(hidden_states.dtype))  # [B,T,1] weighting + selection
    mixed = hidden_states + w * delta
    self._last_sel = sel.detach()
    self._last_rlogits = r.detach()
    if self.training:
        tgt = sel.to(r.dtype)
        m = valid if valid is not None else torch.ones_like(tgt, dtype=torch.bool)
        if bool(m.any()):
            self._last_bce = F.binary_cross_entropy_with_logits(r[m], tgt[m])
    return (mixed, *out[1:]) if returns_tuple else mixed


def patch_model_modd(model, is_routed, capacity=0.125):
    """Attach router + MoD forward layer by layer; returns the list of routed layers. BCE / capacity annealing are aggregated and controlled by the caller."""
    layers = model.model.layers
    routed = []
    for i, layer in enumerate(layers):
        layer.is_routed = bool(is_routed[i])
        layer.route_capacity = float(capacity)
        if layer.is_routed and not hasattr(layer, "router"):
            layer.router = nn.Linear(model.config.hidden_size, 1, bias=False)
            nn.init.normal_(layer.router.weight, std=0.02)
            # keep the router in fp32 (input hidden states are cast to fp32 as well before computing, same trick as our wrapper, avoids mixed-precision mismatch)
        layer._last_sel = None
        layer._last_rlogits = None
        layer._last_bce = None
        if not hasattr(layer, "_mod_dense_forward"):
            layer._mod_dense_forward = layer.forward  # original dense forward (saved as bound method)
        layer.forward = MethodType(_mod_layer_forward, layer)
        if layer.is_routed:
            routed.append(layer)
    return routed


def collect_modd_stats(routed, training: bool):
    """Returns (bce_loss_sum or None, per-token active layer count [B,T] or None)."""
    bce = None
    sels = []
    for layer in routed:
        if training and layer._last_bce is not None:
            bce = layer._last_bce if bce is None else bce + layer._last_bce
        if layer._last_sel is not None:
            sels.append(layer._last_sel)
    k = torch.stack(sels, dim=-1).float().sum(-1) if sels else None  # [B,T]
    return bce, k


# ==================== MoDification (threshold-p + R load target) ====================

def _select_threshold(g: torch.Tensor, valid: torch.Tensor | None, p: float) -> torch.Tensor:
    """Per-token absolute threshold [B,T] bool: selected <=> g>=p and the position is valid. No cross-token ranking."""
    sel = g >= p
    return sel & valid if valid is not None else sel


def _mdf_layer_forward(self, hidden_states, attention_mask=None, position_ids=None,
                       past_key_values=None, use_cache=False, position_embeddings=None, **kwargs):
    """MoDification layer: h' = h + sel·g·(block(h) − h). The R term backprops through G (soft mean)."""
    self._last_sel = None
    self._last_F = 0.0
    self._last_G = None
    if not self.is_routed:
        return self._mdf_dense_forward(hidden_states, attention_mask=attention_mask,
                                       position_ids=position_ids, past_key_values=past_key_values,
                                       use_cache=use_cache, position_embeddings=position_embeddings,
                                       **kwargs)
    hf = hidden_states.float() if hidden_states.dtype in (torch.bfloat16, torch.float16) else hidden_states
    g = torch.sigmoid(self.router(hf)).squeeze(-1)  # [B,T] paper: g ∈ [0,1]
    valid = attention_mask.bool() if attention_mask is not None and attention_mask.dim() == 2 else None
    sel = _select_threshold(g, valid, self.route_p)
    out = self._mdf_dense_forward(hidden_states, attention_mask=attention_mask,
                                  position_ids=position_ids, past_key_values=past_key_values,
                                  use_cache=use_cache, position_embeddings=position_embeddings,
                                  **kwargs)
    returns_tuple = isinstance(out, tuple)
    layer_hidden = out[0] if returns_tuple else out
    delta = layer_hidden - hidden_states
    w = (g.to(hidden_states.dtype).unsqueeze(-1)
         * sel.unsqueeze(-1).to(hidden_states.dtype))  # [B,T,1] shared-gate weighting + selection
    mixed = hidden_states + w * delta
    m = valid if valid is not None else torch.ones_like(sel)
    self._last_sel = sel.detach()
    self._last_F = float(sel[m].float().mean()) if bool(m.any()) else 0.0
    self._last_G = g[m].mean() if bool(m.any()) else None
    return (mixed, *out[1:]) if returns_tuple else mixed


def patch_model_mdf(model, is_routed, p=0.5):
    """Attach gating + MoDification forward layer by layer; returns the list of routed layers. R is aggregated and controlled by the caller."""
    layers = model.model.layers
    routed = []
    for i, layer in enumerate(layers):
        layer.is_routed = bool(is_routed[i])
        layer.route_p = float(p)
        if layer.is_routed and not hasattr(layer, "router"):
            layer.router = nn.Linear(model.config.hidden_size, 1)
            nn.init.zeros_(layer.router.weight)  # zero init: g=0.5, dense start with everything executing
            nn.init.zeros_(layer.router.bias)
        layer._last_sel = None
        layer._last_F = 0.0
        layer._last_G = None
        if not hasattr(layer, "_mdf_dense_forward"):
            layer._mdf_dense_forward = layer.forward
        layer.forward = MethodType(_mdf_layer_forward, layer)
        if layer.is_routed:
            routed.append(layer)
    return routed


def collect_mdf_stats(routed, training: bool):
    """Returns (ΣF·G or None, per-token active layer count [B,T] or None). The α coefficient is multiplied by the caller."""
    fg = None
    sels = []
    for layer in routed:
        if training and layer._last_G is not None:
            term = layer._last_F * layer._last_G
            fg = term if fg is None else fg + term
        if layer._last_sel is not None:
            sels.append(layer._last_sel)
    k = torch.stack(sels, dim=-1).float().sum(-1) if sels else None  # [B,T]
    return fg, k


# ==================== Router-Tuning (EMNLP'25, frozen base, trains gates only) ====================

def _rt_layer_forward(self, hidden_states, attention_mask=None, position_ids=None,
                      past_key_values=None, use_cache=False, position_embeddings=None,
                      **kwargs):
    self._last_mod_capacity = None
    self._last_mod_loss = None
    block_residual = hidden_states
    attn_inputs = self.input_layernorm(hidden_states)
    routing_mask = None
    if self.is_mod:
        routing_mask, cap, mloss = _compute_routing_state(self, attn_inputs, self.training)
        self._last_mod_capacity = cap
        self._last_mod_loss = mloss
    attn_out, _ = self.self_attn(
        hidden_states=attn_inputs, position_embeddings=position_embeddings,
        attention_mask=attention_mask,
        past_key_values=past_key_values if use_cache else None)
    if self.is_mod and self.route_target == "attn":
        hidden_states = _apply_routing_mask(block_residual, attn_out, routing_mask)
    else:
        hidden_states = block_residual + attn_out
    mlp_residual = hidden_states
    mlp_out = self.mlp(self.post_attention_layernorm(hidden_states))
    if self.is_mod and self.route_target == "mlp":
        hidden_states = _apply_routing_mask(mlp_residual, mlp_out, routing_mask)
    else:
        hidden_states = mlp_residual + mlp_out
    if self.is_mod and self.route_target == "block":
        hidden_states = _apply_routing_mask(
            block_residual, hidden_states - block_residual, routing_mask)
    return hidden_states


def patch_model_rt(model, is_mod, granularity="block_token", threshold=0.5,
                   target=None, scale=0.0):
    """Attach router + rewritten-convention forward layer by layer; returns the list of gated layers. Loss aggregation is done by the caller
    (loss = LM + Σ_last_mod_loss, equivalent to upstream _patched_model_forward)."""
    route_target, route_level = _parse_granularity(granularity)
    layers = model.model.layers
    try:
        want_dtype = next(model.parameters()).dtype
    except StopIteration:
        want_dtype = None
    gated = []
    for i, layer in enumerate(layers):
        layer.is_mod = bool(is_mod[i])
        layer.route_target, layer.route_level = route_target, route_level
        layer.threshold = threshold
        layer.target_mod_capacity = target
        layer.gradient_scale = scale or 0.0
        if layer.is_mod and not hasattr(layer, "router"):
            layer.router = nn.Linear(model.config.hidden_size, 1, bias=False)
            nn.init.zeros_(layer.router.weight)  # paper §4: zero init, dense start
        if want_dtype is not None and hasattr(layer, "router"):
            layer.router.to(want_dtype)  # a fresh Linear defaults to fp32, cast to bf16 along with the base (upstream relies on deepspeed to do this implicitly)
        layer._last_mod_capacity = None
        layer._last_mod_loss = None
        layer.forward = MethodType(_rt_layer_forward, layer)
        if layer.is_mod:
            gated.append(layer)
    return gated


def collect_rt_stats(gated, training):
    caps, mloss = [], None
    for layer in gated:
        if layer._last_mod_capacity is not None:
            caps.append(layer._last_mod_capacity)
        if training and layer._last_mod_loss is not None:
            mloss = layer._last_mod_loss if mloss is None else mloss + layer._last_mod_loss
    cap = sum(caps) / max(len(caps), 1) if caps else 0.0
    return cap, mloss


# ==================== Unified eval adapter (valid=labels, k=total protocol) ====================

def _k_total_from(routed, n_dense, collect_fn):
    """k_provider factory: routed-layer k [B,T] from collect_fn(routed, False) + n_dense = total k."""
    def fn(b):
        _, k = collect_fn(routed, training=False)
        return (k + n_dense) if k is not None else None
    return fn


def eval_heldout_modd(model, routed, n_dense, texts, coll, batch_size=4):
    """MoD held-out: loss/acc/mean_k±std (k includes the fixed layers), same distribution, same protocol."""
    return eval_heldout(model, texts, coll, batch_size, valid_mode="labels",
                        k_provider=_k_total_from(routed, n_dense, collect_modd_stats))


def eval_heldout_mdf(model, routed, n_dense, texts, coll, batch_size=4):
    """MoDification held-out: loss/acc/mean_k±std (k includes the fixed layers), same distribution, same protocol."""
    return eval_heldout(model, texts, coll, batch_size, valid_mode="labels",
                        k_provider=_k_total_from(routed, n_dense, collect_mdf_stats))


def eval_heldout_rt(model, gated, texts, coll, batch_size=4, n_always: int = 0):
    """RT held-out: loss/acc + exec_rate (token-weighted mean of cap).
    k_provider carries it via a constant tensor n_always + n_mod·cap (RT cap is a sequence-level scalar;
    the aggregation formula is mathematically identical to the original implementation: exec = Σcap_b·n_b / Σn_b)."""
    n_mod = len(gated)

    def fn(b):
        cap, _ = collect_rt_stats(gated, training=False)
        k = n_always + n_mod * cap
        return torch.full(b["input_ids"].shape, float(k), device=b["input_ids"].device)

    res = eval_heldout(model, texts, coll, batch_size, valid_mode="labels", k_provider=fn)
    res["exec_rate"] = (res["mean_k"] - n_always) / max(n_mod, 1) \
        if res.get("mean_k") is not None else 0.0
    return res


# ==================== Baseline ckpt saving ====================

def save_baseline_ckpt(model, tok, save_dir: str, use_lora: bool, config: dict,
                       config_name: str):
    """Baseline ckpt: LoRA version = small ckpt (routers.pt: gate + lora keys); full-parameter version = full base (gate keys stripped).
    The config is written to config_name (modd_config.json / mdf_config.json / rt_config.json)."""
    sd = model.state_dict()
    if use_lora:
        torch.save({k: v.cpu() for k, v in gate_state_dict(sd, extra_marks=("lora_",)).items()},
                   os.path.join(save_dir, "routers.pt"))
    else:
        model.save_pretrained(
            save_dir, state_dict={strip_wrapper_prefix(k): v.cpu()
                                  for k, v in sd.items() if is_base_key(k)})
    tok.save_pretrained(save_dir)
    with open(os.path.join(save_dir, config_name), "w", encoding="utf-8") as f:
        json.dump(config, f)
