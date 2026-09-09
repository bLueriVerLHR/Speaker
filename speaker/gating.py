"""Gating components (dual scheme coexists, gate_mode switch; the two schemes' checkpoints
are not interchangeable):

- moe (default): JointRouter(H->G) single-point joint routing + log p selection + RouteDecision.
  The hidden state at the gated-region entry computes logits for all G gated layers in one shot,
  log_softmax gives log p — the layers are serial, so the joint distribution must be fixed before
  execution. "MoE gating applied directly on the layers": the gated layers are the experts, the
  shared layers sit outside the routing pool. Selection: top-p (default) accumulates probability
  in descending log p order and stops upon reaching p, k is adaptive per token; top-k (optional)
  fixes k. Mixing: the selected layers' probabilities are renormalized into w, weighted residual
  h += w·(F(h)-h); unselected layers are not executed (decode writes no K/V). Gradient: forward
  hard, backward soft-inclusive sigmoid((p - cum)/T) STE, budget λ·mean(k) differentiable via the
  soft count.

- threshold (legacy scheme): Router gives per-layer independent a_l(t) affinity logits; the
  threshold is carried solely by tau (no bias, avoiding bias-tau collinearity); each gated layer
  has one Router deciding whether the token executes that layer in full.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GatingOutput:
    """Gating result of a single gated layer.
    threshold: full per-layer gating; moe: the slot-cut slice of a RouteDecision."""
    mask: torch.Tensor               # mask/weights used for mixing (STE or soft/hard/renormalized)
    soft_mask: torch.Tensor          # differentiable soft
    hard_mask: torch.Tensor          # hard 0/1
    router_logits: torch.Tensor      # this layer's logits (diagnostics)
    aux_loss: Optional[torch.Tensor] = None  # z-loss (attached here by threshold; moe computes it in the model wrapper)


class Router(nn.Module):
    """threshold scheme: per-layer independent a_l(t) affinity logit. No bias: the threshold is
    carried solely by tau, avoiding bias-tau collinearity that would make tau look frozen (in the
    old version both scalars managed the constant term, so all learning went into the bias)."""

    def __init__(self, hidden_size: int, hidden_dim: Optional[int] = None):
        super().__init__()
        if hidden_dim:
            self.net = nn.Sequential(
                nn.Linear(hidden_size, hidden_dim, bias=False),
                nn.SiLU(),
                nn.Linear(hidden_dim, 1, bias=False),
            )
        else:
            self.net = nn.Linear(hidden_size, 1, bias=False)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # [B,T,1] affinity logit


class JointRouter(nn.Module):
    """moe scheme: gated-region entry single-point routing Linear(H->G, no bias)
    + per-layer learnable prior bias (G,).

    Outputs fp32 logits [B,T,G]; softmax temperature/annealing/Gumbel are applied by the caller
    (temperature acts on the logits, selection happens in log p space)."""

    def __init__(self, hidden_size: int, n_gated: int):
        super().__init__()
        self.net = nn.Linear(hidden_size, n_gated, bias=False)
        self.layer_bias = nn.Parameter(torch.zeros(n_gated))
        nn.init.normal_(self.net.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype in (torch.bfloat16, torch.float16):
            x = x.float()
        return self.net(x) + self.layer_bias


@dataclass
class RouteDecision:
    """moe scheme: joint routing result of one forward (computed at the entry layer,
    shared slices for all gated layers)."""
    logits: torch.Tensor    # [B,T,G] logits fed to softmax (with temperature/noise)
    log_p: torch.Tensor     # [B,T,G]
    probs: torch.Tensor     # [B,T,G] softmax probabilities
    selected: torch.Tensor  # [B,T,G] hard 0/1 (padding=0)
    weights: torch.Tensor   # [B,T,G] selected layers' renormalized probabilities (weighted residual), 0 elsewhere
    valid: torch.Tensor     # [B,T] float padding mask
    k: torch.Tensor         # [B,T] hard count of activated gated layers (detached, for statistics)
    k_soft: torch.Tensor    # [B,T] differentiable soft count (degenerates to k in topk mode)


def select_and_weight(logits: torch.Tensor, select_mode: str = "topp",
                      top_p: float = 0.9, top_k: int = 6, min_layers: int = 1,
                      kmax: int = 10, count_temp: float = 0.1,
                      valid: torch.Tensor = None) -> RouteDecision:
    """moe scheme: logits [B,T,G] -> selection + renormalized weights + soft count. All fp32.

    top-p: the j-th largest layer is selected iff its preceding cumulative probability
    cum_{j-1} < top_p (the first layer is always selected);
    k = number of qualifying positions, clamped to [min_layers, min(kmax, G)].
    Soft inclusion (backward only): sigmoid((top_p - cum_{j-1}) / count_temp); gradients flow
    through cum (softmax probabilities) back into the router — pressing k is equivalent to
    pressing distribution entropy (the sharper the distribution, the smaller k).
    """
    G = logits.shape[-1]
    log_p = F.log_softmax(logits.float(), dim=-1)
    probs = log_p.exp()
    sorted_probs, order = torch.sort(probs, dim=-1, descending=True)
    cum = sorted_probs.cumsum(-1)  # cum_j = sum of the top j+1 probabilities
    if select_mode == "topk":
        cnt = min(max(top_k, min_layers), min(kmax, G))
        rank = torch.arange(G, device=logits.device, dtype=probs.dtype)
        sel_sorted = (rank < float(cnt)).expand_as(sorted_probs).to(probs.dtype)
        soft_sorted = sel_sorted
    else:  # topp
        exceed = cum >= top_p
        count = exceed.float().argmax(-1) + 1.0
        # Numeric fallback: cumsum float error at p≈1 may leave no qualifying position -> select all
        count = torch.where(exceed.any(-1), count,
                            torch.full_like(count, float(G)))
        count = count.clamp(min=float(min_layers), max=float(min(kmax, G)))
        rank = torch.arange(G, device=logits.device, dtype=probs.dtype)
        sel_sorted = (rank < count.unsqueeze(-1)).to(probs.dtype)
        prev = torch.cat([torch.zeros_like(cum[..., :1]), cum[..., :-1]], dim=-1)
        soft_sorted = torch.sigmoid((float(top_p) - prev) / float(count_temp))
    selected = torch.zeros_like(probs).scatter(-1, order, sel_sorted)
    soft_selected = torch.zeros_like(probs).scatter(-1, order, soft_sorted)
    ste = soft_selected + (selected - soft_selected).detach()  # forward hard, backward soft
    if valid is not None:
        v = valid.to(probs.dtype)
        selected = selected * v.unsqueeze(-1)
        ste = ste * v.unsqueeze(-1)
    weights = probs * ste
    weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-6)
    if valid is not None:
        weights = weights * valid.to(weights.dtype).unsqueeze(-1)
    k = selected.sum(-1).detach()
    k_soft = ste.sum(-1)
    return RouteDecision(logits=logits, log_p=log_p, probs=probs,
                         selected=selected, weights=weights,
                         valid=(valid.to(probs.dtype) if valid is not None
                                else torch.ones_like(k)),
                         k=k, k_soft=k_soft)
