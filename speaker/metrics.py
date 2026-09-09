"""Training/evaluation metrics: per-token NLL, per-token correct mask, activation memory estimate."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def per_token_nll(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Per-token NLL [B,T], zero at padding (labels==-100).
    Reflects the model's own ability: high NLL = hard.
    Causal LM: logits[t] predicts labels[t+1], must shift, last position padded with 0."""
    B, T, V = logits.shape
    nll = F.cross_entropy(
        logits[:, :-1].reshape(-1, V),
        labels[:, 1:].reshape(-1),
        reduction="none",
        ignore_index=-100,
    )
    out = torch.zeros(B, T, device=logits.device)
    out[:, :-1] = nll.reshape(B, T - 1)
    return out


def per_token_correct(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Per-token correct mask [B,T] bool, False at padding/last position.
    Difficulty label for the accuracy accounting: wrong = hard.
    Causal LM: logits[t] predicts labels[t+1], must shift (the old unshifted version even
    scored dense at 0)."""
    B, T = logits.shape[:2]
    pred = logits[:, :-1].argmax(-1)
    tgt = labels[:, 1:]
    out = torch.zeros(B, T, dtype=torch.bool, device=logits.device)
    out[:, :-1] = (pred == tgt) & (tgt != -100)
    return out


def estimate_act_mb(mean_k: float, batch: int, seq: int, hidden: int,
                    bytes_per_hidden: float = 28.0) -> float:
    """Activation memory estimate (MB): skipped layers allocate no activations, savings ≈
    k × T × H × coefficient. The coefficient covers QKVO/mlp/residual/attention scores; the
    bf16 default of 28 is empirical, for relative comparison only."""
    return mean_k * batch * seq * hidden * bytes_per_hidden / 1e6


def distill_kl_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor,
                    labels: torch.Tensor, temp: float = 1.0) -> torch.Tensor:
    """Dense self-distillation KL(student‖teacher), averaged only over shifted positions where
    labels!=-100. SFT labels are a sparse signal (rewarding only correct tokens); KL is a dense
    signal (learning the full dense distribution), letting the gating learn "what dense would
    have done" and preserving coherent autoregressive generation. The caller detaches the teacher
    side."""
    T = max(float(temp), 1e-3)
    s = student_logits[:, :-1].float()
    t = teacher_logits[:, :-1].float()
    mask = (labels[:, 1:] != -100)
    if not bool(mask.any()):
        return (s.sum() * 0.0)
    log_p = torch.log_softmax(s / T, dim=-1)
    log_q = torch.log_softmax(t / T, dim=-1)
    kl = (log_p.exp() * (log_p - log_q)).sum(-1)  # [B,T-1]
    w = mask.float()
    return (kl * w).sum() / w.sum().clamp_min(1) * (T * T)
