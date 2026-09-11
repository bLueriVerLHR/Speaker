"""Held-out evaluation and training-time EMA (decoupled from the data pipeline: collate_fn is
injected by the caller)."""
from __future__ import annotations

import torch

from .metrics import per_token_correct, per_token_nll
from .ruler import valid_positions


def ema_update(prev, value, beta=0.95):
    return value if prev is None else beta * prev + (1 - beta) * value


def eval_heldout(model, texts, collate_fn, batch_size=4,
                 valid_mode: str = "attention_mask", k_provider=None):
    """Same-distribution held-out: loss/acc; gating models additionally get mean_k±std and
    per-position quartile k.

    collate_fn(texts) -> batch dict (input_ids/attention_mask/labels); the accounting is defined
    by the caller (keep chat masking/truncation length consistent with training).
    valid_mode: denominator accounting for the statistics — "attention_mask" (ours, all tokens) |
                "labels" (baseline accounting, supervised segment labels!=-100; the two are
                equivalent under plain collate).
    k_provider(batch) -> [B,T] per-token active layer count, or None; by default duck-probes
                model.get_active_counts() (SpeakerModelWrapper). Baselines inject their own k
                accounting (including fixed layers) via the baselines.lib adapters.
    """
    was_training = model.training
    model.eval()
    sum_nll = 0.0
    n_tok = 0
    n_ok = 0
    sum_k = 0.0
    sum_k2 = 0.0
    quart = [0.0] * 4
    qn = [0] * 4
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            b = collate_fn(texts[i:i + batch_size])
            out = model(**b)
            logits = out.logits.float()
            nll = per_token_nll(logits, b["labels"])
            correct = per_token_correct(logits, b["labels"])
            valid = valid_positions(b, valid_mode)
            sum_nll += nll[valid].sum().item()
            n_tok += valid.sum().item()
            n_ok += correct[valid].sum().item()
            if k_provider is not None:
                k = k_provider(b)
            else:
                getk = getattr(model, "get_active_counts", None)
                k = getk() if callable(getk) else None
            if k is not None:
                kv = k[valid].float()
                sum_k += kv.sum().item()
                sum_k2 += (kv * kv).sum().item()
                B, T = b["input_ids"].shape
                for bi in range(B):
                    L = valid[bi].sum().item()
                    for t in range(T):
                        if not valid[bi, t]:
                            continue
                        q = min(3, int(t / max(L - 1, 1) * 4))
                        quart[q] += k[bi, t].item()
                        qn[q] += 1
    if was_training:
        model.train()
    res = {"loss": sum_nll / max(n_tok, 1), "acc": n_ok / max(n_tok, 1)}
    if n_tok and any(qn):
        mean_k = sum_k / n_tok
        var_k = max(sum_k2 / n_tok - mean_k * mean_k, 0.0)
        res["mean_k"] = mean_k
        res["std_k"] = var_k ** 0.5
        res["quartile_k"] = [quart[q] / max(qn[q], 1) for q in range(4)]
    else:
        res["mean_k"] = res["std_k"] = res["quartile_k"] = None
    return res
