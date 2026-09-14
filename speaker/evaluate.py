"""Held-out evaluation and training-time EMA (decoupled from the data pipeline: collate_fn is
injected by the caller)."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .ruler import valid_positions
from .terminal import track

# Sequence chunk for the eval CE/argmax: a full fp32 logits copy (B*T*V*4B,
# ~1.2GB @bs2x1024x152k) OOMs once optimizer states are resident post-train.
_EVAL_TOK_CHUNK = 256


def chunked_nll_correct(logits: torch.Tensor, labels: torch.Tensor):
    """Memory-capped per-token NLL [B,T] (fp32) + correct mask [B,T] (bool).

    Same layout/semantics as metrics.per_token_{nll,correct} (source position t
    predicts labels[t+1]; trailing position holds 0/False) but the fp32 CE and
    argmax run in T-chunks, so peak memory is O(B*chunk*V) instead of O(B*T*V).
    Shared by eval_heldout and the probe tools (probe_kdist/probe_nll_ablation).
    """
    B, T, V = logits.shape
    dev = logits.device
    nll = torch.zeros(B, T, device=dev)
    corr = torch.zeros(B, T, dtype=torch.bool, device=dev)
    tgt_full = labels.to(dev)
    for s in range(0, T - 1, _EVAL_TOK_CHUNK):
        e = min(s + _EVAL_TOK_CHUNK, T - 1)
        p = logits[:, s:e].float()
        tgt = tgt_full[:, s + 1:e + 1]
        n = F.cross_entropy(p.reshape(-1, V), tgt.reshape(-1), reduction="none",
                            ignore_index=-100)
        nll[:, s:e] = n.reshape(B, e - s)
        corr[:, s:e] = (p.argmax(-1) == tgt) & (tgt != -100)
        del p, n
    return nll, corr


def ema_update(prev, value, beta=0.95):
    return value if prev is None else beta * prev + (1 - beta) * value


def format_k_quartile(res) -> str:
    """One-line k summary shared by the terminal held-out reports."""
    return (f"k {res['mean_k']:.1f}±{res['std_k']:.1f} "
            f"quartile {[round(v, 1) for v in res['quartile_k']]}")


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
        for i in track(range(0, len(texts), batch_size),
                       total=(len(texts) + batch_size - 1) // batch_size,
                       desc="heldout"):
            b = collate_fn(texts[i:i + batch_size])
            out = model(**b)
            # NOTE: no full-tensor .float() — materializing B*T*V fp32 OOMs
            # post-train (optimizer resident); see chunked_nll_correct.
            lg = out.logits
            # sharded backbones: logits may land on the last card while labels stay on
            # the input device — index with a mask that followed the logits (no-op single)
            valid = valid_positions(b, valid_mode).to(lg.device)
            nll, correct = chunked_nll_correct(lg, b["labels"])
            sum_nll += nll[valid].sum().item()
            n_tok += valid.sum().item()
            n_ok += correct[valid].sum().item()
            del nll, correct
            if k_provider is not None:
                k = k_provider(b)
            else:
                getk = getattr(model, "get_active_counts", None)
                k = getk() if callable(getk) else None
            if k is not None:
                kv = k.to(valid.device)[valid].float()
                sum_k += kv.sum().item()
                sum_k2 += (kv * kv).sum().item()
                # quartile bins, vectorized (identical math to the old
                # per-position loop: q = min(3, t*4 // max(L-1, 1))).
                B, T = b["input_ids"].shape
                tidx = torch.arange(T, device=valid.device)
                for bi in range(B):
                    vb = valid[bi]
                    L = int(vb.sum())
                    if not L:
                        continue
                    qb = ((tidx[vb] * 4) // max(L - 1, 1)).clamp_max(3)
                    kb = k[bi][vb].double()
                    qa = torch.zeros(4, dtype=torch.float64,
                                     device=valid.device).index_add_(0, qb, kb)
                    na = torch.bincount(qb, minlength=4)
                    for q in range(4):
                        quart[q] += float(qa[q])
                        qn[q] += int(na[q])
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
