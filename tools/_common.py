"""Shared helpers for experiment tooling (tools/*, baselines/eval_compare.py).

Covers small statistics and model-handle utilities. (The old argparse CLI flag
builders died with the Typer migration: every entry point declares its flags
as a Typer signature directly.)
Deliberately OUT of scope (history-pinned numbers stay in owning tools):
prompt construction, ROUGE/rep/tokF1 implementations, plots.
"""
import gc
import json

import torch


# ----- small statistics -----

def pearson_r(x: torch.Tensor, y: torch.Tensor) -> float:
    xc, yc = x - x.mean(), y - y.mean()
    return float((xc * yc).sum() / (xc.pow(2).sum() * yc.pow(2).sum()).sqrt().clamp_min(1e-12))


def quartile_means(x: torch.Tensor, by: torch.Tensor, ndigits=3):
    """Mean of x within quartiles of by (edges -inf/q25/q50/q75/+inf)."""
    qs = by.quantile(torch.tensor([0.25, 0.5, 0.75]))
    edges = [float("-inf")] + [float(q) for q in qs] + [float("inf")]
    out = []
    for i in range(4):
        sel = (by > edges[i]) & (by <= edges[i + 1])
        out.append(round(float(x[sel].mean()), ndigits) if sel.any() else None)
    return out


# ----- model handles -----

def find_layers(model):
    for path in (["model", "layers"], ["transformer", "h"], ["layers"]):
        cur = model
        try:
            for a in path:
                cur = getattr(cur, a)
            return cur
        except AttributeError:
            pass
    raise ValueError("layers not found")


def passthrough_hook(_module, margs, output):
    """Forward hook that bypasses one decoder layer (returns its input)."""
    hs = margs[0] if margs else output[0]
    return (hs,) + tuple(output[1:]) if isinstance(output, tuple) else hs


def module_gb(mods, include_buffers=False):
    """Weight footprint in GB; accepts one module or an iterable (skips None)."""
    if isinstance(mods, torch.nn.Module):
        mods = [mods]
    n = 0
    for m in mods:
        if m is None:
            continue
        n += sum(p.numel() * p.element_size() for p in m.parameters())
        if include_buffers:
            n += sum(b.numel() * b.element_size() for b in m.buffers())
    return n / 1e9


# ----- IO / memory hygiene -----

def dump_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)


def collect_gc():
    gc.collect()
    torch.cuda.empty_cache()
