"""Layer-load profiling: load -> fixed-layer promotion and GPU/CPU heterogeneous placement plan.

Theoretical accounting (user's design):
- Fixed (shared) layers = layers with load >= threshold (90%~95%), participating in every
  inference round;
- From-scratch training: structural prior — directly fix the first/last k layers as shared layers;
- Finetune: the distribution differs — train the gating first, then fix layers based on measured
  load, the rest stay under gating control;
- Placement: with enough VRAM -> all GPU + layer skipping for lower latency; otherwise fixed +
  high-load layers resident on GPU, low-load layers inferred on CPU (decode stage); prefill
  degenerates to dense like MoE.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple


def profile_layer_load(model, batches: Iterable[dict]) -> Dict[int, float]:
    """Runs several batches accumulating each layer's hard usage {layer_idx: load}; fixed layers
    are always 1.0.

    model must be a SpeakerModelWrapper (provides get_layer_usage); eval + soft mode is
    recommended (all layers execute, trajectories do not drift from skipping, the hard mask is
    the deployment-accounting load)."""
    model.get_layer_usage()  # zero the starting point
    was_training = model.training
    model.eval()
    import torch
    with torch.no_grad():
        n_batch = 0
        for b in batches:
            model(**b)
            n_batch += 1
    if was_training:
        model.train()
    if n_batch == 0:
        return {}
    usage = model.get_layer_usage()  # read-and-clear
    return {i: float(v[0]) for i, v in usage.items()}


def select_fixed_layers(load: Dict[int, float], threshold: float = 0.9,
                        always_on: Sequence[int] = (),
                        keep_gated_min: int = 1, top_k: int = 0) -> Tuple[List[int], List[int]]:
    """Selects fixed layers by load: gated layers with load >= threshold are promoted to
    fixed (shared) layers.

    When top_k>0, threshold is ignored and the top_k gated layers by load are promoted directly
    (user-specified scheme). Returns (fixed_layers, promoted): fixed is the full fixed-layer list
    (including pre-existing always_on), promoted are the layers newly promoted this round.
    Guarantees at least keep_gated_min layers remain gated (promoting everything would degenerate
    to dense, defeating the purpose of savings)."""
    ao = set(int(i) for i in always_on)
    gated = [i for i in sorted(load) if i not in ao]
    if top_k and top_k > 0:
        hot = sorted(gated, key=lambda i: -load[i])[:top_k]
    else:
        if not 0.0 < threshold <= 1.0:
            raise ValueError(f"threshold must be in (0,1], got {threshold}")
        hot = [i for i in gated if load[i] >= threshold]
    if len(gated) - len(hot) < keep_gated_min:  # gated floor: return the coldest layers from the hottest set
        hot = sorted(hot, key=lambda i: load[i])[:max(len(gated) - keep_gated_min, 0)]
    promoted = sorted(hot)
    fixed = sorted(ao | set(promoted))
    return fixed, promoted


def estimate_layer_bytes(model) -> Dict[int, int]:
    """Per-layer parameter bytes (including gating router/tau/comp). model needs .layers
    (wrapper list)."""
    out: Dict[int, int] = {}
    for w in model.layers:
        n = sum(p.numel() * p.element_size() for p in w.parameters())
        n += sum(b.numel() * b.element_size() for b in w.buffers())
        out[w.layer_idx] = int(n)
    return out


def plan_placement(load: Dict[int, float], layer_bytes: Dict[int, int],
                   gpu_budget_bytes: float, always_on: Sequence[int] = ()) -> dict:
    """Greedy placement plan: fixed layers are forced onto GPU; the rest are packed in descending
    load order until the budget runs out, leftovers go to CPU.

    Takes effect at decode: high-load layers resident on GPU for fast response, occasionally
    activated low-load layers on CPU to save VRAM; prefill degenerates to dense like MoE
    (compute everything), only slower for cold layers on CPU."""
    resident = set(int(i) for i in always_on)
    used = sum(layer_bytes.get(i, 0) for i in resident)
    rest = sorted((i for i in load if i not in resident),
                  key=lambda i: (-load.get(i, 0.0), i))
    for i in rest:
        b = layer_bytes.get(i, 0)
        if used + b > gpu_budget_bytes:
            continue  # not enough budget now, skip; smaller layers later may still fit
        resident.add(i)
        used += b
    total = sum(layer_bytes.values())
    return {
        "gpu_layers": sorted(resident),
        "cpu_layers": sorted(i for i in load if i not in resident),
        "gpu_bytes": int(used),
        "total_bytes": int(total),
        "gpu_gb": used / 1e9,
        "total_gb": total / 1e9,
    }


def render_load_table(load: Dict[int, float], always_on: Sequence[int] = (),
                      width: int = 40) -> str:
    """Text load table (for profiling scripts/logs): layer | load | bar chart."""
    lines = []
    ao = set(always_on)
    for i, v in sorted(load.items()):
        bar = "#" * round(v * width)
        tag = " fixed" if i in ao else ("" if v < 0.9 else " >=90%")
        lines.append(f"  L{i:2d} {v:6.1%} |{bar:<{width}s}|{tag}")
    return "\n".join(lines)
