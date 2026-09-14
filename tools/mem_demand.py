"""Per-token memory demand accounting (single source of truth).

MoDification-style claim needs one number per method: at the same loss, how
much memory does one token demand, and at the same memory, what loss.
This module owns that conversion so eval_compare / probe_kdist / edge_bench
cannot drift apart:

  active_weights(k) = k/N * decoder_gb          (bf16 measured, not estimated)
  kv_cache(k, ctx)  = k * kv_layer_bytes * ctx  (sparse KV: skipped layers write nothing)
  flop_ratio(k)     = k/N

`decoder_gb` is measured from real weights (module_gb over the decoder
layers); the pure-math part (`kv_layer_bytes`, `demand`) needs only the HF
config values and is covered offline without weights.
"""
import json
import sys
from pathlib import Path
from typing import Annotated, Any

import typer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

app = typer.Typer(add_completion=False)


def kv_layer_bytes(hidden: int, n_attn_heads: int, n_kv_heads: int,
                   dtype_bytes: int = 2) -> int:
    """KV bytes for ONE layer and ONE token (K+V)."""
    heads_kv = n_kv_heads or n_attn_heads
    head_dim = hidden // max(n_attn_heads, 1)
    return 2 * heads_kv * head_dim * dtype_bytes


def kv_layer_bytes_from_config(hf_config: Any, dtype_bytes: int = 2) -> int:
    """Same as kv_layer_bytes but reads a HF config object/dict (GQA/MQA safe)."""
    if isinstance(hf_config, dict):
        hidden = hf_config["hidden_size"]
        n_q = hf_config.get("num_attention_heads")
        n_kv = hf_config.get("num_key_value_heads", n_q)
    else:
        if hasattr(hf_config, "text_config") and hf_config.text_config is not None:
            hf_config = hf_config.text_config
        hidden = hf_config.hidden_size
        n_q = getattr(hf_config, "num_attention_heads", None)
        n_kv = getattr(hf_config, "num_key_value_heads", n_q)
    return kv_layer_bytes(int(hidden), int(n_q), int(n_kv or n_q), dtype_bytes)


def demand(k_total: float, n_layers: int, decoder_gb: float,
           kv_layer_B: int, ctx: int = 1024) -> dict:
    """Pure conversion: per-token k -> weight/KV/FLOP demand. No weights needed."""
    k = max(float(k_total), 0.0)
    n = max(int(n_layers), 1)
    active_w_gb = k / n * float(decoder_gb)
    kv_gb = k * float(kv_layer_B) * int(ctx) / 1e9
    return {"k": k,
            "active_w_gb": active_w_gb,
            "kv_mb_ctx": kv_gb * 1000.0,
            "kv_gb_ctx": kv_gb,
            "flop_ratio": k / n}


def decoder_gb_from_model(model) -> tuple:
    """Measured (decoder_gb, per_layer_gb list, n_layers) from a loaded model.

    Works on plain HF models (model.layers) and on ours wrappers
    (SpeakerModelWrapper.layers of SpeakerLayerWrapper).
    """
    from tools._common import find_layers, module_gb

    layers = None
    inner = getattr(model, "layers", None)
    if inner is not None and hasattr(inner, "__len__"):
        try:
            if len(inner) == getattr(
                    getattr(getattr(model, "mod_config", None),
                            "num_hidden_layers", None) or len(inner), -1):
                layers = inner
        except Exception:
            layers = None
    if layers is None:
        try:
            layers = find_layers(model.hf_model if hasattr(model, "hf_model") else model)
        except ValueError:
            layers = getattr(model, "layers", None)
    if layers is None:
        raise ValueError("decoder layers not found")
    per_layer = []
    for lyr in layers:
        base = getattr(lyr, "layer", lyr)  # unwrap SpeakerLayerWrapper
        per_layer.append(module_gb(base))
    return sum(per_layer), per_layer, len(per_layer)


@app.command()
def main(
    model_id: Annotated[str, typer.Option("--model_id")] = "/home/hdd/model/Qwen2.5-7B-Instruct",
    k_mean: Annotated[float, typer.Option("--k_mean", help="per-token total active layers k")] = 12.0,
    k_std: Annotated[float, typer.Option("--k_std")] = 0.0,
    ctx: Annotated[int, typer.Option("--ctx", help="context length for the KV term")] = 1024,
    dtype_bytes: Annotated[int, typer.Option("--dtype_bytes")] = 2,
    out: Annotated[str, typer.Option("--out")] = "",
) -> None:
    """One-row demand report: k -> weight/KV/FLOP (loads the base on CPU to measure)."""
    from speaker.terminal import setup_terminal
    setup_terminal()
    from transformers import AutoConfig
    from transformers import AutoModelForCausalLM
    import torch
    from speaker.log import logger

    cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    kv_B = kv_layer_bytes_from_config(cfg, dtype_bytes)
    n = int(cfg.num_hidden_layers)
    m = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map=None,
        trust_remote_code=True, low_cpu_mem_usage=True).to("cpu")
    dec_gb, _, _ = decoder_gb_from_model(m)
    row = demand(k_mean, n, dec_gb, kv_B, ctx)
    row.update({"model_id": model_id, "n_layers": n, "decoder_gb": dec_gb,
                "kv_layer_B": kv_B, "ctx": ctx, "k_std": k_std})
    logger.info(f"k {k_mean:.1f}±{k_std:.1f}/{n} ctx {ctx}: "
                f"active_w {row['active_w_gb']:.2f}GB "
                f"KV {row['kv_mb_ctx']:.1f}MB flop {row['flop_ratio']:.2f}")
    if out:
        with open(out, "w", encoding="utf-8") as f:
            json.dump(row, f, indent=1)
        logger.info(f"saved {out}")


if __name__ == "__main__":
    app()
