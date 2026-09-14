"""Checkpoint I/O consolidation (ed5): gate key filtering / strip wrapper prefix /
self-contained save & load.

Key accounting conventions (bit-identical to history, old ckpt compatible):
- gate.pt: keys in the state_dict containing GATE_KEY_MARKS substrings (the finetune version
  additionally adds "lora_");
- Clean base: after removing gating keys, strip the layer-wrapper prefix
  model.layers.<i>.layer.X -> model.layers.<i>.X;
- Profile promotion: drop joint_router entirely (moe dimension mismatch) + promoted layers'
  per-layer gating keys (threshold).
"""
from __future__ import annotations

import os
import re

import torch

from .log import logger

# Gating key markers: "router" covers both per-layer router.* and moe's joint_router.*
GATE_KEY_MARKS = ("router", "tau", "comp")


def gate_state_dict(state_dict: dict, extra_marks=()) -> dict:
    """Filters gating keys by substring markers (same accounting as the historical
    SAVE_KEY_SUBSTRINGS)."""
    marks = GATE_KEY_MARKS + tuple(extra_marks)
    return {k: v for k, v in state_dict.items() if any(m in k for m in marks)}


def is_gate_key(k: str) -> bool:
    return any(m in k for m in GATE_KEY_MARKS)


def is_base_key(k: str) -> bool:
    """Clean-base key predicate (same accounting as the historical --save_full filter):
    exclude keys containing .router. / joint_router or ending with .tau / .comp."""
    return not (".router." in k or "joint_router" in k
                or k.endswith(".tau") or k.endswith(".comp"))


def strip_wrapper_prefix(k: str) -> str:
    """Strips the layer-wrapper prefix: model.layers.<i>.layer.X -> model.layers.<i>.X."""
    return re.sub(r"^((?:\w+\.)*layers\.\d+)\.layer\.", r"\1.", k)


def clean_base_state_dict(mod_model) -> dict:
    """Full base state_dict (gating keys removed + wrapper prefix stripped, cpu)."""
    return {strip_wrapper_prefix(k): v.cpu()
            for k, v in mod_model.hf_model.state_dict().items() if is_base_key(k)}


def save_gate(mod_model, save_dir: str, extra_marks=(), filename: str = "gate.pt",
              fmt: str = "pt") -> dict:
    """Saves only gating (+ optional lora) keys; returns the saved state_dict.

    fmt: "pt" (default, legacy pickle, bit-compatible reruns) | "safetensors"
    (requires the safetensors package; safer for sharing, no pickle). The
    filename extension follows fmt unless an explicit filename is given.
    load_gate auto-detects either file, so readers need no flag.
    """
    os.makedirs(save_dir, exist_ok=True)
    sd = {k: v.cpu() for k, v in
          gate_state_dict(mod_model.state_dict(), extra_marks).items()}
    if fmt == "safetensors":
        try:
            from safetensors.torch import save_file
        except ImportError as e:
            raise ImportError(
                "safetensors format requested but the 'safetensors' package is "
                "not installed; use fmt='pt' (default) or pip install safetensors") from e
        if filename == "gate.pt":
            filename = "gate.safetensors"
        save_file(sd, os.path.join(save_dir, filename))
    elif fmt == "pt":
        torch.save(sd, os.path.join(save_dir, filename))
    else:
        raise ValueError(f"fmt must be pt|safetensors, got {fmt!r}")
    return sd


def save_clean_base(mod_model, tok=None, cfg=None, save_dir: str = "."):
    """Self-contained ckpt base part: clean base (+ tokenizer + mod_config.json)."""
    os.makedirs(save_dir, exist_ok=True)
    mod_model.hf_model.save_pretrained(save_dir, state_dict=clean_base_state_dict(mod_model))
    if tok is not None:
        tok.save_pretrained(save_dir)
    if cfg is not None:
        cfg.to_json(os.path.join(save_dir, "mod_config.json"))


def _resolve_gate_file(ckpt_dir: str, filename: str) -> str:
    """Auto-detects gate.pt vs gate.safetensors: an explicit existing filename wins;
    otherwise a gate.pt request falls back to a sibling gate.safetensors (and vice
    versa), so writers and readers never need to agree on a flag."""
    path = os.path.join(ckpt_dir, filename)
    if os.path.exists(path):
        return path
    stem, ext = os.path.splitext(filename)
    alt = stem + (".safetensors" if ext == ".pt" else ".pt")
    alt_path = os.path.join(ckpt_dir, alt)
    return alt_path if os.path.exists(alt_path) else path


def load_gate(mod_model, ckpt_dir: str, filename: str = "gate.pt"):
    """Loads gate.pt (strict=False, returns (missing, unexpected) for the caller to
    assert/print). Warns loudly when gating/LoRA keys end up missing — the silent
    strict=False drop historically masked wrong-scheme ckpts and unwrapped LoRA
    (r1 pitfall: eval quietly scored a half-initialized model)."""
    path = _resolve_gate_file(ckpt_dir, filename)
    if path.endswith(".safetensors"):
        try:
            from safetensors.torch import load_file
        except ImportError as e:
            raise ImportError(
                f"{path} needs the 'safetensors' package to load; "
                "pip install safetensors or use a gate.pt ckpt") from e
        sd = load_file(path, device="cpu")
    else:
        sd = torch.load(path, map_location="cpu")
    missing, unexp = mod_model.load_state_dict(sd, strict=False)
    gate_missing = [k for k in missing if is_gate_key(k) or "lora_" in k]
    if gate_missing:
        logger.warning(f"{len(gate_missing)} gating/LoRA keys missing after loading "
                       f"{filename} (e.g. {gate_missing[:3]}); ckpt scheme or LoRA spec likely "
                       f"mismatched — those gates silently run at fresh initialization")
    return missing, unexp


def strip_promoted_gate(state_dict: dict, promoted) -> dict:
    """Gate key filtering after profile promotion: drop joint_router entirely + promoted layers'
    per-layer keys.
    (moe: promotion changes the gated layer count G, joint_router dimensions mismatch, reinit on
     resumed training;
     threshold: only drop the promoted layers' router/tau/comp.)"""
    promoted = set(promoted)
    return {k: v for k, v in state_dict.items()
            if not ("joint_router" in k
                    or (is_gate_key(k) and any(f"layers.{i}." in k for i in promoted)))}
