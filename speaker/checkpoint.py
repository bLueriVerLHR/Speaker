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


def save_gate(mod_model, save_dir: str, extra_marks=(), filename: str = "gate.pt") -> dict:
    """Saves only gating (+ optional lora) keys; returns the saved state_dict."""
    os.makedirs(save_dir, exist_ok=True)
    sd = {k: v.cpu() for k, v in
          gate_state_dict(mod_model.state_dict(), extra_marks).items()}
    torch.save(sd, os.path.join(save_dir, filename))
    return sd


def save_clean_base(mod_model, tok=None, cfg=None, save_dir: str = "."):
    """Self-contained ckpt base part: clean base (+ tokenizer + mod_config.json)."""
    os.makedirs(save_dir, exist_ok=True)
    mod_model.hf_model.save_pretrained(save_dir, state_dict=clean_base_state_dict(mod_model))
    if tok is not None:
        tok.save_pretrained(save_dir)
    if cfg is not None:
        cfg.to_json(os.path.join(save_dir, "mod_config.json"))


def load_gate(mod_model, ckpt_dir: str, filename: str = "gate.pt"):
    """Loads gate.pt (strict=False, returns (missing, unexpected) for the caller to
    assert/print)."""
    sd = torch.load(os.path.join(ckpt_dir, filename), map_location="cpu")
    return mod_model.load_state_dict(sd, strict=False)


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
