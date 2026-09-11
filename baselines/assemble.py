"""Single source of truth for assembling an evaluable model from a ckpt dir.

Every eval tool (eval_compare / eval_gen / probe_kdist) builds models through
ModelBuilder, so the per-family recipe — base source, patch step, LoRA wrap
order, adapter file, load checks — lives here exactly once. Before this module
the recipe was copy-pasted per tool and had already drifted: eval_gen's RT
loader skipped the router-key assert and its ours loader lacked save_full base
support (both silent-divergence bugs found 0910).

Canonical pipelines (order identical to training, verified by the r3 patch->peft
roundtrip):

  ours      base -> LoRA(CLI spec) -> convert_to_speaker -> gate.pt -> hard mode
  dense_ft  base -> LoRA(ckpt spec) -> lora.pt
  modd      base(ckpt base if present) -> patch -> LoRA(ckpt spec) -> routers.pt
  mdf       base(ckpt base if present) -> patch -> LoRA(ckpt spec) -> routers.pt
  rt        base -> patch -> routers.pt   (gates only, no LoRA path)

Usage:
    asm = (ModelBuilder(args.model_id, args)
           .from_ckpt(ckpt_dir)     # family auto-detect + config + base source
           .build(device))           # executes the pipeline, loud load report
    asm.model / asm.name / asm.routed / asm.n_dense / asm.n_always ...
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

import torch

from speaker.checkpoint import load_gate

from .lib import patch_model_mdf, patch_model_modd, patch_model_rt  # noqa: E402

FAMILY_CONFIG = {
    "modd": "modd_config.json",
    "mdf": "mdf_config.json",
    "rt": "rt_config.json",
    "dense_ft": "denseft_config.json",
    "ours": "mod_config.json",
}
DISPLAY_PREFIX = {"ours": "ours", "dense_ft": "denseft", "modd": "modd", "mdf": "mdf", "rt": "rt"}


def detect_family(ckpt_dir: str) -> str:
    """Family = which config file the ckpt dir carries."""
    for fam, fn in FAMILY_CONFIG.items():
        if os.path.exists(os.path.join(ckpt_dir, fn)):
            return fam
    raise FileNotFoundError(
        f"[{ckpt_dir}] no family config found (expected one of {sorted(FAMILY_CONFIG.values())})")


@dataclass
class Assembly:
    """The built, adapter-loaded model plus everything downstream eval needs."""
    model: Any
    family: str
    ckpt: str
    cfg: dict = field(default_factory=dict)
    base_src: str = ""
    routed: list | None = None      # modd/mdf: is_routed flags
    gated: list | None = None       # rt: is_mod flags
    n_dense: int = 0                # modd/mdf: non-routed (always-on) layers
    n_always: int = 0               # rt: non-modulated layers

    @property
    def name(self) -> str:
        return f"{DISPLAY_PREFIX[self.family]}:{os.path.basename(self.ckpt)}"

    @property
    def n_mod(self) -> int:
        return len(self.gated) if self.gated else 0


def _lora_from_cli(args) -> bool:
    return bool(getattr(args, "use_lora", False))


def _norm_targets(targets) -> list:
    """Configs store a list, CLI passes a comma string — normalize to a list."""
    if isinstance(targets, str):
        return [t.strip() for t in targets.split(",") if t.strip()]
    return list(targets or [])


def _wrap_peft(model, rank: int, alpha: int, targets, dropout: float = 0.05):
    from peft import LoraConfig, TaskType, get_peft_model
    return get_peft_model(model, LoraConfig(
        r=rank, lora_alpha=alpha, target_modules=_norm_targets(targets),
        lora_dropout=dropout, bias="none", task_type=TaskType.CAUSAL_LM))


def _cli_lora_spec(args) -> dict:
    return dict(rank=getattr(args, "lora_rank", 8),
                alpha=getattr(args, "lora_alpha", 16),
                targets=getattr(args, "lora_targets", "q_proj,v_proj"))


def _cfg_lora_spec(cfg: dict, args) -> dict:
    """LoRA spec from the ckpt config, falling back to CLI values (historical behavior)."""
    return dict(rank=cfg.get("lora_rank", getattr(args, "lora_rank", 8)),
                alpha=cfg.get("lora_alpha", getattr(args, "lora_alpha", 16)),
                targets=cfg.get("lora_targets") or getattr(args, "lora_targets", "q_proj,v_proj"))


class ModelBuilder:
    """Fluent builder; from_ckpt records intent, build() runs the family pipeline."""

    def __init__(self, model_id: str, args=None, dtype=torch.bfloat16):
        self.model_id = model_id
        self.args = args
        self.dtype = dtype
        self.ckpt_dir: str | None = None
        self.family: str | None = None
        self.cfg: dict = {}
        self.base_src = model_id
        self._skip_mode: str | None = None

    def from_ckpt(self, ckpt_dir: str, family: str | None = None) -> "ModelBuilder":
        self.ckpt_dir = ckpt_dir
        self.family = family or detect_family(ckpt_dir)
        with open(os.path.join(ckpt_dir, FAMILY_CONFIG[self.family])) as f:
            self.cfg = json.load(f)
        # joint-training ckpts ship their own full base (config.json) — only
        # ours/modd/mdf ever save one; dense-ft/rt always load the CLI base
        if self.family in ("ours", "modd", "mdf") and \
                os.path.exists(os.path.join(ckpt_dir, "config.json")):
            self.base_src = ckpt_dir
        return self

    def skip_mode(self, mode: str) -> "ModelBuilder":
        self._skip_mode = mode
        return self

    # ---- pipeline primitives -------------------------------------------------

    def _load_base(self):
        from transformers import AutoModelForCausalLM
        return AutoModelForCausalLM.from_pretrained(
            self.base_src, dtype=self.dtype, device_map=None,
            trust_remote_code=True, low_cpu_mem_usage=True)

    def _report(self, missing, unexp, filename, extra: str = ""):
        print(f"[{os.path.basename(self.ckpt_dir)}] {filename} missing {len(missing)} "
              f"unexpected {len(unexp)}{extra}", flush=True)

    def _load_adapters(self, model, filename: str):
        sd = torch.load(os.path.join(self.ckpt_dir, filename), map_location="cpu")
        missing, unexp = model.load_state_dict(sd, strict=False)
        n_drop = sum(1 for k in missing if "lora" in k or "router" in k)
        assert n_drop == 0, (f"[{self.ckpt_dir}] {filename} keys did not match "
                             f"(LoRA spec inconsistent with training?)")
        extra = f" (base from {os.path.basename(self.base_src)})" if self.family in ("modd", "mdf") else ""
        self._report(missing, unexp, filename, extra)
        return model

    # ---- family pipelines ----------------------------------------------------

    def build(self, device=None) -> Assembly:
        if not self.family:
            raise RuntimeError("call from_ckpt() before build()")
        fam = self.family
        asm = Assembly(model=None, family=fam, ckpt=self.ckpt_dir, cfg=self.cfg,
                       base_src=self.base_src)
        if fam == "ours":
            m = self._load_base()
            if _lora_from_cli(self.args):
                m = _wrap_peft(m, **_cli_lora_spec(self.args))
            from speaker.config import SpeakerConfig
            from speaker.wrapper import convert_to_speaker, apply_decode_config
            m = convert_to_speaker(m, SpeakerConfig.from_json(
                os.path.join(self.ckpt_dir, "mod_config.json")))
            missing, unexp = load_gate(m, self.ckpt_dir)
            self._report(missing, unexp, "gate.pt")
            if self._skip_mode:
                m.set_skip_mode(self._skip_mode)
            apply_decode_config(m, (self.cfg or {}).get("decode"))
            asm.model = m
        elif fam == "dense_ft":
            m = self._load_base()
            if self.cfg.get("use_lora", True):
                m = _wrap_peft(m, **_cfg_lora_spec(self.cfg, self.args))
            sd = torch.load(os.path.join(self.ckpt_dir, "lora.pt"), map_location="cpu")
            missing, unexp = m.load_state_dict(sd, strict=False)
            n_lora = sum(1 for k in missing if "lora" in k)
            assert n_lora == 0, (f"[{self.ckpt_dir}] lora.pt keys did not match "
                                 f"(rank/targets inconsistent with training?)")
            print(f"[{os.path.basename(self.ckpt_dir)}] lora.pt missing {len(missing)} "
                  f"(lora {n_lora}) unexpected {len(unexp)}", flush=True)
            asm.model = m
        elif fam in ("modd", "mdf"):
            m = self._load_base()
            if fam == "modd":
                asm.routed = patch_model_modd(m, self.cfg["is_routed"],
                                              capacity=self.cfg.get("capacity", 0.125))
            else:
                asm.routed = patch_model_mdf(m, self.cfg["is_routed"], p=self.cfg.get("p", 0.5))
            if self.cfg.get("use_lora"):  # same order as training (patch -> peft)
                m = _wrap_peft(m, **_cfg_lora_spec(self.cfg, self.args))
            self._load_adapters(m, "routers.pt")
            asm.n_dense = len(self.cfg["is_routed"]) - sum(self.cfg["is_routed"])
            asm.model = m
        elif fam == "rt":
            m = self._load_base()
            asm.gated = patch_model_rt(m, self.cfg["is_mod"],
                                       self.cfg.get("granularity", "block_token"),
                                       self.cfg.get("threshold", 0.5), self.cfg.get("target"),
                                       self.cfg.get("scale", 0.0))
            self._load_adapters(m, "routers.pt")
            asm.n_always = len(self.cfg["is_mod"]) - sum(self.cfg["is_mod"])
            asm.model = m
        else:  # pragma: no cover
            raise ValueError(f"unknown family {fam}")
        if device is not None:
            asm.model = asm.model.to(device)
        return asm


def assemble(ckpt_dir: str, model_id: str, args=None, device=None,
             dtype=torch.bfloat16, skip_mode: str | None = "hard") -> Assembly:
    """One-liner for the common eval path (ours defaults to hard skip mode)."""
    return (ModelBuilder(model_id, args, dtype)
            .from_ckpt(ckpt_dir)
            .skip_mode(skip_mode)
            .build(device))
