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
    asm = (ModelBuilder(model_id, lora={"rank": 8, "alpha": 16, "targets": "q_proj,v_proj"})
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
from speaker.log import logger

from .lib import patch_model_mdf, patch_model_modd, patch_model_rt  # noqa: E402

FAMILY_CONFIG = {
    "modd": "modd_config.json",
    "mdf": "mdf_config.json",
    "rt": "rt_config.json",
    "dense_ft": "denseft_config.json",
    "ours": "mod_config.json",
}
DISPLAY_PREFIX = {"ours": "ours", "dense_ft": "denseft", "modd": "modd", "mdf": "mdf", "rt": "rt"}

# ---- Decorator registry (P1 structure): identifier -> implementation mapping as
# a first-class boundary (transformers AutoModel lesson). New families register
# their pipeline fn here; ModelBuilder.build() dispatches through it instead of
# an if/elif chain. FAMILY_CONFIG / DISPLAY_PREFIX stay as derived views so
# existing imports keep working. ----
FAMILY_REGISTRY: dict = {}


def register_family(name: str, config_file: str, display_prefix: str):
    """Decorator: registers a family pipeline ``fn(builder, asm) -> Assembly``."""
    def deco(fn):
        if name in FAMILY_REGISTRY:
            raise ValueError(f"family {name!r} already registered "
                             f"(duplicate key, check for copy-paste)")
        FAMILY_REGISTRY[name] = {
            "config": config_file, "prefix": display_prefix, "fn": fn,
        }
        return fn
    return deco


def list_families() -> list:
    return sorted(FAMILY_REGISTRY)


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


def _lora_spec(cfg: dict, cli: dict | None) -> dict:
    """LoRA spec from the ckpt config, falling back to CLI values (historical
    behavior). cli is None when the caller wants no LoRA fallback."""
    cli = cli or {}
    return dict(rank=cfg.get("lora_rank", cli.get("rank", 8)),
                alpha=cfg.get("lora_alpha", cli.get("alpha", 16)),
                targets=cfg.get("lora_targets") or cli.get("targets", "q_proj,v_proj"))


def lora_from_checkpoint(cfg: dict, cli: dict | None = None) -> dict | None:
    """Return the persisted adapter recipe, with legacy CLI fallback.

    Older checkpoints did not record ``use_lora``; in that case the caller's
    explicit CLI choice remains authoritative. New checkpoints are self
    describing and cannot silently load adapter weights into a dense model.
    """
    if "use_lora" in cfg:
        return _lora_spec(cfg, cli) if cfg["use_lora"] else None
    if cli is not None:
        logger.warning("checkpoint has no persisted LoRA spec; falling back to CLI "
                       "rank/alpha/targets (legacy checkpoint)")
    return _lora_spec(cfg, cli) if cli is not None else None


class ModelBuilder:
    """Fluent builder; from_ckpt records intent, build() runs the family pipeline."""

    def __init__(self, model_id: str, lora: dict | None = None,
                 dtype=torch.bfloat16, device_map=None):
        self.model_id = model_id
        self.lora = lora  # None = no LoRA; else {rank, alpha, targets}
        self.dtype = dtype
        # device_map="auto": accelerate sharding across visible GPUs (backbones larger than
        # one card); build(device) must then be None — the shards decide placement
        self.device_map = device_map
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
        # shared loader (train_common.build_model): VL/nested-text backbones dispatch to
        # AutoModelForImageTextToText, traditional ones to AutoModelForCausalLM
        from speaker.train_common import build_model
        return build_model(self.base_src, torch.device("cpu"), self.dtype,
                           device_map=self.device_map)

    def _report(self, missing, unexp, filename, extra: str = ""):
        logger.info(f"[{os.path.basename(self.ckpt_dir)}] {filename} missing {len(missing)} "
                    f"unexpected {len(unexp)}{extra}")

    def _load_adapters(self, model, filename: str):
        sd = torch.load(os.path.join(self.ckpt_dir, filename), map_location="cpu")
        missing, unexp = model.load_state_dict(sd, strict=False)
        n_drop = sum(1 for k in missing if "lora" in k or "router" in k)
        assert n_drop == 0, (f"[{self.ckpt_dir}] {filename} keys did not match "
                             f"(LoRA spec inconsistent with training?)")
        extra = f" (base from {os.path.basename(self.base_src)})" if self.family in ("modd", "mdf") else ""
        self._report(missing, unexp, filename, extra)
        return model

    # ---- family pipelines (Registry dispatch; bodies moved verbatim) --------

    def build(self, device=None) -> Assembly:
        if not self.family:
            raise RuntimeError("call from_ckpt() before build()")
        try:
            entry = FAMILY_REGISTRY[self.family]
        except KeyError:
            raise ValueError(f"unknown family {self.family}") from None
        asm = Assembly(model=None, family=self.family, ckpt=self.ckpt_dir,
                       cfg=self.cfg, base_src=self.base_src)
        entry["fn"](self, asm)
        if self.device_map is not None and device is not None:
            raise ValueError("device_map sharding and build(device) are mutually exclusive "
                             "(.to would collapse the shards); pass device=None")
        if device is not None:
            asm.model = asm.model.to(device)
        return asm


@register_family("ours", "mod_config.json", "ours")
def _build_ours(b: "ModelBuilder", asm: Assembly) -> Assembly:
    m = b._load_base()
    spec = lora_from_checkpoint(b.cfg, b.lora)
    if spec is not None:
        m = _wrap_peft(m, **spec)
    from speaker.config import SpeakerConfig
    from speaker.wrapper import convert_to_speaker, apply_decode_config
    m = convert_to_speaker(m, SpeakerConfig.from_json(
        os.path.join(b.ckpt_dir, "mod_config.json")))
    missing, unexp = load_gate(m, b.ckpt_dir)
    b._report(missing, unexp, "gate.pt")
    if b._skip_mode:
        m.set_skip_mode(b._skip_mode)
    apply_decode_config(m, (b.cfg or {}).get("decode"))
    asm.model = m
    return asm


@register_family("dense_ft", "denseft_config.json", "denseft")
def _build_dense_ft(b: "ModelBuilder", asm: Assembly) -> Assembly:
    m = b._load_base()
    if b.cfg.get("use_lora", True):
        m = _wrap_peft(m, **_lora_spec(b.cfg, b.lora))
    sd = torch.load(os.path.join(b.ckpt_dir, "lora.pt"), map_location="cpu")
    missing, unexp = m.load_state_dict(sd, strict=False)
    n_lora = sum(1 for k in missing if "lora" in k)
    assert n_lora == 0, (f"[{b.ckpt_dir}] lora.pt keys did not match "
                         f"(rank/targets inconsistent with training?)")
    logger.info(f"[{os.path.basename(b.ckpt_dir)}] lora.pt missing {len(missing)} "
                f"(lora {n_lora}) unexpected {len(unexp)}")
    asm.model = m
    return asm


@register_family("modd", "modd_config.json", "modd")
def _build_modd(b: "ModelBuilder", asm: Assembly) -> Assembly:
    return _build_routed(b, asm, kind="modd")


@register_family("mdf", "mdf_config.json", "mdf")
def _build_mdf(b: "ModelBuilder", asm: Assembly) -> Assembly:
    return _build_routed(b, asm, kind="mdf")


def _build_routed(b: "ModelBuilder", asm: Assembly, kind: str) -> Assembly:
    m = b._load_base()
    if kind == "modd":
        asm.routed = patch_model_modd(m, b.cfg["is_routed"],
                                      capacity=b.cfg.get("capacity", 0.125))
    else:
        asm.routed = patch_model_mdf(m, b.cfg["is_routed"], p=b.cfg.get("p", 0.5))
    if b.cfg.get("use_lora"):  # same order as training (patch -> peft)
        m = _wrap_peft(m, **_lora_spec(b.cfg, b.lora))
    b._load_adapters(m, "routers.pt")
    asm.n_dense = len(b.cfg["is_routed"]) - sum(b.cfg["is_routed"])
    asm.model = m
    return asm


@register_family("rt", "rt_config.json", "rt")
def _build_rt(b: "ModelBuilder", asm: Assembly) -> Assembly:
    m = b._load_base()
    asm.gated = patch_model_rt(m, b.cfg["is_mod"],
                               b.cfg.get("granularity", "block_token"),
                               b.cfg.get("threshold", 0.5), b.cfg.get("target"),
                               b.cfg.get("scale", 0.0))
    b._load_adapters(m, "routers.pt")
    asm.n_always = len(b.cfg["is_mod"]) - sum(b.cfg["is_mod"])
    asm.model = m
    return asm


def assemble(ckpt_dir: str, model_id: str, lora: dict | None = None,
             device=None, dtype=torch.bfloat16, skip_mode: str | None = "hard",
             device_map=None) -> Assembly:
    """One-liner for the common eval path (ours defaults to hard skip mode).
    lora: None = no LoRA; else {rank, alpha, targets} (CLI spec, ckpt config
    takes precedence per family). device_map="auto": shard the base across
    visible GPUs (pass device=None)."""
    return (ModelBuilder(model_id, lora, dtype, device_map=device_map)
            .from_ckpt(ckpt_dir)
            .skip_mode(skip_mode)
            .build(device))
