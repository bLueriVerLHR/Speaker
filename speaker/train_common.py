"""Shared boilerplate for training/evaluation scripts (ed5 consolidation): device / tokenizer /
model loading / data slicing / LoRA / parameter grouping.

Accounting convention: only deduplicate repeated lines, numerical paths untouched (same-seed
smoke metrics bit-identical).
"""
from __future__ import annotations

import random

import torch
from transformers import AutoTokenizer

from .log import logger

# Name markers of gating parameters in named_parameters (baseline models only have router;
# the other markers never match)
GATE_KEYS = ("router", "tau", "comp")


def resolve_device(name: str) -> torch.device:
    """Unified device resolution + VRAM print (after worker device assignment only cuda:0 is
    visible; hardcoding other indices is forbidden)."""
    device = torch.device(name)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        logger.info(f"Using {device} free "
                    f"{torch.cuda.mem_get_info(device)[0] / 1024**3:.1f}GB")
    else:
        logger.info(f"Using {device}")
    return device


def dtype_of(name: str):
    return {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[name]


def build_tok(model_id: str):
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def build_model(model_id: str, device: torch.device, dtype=torch.bfloat16,
                device_map=None):
    """Loads the base model. device_map=None (default) = whole-card placement (.to(device),
    consistent with history); device_map="auto" = accelerate sharding across all visible GPUs
    for backbones larger than one card (e.g. 27B on 24GB cards) — no .to(device), the training
    loop must then keep inputs on the first device and tolerate logits landing on the last.
    Nested-text (VL/multimodal) backbones load through AutoModelForImageTextToText: the vision
    tower stays in the way (inert without pixel_values), the decoder stack is what we gate."""
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    if getattr(cfg, "text_config", None) is not None:
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(
            model_id, dtype=dtype, device_map=device_map,
            trust_remote_code=True, low_cpu_mem_usage=True)
    else:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=dtype, device_map=device_map,
            trust_remote_code=True, low_cpu_mem_usage=True)
    if device_map is None:
        model = model.to(device)
    return model


def split_train_eval(data_path: str, max_samples: int, eval_samples: int,
                     tok=None, use_chat: bool = False):
    """Standard slice: first max_samples for training / the following eval_samples for eval.
    Returns (train_ds, eval_texts)."""
    from data.sft import SFTDataset
    full = SFTDataset(data_path, max_samples + eval_samples, tok=tok, use_chat=use_chat)
    eval_texts = full.samples[max_samples:max_samples + eval_samples]
    full.samples = full.samples[:max_samples]
    return full, eval_texts


def eval_slice(data_path: str, offset: int, n: int, tok=None, use_chat: bool = False):
    """Eval slice: n samples starting at offset (shared by eval_ckpt/profile_layers/eval_compare)."""
    from data.sft import SFTDataset
    full = SFTDataset(data_path, offset + n, tok=tok, use_chat=use_chat)
    texts = full.samples[offset:offset + n]
    assert texts, (f"eval slice is empty: file has {len(full.samples)} samples, is offset {offset} out of range? "
                   "check that build-time filtering matches the SFTDataset accounting")
    return texts


def wrap_lora(model, rank: int = 8, alpha: int = 16, targets: str = "q_proj,v_proj",
              dropout: float = 0.05):
    """Unified LoRA spec (same rank8 q,v as ours/baselines; key layout consistent with training)."""
    from peft import LoraConfig, TaskType, get_peft_model
    return get_peft_model(model, LoraConfig(
        r=rank, lora_alpha=alpha,
        target_modules=[t.strip() for t in targets.split(",") if t.strip()],
        lora_dropout=dropout, bias="none", task_type=TaskType.CAUSAL_LM))


def build_param_groups(mod_model, lr: float, router_lr: float,
                       gate_keys=GATE_KEYS):
    """Base/gating parameter grouping (joint training). named_parameters
    auto-deduplicates the hf_model/layers dual registration paths; a single pass suffices."""
    base_params = [p for n, p in mod_model.named_parameters()
                   if not any(k in n for k in gate_keys)]
    gate_params = mod_model.get_router_parameters()
    groups = [{"params": base_params, "lr": lr}, {"params": gate_params, "lr": router_lr}]
    return groups, base_params, gate_params


def parse_csv_list(s: str) -> list:
    """Comma-separated int list (e.g. --always_layers); blank entries skipped."""
    return [int(x) for x in (s or "").split(",") if x.strip() != ""]


def seed_all(seed):
    random.seed(seed)
    torch.manual_seed(seed)


def build_optimizer(groups, weight_decay: float = 0.01):
    return torch.optim.AdamW(groups, weight_decay=weight_decay)


def enable_checkpointing(model, use_lora: bool = False):
    """Gradient checkpointing + the LoRA frozen-embed fix; no-op without support."""
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        if use_lora:
            # peft: with a frozen base, checkpointing requires inputs to carry
            # gradients, otherwise backward breaks (frozen-embed pitfall)
            model.enable_input_require_grads()
