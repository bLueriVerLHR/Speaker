"""Shared boilerplate for training/evaluation scripts (ed5 consolidation): device / tokenizer /
model loading / data slicing / LoRA / parameter grouping.

Accounting convention: only deduplicate repeated lines, numerical paths untouched (same-seed
smoke metrics bit-identical).
"""
from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Name markers of gating parameters in named_parameters (baseline models only have router;
# the other markers never match)
GATE_KEYS = ("router", "tau", "comp")


def resolve_device(name: str) -> torch.device:
    """Unified device resolution + VRAM print (after worker device assignment only cuda:0 is
    visible; hardcoding other indices is forbidden)."""
    device = torch.device(name)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        print(f"Using {device} free {torch.cuda.mem_get_info(device)[0] / 1024**3:.1f}GB",
              flush=True)
    else:
        print(f"Using {device}", flush=True)
    return device


def dtype_of(name: str):
    return {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[name]


def build_tok(model_id: str):
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def build_model(model_id: str, device: torch.device, dtype=torch.bfloat16):
    """Loads the base model (device_map=None whole-card placement, consistent with history)."""
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=dtype, device_map=None,
        trust_remote_code=True, low_cpu_mem_usage=True).to(device)
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
                       freeze_base: bool = False, freeze_gate: bool = False,
                       gate_keys=GATE_KEYS):
    """Base/gating parameter grouping. freeze_base = finetune the router (gating) only;
    freeze_gate = train the base only (ablation). named_parameters auto-deduplicates the
    hf_model/layers dual registration paths; a single pass suffices."""
    base_params = [p for n, p in mod_model.named_parameters()
                   if not any(k in n for k in gate_keys)]
    gate_params = mod_model.get_router_parameters()
    if freeze_base:
        for p in base_params:
            p.requires_grad_(False)
        groups = [{"params": gate_params, "lr": router_lr}]
    elif freeze_gate:
        for p in gate_params:
            p.requires_grad_(False)
        groups = [{"params": base_params, "lr": lr}]
    else:
        groups = [{"params": base_params, "lr": lr}, {"params": gate_params, "lr": router_lr}]
    return groups, base_params, gate_params
