"""Training hyperparameter bundles (typed dataclasses, not argparse).

Entry points (Typer commands) take flat typed flags and pack them into one of
these bundles for the shared workers (finetune.pipeline.run_finetune). The
bundle is constructed with explicit kwargs at a single call site per entry —
no argparse.Namespace anywhere in the codebase.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class FinetuneConfig:
    """All finetune-track knobs (mirrors finetune.cli.train flags/defaults)."""
    model_id: str = "/home/hdd/model/Qwen2.5-7B-Instruct"
    data_path: str = "./data/sft_t2t_mini.jsonl"
    device: str = "cuda:0"
    device_map: str = ""
    dtype: Literal["float32", "bfloat16", "float16"] = "bfloat16"
    batch_size: int = 2
    max_length: int = 1024
    anneal_steps: int = 4000
    max_steps: int = 4000
    lr: float = 2e-5
    router_lr: float = 1e-4
    kmax: int = 16
    decode_rep_penalty: Optional[float] = None
    decode_no_repeat_ngram: Optional[int] = None
    gate_mode: Literal["moe", "mol", "threshold", "speaker"] = "threshold"
    select_mode: Literal["topp", "topk"] = "topp"
    top_p: float = 0.9
    top_k: int = 6
    weight_mode: Literal["pmax", "renorm"] = "pmax"
    min_layers: int = 1
    always_head: int = 2
    always_tail: int = 2
    always_layers: str = ""
    temp_affinity: float = 1.0
    ta_end: float = 0.3
    gumbel_scale: float = 1.0
    tau_init: float = 0.0
    sparsity_price: float = 0.0005
    price_adapt: bool = True
    acc_target: str = "none"
    acc_margin: float = 0.03
    price_warmup: int = 200
    budget_ramp: int = 400
    budget_form: Literal["mean", "hinge", "tail"] = "mean"
    budget_target: float = 0.0
    tail_coef: float = 1.0
    tail_temp: float = 0.5
    diff_mode: Literal["off", "teacher"] = "off"
    diff_easy_nll: float = 0.5
    diff_hard_nll: float = 2.5
    diff_easy_mult: float = 2.0
    diff_hard_mult: float = 0.5
    eval_samples: int = 300
    max_samples: int = 904000
    log_interval: int = 20
    save_dir: str = "/tmp/mod_ckpt"
    seed: Optional[int] = 42
    use_lora: bool = True
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_targets: str = "q_proj,v_proj"
    gradient_checkpointing: bool = True
    use_chat_template: bool = False
    mask_user_tokens: bool = True
    kl_coef: float = 0.0
    kl_temp: float = 1.0
    teacher_model_id: str = ""
    teacher_device: str = "cuda:0"
    cos_reg_coef: float = 0.01
    tau_calib_batches: int = 8
    tau_spread: float = 0.5
    router_calib_batches: int = 0
    router_start_k: int = 0
    resume_dir: str = ""
    eval_every: int = 500
    patience: int = 100
    save_full: bool = False
    ul_mode: Literal["none", "gt", "rollout"] = "rollout"
    ul_coef: float = 0.3
    ul_n: int = 3
    rollout_every: int = 20
    rollout_tokens: int = 128
    rollout_prompt: int = 32
    rep_probe: bool = True
    accelerator: Literal["none", "accelerate", "fabric"] = "none"


__all__ = ["FinetuneConfig"]
