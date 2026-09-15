"""Finetune CLI definition (Typer, single source of truth).

The 70+ ``add_argument`` lines are replaced by one Typer ``train()``
signature: each parameter is a single
``Annotated[type, typer.Option("--flag_name", help=...)]`` line, defaults and
``Literal`` choices inline, ``--help`` rendered by Typer. Flag names keep the
historical underscore style (``--use_lora`` not ``--use-lora``) and bool
switches keep their ``--x/--no-x`` dual form, so existing job-queue submit
lines work unchanged.

``train()`` packs its flags into a ``speaker.hparams.FinetuneConfig`` (explicit
kwargs, typed dataclass) for ``finetune.pipeline.run_finetune`` — no argparse
anywhere.
"""
from __future__ import annotations

from typing import Annotated, Literal, Optional

import typer

app = typer.Typer(add_completion=False)


@app.command()
def train(
    model_id: Annotated[str, typer.Option("--model_id", help="finetune route defaults to 7B; only the pretrain route uses small models")] = "/home/hdd/model/Qwen2.5-7B-Instruct",
    data_path: Annotated[str, typer.Option("--data_path", help="SFT jsonl path")] = "./data/sft_t2t_mini.jsonl",
    device: Annotated[str, typer.Option("--device", help="training device (worker sees cuda:0..N-1 only)")] = "cuda:0",
    device_map: Annotated[str, typer.Option("--device_map", help="empty = whole-card placement (default); 'auto' = accelerate sharding across ALL visible GPUs")] = "",
    dtype: Annotated[Literal["float32", "bfloat16", "float16"], typer.Option("--dtype", help="model dtype")] = "bfloat16",
    batch_size: Annotated[int, typer.Option("--batch_size", help="per-device batch size")] = 2,
    max_length: Annotated[int, typer.Option("--max_length", help="truncation length")] = 1024,
    anneal_steps: Annotated[int, typer.Option("--anneal_steps", help="Ta/gumbel annealing horizon (denominator); stopping is governed solely by --max_steps")] = 4000,
    max_steps: Annotated[int, typer.Option("--max_steps", help="hard step cap; actual stopping is the StopOnPlateau rule")] = 4000,
    lr: Annotated[float, typer.Option("--lr", help="base learning rate")] = 2e-5,
    router_lr: Annotated[float, typer.Option("--router_lr", help="gating-parameter learning rate")] = 1e-4,
    kmax: Annotated[int, typer.Option("--kmax", help="moe: hard cap during selection; threshold: NOT enforced in selection — binds only via --over_budget_coef>0 (loss-side cap)")] = 16,
    decode_rep_penalty: Annotated[Optional[float], typer.Option("--decode_rep_penalty", help="bake repetition_penalty into mod_config.json; None = legacy (validated 1.15)")] = None,
    decode_no_repeat_ngram: Annotated[Optional[int], typer.Option("--decode_no_repeat_ngram", help="bake no_repeat_ngram_size into mod_config.json; None = legacy (validated 3)")] = None,
    gate_mode: Annotated[Literal["moe", "mol", "threshold", "speaker"], typer.Option("--gate_mode", help="threshold = per-layer gating (default) / moe = joint routing; ckpt takes precedence on resume")] = "threshold",
    select_mode: Annotated[Literal["topp", "topk"], typer.Option("--select_mode", help="moe selection rule: topp = adaptive k / topk = fixed k")] = "topp",
    top_p: Annotated[float, typer.Option("--top_p", help="moe topp cumulative-probability threshold")] = 0.9,
    top_k: Annotated[int, typer.Option("--top_k", help="moe topk fixed k")] = 6,
    weight_mode: Annotated[Literal["pmax", "renorm"], typer.Option("--weight_mode", help="moe residual weighting: pmax (default, dual lever connected) / renorm (legacy, collapses)")] = "pmax",
    min_layers: Annotated[int, typer.Option("--min_layers", help="moe: minimum gated layers activated per token")] = 1,
    always_head: Annotated[int, typer.Option("--always_head", help="first m layers fixed (always on); finetune default 0 = every layer gated during training, use --always_layers to pin fixed layers for deployment runs")] = 0,
    always_tail: Annotated[int, typer.Option("--always_tail", help="last n layers fixed (always on); finetune default 0 = every layer gated during training")] = 0,
    always_layers: Annotated[str, typer.Option("--always_layers", help="explicit fixed-layer list (comma-separated), overrides head/tail; e.g. '0,1,2,22,23'")] = "",
    temp_affinity: Annotated[float, typer.Option("--temp_affinity", help="initial gating temperature Ta")] = 1.0,
    ta_end: Annotated[float, typer.Option("--ta_end", help="annealed Ta end value")] = 0.3,
    gumbel_scale: Annotated[float, typer.Option("--gumbel_scale", help="training noise strength")] = 1.0,
    tau_init: Annotated[float, typer.Option("--tau_init", help="threshold: initial gating threshold; negative = dense start")] = 0.0,
    sparsity_price: Annotated[float, typer.Option("--sparsity_price", help="lambda: loss += lambda*mean(k)")] = 0.0005,
    price_adapt: Annotated[bool, typer.Option("--price_adapt/--no-price_adapt", help="dual adjustment of lambda")] = True,
    acc_target: Annotated[str, typer.Option("--acc_target", help="dual floor: 'auto' = dense_acc-margin | float = legacy floor | 'none' = dual off (default)")] = "none",
    acc_margin: Annotated[float, typer.Option("--acc_margin", help="auto policy: allowed drop below the dense reference")] = 0.03,
    price_warmup: Annotated[int, typer.Option("--price_warmup", help="lambda frozen for the first N steps")] = 200,
    budget_ramp: Annotated[int, typer.Option("--budget_ramp", help="budget-loss ramp steps (task*=min(1,step/ramp))")] = 400,
    budget_form: Annotated[Literal["mean", "hinge", "tail", "sqdev"], typer.Option("--budget_form", help="budget shape: mean (legacy λ·mean) | hinge | tail | sqdev (two-sided push toward target T; acts on TOTAL k incl. fixed layers; converges to T+2.5~8)")] = "mean",
    budget_target: Annotated[float, typer.Option("--budget_target", help="hinge setpoint T / tail budget B / sqdev target; 0 = auto -> kmax")] = 0.0,
    over_budget_coef: Annotated[float, typer.Option("--over_budget_coef", help="quadratic one-sided pin coef*mean(max(k_gated-kmax,0)^2); 0.05 = r8c recipe depth pin (threshold mode: the only kmax enforcement); 0 = k only λ-priced, drifts near-dense")] = 0.0,
    tail_coef: Annotated[float, typer.Option("--tail_coef", help="tail-violation weight relative to the mean term")] = 1.0,
    tail_temp: Annotated[float, typer.Option("--tail_temp", help="sigmoid softness for the P(k>B) counter")] = 0.5,
    diff_mode: Annotated[Literal["off", "teacher"], typer.Option("--diff_mode", help="difficulty-conditioned budget: teacher = per-token lambda shaping; off = uniform")] = "off",
    diff_easy_nll: Annotated[float, typer.Option("--diff_easy_nll", help="teacher NLL below this = easy token")] = 0.5,
    diff_hard_nll: Annotated[float, typer.Option("--diff_hard_nll", help="teacher NLL above this = hard token")] = 2.5,
    diff_easy_mult: Annotated[float, typer.Option("--diff_easy_mult", help="lambda multiplier on easy tokens")] = 2.0,
    diff_hard_mult: Annotated[float, typer.Option("--diff_hard_mult", help="lambda multiplier on hard tokens")] = 0.5,
    eval_samples: Annotated[int, typer.Option("--eval_samples", help="held-out eval samples right after the training slice")] = 300,
    max_samples: Annotated[int, typer.Option("--max_samples", help="training slice size")] = 904000,
    log_interval: Annotated[int, typer.Option("--log_interval", help="metric log cadence (steps)")] = 20,
    save_dir: Annotated[str, typer.Option("--save_dir", help="ckpt output dir")] = "/tmp/mod_ckpt",
    seed: Annotated[Optional[int], typer.Option("--seed", help="random seed (unset = legacy non-deterministic)")] = 42,
    use_lora: Annotated[bool, typer.Option("--use_lora/--no-use_lora", help="LoRA + joint-train with the gating")] = True,
    lora_rank: Annotated[int, typer.Option("--lora_rank", help="LoRA rank")] = 8,
    lora_alpha: Annotated[int, typer.Option("--lora_alpha", help="LoRA alpha")] = 16,
    lora_targets: Annotated[str, typer.Option("--lora_targets", help="LoRA target modules (comma-separated)")] = "q_proj,v_proj",
    gradient_checkpointing: Annotated[bool, typer.Option("--gradient_checkpointing/--no-gradient_checkpointing", help="gradient checkpointing (required for long seq/large batch)")] = True,
    use_chat_template: Annotated[bool, typer.Option("--use_chat_template/--no-use_chat_template", help="chat_template assembly (required for Instruct models)")] = False,
    single_turn: Annotated[bool, typer.Option("--single_turn/--no-single_turn", help="first-round truncation: keep only the first user+assistant round per sample (default off, P1 red-cell experiment)")] = False,
    mask_user_tokens: Annotated[bool, typer.Option("--mask_user_tokens/--no-mask_user_tokens", help="in chat mode, loss/acc only on assistant replies")] = True,
    kl_coef: Annotated[float, typer.Option("--kl_coef", help="dense self-distillation weight, 0 = off")] = 0.0,
    kl_temp: Annotated[float, typer.Option("--kl_temp", help="distillation temperature")] = 1.0,
    teacher_model_id: Annotated[str, typer.Option("--teacher_model_id", help="teacher path; empty = same as model_id")] = "",
    teacher_device: Annotated[str, typer.Option("--teacher_device", help="device hosting the frozen teacher")] = "cuda:0",
    cos_reg_coef: Annotated[float, typer.Option("--cos_reg_coef", help="StableSkip routing regularization weight, 0 = off")] = 0.01,
    tau_calib_batches: Annotated[int, typer.Option("--tau_calib_batches", help="threshold: model-aware tau calibration batches (0 = off)")] = 8,
    tau_spread: Annotated[float, typer.Option("--tau_spread", help="threshold: tau calibration spread")] = 0.5,
    router_calib_batches: Annotated[int, typer.Option("--router_calib_batches", help="moe: router-temp calibration batches (0 = off)")] = 0,
    router_start_k: Annotated[int, typer.Option("--router_start_k", help="moe: target initial mean k (0 = auto ~0.6*gated)")] = 0,
    resume_dir: Annotated[str, typer.Option("--resume_dir", help="resume ckpt dir (structure follows its mod_config.json)")] = "",
    eval_every: Annotated[int, typer.Option("--eval_every", help="plateau check cadence (steps)")] = 500,
    patience: Annotated[int, typer.Option("--patience", help="plateau tolerance count")] = 100,
    save_full: Annotated[bool, typer.Option("--save_full", help="also save the full base (non-LoRA only)")] = False,
    ul_mode: Annotated[Literal["none", "gt", "rollout"], typer.Option("--ul_mode", help="anti-repetition regularizer (default rollout)")] = "rollout",
    ul_coef: Annotated[float, typer.Option("--ul_coef", help="unlikelihood term weight (0 = off)")] = 0.3,
    ul_n: Annotated[int, typer.Option("--ul_n", help="n-gram length triggering UL")] = 3,
    rollout_every: Annotated[int, typer.Option("--rollout_every", help="rollout UL cadence (steps)")] = 20,
    rollout_tokens: Annotated[int, typer.Option("--rollout_tokens", help="tokens generated per rollout")] = 128,
    rollout_prompt: Annotated[int, typer.Option("--rollout_prompt", help="real-prefix length taken from the batch")] = 32,
    rep_probe: Annotated[bool, typer.Option("--rep_probe/--no-rep_probe", help="hard-greedy rollout probe at plateau beats")] = True,
    accelerator: Annotated[Literal["none", "accelerate", "fabric"], typer.Option("--accelerator", help="distributed backend: none (default, legacy) | accelerate | fabric")] = "none",
) -> None:
    """Gating-first finetune: train the gating on an existing model (see README track B)."""
    from speaker.hparams import FinetuneConfig
    from finetune.pipeline import run_finetune
    run_finetune(FinetuneConfig(
        model_id=model_id, data_path=data_path, device=device,
        device_map=device_map, dtype=dtype, batch_size=batch_size,
        max_length=max_length, single_turn=single_turn,
        anneal_steps=anneal_steps, max_steps=max_steps,
        lr=lr, router_lr=router_lr, kmax=kmax,
        decode_rep_penalty=decode_rep_penalty,
        decode_no_repeat_ngram=decode_no_repeat_ngram, gate_mode=gate_mode,
        select_mode=select_mode, top_p=top_p, top_k=top_k,
        weight_mode=weight_mode, min_layers=min_layers,
        always_head=always_head, always_tail=always_tail,
        always_layers=always_layers, temp_affinity=temp_affinity, ta_end=ta_end,
        gumbel_scale=gumbel_scale, tau_init=tau_init,
        sparsity_price=sparsity_price, price_adapt=price_adapt,
        acc_target=acc_target, acc_margin=acc_margin,
        price_warmup=price_warmup, budget_ramp=budget_ramp,
        budget_form=budget_form, budget_target=budget_target,
        over_budget_coef=over_budget_coef,
        tail_coef=tail_coef, tail_temp=tail_temp, diff_mode=diff_mode,
        diff_easy_nll=diff_easy_nll, diff_hard_nll=diff_hard_nll,
        diff_easy_mult=diff_easy_mult, diff_hard_mult=diff_hard_mult,
        eval_samples=eval_samples, max_samples=max_samples,
        log_interval=log_interval, save_dir=save_dir, seed=seed,
        use_lora=use_lora, lora_rank=lora_rank, lora_alpha=lora_alpha,
        lora_targets=lora_targets,
        gradient_checkpointing=gradient_checkpointing,
        use_chat_template=use_chat_template,
        mask_user_tokens=mask_user_tokens, kl_coef=kl_coef, kl_temp=kl_temp,
        teacher_model_id=teacher_model_id, teacher_device=teacher_device,
        cos_reg_coef=cos_reg_coef, tau_calib_batches=tau_calib_batches,
        tau_spread=tau_spread, router_calib_batches=router_calib_batches,
        router_start_k=router_start_k, resume_dir=resume_dir,
        eval_every=eval_every, patience=patience, save_full=save_full,
        ul_mode=ul_mode, ul_coef=ul_coef, ul_n=ul_n,
        rollout_every=rollout_every, rollout_tokens=rollout_tokens,
        rollout_prompt=rollout_prompt, rep_probe=rep_probe,
        accelerator=accelerator))


__all__ = ["app", "train"]
