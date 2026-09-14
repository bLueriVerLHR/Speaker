"""From-scratch training (design-first): shared (fixed) layers + gated layers trained
from the source (dual scheme, gate_mode switch).

The main path of the Speaker idea: grammatical ability is stored in the shared (fixed)
layers, reasoning ability in the middle gated layers. When the speaker talks, easy
tokens pass through the shared layers only to finish syntactic-structure inference,
while hard tokens are let through by the gating into more middle layers for complex
logical reasoning -- the budget loss (lambda * mean(k)) pressures the gating into
learning this division of labor.

- Structural prior: the first and last k layers are shared layers (--shared_head/
  --shared_tail, theoretically the layers with load >= 90%-95%; from-scratch training
  has no prior load profile, so they are specified by structure directly);
- Dual gating scheme: moe = hierarchical MoE (a single JointRouter at the entry emits
  log p, top-p/top-k selection, renormalized weighted residual over selected layers;
  default mainline) / threshold = legacy per-layer threshold gating; the two schemes'
  checkpoints are mutually incompatible;
- Budget-regulated training (LM + lambda*mean(k), optional acc dual adjustment),
  price basis = actual demand per inference (avg);
- Random initialization: --arch_from only borrows the architecture config and the
  tokenizer, weights are trained from scratch;
- Self-contained ckpt: full weights (clean base with the gating params stripped)
  + tokenizer + mod_config.json + gate.pt; see the comment at the bottom of the file
  for reuse instructions.
- Logging uses the shared logger (speaker.log, loguru-based): console + run.log
  mirror in save_dir, metrics.jsonl / layers.jsonl via JSONL sinks.

Usage (smoke):
  python3 pretrain/train.py --device cuda:0 --max_steps 8 --max_samples 60 --eval_samples 8 \
      --log_interval 4 --price_warmup 3 --budget_ramp 6 --save_dir /tmp/speaker_pretrain_smoke
"""
import os
import sys
import time
from pathlib import Path
from typing import Annotated, Literal, Optional

import torch
import typer
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModelForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from speaker import SpeakerConfig, convert_to_speaker  # noqa: E402
from speaker.metrics import estimate_act_mb  # noqa: E402
from speaker.evaluate import ema_update, eval_heldout, format_k_quartile  # noqa: E402
from speaker.ruler import batch_accuracy  # noqa: E402 (P0: one accuracy scale everywhere)
from speaker.dual import DualController  # noqa: E402 (P0.5: one dual orchestration point)
from speaker.log import add_jsonl, emit, logger, setup_logger  # noqa: E402
from speaker.checkpoint import save_clean_base, save_gate  # noqa: E402
from speaker.train_common import (  # noqa: E402
    build_optimizer,
    build_param_groups,
    build_tok,
    dtype_of,
    enable_checkpointing,
    parse_csv_list,
    resolve_device,
    seed_all,
    split_train_eval,
)
from data.sft import make_collate  # noqa: E402


app = typer.Typer(add_completion=False)


def patch_arch_config(hf_cfg, *, num_layers, hidden_size, num_heads,
                      intermediate_size):
    """Override architecture hyperparameters; 0 = keep the arch_from original value.
    layer_types is tiled/trimmed to match the number of layers (newer transformers
    strictly validate that the layer_types length equals the number of layers)."""
    if num_layers > 0:
        hf_cfg.num_hidden_layers = num_layers
    if hidden_size > 0:
        hf_cfg.hidden_size = hidden_size
    if num_heads > 0:
        hf_cfg.num_attention_heads = num_heads
        if hasattr(hf_cfg, "num_key_value_heads"):
            hf_cfg.num_key_value_heads = num_heads
    if intermediate_size > 0 and hasattr(hf_cfg, "intermediate_size"):
        hf_cfg.intermediate_size = intermediate_size
    lt = getattr(hf_cfg, "layer_types", None)
    if lt and len(lt) != hf_cfg.num_hidden_layers:
        hf_cfg.layer_types = [lt[i % len(lt)] for i in range(hf_cfg.num_hidden_layers)]
    return hf_cfg


@app.command()
def main(
    arch_from: Annotated[str, typer.Option("--arch_from", help="only borrows the architecture config and tokenizer; weights are randomly initialized")] = "/home/hdd/model/Qwen1.5-0.5B",
    num_layers: Annotated[int, typer.Option("--num_layers", help=">0 overrides the number of layers (start small, e.g. 12)")] = 0,
    hidden_size: Annotated[int, typer.Option("--hidden_size", help=">0 overrides the hidden size (e.g. 512; must be divisible by num_heads)")] = 0,
    num_heads: Annotated[int, typer.Option("--num_heads", help=">0 overrides the number of attention heads (e.g. 8)")] = 0,
    intermediate_size: Annotated[int, typer.Option("--intermediate_size", help=">0 overrides the FFN intermediate dim (e.g. 1408); shrink it too for small scales")] = 0,
    data_path: Annotated[str, typer.Option("--data_path")] = "./data/sft_t2t_mini.jsonl",
    device: Annotated[str, typer.Option("--device")] = "cuda:0",
    dtype: Annotated[Literal["float32", "bfloat16", "float16"], typer.Option("--dtype")] = "bfloat16",
    batch_size: Annotated[int, typer.Option("--batch_size")] = 1,
    max_length: Annotated[int, typer.Option("--max_length")] = 256,
    max_steps: Annotated[int, typer.Option("--max_steps", help="hard stop cap (ed5: renamed from --steps to match the finetune/baselines convention — one name, one meaning across all training scripts)")] = 100,
    lr: Annotated[float, typer.Option("--lr", help="from-scratch training defaults to a larger lr (finetune uses 2e-5)")] = 3e-4,
    router_lr: Annotated[float, typer.Option("--router_lr")] = 1e-3,
    seed: Annotated[int, typer.Option("--seed", help="fixed seed for from-scratch training (reproducibility first)")] = 42,
    shared_head: Annotated[int, typer.Option("--shared_head", help="first k layers fixed (shared), storing the grammar foundation")] = 2,
    shared_tail: Annotated[int, typer.Option("--shared_tail", help="last k layers fixed (shared), storing output/speaking")] = 2,
    always_layers: Annotated[str, typer.Option("--always_layers", help="explicit list of fixed layers (comma-separated), overrides shared_head/tail")] = "",
    kmax: Annotated[int, typer.Option("--kmax")] = 10,
    gate_mode: Annotated[Literal["moe", "mol", "threshold", "speaker"], typer.Option("--gate_mode", help="moe=hierarchical MoE joint routing (default mainline) / threshold=legacy per-layer threshold gating")] = "moe",
    select_mode: Annotated[Literal["topp", "topk"], typer.Option("--select_mode")] = "topp",
    top_p: Annotated[float, typer.Option("--top_p")] = 0.9,
    top_k: Annotated[int, typer.Option("--top_k")] = 6,
    weight_mode: Annotated[Literal["pmax", "renorm"], typer.Option("--weight_mode", help="moe residual weighting: pmax (ed7 fix, default) / renorm (legacy)")] = "pmax",
    min_layers: Annotated[int, typer.Option("--min_layers")] = 1,
    temp_affinity: Annotated[float, typer.Option("--temp_affinity")] = 1.0,
    ta_end: Annotated[float, typer.Option("--ta_end")] = 0.3,
    gumbel_scale: Annotated[float, typer.Option("--gumbel_scale")] = 1.0,
    tau_init: Annotated[float, typer.Option("--tau_init")] = 0.0,
    sparsity_price: Annotated[float, typer.Option("--sparsity_price")] = 0.03,
    price_adapt: Annotated[bool, typer.Option("--price_adapt/--no-price_adapt", help="early in from-scratch training acc is meaningless, dual adjustment off by default, lambda+kmax serve as the backstop")] = False,
    acc_target: Annotated[Optional[float], typer.Option("--acc_target", help="absolute floor for the dual (numeric only: the from-scratch dense reference is a random init, so the finetune-style 'auto' derivation is meaningless here)")] = None,
    price_warmup: Annotated[int, typer.Option("--price_warmup")] = 100,
    budget_ramp: Annotated[int, typer.Option("--budget_ramp")] = 100,
    cos_reg_coef: Annotated[float, typer.Option("--cos_reg_coef")] = 0.01,
    balance_loss_coef: Annotated[float, typer.Option("--balance_loss_coef")] = 0.01,
    eval_samples: Annotated[int, typer.Option("--eval_samples")] = 100,
    max_samples: Annotated[int, typer.Option("--max_samples")] = 500,
    log_interval: Annotated[int, typer.Option("--log_interval")] = 10,
    save_dir: Annotated[str, typer.Option("--save_dir")] = "/tmp/speaker_pretrain",
    gradient_checkpointing: Annotated[bool, typer.Option("--gradient_checkpointing/--no-gradient_checkpointing")] = True,
) -> None:
    """From-scratch training: shared fixed layers + gated layers from random init."""
    from speaker.terminal import setup_terminal
    setup_terminal()
    setup_logger(run_dir=save_dir)
    add_jsonl(os.path.join(save_dir, "metrics.jsonl"), "metrics")
    add_jsonl(os.path.join(save_dir, "layers.jsonl"), "layers")
    seed_all(seed)
    device = resolve_device(device)

    tok = build_tok(arch_from)
    # Random initialization: borrow only the architecture config (layers/hidden/vocab), no pretrained weights
    hf_cfg = patch_arch_config(
        AutoConfig.from_pretrained(arch_from, trust_remote_code=True),
        num_layers=num_layers, hidden_size=hidden_size, num_heads=num_heads,
        intermediate_size=intermediate_size)
    model = AutoModelForCausalLM.from_config(hf_cfg, dtype=dtype_of(dtype))
    model.to(device)
    n_layers = model.config.num_hidden_layers
    logger.info(f"from-scratch model: N={n_layers} H={model.config.hidden_size} "
                f"params {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M (seed {seed})")

    full, eval_texts = split_train_eval(data_path, max_samples, eval_samples)
    ds = full
    coll_fn = make_collate(tok, device, max_length)

    overrides = dict(kmax=kmax, gate_mode=gate_mode,
                     select_mode=select_mode, top_p=top_p,
                     top_k=top_k, min_layers=min_layers,
                     weight_mode=weight_mode,
                     always_on_head=shared_head,
                     always_on_tail=shared_tail, temp_affinity=temp_affinity,
                     gumbel_scale=gumbel_scale, sparsity_price=sparsity_price,
                     price_adapt=price_adapt, acc_target=acc_target,
                     price_warmup_steps=price_warmup, tau_init=tau_init,
                     cos_reg_coef=cos_reg_coef,
                     balance_loss_coef=balance_loss_coef)
    if always_layers.strip():
        overrides["always_on_layers"] = parse_csv_list(always_layers)
    cfg = SpeakerConfig.from_model_config(model.config, **overrides)
    logger.info(f"[pretrain] fixed (shared) layers head_k={shared_head} tail_k={shared_tail} "
                f"-> {cfg.always_on_layers}; gated layers {len(cfg.gated_layers)} "
                f"(grammar in shared layers, reasoning in gated layers, budget pressures mean(k))")
    logger.info(cfg.summary())
    mod_model = convert_to_speaker(model, cfg).to(device)

    if gradient_checkpointing:
        enable_checkpointing(model)

    groups, base_params, gate_params = build_param_groups(
        mod_model, lr, router_lr)
    logger.info(f"base {len(base_params)} gate {len(gate_params)} "
                f"(from-scratch full-parameter joint training, no LoRA/distillation)")
    opt = build_optimizer(groups)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=coll_fn)

    step = 0
    ema_lm = None
    dual = DualController(mod_model, cfg)  # accuracy dual: ema + warmup gate + λ adaptation
    window_t0 = time.time()
    window_tokens = 0
    mod_model.train()
    for epoch in range(100):
        for b in dl:
            step += 1
            if step > max_steps:
                break
            mod_model.anneal(step, max_steps, ta_end=ta_end, g_end=0.0)
            out = mod_model(**b)
            lm_loss = out.loss
            aux = mod_model.get_aux_loss()
            with torch.no_grad():
                acc_item = batch_accuracy(out.logits, b)
            task = mod_model.get_budget_loss(b["attention_mask"])
            if task is not None and budget_ramp > 0 and step < budget_ramp:
                task = task * (step / budget_ramp)
            loss = lm_loss + (aux or 0) + (task or 0)
            ema_lm = ema_update(ema_lm, lm_loss.item())
            ema_acc = dual.observe(step, acc_item)
            with torch.no_grad():
                counts = mod_model.get_active_counts()
                valid = b["attention_mask"].bool()
                if counts is not None and valid.any():
                    ks = counts[valid].float()
                    mean_k = ks.mean().item()
                    std_k = ks.std().item() if ks.numel() > 1 else 0.0
                    est_mb = estimate_act_mb(mean_k, *b["input_ids"].shape,
                                             cfg.hidden_size, cfg.mem_bytes_per_hidden)
                else:
                    mean_k = std_k = est_mb = float("nan")
                window_tokens += int(valid.sum())
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(mod_model.parameters(), 1.0)
            opt.step()
            if step % log_interval == 0:
                rate = window_tokens / max(time.time() - window_t0, 1e-6)
                sp = mod_model.get_layer_sparsity()
                spars = sum(sp[i] for i in cfg.gated_layers) / max(len(cfg.gated_layers), 1)
                usage = mod_model.get_layer_usage()  # window usage rate; read-and-reset
                mem_gb = torch.cuda.memory_allocated(device) / 1024**3 \
                    if device.type == "cuda" else 0.0
                emit("metrics", step=step, lm=round(lm_loss.item(), 4), ema_lm=round(ema_lm, 4),
                           acc=round(acc_item, 4), ema_acc=round(ema_acc, 4),
                           k=round(mean_k, 3), k_std=round(std_k, 3), spars=round(spars, 4),
                           aux=round(float(aux.detach()), 5) if aux is not None else 0.0,
                           task=round(float(task.detach()), 4) if task is not None else 0.0,
                           tot=round(float(loss.detach()), 4),
                           price=round(cfg.sparsity_price, 5), ta=round(cfg.temp_affinity, 3),
                           gumbel=round(cfg.gumbel_scale, 3), est_mb=round(est_mb, 2),
                           tok_s=round(rate, 1), mem_gb=round(mem_gb, 2))
                logger.info(
                    f"step {step} lm {ema_lm:.3f} "
                    f"acc {ema_acc:.2f} k {mean_k:.1f}±{std_k:.1f} spars {spars:.0%} "
                    f"price {cfg.sparsity_price:.4f} {rate:.0f}tok/s"
                    + (f" mem {mem_gb:.1f}GB" if device.type == "cuda" else ""))
                if step % (log_interval * 5) == 0 and usage:
                    emit("layers", step=step,
                         exec={str(i): round(v[0], 4) for i, v in sorted(usage.items())},
                         always_on=sorted(cfg.always_on_layers))
                window_t0 = time.time()
                window_tokens = 0
            if step >= max_steps:
                break
        if step >= max_steps:
            break

    save_clean_base(mod_model, tok, cfg, save_dir)
    save_gate(mod_model, save_dir)
    logger.info(f"saved to {save_dir} (clean base+tokenizer+mod_config.json+gate.pt)")
    if eval_texts:
        mod_model.set_skip_mode("hard")
        res = eval_heldout(mod_model, eval_texts, coll_fn, valid_mode="labels")
        logger.info(f"heldout (hard): loss {res['loss']:.3f} acc {res['acc']:.3f} "
                    f"| {format_k_quartile(res)}")
        mod_model.set_skip_mode("soft")


if __name__ == "__main__":
    app()
