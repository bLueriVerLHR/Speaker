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
- Logging uses the same RunLogger as finetune (unified in ed5): metrics.jsonl /
  layers.jsonl / one-line stdout status.

Usage (smoke):
  python3 pretrain/train.py --device cuda:0 --steps 8 --max_samples 60 --eval_samples 8 \
      --log_interval 4 --price_warmup 3 --budget_ramp 6 --save_dir /tmp/speaker_pretrain_smoke
"""
import argparse
import os
import random
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModelForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from speaker import SpeakerConfig, convert_to_speaker  # noqa: E402
from speaker.metrics import estimate_act_mb, per_token_correct  # noqa: E402
from speaker.evaluate import ema_update, eval_heldout  # noqa: E402
from speaker.log import RunLogger  # noqa: E402
from speaker.checkpoint import save_clean_base, save_gate  # noqa: E402
from speaker.train_common import (  # noqa: E402
    build_param_groups,
    build_tok,
    dtype_of,
    resolve_device,
    split_train_eval,
)
from data.sft import make_collate  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--arch_from", default="/home/hdd/model/Qwen1.5-0.5B",
                   help="only borrows the architecture config and tokenizer; weights are randomly initialized")
    p.add_argument("--num_layers", type=int, default=0,
                   help=">0 overrides the number of layers (start small, e.g. 12)")
    p.add_argument("--hidden_size", type=int, default=0,
                   help=">0 overrides the hidden size (e.g. 512; must be divisible by num_heads)")
    p.add_argument("--num_heads", type=int, default=0, help=">0 overrides the number of attention heads (e.g. 8)")
    p.add_argument("--intermediate_size", type=int, default=0,
                   help=">0 overrides the FFN intermediate dim (e.g. 1408); shrink it too for small scales")
    p.add_argument("--data_path", default="./data/sft_t2t_mini.jsonl")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16", "float16"])
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--lr", type=float, default=3e-4, help="from-scratch training defaults to a larger lr (finetune uses 2e-5)")
    p.add_argument("--router_lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42, help="fixed seed for from-scratch training (reproducibility first)")
    # Shared (fixed) layers: structural prior for from-scratch training, k layers at each end
    p.add_argument("--shared_head", type=int, default=2, help="first k layers fixed (shared), storing the grammar foundation")
    p.add_argument("--shared_tail", type=int, default=2, help="last k layers fixed (shared), storing output/speaking")
    p.add_argument("--always_layers", default="",
                   help="explicit list of fixed layers (comma-separated), overrides shared_head/tail")
    # Gating/budget (same Speaker mechanism as finetune, gate_mode dual scheme)
    p.add_argument("--kmax", type=int, default=10)
    p.add_argument("--gate_mode", default="moe", choices=["moe", "threshold"],
                   help="moe=hierarchical MoE joint routing (default mainline) / threshold=legacy per-layer threshold gating")
    p.add_argument("--select_mode", default="topp", choices=["topp", "topk"])
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--top_k", type=int, default=6)
    p.add_argument("--min_layers", type=int, default=1)
    p.add_argument("--temp_affinity", type=float, default=1.0)
    p.add_argument("--ta_end", type=float, default=0.3)
    p.add_argument("--gumbel_scale", type=float, default=1.0)
    p.add_argument("--tau_init", type=float, default=0.0)
    p.add_argument("--sparsity_price", type=float, default=0.03)
    p.add_argument("--price_adapt", default=False, action=argparse.BooleanOptionalAction,
                   help="early in from-scratch training acc is meaningless, dual adjustment off by default, lambda+kmax serve as the backstop")
    p.add_argument("--acc_target", type=float, default=None)
    p.add_argument("--price_warmup", type=int, default=100)
    p.add_argument("--budget_ramp", type=int, default=100)
    p.add_argument("--cos_reg_coef", type=float, default=0.01)
    p.add_argument("--balance_loss_coef", type=float, default=0.01)
    # Data/eval/output
    p.add_argument("--eval_samples", type=int, default=100)
    p.add_argument("--max_samples", type=int, default=500)
    p.add_argument("--log_interval", type=int, default=10)
    p.add_argument("--save_dir", default="/tmp/speaker_pretrain")
    p.add_argument("--gradient_checkpointing", default=True, action=argparse.BooleanOptionalAction)
    return p.parse_args()


def patch_arch_config(hf_cfg, args):
    """Override architecture hyperparameters; 0 = keep the arch_from original value.
    layer_types is tiled/trimmed to match the number of layers (newer transformers
    strictly validate that the layer_types length equals the number of layers)."""
    if args.num_layers > 0:
        hf_cfg.num_hidden_layers = args.num_layers
    if args.hidden_size > 0:
        hf_cfg.hidden_size = args.hidden_size
    if args.num_heads > 0:
        hf_cfg.num_attention_heads = args.num_heads
        if hasattr(hf_cfg, "num_key_value_heads"):
            hf_cfg.num_key_value_heads = args.num_heads
    if args.intermediate_size > 0 and hasattr(hf_cfg, "intermediate_size"):
        hf_cfg.intermediate_size = args.intermediate_size
    lt = getattr(hf_cfg, "layer_types", None)
    if lt and len(lt) != hf_cfg.num_hidden_layers:
        hf_cfg.layer_types = [lt[i % len(lt)] for i in range(hf_cfg.num_hidden_layers)]
    return hf_cfg


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)

    tok = build_tok(args.arch_from)
    # Random initialization: borrow only the architecture config (layers/hidden/vocab), no pretrained weights
    hf_cfg = patch_arch_config(
        AutoConfig.from_pretrained(args.arch_from, trust_remote_code=True), args)
    model = AutoModelForCausalLM.from_config(hf_cfg, dtype=dtype_of(args.dtype))
    model.to(device)
    n_layers = model.config.num_hidden_layers
    print(f"from-scratch model: N={n_layers} H={model.config.hidden_size} "
          f"params {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M (seed {args.seed})",
          flush=True)

    full, eval_texts = split_train_eval(args.data_path, args.max_samples, args.eval_samples)
    ds = full
    coll_fn = make_collate(tok, device, args.max_length)
    eval_coll = make_collate(tok, device, args.max_length)

    overrides = dict(kmax=args.kmax, gate_mode=args.gate_mode,
                     select_mode=args.select_mode, top_p=args.top_p,
                     top_k=args.top_k, min_layers=args.min_layers,
                     always_on_head=args.shared_head,
                     always_on_tail=args.shared_tail, temp_affinity=args.temp_affinity,
                     gumbel_scale=args.gumbel_scale, sparsity_price=args.sparsity_price,
                     price_adapt=args.price_adapt, acc_target=args.acc_target,
                     price_warmup_steps=args.price_warmup, tau_init=args.tau_init,
                     cos_reg_coef=args.cos_reg_coef,
                     balance_loss_coef=args.balance_loss_coef)
    if args.always_layers.strip():
        overrides["always_on_layers"] = [int(x) for x in args.always_layers.split(",")
                                         if x.strip() != ""]
    cfg = SpeakerConfig.from_model_config(model.config, **overrides)
    print(f"[pretrain] fixed (shared) layers head_k={args.shared_head} tail_k={args.shared_tail} "
          f"-> {cfg.always_on_layers}; gated layers {len(cfg.gated_layers)} "
          f"(grammar in shared layers, reasoning in gated layers, budget pressures mean(k))", flush=True)
    print(cfg.summary(), flush=True)
    mod_model = convert_to_speaker(model, cfg).to(device)

    if args.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    groups, base_params, gate_params = build_param_groups(
        mod_model, args.lr, args.router_lr)
    print(f"base {len(base_params)} gate {len(gate_params)} "
          f"(from-scratch full-parameter joint training, no LoRA/distillation)", flush=True)
    opt = torch.optim.AdamW(groups, weight_decay=0.01)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=coll_fn)

    rl = RunLogger(args.save_dir)
    step = 0
    ema_lm = None
    ema_acc = None
    window_t0 = time.time()
    window_tokens = 0
    mod_model.train()
    for epoch in range(100):
        for b in dl:
            step += 1
            if step > args.steps:
                break
            mod_model.anneal(step, args.steps, ta_end=args.ta_end, g_end=0.0)
            out = mod_model(**b)
            lm_loss = out.loss
            aux = mod_model.get_aux_loss()
            with torch.no_grad():
                correct = per_token_correct(out.logits.float(), b["labels"])
                valid_tok = (b["labels"] != -100)
                acc_item = correct[valid_tok].float().mean().item() if valid_tok.any() else 0.0
            task = mod_model.get_budget_loss(b["attention_mask"])
            if task is not None and args.budget_ramp > 0 and step < args.budget_ramp:
                task = task * (step / args.budget_ramp)
            loss = lm_loss + (aux or 0) + (task or 0)
            ema_lm = ema_update(ema_lm, lm_loss.item())
            ema_acc = ema_update(ema_acc, acc_item)
            if step > cfg.price_warmup_steps:
                mod_model.adapt_price(ema_acc)
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
            if step % args.log_interval == 0:
                rate = window_tokens / max(time.time() - window_t0, 1e-6)
                sp = mod_model.get_layer_sparsity()
                spars = sum(sp[i] for i in cfg.gated_layers) / max(len(cfg.gated_layers), 1)
                usage = mod_model.get_layer_usage()  # window usage rate; read-and-reset
                mem_gb = torch.cuda.memory_allocated(device) / 1024**3 \
                    if device.type == "cuda" else 0.0
                rl.metrics(step=step, lm=round(lm_loss.item(), 4), ema_lm=round(ema_lm, 4),
                           acc=round(acc_item, 4), ema_acc=round(ema_acc, 4),
                           k=round(mean_k, 3), k_std=round(std_k, 3), spars=round(spars, 4),
                           aux=round(float(aux.detach()), 5) if aux is not None else 0.0,
                           task=round(float(task.detach()), 4) if task is not None else 0.0,
                           tot=round(float(loss.detach()), 4),
                           price=round(cfg.sparsity_price, 5), ta=round(cfg.temp_affinity, 3),
                           gumbel=round(cfg.gumbel_scale, 3), est_mb=round(est_mb, 2),
                           tok_s=round(rate, 1), mem_gb=round(mem_gb, 2))
                RunLogger.status(
                    f"[{time.strftime('%H:%M:%S')}] step {step} lm {ema_lm:.3f} "
                    f"acc {ema_acc:.2f} k {mean_k:.1f}±{std_k:.1f} spars {spars:.0%} "
                    f"price {cfg.sparsity_price:.4f} {rate:.0f}tok/s"
                    + (f" mem {mem_gb:.1f}GB" if device.type == "cuda" else ""))
                if step % (args.log_interval * 5) == 0 and usage:
                    rl.layers(step, usage, cfg.always_on_layers)
                window_t0 = time.time()
                window_tokens = 0
            if step >= args.steps:
                break
        if step >= args.steps:
            break
    rl.close()

    save_clean_base(mod_model, tok, cfg, args.save_dir)
    save_gate(mod_model, args.save_dir)
    print(f"saved to {args.save_dir} (clean base+tokenizer+mod_config.json+gate.pt)", flush=True)
    if eval_texts:
        mod_model.set_skip_mode("hard")
        res = eval_heldout(mod_model, eval_texts, eval_coll)
        print(f"heldout (hard): loss {res['loss']:.3f} acc {res['acc']:.3f} "
              f"| k {res['mean_k']:.1f}±{res['std_k']:.1f} "
              f"quartile {[round(v, 1) for v in res['quartile_k']]}", flush=True)
        mod_model.set_skip_mode("soft")


if __name__ == "__main__":
    main()
