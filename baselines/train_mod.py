"""MoD original baseline (main baseline, token-choice top-k capacity scheme).

Paper (Mixture-of-Depths, arXiv 2404.06604; PDF in baselines/papers/):
  - per layer a raw router weight r (no sigmoid); token-choice top-k with capacity:
    k = clamp(round(cap * valid tokens), 1, n_valid), tokens above the threshold pass;
  - routed mixing x' = x + sel·r·(block(x) − x) (raw-score weighting, no normalization);
  - BCE auxiliary loss aligning r with the hard selection (paper §3.5 method 1);
  - interleaved routing (every other block), conversion training.

ed3 rule: everything trains to convergence (StopOnPlateau, see speaker/converge.py); --max_steps is a hard cap.

Sandbox-comparable protocol: same data, same slice, same eval; joint training with the base.
Deviation notes (beyond the paper): 500-step-scale regime instead of the paper's full schedule;
known numeric fragility — MoD's raw-score router amplifies padding/batch noise, honest eval is bs1.

ckpt: modd_config.json + routers.pt (LoRA version, small ckpt) / full base (joint version).
Mechanisms (layer forward/patch/stats/eval) live in baselines/lib.py; the training loop lives in
baselines/train_loop.py (P1 shared template); this file keeps the CLI + the family recipe.
Usage:
  python3 baselines/train_mod.py --model_id /home/hdd/model/Qwen1.5-0.5B \
      --max_steps 500 --save_dir ./ckpt/modd_c0125
"""
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from baselines.lib import (  # noqa: E402
    GATE_KEYS,
    collect_modd_stats,
    eval_heldout_modd,
    patch_model_modd,
    save_baseline_ckpt,
)
from baselines.train_loop import BaselineRecipe, run_baseline_training  # noqa: E402
from speaker.evaluate import ema_update  # noqa: E402
from speaker.train_common import build_model, build_tok, resolve_device, split_train_eval, wrap_lora  # noqa: E402
from data.sft import make_collate  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_id", default="/home/hdd/model/Qwen2.5-7B-Instruct",
                   help="same model as the fine-tuning route for comparison (7B)")
    p.add_argument("--data_path", default="./data/sft_t2t_mini.jsonl")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--max_steps", type=int, default=3000,
                   help="hard cap on steps; actual stopping is decided by the ed3 unified convergence rule (StopOnPlateau)")
    p.add_argument("--lr", type=float, default=3e-5,
                   help="unified single-group lr (the paper has no fine-tuning recipe; same regime-conversion setting as the MoDification paper)")
    p.add_argument("--max_samples", type=int, default=1000)
    p.add_argument("--eval_samples", type=int, default=100)
    p.add_argument("--capacity", type=float, default=0.125,
                   help="per-layer top-k capacity ratio (the paper's optimum 12.5%%, fixed throughout, no annealing)")
    p.add_argument("--bce_coef", type=float, default=1.0,
                   help="BCE auxiliary loss weight (paper §3.5 method 1; coefficient value not given in the paper)")
    p.add_argument("--log_interval", type=int, default=10)
    p.add_argument("--save_dir", default="/tmp/modd_baseline")
    p.add_argument("--seed", type=int, default=None,
                   help="random seed (unset by default, preserving legacy behavior)")
    p.add_argument("--patience", type=int, default=3,
                   help="StopOnPlateau patience (enlarge to guarantee running to --max_steps)")
    p.add_argument("--use_lora", default=True, action=argparse.BooleanOptionalAction,
                   help="add LoRA to the base and jointly train it with the router (the paper is full-parameter; 7B full-parameter "
                        "does not fit in 24GB, hardware adaptation, same rank/targets spec as ours)")
    p.add_argument("--lora_rank", type=int, default=8, help="same spec as finetune/train.py")
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_targets", default="q_proj,v_proj")
    return p.parse_args()


class ModdRecipe(BaselineRecipe):
    """MoD family strategy: BCE aux term + capacity-selected k accounting."""

    def __init__(self, model, routed, is_routed, n_dense, coll_eval, dense_res):
        self.model, self.routed, self.is_routed = model, routed, is_routed
        self.n_dense = n_dense
        self.coll_eval, self.d = coll_eval, dense_res
        self.ema_k = None
        self._bce = None

    def loss_term(self, b, out, args):
        bce, k = collect_modd_stats(self.routed, training=True)
        self._bce = bce
        if k is not None:
            valid = b["attention_mask"].bool()
            kt = (k[valid] + self.n_dense)
            self.ema_k = ema_update(self.ema_k, kt.mean().item())
        return (args.bce_coef * bce if bce is not None and args.bce_coef > 0 else None)

    def clip_params(self):
        return [p for p in self.model.parameters() if p.requires_grad]

    def log_line(self, step, lm, ema_lm, rate, mem, elapsed, args):
        cap_now = self.routed[0].route_capacity if self.routed else args.capacity
        return (f"[{time.strftime('%H:%M:%S')}] step {step:4d}/{args.max_steps} lm {lm.item():.3f} "
                f"ema {ema_lm:.3f} bce {self._bce.item() if self._bce is not None else 0:.3f} "
                f"cap {cap_now:.3f} k {self.ema_k:.1f} {rate:.0f}tok/s{mem} "
                f"{elapsed:.0f}s")

    def eval_model(self, model, texts):
        return eval_heldout_modd(model, self.routed, self.n_dense, texts, self.coll_eval)

    def converge_row(self, chk):
        return {"k": self.ema_k}

    def save(self, model, tok, args, step, stopper):
        save_baseline_ckpt(model, tok, args.save_dir, args.use_lora,
                           {"is_routed": self.is_routed, "capacity": args.capacity,
                            "bce_coef": args.bce_coef, "lr": args.lr,
                            "use_lora": args.use_lora,
                            "lora_rank": args.lora_rank, "lora_alpha": args.lora_alpha,
                            "lora_targets": [t.strip() for t in args.lora_targets.split(",") if t.strip()],
                            "converged_step": step, "best_subset_loss": stopper.best},
                           "modd_config.json")

    def final_line(self, m, args):
        d, mm = self.d, m
        return (f"heldout | dense loss {d['loss']:.3f} acc {d['acc']:.3f} "
                f"| modd loss {mm['loss']:.3f} acc {mm['acc']:.3f} "
                f"(Δloss {mm['loss'] - d['loss']:+.3f} Δacc {mm['acc'] - d['acc']:+.3f}) "
                f"| k {mm['mean_k']:.1f}±{mm['std_k']:.1f}")


def main():
    args = parse_args()
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    tok = build_tok(args.model_id)
    model = build_model(args.model_id, device)

    full, eval_texts = split_train_eval(args.data_path, args.max_samples, args.eval_samples)
    assert eval_texts, "empty eval slice"
    coll_fn = make_collate(tok, device, args.max_length)

    n = model.config.num_hidden_layers
    is_routed = [(i % 2 == 1) for i in range(n)]  # paper: interleaved (every other block); parity unspecified, odd layers as in mdf
    n_dense = n - sum(is_routed)
    # dense baseline on the pristine base (before routing is attached)
    coll_eval = make_collate(tok, device, args.max_length)
    d = eval_heldout_modd(model, [], n, eval_texts, coll_eval)
    print(f"heldout dense baseline: loss {d['loss']:.3f} acc {d['acc']:.3f} "
          f"({len(eval_texts)} samples)", flush=True)

    init_cap = args.capacity  # capacity fixed throughout, no annealing (none in the paper)
    routed = patch_model_modd(model, is_routed, capacity=init_cap)
    model.to(device)
    k_est = n_dense + args.capacity * len(routed)
    print(f"MoD patched: {n} layers, routed {len(routed)} (interleave odd, paper §4), "
          f"capacity {args.capacity} fixed, k_est {k_est:.1f}, bce {args.bce_coef}, "
          f"lr {args.lr} unified", flush=True)
    init = eval_heldout_modd(model, routed, n_dense, eval_texts[:10], coll_eval)
    print(f"patched-init: loss {init['loss']:.3f} acc {init['acc']:.3f} "
          f"k {init['mean_k']:.1f}±{init['std_k']:.1f}", flush=True)

    if args.use_lora:
        model = wrap_lora(model, args.lora_rank, args.lora_alpha, args.lora_targets)
        # peft froze the router attached during patching; unfreeze it (LoRA+router joint training, same protocol as ours)
        for nm, p in model.named_parameters():
            if any(g in nm for g in GATE_KEYS):
                p.requires_grad_(True)
    trainable = [p for p in model.parameters() if p.requires_grad]
    groups = [{"params": trainable, "lr": args.lr}]  # unified single group (no fine-tuning recipe in the paper; with LoRA = lora+router)
    n_router_train = sum(1 for nm, p in model.named_parameters()
                         if any(g in nm for g in GATE_KEYS) and p.requires_grad)
    print(f"trainable {sum(p.numel() for p in trainable) / 1e6:.1f}M params "
          f"(lora={args.use_lora}, routers train {n_router_train})", flush=True)
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        if args.use_lora:
            # r7 OOM fix: bs2 x len1024 x 7B LoRA needs checkpointing; the frozen-embed backward
            # pitfall is solved by enable_input_require_grads (same as finetune/train.py since r5)
            model.enable_input_require_grads()

    opt = torch.optim.AdamW(groups, weight_decay=0.01)
    recipe = ModdRecipe(model, routed, is_routed, n_dense, coll_eval, d)
    run_baseline_training(args, model, tok, full, eval_texts, coll_fn, device, opt, recipe)


if __name__ == "__main__":
    main()
