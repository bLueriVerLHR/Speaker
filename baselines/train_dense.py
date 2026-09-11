"""Secondary baseline: dense fine-tuned version (same-spec LoRA pure SFT), the "trained dense" for a fair comparison.

Same protocol as train_mod/train_mdf/train_rt/ours (r3, 7B):
  - same data same slice (first max_samples train / last eval_samples eval, plain full-text LM);
  - same LoRA spec (rank 8, q_proj,v_proj, dropout 0.05, same as ours/LoRA-version baselines);
  - same single lr group (3e-5) + AdamW(wd 0.01) + the same StopOnPlateau convergence rule;
  - the only difference is "no gating/sparsity mechanism at all": loss = LM, k = all layers (upper reference with no sparsity savings).

ckpt: lora.pt (lora_ keys) + denseft_config.json; base loaded from --model_id (eval_compare wraps LoRA per the config).
The training loop lives in baselines/train_loop.py (P1 shared template); this file keeps the CLI + the family recipe.
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
from baselines.train_loop import BaselineRecipe, run_baseline_training  # noqa: E402
from speaker.evaluate import ema_update, eval_heldout  # noqa: E402
from speaker.ruler import batch_accuracy  # noqa: E402 (P0: one accuracy scale)
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
    p.add_argument("--max_steps", type=int, default=500,
                   help="display/annealing horizon alignment slot (actual stopping is decided by the ed3 unified convergence rule)")
    p.add_argument("--lr", type=float, default=3e-5, help="unified single-group lr (same as the LoRA-version baselines)")
    p.add_argument("--max_samples", type=int, default=1000)
    p.add_argument("--eval_samples", type=int, default=100)
    p.add_argument("--log_interval", type=int, default=10)
    p.add_argument("--save_dir", default="/tmp/denseft_baseline")
    p.add_argument("--seed", type=int, default=None,
                   help="random seed (unset by default, preserving legacy behavior)")
    p.add_argument("--patience", type=int, default=3,
                   help="StopOnPlateau patience (enlarge to guarantee running to --max_steps)")
    p.add_argument("--use_lora", default=True, action=argparse.BooleanOptionalAction,
                   help="same-spec LoRA as ours/baselines (fairly fine-tuned dense); off = pure raw dense, for smoke tests")
    p.add_argument("--lora_rank", type=int, default=8, help="same spec as finetune/train.py")
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_targets", default="q_proj,v_proj")
    return p.parse_args()


class DenseRecipe(BaselineRecipe):
    """Dense-ft family strategy: pure LM loss, k fixed = all layers."""

    def __init__(self, model, trainable, n_layers, coll_eval, dense_res):
        self.model, self.trainable, self.n_layers = model, trainable, n_layers
        self.coll_eval, self.d = coll_eval, dense_res
        self.ema_acc = None
        self.acc_item = 0.0

    def loss_term(self, b, out, args):
        with torch.no_grad():
            self.acc_item = batch_accuracy(out.logits, b)
        self.ema_acc = ema_update(self.ema_acc, self.acc_item)
        return None

    def clip_params(self):
        return self.trainable

    def log_line(self, step, lm, ema_lm, rate, mem, elapsed, args):
        return (f"[{time.strftime('%H:%M:%S')}] step {step:4d}/{args.max_steps} lm {lm.item():.3f} "
                f"ema {ema_lm:.3f} acc {self.acc_item:.2f}/{self.ema_acc:.2f} "
                f"{rate:.0f}tok/s{mem} {elapsed:.0f}s")

    def eval_model(self, model, texts):
        return eval_heldout(model, texts, self.coll_eval)

    def save(self, model, tok, args, step, stopper):
        sd = model.state_dict()
        if args.use_lora:
            torch.save({k: v.cpu() for k, v in sd.items() if "lora_" in k},
                       os.path.join(args.save_dir, "lora.pt"))
        else:
            model.save_pretrained(args.save_dir)
        tok.save_pretrained(args.save_dir)
        with open(os.path.join(args.save_dir, "denseft_config.json"), "w") as f:
            json.dump({"use_lora": args.use_lora,
                       "lora_rank": args.lora_rank, "lora_alpha": args.lora_alpha,
                       "lora_targets": [t.strip() for t in args.lora_targets.split(",") if t.strip()],
                       "lr": args.lr, "n_layers": self.n_layers,
                       "converged_step": step, "best_subset_loss": stopper.best,
                       "dense_raw": self.d}, f)

    def final_line(self, m, args):
        d, mm = self.d, m
        return (f"heldout | dense(raw) loss {d['loss']:.3f} acc {d['acc']:.3f} "
                f"| dense-ft loss {mm['loss']:.3f} acc {mm['acc']:.3f} "
                f"(Δloss {mm['loss'] - d['loss']:+.3f} Δacc {mm['acc'] - d['acc']:+.3f}) "
                f"| k {self.n_layers} (no sparsity)")


def main():
    args = parse_args()
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    tok = build_tok(args.model_id)
    model = build_model(args.model_id, device)
    n_layers = model.config.num_hidden_layers

    full, eval_texts = split_train_eval(args.data_path, args.max_samples, args.eval_samples)
    assert eval_texts, "empty eval slice"
    coll_fn = make_collate(tok, device, args.max_length)
    coll_eval = make_collate(tok, device, args.max_length)

    # raw dense baseline (before LoRA is attached; should match the dense row of eval_compare)
    d = eval_heldout(model, eval_texts, coll_eval)
    print(f"heldout dense baseline (raw): loss {d['loss']:.3f} acc {d['acc']:.3f} "
          f"({len(eval_texts)} samples)", flush=True)

    if args.use_lora:
        model = wrap_lora(model, args.lora_rank, args.lora_alpha, args.lora_targets)
    trainable = [p for p in model.parameters() if p.requires_grad]
    groups = [{"params": trainable, "lr": args.lr}]
    print(f"trainable {sum(p.numel() for p in trainable) / 1e6:.1f}M params "
          f"(lora={args.use_lora}), k fixed {n_layers}", flush=True)
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        if args.use_lora:
            # r7 OOM fix: bs2 x len1024 x 7B LoRA needs checkpointing; the frozen-embed backward
            # pitfall is solved by enable_input_require_grads (same as finetune/train.py since r5)
            model.enable_input_require_grads()

    opt = torch.optim.AdamW(groups, weight_decay=0.01)
    recipe = DenseRecipe(model, trainable, n_layers, coll_eval, d)
    run_baseline_training(args, model, tok, full, eval_texts, coll_fn, device, opt, recipe)


if __name__ == "__main__":
    main()
