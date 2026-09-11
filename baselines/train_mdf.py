"""MoDification baseline (second-newest SOTA): threshold-p selection + gated weighted whole layer + R load target.

Paper (MoDification: Mixture of Depths Made Easy, arXiv 2410.14268, BIT/HIT-Shenzhen/Xiaohongshu;
PDF in baselines/papers/):
  - threshold-p instead of top-k: g_i = sigmoid(Gate(x_i)), f_i = [g_i >= p] (p=0.5, 0.55 for large models).
    Per-token absolute threshold, no cross-token ranking -> any number of tokens can be kept, no capacity wall;
  - the gate multiplies both Attention and MLP (shared gate); HF fuses attn/mlp inside a layer, so implemented here as a
    whole-layer shared gate: when executed h' = h + g·(block(h) − h), when skipped h' = h (equivalent to the paper's Eq.3 fused form);
  - load-reducing target R = α·Σ_j F_j·G_j (α=0.01, paper value):
    F_j = fraction of tokens selected at that layer (hard, no gradient), G_j = gate mean (soft, gradient flows back through it);
  - interleaved (every other block): route only one of every two adjacent layers, odd layers 1,3,...,23 (12 layers in total);
  - conversion training on 10B diverse tokens; unified single-group lr 3e-5 (paper §4).
ed3 rule: everything trains to convergence (StopOnPlateau, see speaker/converge.py); --max_steps is a hard cap.

Sandbox-comparable protocol: same data, same slice, same eval; joint training with the base.
Deviation notes (beyond the paper): fused whole-layer shared gate (the paper multiplies g separately on attn/mlp, mathematically equivalent here);
router zero-initialized (initial value unspecified in the paper); training scale at the 500-step level rather than 10B (see the ed3 convergence rule).

ckpt: mdf_config.json + routers.pt + full base (joint training modifies the base, eval needs a full load).
Mechanisms (layer forward/patch/stats/eval) live in baselines/lib.py; the training loop lives in
baselines/train_loop.py (P1 shared template); this file keeps the CLI + the family recipe.
Usage:
  python3 baselines/train_mdf.py --model_id /home/hdd/model/Qwen1.5-0.5B \
      --max_steps 500 --save_dir ./ckpt/mdf_q05
"""
import argparse
import random
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from baselines.lib import (  # noqa: E402
    GATE_KEYS,
    collect_mdf_stats,
    eval_heldout_mdf,
    patch_model_mdf,
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
    p.add_argument("--lr", type=float, default=3e-5, help="unified single-group lr (paper §4)")
    p.add_argument("--max_samples", type=int, default=1000)
    p.add_argument("--eval_samples", type=int, default=100)
    p.add_argument("--p", type=float, default=0.5, help="threshold-p gate threshold (paper 0.5, 0.55 for large models)")
    p.add_argument("--alpha", type=float, default=0.01, help="coefficient of R=α·ΣFG (paper 0.01)")
    p.add_argument("--log_interval", type=int, default=10)
    p.add_argument("--save_dir", default="/tmp/mdf_baseline")
    p.add_argument("--seed", type=int, default=None,
                   help="random seed (unset by default, preserving legacy behavior)")
    p.add_argument("--patience", type=int, default=3,
                   help="StopOnPlateau patience (enlarge to guarantee running to --max_steps)")
    p.add_argument("--use_lora", default=True, action=argparse.BooleanOptionalAction,
                   help="add LoRA to the base and jointly train it with the gates (the paper is full-parameter; 7B full-parameter "
                        "does not fit in 24GB, hardware adaptation, same rank/targets spec as ours)")
    p.add_argument("--lora_rank", type=int, default=8, help="same spec as finetune/train.py")
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_targets", default="q_proj,v_proj")
    return p.parse_args()


class MdfRecipe(BaselineRecipe):
    """MoDification family strategy: R = alpha*Sigma(F*G) load target + threshold-p k accounting."""

    def __init__(self, model, routed, is_routed, n_dense, coll_eval, dense_res):
        self.model, self.routed, self.is_routed = model, routed, is_routed
        self.n_dense = n_dense
        self.coll_eval, self.d = coll_eval, dense_res
        self.ema_k = None
        self._fg = None

    def loss_term(self, b, out, args):
        fg, k = collect_mdf_stats(self.routed, training=True)
        self._fg = fg
        if k is not None:
            valid = b["attention_mask"].bool()
            kt = (k[valid] + self.n_dense)
            self.ema_k = ema_update(self.ema_k, kt.mean().item())
        return (args.alpha * fg if fg is not None and args.alpha > 0 else None)

    def clip_params(self):
        return [p for p in self.model.parameters() if p.requires_grad]

    def log_line(self, step, lm, ema_lm, rate, mem, elapsed, args):
        exec_rate = (self.ema_k - self.n_dense) / max(len(self.routed), 1) if self.ema_k else 0
        return (f"[{time.strftime('%H:%M:%S')}] step {step:4d}/{args.max_steps} lm {lm.item():.3f} "
                f"ema {ema_lm:.3f} R {args.alpha * self._fg.item() if self._fg is not None else 0:.4f} "
                f"exec {exec_rate:.2f} k {self.ema_k:.1f} {rate:.0f}tok/s{mem} "
                f"{elapsed:.0f}s")

    def eval_model(self, model, texts):
        return eval_heldout_mdf(model, self.routed, self.n_dense, texts, self.coll_eval)

    def converge_row(self, chk):
        return {"k": self.ema_k}

    def save(self, model, tok, args, step, stopper):
        save_baseline_ckpt(model, tok, args.save_dir, args.use_lora,
                           {"is_routed": self.is_routed, "p": args.p, "alpha": args.alpha,
                            "lr": args.lr,
                            "use_lora": args.use_lora,
                            "lora_rank": args.lora_rank, "lora_alpha": args.lora_alpha,
                            "lora_targets": [t.strip() for t in args.lora_targets.split(",") if t.strip()],
                            "converged_step": step, "best_subset_loss": stopper.best},
                           "mdf_config.json")

    def final_line(self, m, args):
        d, mm = self.d, m
        return (f"heldout | dense loss {d['loss']:.3f} acc {d['acc']:.3f} "
                f"| mdf loss {mm['loss']:.3f} acc {mm['acc']:.3f} "
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
    is_routed = [(i % 2 == 1) for i in range(n)]  # paper's interleaved (every other block): odd layers, 12/24
    n_dense = n - sum(is_routed)
    coll_eval = make_collate(tok, device, args.max_length)
    d = eval_heldout_mdf(model, [], n, eval_texts, coll_eval)
    print(f"heldout dense baseline: loss {d['loss']:.3f} acc {d['acc']:.3f} "
          f"({len(eval_texts)} samples)", flush=True)

    routed = patch_model_mdf(model, is_routed, p=args.p)
    model.to(device)
    print(f"MoDification patched: {n} layers, routed {len(routed)} (interleave), "
          f"p {args.p}, alpha {args.alpha}, lr {args.lr} unified", flush=True)
    init = eval_heldout_mdf(model, routed, n_dense, eval_texts[:10], coll_eval)
    print(f"patched-init: loss {init['loss']:.3f} acc {init['acc']:.3f} "
          f"k {init['mean_k']:.1f}±{init['std_k']:.1f}", flush=True)

    if args.use_lora:
        model = wrap_lora(model, args.lora_rank, args.lora_alpha, args.lora_targets)
        # peft froze the gates attached during patching; unfreeze them (LoRA+gate joint training, same protocol as ours)
        for nm, p in model.named_parameters():
            if any(g in nm for g in GATE_KEYS):
                p.requires_grad_(True)
    trainable = [p for p in model.parameters() if p.requires_grad]
    groups = [{"params": trainable, "lr": args.lr}]  # unified single group (paper §4; with LoRA = lora+gates)
    n_router_train = sum(1 for nm, p in model.named_parameters()
                         if any(g in nm for g in GATE_KEYS) and p.requires_grad)
    print(f"trainable {sum(p.numel() for p in trainable) / 1e6:.1f}M params "
          f"(lora={args.use_lora}, gates train {n_router_train})", flush=True)
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        if args.use_lora:
            # r7 OOM fix: bs2 x len1024 x 7B LoRA needs checkpointing; the frozen-embed backward
            # pitfall is solved by enable_input_require_grads (same as finetune/train.py since r5)
            model.enable_input_require_grads()

    opt = torch.optim.AdamW(groups, weight_decay=0.01)
    recipe = MdfRecipe(model, routed, is_routed, n_dense, coll_eval, d)
    run_baseline_training(args, model, tok, full, eval_texts, coll_fn, device, opt, recipe)


if __name__ == "__main__":
    main()
