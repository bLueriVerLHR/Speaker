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
Mechanisms (layer forward/patch/stats/eval) live in baselines/lib.py; this file keeps only the CLI and the training loop.
Usage:
  python3 baselines/train_mdf.py --model_id /home/hdd/model/Qwen1.5-0.5B \
      --max_steps 500 --save_dir ./ckpt/mdf_q05
"""
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from baselines.lib import (  # noqa: E402
    GATE_KEYS,
    collect_mdf_stats,
    eval_heldout_mdf,
    patch_model_mdf,
    save_baseline_ckpt,
)
from speaker.evaluate import ema_update  # noqa: E402
from speaker.converge import StopOnPlateau  # noqa: E402 (ed3 unified convergence rule)
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
    dl = DataLoader(full, batch_size=args.batch_size, shuffle=True, collate_fn=coll_fn)
    os.makedirs(args.save_dir, exist_ok=True)
    stopper = StopOnPlateau(max_steps=args.max_steps, patience=args.patience)  # ed5: --max_steps now actually wired into the cap (was cosmetic-only)
    stop_subset = eval_texts[:40]  # subset for plateau checks (saves time); final eval still uses the full set
    step, ema_lm, ema_k = 0, None, None
    t0 = time.time()
    window_tokens, window_t0 = 0, time.time()
    model.train()
    for epoch in range(1000):
        for b in dl:
            step += 1
            if stopper.capped(step):
                break
            out = model(**b)
            lm = out.loss
            fg, k = collect_mdf_stats(routed, training=True)
            loss = lm + (args.alpha * fg if fg is not None and args.alpha > 0 else 0)
            ema_lm = ema_update(ema_lm, lm.item())
            if k is not None:
                valid = b["attention_mask"].bool()
                kt = (k[valid] + n_dense)
                ema_k = ema_update(ema_k, kt.mean().item())
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            window_tokens += int(b["attention_mask"].sum())
            if step % args.log_interval == 0:
                exec_rate = (ema_k - n_dense) / max(len(routed), 1) if ema_k else 0
                rate = window_tokens / max(time.time() - window_t0, 1e-6)
                mem = (f" mem {torch.cuda.memory_allocated(device) / 1024**3:.2f}GB"
                       if device.type == "cuda" else "")
                window_tokens, window_t0 = 0, time.time()
                print(f"[{time.strftime('%H:%M:%S')}] step {step:4d}/{args.max_steps} lm {lm.item():.3f} "
                      f"ema {ema_lm:.3f} R {args.alpha * fg.item() if fg is not None else 0:.4f} "
                      f"exec {exec_rate:.2f} k {ema_k:.1f} {rate:.0f}tok/s{mem} "
                      f"{time.time() - t0:.0f}s", flush=True)
            if step % stopper.eval_every == 0:
                # plateau check (eval_heldout_mdf restores train mode automatically)
                chk = eval_heldout_mdf(model, routed, n_dense, stop_subset, coll_eval)
                with open(os.path.join(args.save_dir, "converge.jsonl"), "a", encoding="utf-8") as f:
                    f.write(json.dumps({"step": step, "subset_loss": chk["loss"],
                                        "best": stopper.best, "ema_lm": ema_lm,
                                        "k": ema_k}) + "\n")
                if stopper.check(step, chk["loss"]):
                    print(f"[converged] step {step}, best heldout-subset loss {stopper.best:.3f}",
                          flush=True)
                    break
            if stopper.capped(step):
                break
        if stopper.capped(step):
            break

    # ckpt: full-parameter version = full base (gate keys stripped) + routers.pt; LoRA version = small ckpt (gates+lora keys), base loaded from --model_id
    save_baseline_ckpt(model, tok, args.save_dir, args.use_lora,
                       {"is_routed": is_routed, "p": args.p, "alpha": args.alpha,
                        "lr": args.lr,
                        "use_lora": args.use_lora,
                        "lora_rank": args.lora_rank, "lora_alpha": args.lora_alpha,
                        "lora_targets": [t.strip() for t in args.lora_targets.split(",") if t.strip()],
                        "converged_step": step, "best_subset_loss": stopper.best},
                       "mdf_config.json")
    print(f"saved to {args.save_dir}", flush=True)
    m = eval_heldout_mdf(model, routed, n_dense, eval_texts, coll_eval)
    print(f"heldout | dense loss {d['loss']:.3f} acc {d['acc']:.3f} "
          f"| mdf loss {m['loss']:.3f} acc {m['acc']:.3f} "
          f"(Δloss {m['loss'] - d['loss']:+.3f} Δacc {m['acc'] - d['acc']:+.3f}) "
          f"| k {m['mean_k']:.1f}±{m['std_k']:.1f}", flush=True)


if __name__ == "__main__":
    main()
