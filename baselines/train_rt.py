"""MoD SOTA baseline: Router-Tuning (EMNLP'25, arXiv 2410.13184; PDF in baselines/papers/), faithful reproduction of the paper.

Paper mechanism (§3-4):
  - per layer Linear(H->1, no bias) + sigmoid score s, STE binary mask (training) / hard@τ (inference), τ=0.5;
  - gates zero-initialized ("training starts from dense", §4): the initial state executes everything;
  - granularity defaults to Attention + sequence level (§5.2 default; block/MLP/token also explored in the paper, Table 2);
  - mixing y = M⊙F(x) + x, hard binary (Eq.3/5), no gate weighting;
  - budget L = Ltask + λ·ReLU(||M||0 − s) (Eq.9/10, l0 fraction); main experiments s=0.5;
    λ grid {0,0.1,0.01,0.001}, middle 0.01 taken here (official code default 0.0 = unconstrained, noted as well);
  - routed layers: deepest half except the last ("deepest layers except the last one", §5; main experiment 16 layers / one half);
    N=24 means 11..22, the remaining 12 layers always dense (a byproduct of the interleaving, not designated fixed layers);
  - base 100% frozen, only the gates are trained (lr official default 1e-5); small-data few-steps (paper <30min/A6000).
Sandbox adaptations (beyond the paper, noted): same data same slice plain full-text LM objective (r1's chat-mask unified away);
held-out scoring eval from the same distribution (the paper uses LM-Harness downstream tasks); everything trains to convergence (StopOnPlateau).
Differences vs ours: gating mechanism / frozen base / attention granularity (ours is the whole layer).

Mechanisms (layer forward/patch/stats/eval) live in baselines/lib.py; this file keeps only the CLI and the training loop.
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
from baselines.lib import collect_rt_stats, eval_heldout_rt, patch_model_rt  # noqa: E402
from speaker.checkpoint import gate_state_dict  # noqa: E402
from speaker.evaluate import ema_update  # noqa: E402
from speaker.converge import StopOnPlateau  # noqa: E402 (ed3 unified convergence rule)
from speaker.train_common import build_model, build_tok, resolve_device, split_train_eval  # noqa: E402
from data.sft import make_collate  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_id", default="/home/hdd/model/Qwen2.5-7B-Instruct")
    p.add_argument("--data_path", default="./data/sft_t2t_mini.jsonl")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--max_steps", type=int, default=3000,
                   help="hard cap on steps; actual stopping is decided by the ed3 unified convergence rule (StopOnPlateau)")
    p.add_argument("--lr", type=float, default=1e-5, help="RT official default")
    p.add_argument("--max_samples", type=int, default=1000, help="same slice as kl2 (first 1000 train / last 100 eval)")
    p.add_argument("--eval_samples", type=int, default=100)
    p.add_argument("--granularity", default="attn_sequence",
                   help="paper default: Attention+sequence level (§5.2); block/mlp/token also explored in the paper")
    p.add_argument("--rt_target", type=float, default=0.5, help="target execution rate s (paper main experiments 50%%)")
    p.add_argument("--rt_scale", type=float, default=0.01,
                   help="capacity loss weight λ (middle of the paper grid {0,0.1,0.01,0.001}; official default 0 = unconstrained)")
    p.add_argument("--log_interval", type=int, default=10)
    p.add_argument("--save_dir", default="/tmp/rt_baseline")
    p.add_argument("--seed", type=int, default=None,
                   help="random seed (unset by default, preserving legacy behavior)")
    p.add_argument("--patience", type=int, default=3,
                   help="StopOnPlateau patience (enlarge to guarantee running to --max_steps)")
    return p.parse_args()


def main():
    args = parse_args()
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    tok = build_tok(args.model_id)
    model = build_model(args.model_id, device)

    full, eval_texts = split_train_eval(args.data_path, args.max_samples, args.eval_samples,
                                        tok=None, use_chat=False)  # plain full-text LM (same protocol as the other three)
    assert eval_texts, "empty eval slice"
    coll_fn = make_collate(tok, device, args.max_length)

    n = model.config.num_hidden_layers
    n_route = n // 2  # paper: deepest half except the last (N=24 means 11..22)
    is_mod = [(n - 1 - n_route) <= i < (n - 1) for i in range(n)]
    n_dense = n - sum(is_mod)
    # measure the dense baseline on the pristine base first, then attach the gates (otherwise the baseline is polluted by random gates)
    coll_eval = make_collate(tok, device, args.max_length)
    d = eval_heldout_rt(model, [], eval_texts, coll_eval, n_always=n)
    print(f"heldout dense baseline: loss {d['loss']:.3f} acc {d['acc']:.3f} "
          f"({len(eval_texts)} samples)", flush=True)
    gated = patch_model_rt(model, is_mod, args.granularity, threshold=0.5,
                           target=args.rt_target, scale=args.rt_scale)
    model.to(device)  # move the newly attached routers onto the device too
    n_gated = len(gated)
    print(f"RT patched: {n} layers, gated {n_gated} (deepest-half-except-last), "
          f"{args.granularity}, target exec {args.rt_target} scale {args.rt_scale}", flush=True)

    for p in model.parameters():
        p.requires_grad_(False)
    routers = [p for nm, p in model.named_parameters() if "router" in nm]
    for p in routers:
        p.requires_grad_(True)
    print(f"trainable routers {len(routers)} "
          f"({sum(p.numel() for p in routers) / 1e3:.1f}K params)", flush=True)

    init = eval_heldout_rt(model, gated, eval_texts[:10], coll_eval, n_always=n_dense)
    print(f"patched-init (hard@0.5): loss {init['loss']:.3f} acc {init['acc']:.3f} "
          f"exec {init['exec_rate']:.2f}", flush=True)

    opt = torch.optim.AdamW([{"params": routers, "lr": args.lr}], weight_decay=0.0)
    dl = DataLoader(full, batch_size=1, shuffle=True, collate_fn=coll_fn)
    os.makedirs(args.save_dir, exist_ok=True)
    stopper = StopOnPlateau(max_steps=args.max_steps, patience=args.patience)  # ed5: --max_steps now actually wired into the cap (was cosmetic-only)
    stop_subset = eval_texts[:40]  # subset for plateau checks (saves time); final eval still uses the full set
    step, ema_lm, ema_cap = 0, None, None
    t0 = time.time()
    window_tokens, window_t0 = 0, time.time()
    model.train()
    for epoch in range(1000):
        for b in dl:
            step += 1
            if stopper.capped(step):
                break
            out = model(**b)
            cap, mod_loss = collect_rt_stats(gated, training=True)
            lm = out.loss
            loss = lm + (mod_loss if mod_loss is not None else 0)  # upstream math: LM + Σrelu(cap-target)*scale
            ema_lm = ema_update(ema_lm, out.loss.item())
            ema_cap = ema_update(ema_cap, cap)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(routers, 1.0)
            opt.step()
            window_tokens += int(b["attention_mask"].sum())
            if step % args.log_interval == 0:
                k_est = n_dense + n_gated * cap
                rate = window_tokens / max(time.time() - window_t0, 1e-6)
                mem = (f" mem {torch.cuda.memory_allocated(device) / 1024**3:.2f}GB"
                       if device.type == "cuda" else "")
                window_tokens, window_t0 = 0, time.time()
                print(f"step {step:4d}/{args.max_steps} lm {lm.item():.3f} ema {ema_lm:.3f} "
                      f"cap {mod_loss.item() if mod_loss is not None else 0:.3f} "
                      f"exec {cap:.2f}/{ema_cap:.2f} k_est {k_est:.1f} {rate:.0f}tok/s{mem} "
                      f"{time.time() - t0:.0f}s", flush=True)
            if step % stopper.eval_every == 0:
                # plateau check (eval_heldout_rt restores train mode automatically)
                chk = eval_heldout_rt(model, gated, stop_subset, coll_eval, n_always=n_dense)
                with open(os.path.join(args.save_dir, "converge.jsonl"), "a", encoding="utf-8") as f:
                    f.write(json.dumps({"step": step, "subset_loss": chk["loss"],
                                        "best": stopper.best, "ema_lm": ema_lm,
                                        "exec": chk["exec_rate"]}) + "\n")
                if stopper.check(step, chk["loss"]):
                    print(f"[converged] step {step}, best heldout-subset loss {stopper.best:.3f}",
                          flush=True)
                    break
            if stopper.capped(step):
                break
        if stopper.capped(step):
            break

    torch.save({k: v.cpu() for k, v in gate_state_dict(model.state_dict()).items()},
               os.path.join(args.save_dir, "routers.pt"))
    with open(os.path.join(args.save_dir, "rt_config.json"), "w") as f:
        json.dump({"is_mod": is_mod, "granularity": args.granularity, "threshold": 0.5,
                   "target": args.rt_target, "scale": args.rt_scale,
                   "converged_step": step, "best_subset_loss": stopper.best}, f)
    print(f"saved to {args.save_dir}", flush=True)
    m = eval_heldout_rt(model, gated, eval_texts, coll_eval, n_always=n_dense)
    k_m = n_dense + n_gated * (m["exec_rate"] or 0)
    print(f"heldout | dense loss {d['loss']:.3f} acc {d['acc']:.3f} "
          f"| rt loss {m['loss']:.3f} acc {m['acc']:.3f} "
          f"(Δloss {m['loss'] - d['loss']:+.3f} Δacc {m['acc'] - d['acc']:+.3f}) "
          f"| exec {m['exec_rate']:.2f} k_est {k_m:.1f}", flush=True)


if __name__ == "__main__":
    main()
