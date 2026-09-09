"""Secondary baseline: dense fine-tuned version (same-spec LoRA pure SFT), the "trained dense" for a fair comparison.

Same protocol as train_mod/train_mdf/train_rt/ours (r3, 7B):
  - same data same slice (first max_samples train / last eval_samples eval, plain full-text LM);
  - same LoRA spec (rank 8, q_proj,v_proj, dropout 0.05, same as ours/LoRA-version baselines);
  - same single lr group (3e-5) + AdamW(wd 0.01) + the same StopOnPlateau convergence rule;
  - the only difference is "no gating/sparsity mechanism at all": loss = LM, k = all layers (upper reference with no sparsity savings).

ckpt: lora.pt (lora_ keys) + denseft_config.json; base loaded from --model_id (eval_compare wraps LoRA per the config).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from speaker.metrics import per_token_correct  # noqa: E402
from speaker.evaluate import ema_update, eval_heldout  # noqa: E402
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
    p.add_argument("--steps", type=int, default=500,
                   help="display/annealing horizon alignment slot (actual stopping is decided by the ed3 unified convergence rule)")
    p.add_argument("--lr", type=float, default=3e-5, help="unified single-group lr (same as the LoRA-version baselines)")
    p.add_argument("--max_samples", type=int, default=1000)
    p.add_argument("--eval_samples", type=int, default=100)
    p.add_argument("--log_interval", type=int, default=10)
    p.add_argument("--save_dir", default="/tmp/denseft_baseline")
    p.add_argument("--use_lora", default=True, action=argparse.BooleanOptionalAction,
                   help="same-spec LoRA as ours/baselines (fairly fine-tuned dense); off = pure raw dense, for smoke tests")
    p.add_argument("--lora_rank", type=int, default=8, help="same spec as finetune/train.py")
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_targets", default="q_proj,v_proj")
    return p.parse_args()


def main():
    args = parse_args()
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
    if not args.use_lora and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    opt = torch.optim.AdamW(groups, weight_decay=0.01)
    dl = DataLoader(full, batch_size=args.batch_size, shuffle=True, collate_fn=coll_fn)
    os.makedirs(args.save_dir, exist_ok=True)
    stopper = StopOnPlateau()  # ed3 unified convergence rule (same constants as ours and the three baselines)
    stop_subset = eval_texts[:40]  # subset for plateau checks (saves time); final eval still uses the full set
    step, ema_lm, ema_acc = 0, None, None
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
            ema_lm = ema_update(ema_lm, lm.item())
            with torch.no_grad():
                correct = per_token_correct(out.logits.float(), b["labels"])
                valid = (b["labels"] != -100)
                acc_item = correct[valid].float().mean().item() if valid.any() else 0.0
            ema_acc = ema_update(ema_acc, acc_item)
            opt.zero_grad()
            lm.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            window_tokens += int(b["attention_mask"].sum())
            if step % args.log_interval == 0:
                rate = window_tokens / max(time.time() - window_t0, 1e-6)
                mem = (f" mem {torch.cuda.memory_allocated(device) / 1024**3:.2f}GB"
                       if device.type == "cuda" else "")
                window_tokens, window_t0 = 0, time.time()
                print(f"[{time.strftime('%H:%M:%S')}] step {step:4d}/{args.steps} lm {lm.item():.3f} "
                      f"ema {ema_lm:.3f} acc {acc_item:.2f}/{ema_acc:.2f} "
                      f"{rate:.0f}tok/s{mem} {time.time() - t0:.0f}s", flush=True)
            if step % stopper.eval_every == 0 and step > 0:
                chk = eval_heldout(model, stop_subset, coll_eval)
                with open(os.path.join(args.save_dir, "converge.jsonl"), "a", encoding="utf-8") as f:
                    f.write(json.dumps({"step": step, "subset_loss": chk["loss"],
                                        "best": stopper.best, "ema_lm": ema_lm}) + "\n")
                if stopper.check(step, chk["loss"]):
                    print(f"[converged] step {step}, best heldout-subset loss {stopper.best:.3f}",
                          flush=True)
                    break
            if stopper.capped(step):
                break
        if stopper.capped(step):
            break

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
                   "lr": args.lr, "n_layers": n_layers,
                   "converged_step": step, "best_subset_loss": stopper.best,
                   "dense_raw": d}, f)
    print(f"saved to {args.save_dir}", flush=True)
    m = eval_heldout(model, eval_texts, coll_eval)
    print(f"heldout | dense(raw) loss {d['loss']:.3f} acc {d['acc']:.3f} "
          f"| dense-ft loss {m['loss']:.3f} acc {m['acc']:.3f} "
          f"(Δloss {m['loss'] - d['loss']:+.3f} Δacc {m['acc'] - d['acc']:+.3f}) "
          f"| k {n_layers} (no sparsity)", flush=True)


if __name__ == "__main__":
    main()
