"""Shared training loop for the baseline scripts (P1 Template Method consolidation).

Historically train_mod / train_mdf / train_rt / train_dense each carried a ~60-line
near-identical copy of the loop scaffold (step/cap breaks, window tok/s accounting,
log cadence, plateau eval + converge.jsonl, ckpt save, final delta report). One
semantic change to the protocol had to be replicated four times — the exact cascade
the consolidation is killing (ed5's --steps→--max_steps rename and the r7 ruler drift
both paid this tax).

run_baseline_training owns the scaffold once; each family supplies a BaselineRecipe
(Strategy): the extra loss term (BCE / alpha*SigmaFG / capacity loss / none), the
clip set, the log line, the eval adapter, the ckpt save and the final report line.

Behavior contract: op order inside the loop reproduces the historical scripts
bit-for-bit (verified per family by same-seed CPU smoke: normalized stdout numeric
lines + ckpt md5). Known faithful quirks preserved on purpose:
- rt logs without the [HH:MM:SS] prefix (its historical format), batch_size is
  hardcoded to 1 (no --batch_size arg in its CLI);
- mdf's `exec_rate = ... if ema_k else 0` None/0.0 guard;
- modd/mdf clip over `[p for p in model.parameters() if p.requires_grad]`
  (re-evaluated per step), rt over its static routers list, dense over its static
  trainable list — identical param sets, order preserved.
"""
from __future__ import annotations

import json
import os
import time

import torch
from torch.utils.data import DataLoader

from speaker.converge import StopOnPlateau
from speaker.evaluate import ema_update

PLATEAU_SUBSET = 40  # held-out subset size for plateau checks (final eval uses the full set)


class BaselineRecipe:
    """Family strategy for run_baseline_training. All hooks are required unless a
    default is given; hooks must not change global RNG state order (no sampling)."""

    def loss_term(self, batch, out, args):
        """Extra loss term added to LM (already scaled by the family coefficient);
        None = pure LM step. Also the right place to update family EMA stats
        (k / exec / acc) used by the log line and the converge row."""
        return None

    def clip_params(self):
        raise NotImplementedError

    def log_line(self, step, lm, ema_lm, rate, mem, elapsed, args) -> str:
        raise NotImplementedError

    def eval_model(self, model, texts) -> dict:
        raise NotImplementedError

    def converge_row(self, chk) -> dict:
        return {}

    def save(self, model, tok, args, step, stopper) -> None:
        raise NotImplementedError

    def final_line(self, m: dict, args) -> str:
        raise NotImplementedError


def run_baseline_training(args, model, tok, full, eval_texts, coll_fn, device, opt,
                          recipe: BaselineRecipe):
    """Template Method: the shared loop of the four baseline scripts. Everything
    before (patch/LoRA/trainable setup/dense baseline measurement) stays in the
    caller. Returns a small summary dict."""
    dl = DataLoader(full, batch_size=getattr(args, "batch_size", 1), shuffle=True,
                    collate_fn=coll_fn)
    os.makedirs(args.save_dir, exist_ok=True)
    stopper = StopOnPlateau(max_steps=args.max_steps, patience=args.patience)  # ed5: --max_steps is the real cap
    stop_subset = eval_texts[:PLATEAU_SUBSET]  # subset for plateau checks (saves time); final eval still uses the full set
    step, ema_lm = 0, None
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
            term = recipe.loss_term(b, out, args)
            loss = lm + (term if term is not None else 0)
            ema_lm = ema_update(ema_lm, lm.item())
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(recipe.clip_params(), 1.0)
            opt.step()
            window_tokens += int(b["attention_mask"].sum())
            if step % args.log_interval == 0:
                rate = window_tokens / max(time.time() - window_t0, 1e-6)
                mem = (f" mem {torch.cuda.memory_allocated(device) / 1024**3:.2f}GB"
                       if device.type == "cuda" else "")
                window_tokens, window_t0 = 0, time.time()
                print(recipe.log_line(step, lm, ema_lm, rate, mem, time.time() - t0, args),
                      flush=True)
            if step % stopper.eval_every == 0:
                # plateau check (family eval restores train mode automatically)
                chk = recipe.eval_model(model, stop_subset)
                with open(os.path.join(args.save_dir, "converge.jsonl"), "a", encoding="utf-8") as f:
                    f.write(json.dumps({"step": step, "subset_loss": chk["loss"],
                                        "best": stopper.best, "ema_lm": ema_lm,
                                        **recipe.converge_row(chk)}) + "\n")
                if stopper.check(step, chk["loss"]):
                    print(f"[converged] step {step}, best heldout-subset loss {stopper.best:.3f}",
                          flush=True)
                    break
            if stopper.capped(step):
                break
        if stopper.capped(step):
            break

    recipe.save(model, tok, args, step, stopper)
    print(f"saved to {args.save_dir}", flush=True)
    m = recipe.eval_model(model, eval_texts)
    print(recipe.final_line(m, args), flush=True)
    return {"step": step, "best": stopper.best, "final": m}
