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
- rt's historical log line carries no [HH:MM:SS] prefix while the others do;
  under the shared logger both get the logger timestamp (content identical),
  batch_size is hardcoded to 1 (no --batch_size arg in its CLI);
- mdf's `exec_rate = ... if ema_k else 0` None/0.0 guard;
- modd/mdf clip over `[p for p in model.parameters() if p.requires_grad]`
  (re-evaluated per step), rt over its static routers list, dense over its static
  trainable list — identical param sets, order preserved.
"""
from __future__ import annotations

import os
import time

import torch
from torch.utils.data import DataLoader

from baselines.lib import save_baseline_ckpt
from speaker.converge import StopOnPlateau
from speaker.evaluate import ema_update
from speaker.log import add_jsonl, emit, event, logger, setup_logger

PLATEAU_SUBSET = 40  # held-out subset size for plateau checks (final eval uses the full set)


class BaselineRecipe:
    """Family strategy for run_baseline_training. All hooks are required unless a
    default is given; hooks must not change global RNG state order (no sampling)."""

    def loss_term(self, batch, out):
        """Extra loss term added to LM (already scaled by the family coefficient);
        None = pure LM step. Also the right place to update family EMA stats
        (k / exec / acc) used by the log line and the converge row."""
        return None

    def clip_params(self):
        raise NotImplementedError

    def log_line(self, step, lm, ema_lm, rate, mem, elapsed, max_steps) -> str:
        raise NotImplementedError

    def eval_model(self, model, texts) -> dict:
        raise NotImplementedError

    def converge_row(self, chk) -> dict:
        return {}

    def save(self, model, tok, step, stopper) -> None:
        raise NotImplementedError

    def final_line(self, m: dict) -> str:
        raise NotImplementedError


class RoutedRecipe(BaselineRecipe):
    """Shared strategy for the two routed families: MoD (token-choice top-k +
    BCE aux) and MoDification (threshold-p + R load target). Loop-visible
    behavior is identical; only the stats collector, the aux coefficient, the
    log token, the eval adapter, and the ckpt manifest differ."""

    collect_fn = None  # (routed, training) -> (aux, k); subclass sets staticmethod
    fam = ""           # log label ("modd"/"mdf")
    config_name = ""   # ckpt manifest filename
    eval_fn = None     # (model, routed, n_dense, texts, coll) -> res; subclass sets staticmethod

    def __init__(self, model, routed, is_routed, n_dense, coll_eval, dense_res,
                 coef, save_cfg: dict):
        self.model, self.routed, self.is_routed = model, routed, is_routed
        self.n_dense = n_dense
        self.coll_eval, self.d = coll_eval, dense_res
        self.coef = coef  # aux loss scale (bce_coef / alpha), explicit, not via args
        self.save_cfg = save_cfg  # save_dir/use_lora/lr/lora_* manifest fields
        self.ema_k = None
        self._aux = None

    def loss_term(self, b, out):
        aux, k = self.collect_fn(self.routed, training=True)
        self._aux = aux
        if k is not None:
            valid = b["attention_mask"].bool()
            self.ema_k = ema_update(self.ema_k, (k[valid] + self.n_dense).mean().item())
        return (self.coef * aux if aux is not None and self.coef > 0 else None)

    def clip_params(self):
        return [p for p in self.model.parameters() if p.requires_grad]

    def aux_token(self) -> str:
        raise NotImplementedError

    def log_line(self, step, lm, ema_lm, rate, mem, elapsed, max_steps):
        # no manual timestamp: the shared logger stamps every line
        return (f"step {step:4d}/{max_steps} lm {lm.item():.3f} "
                f"ema {ema_lm:.3f} {self.aux_token()}"
                f"k {self.ema_k:.1f} {rate:.0f}tok/s{mem} "
                f"{elapsed:.0f}s")

    def eval_model(self, model, texts):
        return self.eval_fn(model, self.routed, self.n_dense, texts, self.coll_eval)

    def converge_row(self, chk):
        return {"k": self.ema_k}

    def manifest_extra(self) -> dict:
        raise NotImplementedError

    def save(self, model, tok, step, stopper):
        c = self.save_cfg
        save_baseline_ckpt(model, tok, c["save_dir"], c["use_lora"],
                           {"is_routed": self.is_routed, "lr": c["lr"],
                            "use_lora": c["use_lora"],
                            "lora_rank": c["lora_rank"], "lora_alpha": c["lora_alpha"],
                            "lora_targets": [t.strip() for t in c["lora_targets"].split(",") if t.strip()],
                            "converged_step": step, "best_subset_loss": stopper.best,
                            **self.manifest_extra()},
                           self.config_name)

    def final_line(self, m):
        d, mm = self.d, m
        return (f"heldout | dense loss {d['loss']:.3f} acc {d['acc']:.3f} "
                f"| {self.fam} loss {mm['loss']:.3f} acc {mm['acc']:.3f} "
                f"(Δloss {mm['loss'] - d['loss']:+.3f} Δacc {mm['acc'] - d['acc']:+.3f}) "
                f"| k {mm['mean_k']:.1f}±{mm['std_k']:.1f}")


def run_baseline_training(model, tok, full, eval_texts, coll_fn, device, opt,
                          recipe: BaselineRecipe, *, batch_size=1, max_steps,
                          log_interval, save_dir, patience):
    """Template Method: the shared loop of the four baseline scripts. Everything
    before (patch/LoRA/trainable setup/dense baseline measurement) stays in the
    caller. Loop controls are explicit kwargs (no args bundle). Returns a small
    summary dict."""
    dl = DataLoader(full, batch_size=batch_size, shuffle=True, collate_fn=coll_fn)
    os.makedirs(save_dir, exist_ok=True)
    setup_logger(run_dir=save_dir)
    add_jsonl(os.path.join(save_dir, "converge.jsonl"), "converge")
    stopper = StopOnPlateau(max_steps=max_steps, patience=patience)  # ed5: --max_steps is the real cap
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
            term = recipe.loss_term(b, out)
            loss = lm + (term if term is not None else 0)
            ema_lm = ema_update(ema_lm, lm.item())
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(recipe.clip_params(), 1.0)
            opt.step()
            window_tokens += int(b["attention_mask"].sum())
            if step % log_interval == 0:
                rate = window_tokens / max(time.time() - window_t0, 1e-6)
                mem = (f" mem {torch.cuda.memory_allocated(device) / 1024**3:.2f}GB"
                       if device.type == "cuda" else "")
                window_tokens, window_t0 = 0, time.time()
                logger.info(recipe.log_line(step, lm, ema_lm, rate, mem, time.time() - t0,
                                        max_steps))
            if step % stopper.eval_every == 0:
                # plateau check (family eval restores train mode automatically)
                chk = recipe.eval_model(model, stop_subset)
                emit("converge", step=step, subset_loss=chk["loss"],
                     best=stopper.best, ema_lm=ema_lm, **recipe.converge_row(chk))
                if stopper.check(step, chk["loss"]):
                    event(f"converged step {step}, best heldout-subset loss {stopper.best:.3f}")
                    break
            if stopper.capped(step):
                break
        if stopper.capped(step):
            break

    recipe.save(model, tok, step, stopper)
    logger.info(f"saved to {save_dir}")
    m = recipe.eval_model(model, eval_texts)
    logger.info(recipe.final_line(m))
    return {"step": step, "best": stopper.best, "final": m}
