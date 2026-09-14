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

Mechanisms (layer forward/patch/stats/eval) live in baselines/lib.py; the training loop lives in
baselines/train_loop.py (P1 shared template); this file keeps the CLI + the family recipe.
"""
import json
import os
import random
import sys
from pathlib import Path
from typing import Annotated, Optional

import torch
import typer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from baselines.lib import collect_rt_stats, eval_heldout_rt, patch_model_rt  # noqa: E402
from baselines.train_loop import BaselineRecipe, run_baseline_training  # noqa: E402
from speaker.log import logger, setup_logger  # noqa: E402
from speaker.checkpoint import gate_state_dict  # noqa: E402
from speaker.evaluate import ema_update  # noqa: E402
from speaker.train_common import build_model, build_tok, resolve_device, split_train_eval  # noqa: E402
from data.sft import make_collate  # noqa: E402


app = typer.Typer(add_completion=False)


class RtRecipe(BaselineRecipe):
    """Router-Tuning family strategy: frozen base, routers-only training,
    loss = LM + Σ capacity loss (upstream math)."""

    def __init__(self, model, gated, is_mod, n_dense, coll_eval, dense_res,
                 granularity, rt_target, rt_scale, save_dir):
        self.model, self.gated, self.is_mod = model, gated, is_mod
        self.n_dense = n_dense
        self.coll_eval, self.d = coll_eval, dense_res
        self.granularity, self.rt_target, self.rt_scale = granularity, rt_target, rt_scale
        self.save_dir = save_dir
        self.ema_cap = None
        self._cap = 0.0
        self._mloss = None
        self.routers = [p for nm, p in model.named_parameters() if "router" in nm]

    def loss_term(self, b, out):
        cap, mod_loss = collect_rt_stats(self.gated, training=True)
        self._cap, self._mloss = cap, mod_loss
        self.ema_cap = ema_update(self.ema_cap, cap)
        return mod_loss

    def clip_params(self):
        return self.routers

    def log_line(self, step, lm, ema_lm, rate, mem, elapsed, max_steps):
        k_est = self.n_dense + len(self.gated) * self._cap
        # historical rt format: no [HH:MM:SS] prefix
        return (f"step {step:4d}/{max_steps} lm {lm.item():.3f} ema {ema_lm:.3f} "
                f"cap {self._mloss.item() if self._mloss is not None else 0:.3f} "
                f"exec {self._cap:.2f}/{self.ema_cap:.2f} k_est {k_est:.1f} {rate:.0f}tok/s{mem} "
                f"{elapsed:.0f}s")

    def eval_model(self, model, texts):
        return eval_heldout_rt(model, self.gated, texts, self.coll_eval,
                               n_always=self.n_dense)

    def converge_row(self, chk):
        return {"exec": chk["exec_rate"]}

    def save(self, model, tok, step, stopper):
        torch.save({k: v.cpu() for k, v in gate_state_dict(model.state_dict()).items()},
                   os.path.join(self.save_dir, "routers.pt"))
        with open(os.path.join(self.save_dir, "rt_config.json"), "w") as f:
            json.dump({"is_mod": self.is_mod, "granularity": self.granularity, "threshold": 0.5,
                       "target": self.rt_target, "scale": self.rt_scale,
                       "converged_step": step, "best_subset_loss": stopper.best}, f)

    def final_line(self, m):
        d, mm = self.d, m
        k_m = self.n_dense + len(self.gated) * (mm["exec_rate"] or 0)
        return (f"heldout | dense loss {d['loss']:.3f} acc {d['acc']:.3f} "
                f"| rt loss {mm['loss']:.3f} acc {mm['acc']:.3f} "
                f"(Δloss {mm['loss'] - d['loss']:+.3f} Δacc {mm['acc'] - d['acc']:+.3f}) "
                f"| exec {mm['exec_rate']:.2f} k_est {k_m:.1f}")


@app.command()
def main(
    model_id: Annotated[str, typer.Option("--model_id")] = "/home/hdd/model/Qwen2.5-7B-Instruct",
    data_path: Annotated[str, typer.Option("--data_path")] = "./data/sft_t2t_mini.jsonl",
    device: Annotated[str, typer.Option("--device")] = "cuda:0",
    max_length: Annotated[int, typer.Option("--max_length")] = 256,
    max_steps: Annotated[int, typer.Option("--max_steps", help="hard cap on steps; actual stopping is decided by the ed3 unified convergence rule (StopOnPlateau)")] = 3000,
    lr: Annotated[float, typer.Option("--lr", help="RT official default")] = 1e-5,
    max_samples: Annotated[int, typer.Option("--max_samples", help="same slice as kl2 (first 1000 train / last 100 eval)")] = 1000,
    eval_samples: Annotated[int, typer.Option("--eval_samples")] = 100,
    granularity: Annotated[str, typer.Option("--granularity", help="paper default: Attention+sequence level (§5.2); block/mlp/token also explored in the paper")] = "attn_sequence",
    rt_target: Annotated[float, typer.Option("--rt_target", help="target execution rate s (paper main experiments 50%)")] = 0.5,
    rt_scale: Annotated[float, typer.Option("--rt_scale", help="capacity loss weight λ (middle of the paper grid {0,0.1,0.01,0.001}; official default 0 = unconstrained)")] = 0.01,
    log_interval: Annotated[int, typer.Option("--log_interval")] = 10,
    save_dir: Annotated[str, typer.Option("--save_dir")] = "/tmp/rt_baseline",
    seed: Annotated[Optional[int], typer.Option("--seed", help="random seed (unset by default, preserving legacy behavior)")] = None,
    patience: Annotated[int, typer.Option("--patience", help="StopOnPlateau patience (enlarge to guarantee running to --max_steps)")] = 3,
) -> None:
    """Router-Tuning baseline (frozen base, routers-only)."""
    from speaker.terminal import setup_terminal
    setup_terminal()
    setup_logger(run_dir=save_dir)
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
    device = resolve_device(device)
    tok = build_tok(model_id)
    model = build_model(model_id, device)

    full, eval_texts = split_train_eval(data_path, max_samples, eval_samples,
                                        tok=None, use_chat=False)  # plain full-text LM (same protocol as the other three)
    assert eval_texts, "empty eval slice"
    coll_fn = make_collate(tok, device, max_length)

    n = model.config.num_hidden_layers
    n_route = n // 2  # paper: deepest half except the last (N=24 means 11..22)
    is_mod = [(n - 1 - n_route) <= i < (n - 1) for i in range(n)]
    n_dense = n - sum(is_mod)
    # measure the dense baseline on the pristine base first, then attach the gates (otherwise the baseline is polluted by random gates)
    coll_eval = make_collate(tok, device, max_length)
    d = eval_heldout_rt(model, [], eval_texts, coll_eval, n_always=n)
    logger.info(f"heldout dense baseline: loss {d['loss']:.3f} acc {d['acc']:.3f} "
                f"({len(eval_texts)} samples)")
    gated = patch_model_rt(model, is_mod, granularity, threshold=0.5,
                           target=rt_target, scale=rt_scale)
    model.to(device)  # move the newly attached routers onto the device too
    n_gated = len(gated)
    logger.info(f"RT patched: {n} layers, gated {n_gated} (deepest-half-except-last), "
                f"{granularity}, target exec {rt_target} scale {rt_scale}")

    for p in model.parameters():
        p.requires_grad_(False)
    routers = [p for nm, p in model.named_parameters() if "router" in nm]
    for p in routers:
        p.requires_grad_(True)
    logger.info(f"trainable routers {len(routers)} "
                f"({sum(p.numel() for p in routers) / 1e3:.1f}K params)")

    init = eval_heldout_rt(model, gated, eval_texts[:10], coll_eval, n_always=n_dense)
    logger.info(f"patched-init (hard@0.5): loss {init['loss']:.3f} acc {init['acc']:.3f} "
                f"exec {init['exec_rate']:.2f}")

    opt = torch.optim.AdamW([{"params": routers, "lr": lr}], weight_decay=0.0)
    recipe = RtRecipe(model, gated, is_mod, n_dense, coll_eval, d,
                      granularity=granularity, rt_target=rt_target,
                      rt_scale=rt_scale, save_dir=save_dir)
    run_baseline_training(model, tok, full, eval_texts, coll_fn, device, opt, recipe,
                          max_steps=max_steps, log_interval=log_interval,
                          save_dir=save_dir, patience=patience)


if __name__ == "__main__":
    app()
