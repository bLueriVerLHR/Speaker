"""Secondary baseline: dense fine-tuned version (same-spec LoRA pure SFT), the "trained dense" for a fair comparison.

Same protocol as train_mod/train_mdf/train_rt/ours (r3, 7B):
  - same data same slice (first max_samples train / last eval_samples eval, plain full-text LM);
  - same LoRA spec (rank 8, q_proj,v_proj, dropout 0.05, same as ours/LoRA-version baselines);
  - same single lr group (3e-5) + AdamW(wd 0.01) + the same StopOnPlateau convergence rule;
  - the only difference is "no gating/sparsity mechanism at all": loss = LM, k = all layers (upper reference with no sparsity savings).

ckpt: lora.pt (lora_ keys) + denseft_config.json; base loaded from --model_id (eval_compare wraps LoRA per the config).
The training loop lives in baselines/train_loop.py (P1 shared template); this file keeps the CLI + the family recipe.
"""
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Annotated, Optional

import torch
import typer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from baselines.train_loop import BaselineRecipe, run_baseline_training  # noqa: E402
from speaker.log import logger, setup_logger  # noqa: E402
from speaker.evaluate import ema_update, eval_heldout  # noqa: E402
from speaker.ruler import batch_accuracy  # noqa: E402 (P0: one accuracy scale)
from speaker.train_common import build_model, build_tok, resolve_device, split_train_eval, wrap_lora  # noqa: E402
from data.sft import make_collate  # noqa: E402


app = typer.Typer(add_completion=False)


class DenseRecipe(BaselineRecipe):
    """Dense-ft family strategy: pure LM loss, k fixed = all layers."""

    def __init__(self, model, trainable, n_layers, coll_eval, dense_res,
                 save_dir, use_lora, lora_rank, lora_alpha, lora_targets, lr):
        self.model, self.trainable, self.n_layers = model, trainable, n_layers
        self.coll_eval, self.d = coll_eval, dense_res
        self.save_dir, self.use_lora = save_dir, use_lora
        self.lora_rank, self.lora_alpha, self.lora_targets = lora_rank, lora_alpha, lora_targets
        self.lr = lr
        self.ema_acc = None
        self.acc_item = 0.0

    def loss_term(self, b, out):
        with torch.no_grad():
            self.acc_item = batch_accuracy(out.logits, b)
        self.ema_acc = ema_update(self.ema_acc, self.acc_item)
        return None

    def clip_params(self):
        return self.trainable

    def log_line(self, step, lm, ema_lm, rate, mem, elapsed, max_steps):
        return (f"[{time.strftime('%H:%M:%S')}] step {step:4d}/{max_steps} lm {lm.item():.3f} "
                f"ema {ema_lm:.3f} acc {self.acc_item:.2f}/{self.ema_acc:.2f} "
                f"{rate:.0f}tok/s{mem} {elapsed:.0f}s")

    def eval_model(self, model, texts):
        return eval_heldout(model, texts, self.coll_eval)

    def save(self, model, tok, step, stopper):
        sd = model.state_dict()
        if self.use_lora:
            torch.save({k: v.cpu() for k, v in sd.items() if "lora_" in k},
                       os.path.join(self.save_dir, "lora.pt"))
        else:
            model.save_pretrained(self.save_dir)
        tok.save_pretrained(self.save_dir)
        with open(os.path.join(self.save_dir, "denseft_config.json"), "w") as f:
            json.dump({"use_lora": self.use_lora,
                       "lora_rank": self.lora_rank, "lora_alpha": self.lora_alpha,
                       "lora_targets": [t.strip() for t in self.lora_targets.split(",") if t.strip()],
                       "lr": self.lr, "n_layers": self.n_layers,
                       "converged_step": step, "best_subset_loss": stopper.best,
                       "dense_raw": self.d}, f)

    def final_line(self, m):
        d, mm = self.d, m
        return (f"heldout | dense(raw) loss {d['loss']:.3f} acc {d['acc']:.3f} "
                f"| dense-ft loss {mm['loss']:.3f} acc {mm['acc']:.3f} "
                f"(Δloss {mm['loss'] - d['loss']:+.3f} Δacc {mm['acc'] - d['acc']:+.3f}) "
                f"| k {self.n_layers} (no sparsity)")


@app.command()
def main(
    model_id: Annotated[str, typer.Option("--model_id", help="same model as the fine-tuning route for comparison (7B)")] = "/home/hdd/model/Qwen2.5-7B-Instruct",
    data_path: Annotated[str, typer.Option("--data_path")] = "./data/sft_t2t_mini.jsonl",
    device: Annotated[str, typer.Option("--device")] = "cuda:0",
    max_length: Annotated[int, typer.Option("--max_length")] = 256,
    batch_size: Annotated[int, typer.Option("--batch_size")] = 1,
    max_steps: Annotated[int, typer.Option("--max_steps", help="display/annealing horizon alignment slot (actual stopping is decided by the ed3 unified convergence rule)")] = 500,
    lr: Annotated[float, typer.Option("--lr", help="unified single-group lr (same as the LoRA-version baselines)")] = 3e-5,
    max_samples: Annotated[int, typer.Option("--max_samples")] = 1000,
    eval_samples: Annotated[int, typer.Option("--eval_samples")] = 100,
    log_interval: Annotated[int, typer.Option("--log_interval")] = 10,
    save_dir: Annotated[str, typer.Option("--save_dir")] = "/tmp/denseft_baseline",
    seed: Annotated[Optional[int], typer.Option("--seed", help="random seed (unset by default, preserving legacy behavior)")] = None,
    patience: Annotated[int, typer.Option("--patience", help="StopOnPlateau patience (enlarge to guarantee running to --max_steps)")] = 3,
    use_lora: Annotated[bool, typer.Option("--use_lora/--no-use_lora", help="same-spec LoRA as ours/baselines (fairly fine-tuned dense); off = pure raw dense, for smoke tests")] = True,
    lora_rank: Annotated[int, typer.Option("--lora_rank", help="same spec as finetune/train.py")] = 8,
    lora_alpha: Annotated[int, typer.Option("--lora_alpha")] = 16,
    lora_targets: Annotated[str, typer.Option("--lora_targets")] = "q_proj,v_proj",
) -> None:
    """Dense-ft baseline (same-spec LoRA pure SFT, no gating)."""
    from speaker.terminal import setup_terminal
    setup_terminal()
    setup_logger(run_dir=save_dir)
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
    device = resolve_device(device)
    tok = build_tok(model_id)
    model = build_model(model_id, device)
    n_layers = model.config.num_hidden_layers

    full, eval_texts = split_train_eval(data_path, max_samples, eval_samples)
    assert eval_texts, "empty eval slice"
    coll_fn = make_collate(tok, device, max_length)
    coll_eval = make_collate(tok, device, max_length)

    # raw dense baseline (before LoRA is attached; should match the dense row of eval_compare)
    d = eval_heldout(model, eval_texts, coll_eval)
    logger.info(f"heldout dense baseline (raw): loss {d['loss']:.3f} acc {d['acc']:.3f} "
                f"({len(eval_texts)} samples)")

    if use_lora:
        model = wrap_lora(model, lora_rank, lora_alpha, lora_targets)
    trainable = [p for p in model.parameters() if p.requires_grad]
    groups = [{"params": trainable, "lr": lr}]
    logger.info(f"trainable {sum(p.numel() for p in trainable) / 1e6:.1f}M params "
                f"(lora={use_lora}), k fixed {n_layers}")
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        if use_lora:
            # r7 OOM fix: bs2 x len1024 x 7B LoRA needs checkpointing; the frozen-embed backward
            # pitfall is solved by enable_input_require_grads (same as finetune/train.py since r5)
            model.enable_input_require_grads()

    opt = torch.optim.AdamW(groups, weight_decay=0.01)
    recipe = DenseRecipe(model, trainable, n_layers, coll_eval, d,
                         save_dir=save_dir, use_lora=use_lora,
                         lora_rank=lora_rank, lora_alpha=lora_alpha,
                         lora_targets=lora_targets, lr=lr)
    run_baseline_training(model, tok, full, eval_texts, coll_fn, device, opt, recipe,
                          batch_size=batch_size, max_steps=max_steps,
                          log_interval=log_interval, save_dir=save_dir,
                          patience=patience)


if __name__ == "__main__":
    app()
