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
import random
import sys
from pathlib import Path
from typing import Annotated, Optional

import torch
import typer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from baselines.lib import (  # noqa: E402
    GATE_KEYS,
    collect_modd_stats,
    eval_heldout_modd,
    patch_model_modd,
)
from baselines.train_loop import RoutedRecipe, run_baseline_training  # noqa: E402
from speaker.log import logger, setup_logger  # noqa: E402
from speaker.train_common import build_model, build_tok, resolve_device, split_train_eval, wrap_lora  # noqa: E402
from data.sft import make_collate  # noqa: E402


app = typer.Typer(add_completion=False)


class ModdRecipe(RoutedRecipe):
    """MoD family strategy: BCE aux term + capacity-selected k accounting."""
    collect_fn = staticmethod(collect_modd_stats)
    fam = "modd"
    config_name = "modd_config.json"
    eval_fn = staticmethod(eval_heldout_modd)

    def __init__(self, *a, capacity, bce_coef, **k):
        super().__init__(*a, coef=bce_coef, **k)
        self.capacity = capacity
        self.bce_coef = bce_coef

    def aux_token(self):
        cap_now = self.routed[0].route_capacity if self.routed else self.capacity
        return (f"bce {self._aux.item() if self._aux is not None else 0:.3f} "
                f"cap {cap_now:.3f} ")

    def manifest_extra(self):
        return {"capacity": self.capacity, "bce_coef": self.bce_coef}


@app.command()
def main(
    model_id: Annotated[str, typer.Option("--model_id", help="same model as the fine-tuning route for comparison (7B)")] = "/home/hdd/model/Qwen2.5-7B-Instruct",
    data_path: Annotated[str, typer.Option("--data_path")] = "./data/sft_t2t_mini.jsonl",
    device: Annotated[str, typer.Option("--device")] = "cuda:0",
    max_length: Annotated[int, typer.Option("--max_length")] = 256,
    batch_size: Annotated[int, typer.Option("--batch_size")] = 1,
    max_steps: Annotated[int, typer.Option("--max_steps", help="hard cap on steps; actual stopping is decided by the ed3 unified convergence rule (StopOnPlateau)")] = 3000,
    lr: Annotated[float, typer.Option("--lr", help="unified single-group lr (papers have no fine-tuning recipe; regime-conversion setting)")] = 3e-5,
    max_samples: Annotated[int, typer.Option("--max_samples")] = 1000,
    eval_samples: Annotated[int, typer.Option("--eval_samples")] = 100,
    log_interval: Annotated[int, typer.Option("--log_interval")] = 10,
    save_dir: Annotated[str, typer.Option("--save_dir")] = "/tmp/modd_baseline",
    seed: Annotated[Optional[int], typer.Option("--seed", help="random seed (unset by default, preserving legacy behavior)")] = None,
    patience: Annotated[int, typer.Option("--patience", help="StopOnPlateau patience (enlarge to guarantee running to --max_steps)")] = 3,
    use_lora: Annotated[bool, typer.Option("--use_lora/--no-use_lora", help="add LoRA to the base and jointly train it with the gates/routers (papers are full-parameter; 7B full-parameter does not fit in 24GB, hardware adaptation, same rank/targets spec as ours)")] = True,
    lora_rank: Annotated[int, typer.Option("--lora_rank", help="same spec as finetune/train.py")] = 8,
    lora_alpha: Annotated[int, typer.Option("--lora_alpha")] = 16,
    lora_targets: Annotated[str, typer.Option("--lora_targets")] = "q_proj,v_proj",
    capacity: Annotated[float, typer.Option("--capacity", help="per-layer top-k capacity ratio (the paper's optimum 12.5%, fixed throughout, no annealing)")] = 0.125,
    bce_coef: Annotated[float, typer.Option("--bce_coef", help="BCE auxiliary loss weight (paper §3.5 method 1; coefficient value not given in the paper)")] = 1.0,
) -> None:
    """MoD original baseline (token-choice top-k capacity + BCE aux)."""
    from speaker.terminal import setup_terminal
    setup_terminal()
    setup_logger(run_dir=save_dir)
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
    device = resolve_device(device)
    tok = build_tok(model_id)
    model = build_model(model_id, device)

    full, eval_texts = split_train_eval(data_path, max_samples, eval_samples)
    assert eval_texts, "empty eval slice"
    coll_fn = make_collate(tok, device, max_length)

    n = model.config.num_hidden_layers
    is_routed = [(i % 2 == 1) for i in range(n)]  # paper: interleaved (every other block); parity unspecified, odd layers as in mdf
    n_dense = n - sum(is_routed)
    # dense baseline on the pristine base (before routing is attached)
    coll_eval = make_collate(tok, device, max_length)
    d = eval_heldout_modd(model, [], n, eval_texts, coll_eval)
    logger.info(f"heldout dense baseline: loss {d['loss']:.3f} acc {d['acc']:.3f} "
                f"({len(eval_texts)} samples)")

    init_cap = capacity  # capacity fixed throughout, no annealing (none in the paper)
    routed = patch_model_modd(model, is_routed, capacity=init_cap)
    model.to(device)
    k_est = n_dense + capacity * len(routed)
    logger.info(f"MoD patched: {n} layers, routed {len(routed)} (interleave odd, paper §4), "
                f"capacity {capacity} fixed, k_est {k_est:.1f}, bce {bce_coef}, "
                f"lr {lr} unified")
    init = eval_heldout_modd(model, routed, n_dense, eval_texts[:10], coll_eval)
    logger.info(f"patched-init: loss {init['loss']:.3f} acc {init['acc']:.3f} "
                f"k {init['mean_k']:.1f}±{init['std_k']:.1f}")

    if use_lora:
        model = wrap_lora(model, lora_rank, lora_alpha, lora_targets)
        # peft froze the router attached during patching; unfreeze it (LoRA+router joint training, same protocol as ours)
        for nm, p in model.named_parameters():
            if any(g in nm for g in GATE_KEYS):
                p.requires_grad_(True)
    trainable = [p for p in model.parameters() if p.requires_grad]
    groups = [{"params": trainable, "lr": lr}]  # unified single group (no fine-tuning recipe in the paper; with LoRA = lora+router)
    n_router_train = sum(1 for nm, p in model.named_parameters()
                         if any(g in nm for g in GATE_KEYS) and p.requires_grad)
    logger.info(f"trainable {sum(p.numel() for p in trainable) / 1e6:.1f}M params "
                f"(lora={use_lora}, routers train {n_router_train})")
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        if use_lora:
            # r7 OOM fix: bs2 x len1024 x 7B LoRA needs checkpointing; the frozen-embed backward
            # pitfall is solved by enable_input_require_grads (same as finetune/train.py since r5)
            model.enable_input_require_grads()

    opt = torch.optim.AdamW(groups, weight_decay=0.01)
    recipe = ModdRecipe(model, routed, is_routed, n_dense, coll_eval, d,
                        capacity=capacity, bce_coef=bce_coef,
                        save_cfg=dict(save_dir=save_dir, use_lora=use_lora, lr=lr,
                                      lora_rank=lora_rank, lora_alpha=lora_alpha,
                                      lora_targets=lora_targets))
    run_baseline_training(model, tok, full, eval_texts, coll_fn, device, opt, recipe,
                          batch_size=batch_size, max_steps=max_steps,
                          log_interval=log_interval, save_dir=save_dir,
                          patience=patience)


if __name__ == "__main__":
    app()
