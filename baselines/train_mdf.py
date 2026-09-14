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
import random
import sys
from pathlib import Path
from typing import Annotated, Optional

import torch
import typer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from baselines.lib import (  # noqa: E402
    GATE_KEYS,
    collect_mdf_stats,
    eval_heldout_mdf,
    patch_model_mdf,
)
from baselines.train_loop import RoutedRecipe, run_baseline_training  # noqa: E402
from speaker.log import logger, setup_logger  # noqa: E402
from speaker.train_common import build_model, build_tok, resolve_device, split_train_eval, wrap_lora  # noqa: E402
from data.sft import make_collate  # noqa: E402


app = typer.Typer(add_completion=False)


class MdfRecipe(RoutedRecipe):
    """MoDification family strategy: R = alpha*Sigma(F*G) load target + threshold-p k accounting."""
    collect_fn = staticmethod(collect_mdf_stats)
    fam = "mdf"
    config_name = "mdf_config.json"
    eval_fn = staticmethod(eval_heldout_mdf)

    def __init__(self, *a, p, alpha, **k):
        super().__init__(*a, coef=alpha, **k)
        self.p = p
        self.alpha = alpha

    def aux_token(self):
        exec_rate = (self.ema_k - self.n_dense) / max(len(self.routed), 1) if self.ema_k else 0
        return (f"R {self.alpha * self._aux.item() if self._aux is not None else 0:.4f} "
                f"exec {exec_rate:.2f} ")

    def manifest_extra(self):
        return {"p": self.p, "alpha": self.alpha}


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
    save_dir: Annotated[str, typer.Option("--save_dir")] = "/tmp/mdf_baseline",
    seed: Annotated[Optional[int], typer.Option("--seed", help="random seed (unset by default, preserving legacy behavior)")] = None,
    patience: Annotated[int, typer.Option("--patience", help="StopOnPlateau patience (enlarge to guarantee running to --max_steps)")] = 3,
    use_lora: Annotated[bool, typer.Option("--use_lora/--no-use_lora", help="add LoRA to the base and jointly train it with the gates/routers (papers are full-parameter; 7B full-parameter does not fit in 24GB, hardware adaptation, same rank/targets spec as ours)")] = True,
    lora_rank: Annotated[int, typer.Option("--lora_rank", help="same spec as finetune/train.py")] = 8,
    lora_alpha: Annotated[int, typer.Option("--lora_alpha")] = 16,
    lora_targets: Annotated[str, typer.Option("--lora_targets")] = "q_proj,v_proj",
    p: Annotated[float, typer.Option("--p", help="threshold-p gate threshold (paper 0.5, 0.55 for large models)")] = 0.5,
    alpha: Annotated[float, typer.Option("--alpha", help="coefficient of R=α·ΣFG (paper 0.01)")] = 0.01,
) -> None:
    """MoDification baseline (threshold-p + R load target)."""
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
    is_routed = [(i % 2 == 1) for i in range(n)]  # paper's interleaved (every other block): odd layers, 12/24
    n_dense = n - sum(is_routed)
    coll_eval = make_collate(tok, device, max_length)
    d = eval_heldout_mdf(model, [], n, eval_texts, coll_eval)
    logger.info(f"heldout dense baseline: loss {d['loss']:.3f} acc {d['acc']:.3f} "
                f"({len(eval_texts)} samples)")

    routed = patch_model_mdf(model, is_routed, p=p)
    model.to(device)
    logger.info(f"MoDification patched: {n} layers, routed {len(routed)} (interleave), "
                f"p {p}, alpha {alpha}, lr {lr} unified")
    init = eval_heldout_mdf(model, routed, n_dense, eval_texts[:10], coll_eval)
    logger.info(f"patched-init: loss {init['loss']:.3f} acc {init['acc']:.3f} "
                f"k {init['mean_k']:.1f}±{init['std_k']:.1f}")

    if use_lora:
        model = wrap_lora(model, lora_rank, lora_alpha, lora_targets)
        # peft froze the gates attached during patching; unfreeze them (LoRA+gate joint training, same protocol as ours)
        # NOTE: loop var must not be `p` — it would shadow the threshold-p CLI arg below (r8c manifest crash).
        for nm, param in model.named_parameters():
            if any(g in nm for g in GATE_KEYS):
                param.requires_grad_(True)
    trainable = [p for p in model.parameters() if p.requires_grad]
    groups = [{"params": trainable, "lr": lr}]  # unified single group (paper §4; with LoRA = lora+gates)
    n_router_train = sum(1 for nm, param in model.named_parameters()
                         if any(g in nm for g in GATE_KEYS) and param.requires_grad)
    logger.info(f"trainable {sum(p.numel() for p in trainable) / 1e6:.1f}M params "
                f"(lora={use_lora}, gates train {n_router_train})")
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        if use_lora:
            # r7 OOM fix: bs2 x len1024 x 7B LoRA needs checkpointing; the frozen-embed backward
            # pitfall is solved by enable_input_require_grads (same as finetune/train.py since r5)
            model.enable_input_require_grads()

    opt = torch.optim.AdamW(groups, weight_decay=0.01)
    recipe = MdfRecipe(model, routed, is_routed, n_dense, coll_eval, d,
                       p=p, alpha=alpha,
                       save_cfg=dict(save_dir=save_dir, use_lora=use_lora, lr=lr,
                                     lora_rank=lora_rank, lora_alpha=lora_alpha,
                                     lora_targets=lora_targets))
    run_baseline_training(model, tok, full, eval_texts, coll_fn, device, opt, recipe,
                          batch_size=batch_size, max_steps=max_steps,
                          log_interval=log_interval, save_dir=save_dir,
                          patience=patience)


if __name__ == "__main__":
    app()
