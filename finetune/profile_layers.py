"""Finetune profiling: after the gating is trained, decide the shared (fixed) layers
from the measured per-layer load; the remaining layers stay gated.

Pipeline position (the finetune distribution != the training distribution, so the fixed
layers must not be guessed):
  1. finetune/train.py            first train the joint routing (at this stage the
                                  router learns each layer's usage freely)
  2. finetune/profile_layers.py   this script: run a held-out slice to profile the
                                  per-layer load; layers with load >= threshold
                                  (default 0.9, the theoretical shared-layer point)
                                  are promoted to fixed layers;
                                  with --out, writes a new ckpt after promotion
                                  (mod_config.json updates always_on_layers;
                                  gate.pt strips the promotion-related gating keys:
                                  threshold = the promoted layers' router/tau/comp;
                                  moe = promotion changes the gated layer count G so
                                  the joint_router dims mismatch and it is stripped
                                  entirely, the router is reinitialized on resume);
  3. finetune/train.py --resume_dir <new ckpt>   continue training (structure =
                                  fixed layers + remaining gated layers).

Optionally, --gpu_budget_gb also prints a decode-time heterogeneous placement plan
(fixed + high-load layers on GPU, low-load layers on CPU); prefill degrades to dense
just like MoE, so placement does not affect correctness.

Usage:
  python3 finetune/profile_layers.py --ckpt /tmp/mod_ckpt_bal --n 32 --threshold 0.9 \
      --out /tmp/mod_ckpt_bal_fixed [--gpu_budget_gb 4]
"""
import json
import os
import sys
from pathlib import Path
from typing import Annotated

import torch
import typer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from baselines.assemble import ModelBuilder  # noqa: E402
from speaker.checkpoint import strip_promoted_gate  # noqa: E402
from speaker.load_profile import (  # noqa: E402
    estimate_layer_bytes,
    plan_placement,
    profile_layer_load,
    render_load_table,
    select_fixed_layers,
)
from speaker.log import logger  # noqa: E402
from speaker.train_common import build_tok, eval_slice, resolve_device  # noqa: E402
from data.sft import make_collate  # noqa: E402


app = typer.Typer(add_completion=False)


@app.command()
def main(
    model_id: Annotated[str, typer.Option("--model_id", help="matches the finetune/train.py default (7B)")] = "/home/hdd/model/Qwen2.5-7B-Instruct",
    ckpt: Annotated[str, typer.Option("--ckpt")] = ...,
    data_path: Annotated[str, typer.Option("--data_path")] = "./data/sft_t2t_mini.jsonl",
    offset: Annotated[int, typer.Option("--offset", help="start of the profiling slice (avoid segments used by training/held-out)")] = 5000,
    n: Annotated[int, typer.Option("--n", help="number of profiling samples (batch=1, one at a time)")] = 32,
    threshold: Annotated[float, typer.Option("--threshold", help="gated layers with load >= this value are promoted to fixed layers")] = 0.9,
    top_k: Annotated[int, typer.Option("--top_k", help=">0 ignores threshold and directly promotes the top_k gated layers by load")] = 0,
    keep_gated_min: Annotated[int, typer.Option("--keep_gated_min")] = 1,
    out: Annotated[str, typer.Option("--out", help="directory to write the promoted new ckpt; empty = profile only, nothing written")] = "",
    gpu_budget_gb: Annotated[float, typer.Option("--gpu_budget_gb", help=">0 also prints the decode heterogeneous placement plan (GPU budget GB)")] = 0.0,
    use_lora: Annotated[bool, typer.Option("--use_lora/--no-use_lora", help="must be on when the ckpt contains LoRA (same rank/targets as training; default matches train.py)")] = True,
    lora_rank: Annotated[int, typer.Option("--lora_rank", help="matches the train.py default")] = 8,
    lora_alpha: Annotated[int, typer.Option("--lora_alpha")] = 16,
    lora_targets: Annotated[str, typer.Option("--lora_targets")] = "q_proj,v_proj",
    device: Annotated[str, typer.Option("--device")] = "cuda:0",
) -> None:
    """Profile layer loads and optionally promote hot gated layers to fixed."""
    from speaker.terminal import setup_terminal
    setup_terminal()
    device = resolve_device(device)
    tok = build_tok(model_id)
    texts = eval_slice(data_path, offset, n)
    coll = make_collate(tok, device, 256)

    lora = (dict(rank=lora_rank, alpha=lora_alpha, targets=lora_targets)
            if use_lora else None)
    asm = (ModelBuilder(model_id, lora=lora, dtype=torch.bfloat16)
           .from_ckpt(ckpt).skip_mode("soft").build(device))
    mod = asm.model
    cfg = mod.mod_config

    # profile: soft mode (all layers execute; the hard mask is the deployment load;
    # trajectories do not drift from layer skipping)
    mod.set_skip_mode("soft")
    batches = [coll([t]) for t in texts]
    load = profile_layer_load(mod, batches)
    logger.info(f"layer load profile ({len(texts)} texts, "
                + (f"top_k {top_k}" if top_k > 0 else f"threshold {threshold:.0%}") + "):\n"
                + render_load_table(load, always_on=cfg.always_on_layers))

    fixed, promoted = select_fixed_layers(load, threshold=threshold,
                                          always_on=cfg.always_on_layers,
                                          keep_gated_min=keep_gated_min,
                                          top_k=top_k)
    logger.info(f"fixed (shared) layers: {fixed}")
    logger.info(f"promoted this round: {promoted or '-'} "
                f"(load {['L%d %d%%' % (i, round(load[i] * 100)) for i in promoted]})")
    if not promoted:
        logger.warning("no layer promoted: threshold too high or gating not yet differentiated; "
                       "adjust --threshold/--top_k or keep training")

    if gpu_budget_gb > 0:
        layer_bytes = estimate_layer_bytes(mod)
        plan = plan_placement(load, layer_bytes, gpu_budget_gb * 1e9,
                              always_on=fixed)
        logger.info(f"placement plan @ {gpu_budget_gb}GB GPU budget (decode phase):\n"
                    f"  gpu {plan['gpu_layers']} ({plan['gpu_gb']:.2f}GB / total {plan['total_gb']:.2f}GB)\n"
                    f"  cpu {plan['cpu_layers']}")

    if out and promoted:
        os.makedirs(out, exist_ok=True)
        cfg.always_on_layers = fixed
        cfg.to_json(os.path.join(out, "mod_config.json"))
        sd = torch.load(os.path.join(ckpt, "gate.pt"), map_location="cpu")
        sd2 = strip_promoted_gate(sd, promoted)
        torch.save(sd2, os.path.join(out, "gate.pt"))
        prof = {"source_ckpt": os.path.abspath(ckpt),
                "n_texts": len(texts), "threshold": threshold, "top_k": top_k,
                "load": {str(k): v for k, v in load.items()},
                "fixed_layers": fixed, "promoted": promoted}
        with open(os.path.join(out, "profile.json"), "w", encoding="utf-8") as f:
            json.dump(prof, f, indent=1, ensure_ascii=False)
        logger.info(f"promoted ckpt -> {out} "
                    f"(gate keys {len(sd)} -> {len(sd2)}, joint_router stripped, reinitialized on resume)")
        logger.info("next step, continue training:\n"
                    f"  python3 finetune/train.py --resume_dir {out} "
                    f"--model_id {model_id} --data_path {data_path} ...")


if __name__ == "__main__":
    app()
