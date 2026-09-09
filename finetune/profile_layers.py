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
import argparse
import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from speaker import SpeakerConfig, convert_to_speaker  # noqa: E402
from speaker.checkpoint import load_gate, strip_promoted_gate  # noqa: E402
from speaker.load_profile import (  # noqa: E402
    estimate_layer_bytes,
    plan_placement,
    profile_layer_load,
    render_load_table,
    select_fixed_layers,
)
from speaker.train_common import build_model, build_tok, eval_slice, resolve_device, wrap_lora  # noqa: E402
from data.sft import make_collate  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_id", default="/home/hdd/model/Qwen2.5-7B-Instruct",
                   help="matches the finetune/train.py default (7B)")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data_path", default="./data/sft_t2t_mini.jsonl")
    p.add_argument("--offset", type=int, default=5000, help="start of the profiling slice (avoid segments used by training/held-out)")
    p.add_argument("--n", type=int, default=32, help="number of profiling samples (batch=1, one at a time)")
    p.add_argument("--threshold", type=float, default=0.9, help="gated layers with load >= this value are promoted to fixed layers")
    p.add_argument("--top_k", type=int, default=0,
                   help=">0 ignores threshold and directly promotes the top_k gated layers by load")
    p.add_argument("--keep_gated_min", type=int, default=1)
    p.add_argument("--out", default="", help="directory to write the promoted new ckpt; empty = profile only, nothing written")
    p.add_argument("--gpu_budget_gb", type=float, default=0.0,
                   help=">0 also prints the decode heterogeneous placement plan (GPU budget GB)")
    p.add_argument("--use_lora", default=True, action=argparse.BooleanOptionalAction,
                   help="must be on when the ckpt contains LoRA (same rank/targets as training; default matches train.py)")
    p.add_argument("--lora_rank", type=int, default=8, help="matches the train.py default")
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_targets", default="q_proj,v_proj")
    p.add_argument("--device", default="cuda:0")
    return p.parse_args()


def main():
    args = parse_args()
    device = resolve_device(args.device)
    tok = build_tok(args.model_id)
    texts = eval_slice(args.data_path, args.offset, args.n)
    coll = make_collate(tok, device, 256)

    model = build_model(args.model_id, device)
    if args.use_lora:
        model = wrap_lora(model, args.lora_rank, args.lora_alpha, args.lora_targets)
    cfg = SpeakerConfig.from_json(os.path.join(args.ckpt, "mod_config.json"))
    mod = convert_to_speaker(model, cfg).to(device)
    missing, unexp = load_gate(mod, args.ckpt)
    print(f"gate.pt loaded, missing {len(missing)} unexpected {len(unexp)}", flush=True)

    # profile: soft mode (all layers execute; the hard mask is the deployment load;
    # trajectories do not drift from layer skipping)
    mod.set_skip_mode("soft")
    batches = [coll([t]) for t in texts]
    load = profile_layer_load(mod, batches)
    print(f"\nlayer load profile ({len(texts)} texts, "
          + (f"top_k {args.top_k}" if args.top_k > 0 else f"threshold {args.threshold:.0%}") + "):",
          flush=True)
    print(render_load_table(load, always_on=cfg.always_on_layers), flush=True)

    fixed, promoted = select_fixed_layers(load, threshold=args.threshold,
                                          always_on=cfg.always_on_layers,
                                          keep_gated_min=args.keep_gated_min,
                                          top_k=args.top_k)
    print(f"\nfixed (shared) layers: {fixed}", flush=True)
    print(f"promoted this round: {promoted or '-'} "
          f"(load {['L%d %d%%' % (i, round(load[i] * 100)) for i in promoted]})", flush=True)
    if not promoted:
        print("no layer promoted: threshold too high or gating not yet differentiated; "
              "adjust --threshold/--top_k or keep training", flush=True)

    if args.gpu_budget_gb > 0:
        layer_bytes = estimate_layer_bytes(mod)
        plan = plan_placement(load, layer_bytes, args.gpu_budget_gb * 1e9,
                              always_on=fixed)
        print(f"\nplacement plan @ {args.gpu_budget_gb}GB GPU budget (decode phase):", flush=True)
        print(f"  gpu {plan['gpu_layers']} ({plan['gpu_gb']:.2f}GB / total {plan['total_gb']:.2f}GB)",
              flush=True)
        print(f"  cpu {plan['cpu_layers']}", flush=True)

    if args.out and promoted:
        os.makedirs(args.out, exist_ok=True)
        cfg.always_on_layers = fixed
        cfg.to_json(os.path.join(args.out, "mod_config.json"))
        sd = torch.load(os.path.join(args.ckpt, "gate.pt"), map_location="cpu")
        sd2 = strip_promoted_gate(sd, promoted)
        torch.save(sd2, os.path.join(args.out, "gate.pt"))
        prof = {"source_ckpt": os.path.abspath(args.ckpt),
                "n_texts": len(texts), "threshold": args.threshold, "top_k": args.top_k,
                "load": {str(k): v for k, v in load.items()},
                "fixed_layers": fixed, "promoted": promoted}
        with open(os.path.join(args.out, "profile.json"), "w", encoding="utf-8") as f:
            json.dump(prof, f, indent=1, ensure_ascii=False)
        print(f"\npromoted ckpt -> {args.out} "
              f"(gate keys {len(sd)} -> {len(sd2)}, joint_router stripped, reinitialized on resume)",
              flush=True)
        print("next step, continue training:", flush=True)
        print(f"  python3 finetune/train.py --resume_dir {args.out} "
              f"--model_id {args.model_id} --data_path {args.data_path} ...", flush=True)


if __name__ == "__main__":
    main()
