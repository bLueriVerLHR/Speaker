"""Final ckpt evaluation (can be rerun standalone, no dependency on the training tail):
dense baseline vs Speaker holdout comparison.
loss/acc/delta + mean_k±std + k quartiles, same protocol as the finetune/train.py
final eval.
Usage: python3 finetune/eval_ckpt.py --ckpt /tmp/mod_ckpt_mix1 --data_path ... --offset 3000 --n 100
"""
import argparse
import gc
import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from speaker import SpeakerConfig, convert_to_speaker
from speaker.checkpoint import load_gate
from speaker.evaluate import eval_heldout
from speaker.train_common import build_model, build_tok, eval_slice, resolve_device, wrap_lora
from data.sft import make_collate


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_id", default="/home/hdd/model/Qwen2.5-7B-Instruct")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data_path", default="./data/sft_t2t_mini.jsonl")
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--max_len", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--use_chat", default=True, action=argparse.BooleanOptionalAction)
    p.add_argument("--mask_user", default=True, action=argparse.BooleanOptionalAction)
    p.add_argument("--use_lora", default=True, action=argparse.BooleanOptionalAction,
                   help="must be on when the ckpt contains LoRA (same rank/targets as training), otherwise the lora_ weights are silently dropped")
    p.add_argument("--lora_rank", type=int, default=8, help="matches the train.py default")
    p.add_argument("--lora_alpha", type=int, default=16, help="matches training (train.py default 16)")
    p.add_argument("--lora_targets", default="q_proj,v_proj")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out", default="")
    return p.parse_args()


def main():
    args = parse_args()
    device = resolve_device(args.device)
    tok = build_tok(args.model_id)
    texts = eval_slice(args.data_path, args.offset, args.n,
                       tok=tok if args.use_chat else None, use_chat=args.use_chat)
    print(f"eval {len(texts)} texts from {args.data_path}@{args.offset}", flush=True)

    dense = build_model(args.model_id, device)
    coll = make_collate(tok, device, args.max_len, args.use_chat, args.mask_user)
    d = eval_heldout(dense, texts, coll, args.batch_size)
    print(f"dense loss {d['loss']:.3f} acc {d['acc']:.3f}", flush=True)
    del dense
    if device.type == "cuda":
        gc.collect()  # 7B: gc to break reference cycles before empty_cache, otherwise loading the second model OOMs
        torch.cuda.empty_cache()

    m2 = build_model(args.model_id, device, dtype=torch.bfloat16)
    if args.use_lora:
        m2 = wrap_lora(m2, args.lora_rank, args.lora_alpha, args.lora_targets)
    cfg = SpeakerConfig.from_json(os.path.join(args.ckpt, "mod_config.json"))
    mod = convert_to_speaker(m2, cfg).to(device)
    missing, unexp = load_gate(mod, args.ckpt)
    print(f"gate.pt loaded, missing {len(missing)} unexpected {len(unexp)}", flush=True)
    mod.set_skip_mode("hard")
    m = eval_heldout(mod, texts, coll, args.batch_size)
    print(f"mod   loss {m['loss']:.3f} acc {m['acc']:.3f} "
          f"(Δloss {m['loss'] - d['loss']:+.3f} Δacc {m['acc'] - d['acc']:+.3f}) "
          f"| k {m['mean_k']:.1f}±{m['std_k']:.1f} "
          f"quartile {[round(v, 1) for v in m['quartile_k']]}", flush=True)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"dense": d, "mod": m,
                       "dloss": m["loss"] - d["loss"], "dacc": m["acc"] - d["acc"]},
                      f, indent=1, ensure_ascii=False)
        print(f"saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
