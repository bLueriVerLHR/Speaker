"""Per-layer importance probe: bypass one layer at a time in the dense model and see how much
held-out loss/acc drops.
Used to validate the "syntax at the ends, logic in the middle" hypothesis and to guide
shared (fixed) layer selection."""
import argparse, json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os, sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
from data.sft import SFTDataset, make_collate
from speaker.evaluate import eval_heldout

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_id", default="/home/hdd/model/Qwen2.5-7B-Instruct",
                   help="same as the finetune-track default (7B)")
    p.add_argument("--data_path", default="./data/sft_t2t_mini.jsonl")
    p.add_argument("--offset", type=int, default=1000, help="held-out start (offset from the training set)")
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out", default="/tmp/layer_probe.json")
    return p.parse_args()

def find_layers(model):
    for path in (["model", "layers"], ["transformer", "h"], ["layers"]):
        cur = model
        try:
            for a in path: cur = getattr(cur, a)
            return cur
        except AttributeError: pass
    raise ValueError("layers not found")

def main():
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda": torch.cuda.set_device(device)
    tok = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    full = SFTDataset(args.data_path, args.offset + args.n)
    texts = full.samples[args.offset:args.offset + args.n]
    print(f"heldout {len(texts)}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_id, dtype=torch.bfloat16,
        device_map=None, trust_remote_code=True, low_cpu_mem_usage=True).to(device)
    layers = find_layers(model)
    N = len(layers)
    coll = make_collate(tok, device, 256)
    base = eval_heldout(model, texts, coll)
    print(f"baseline loss {base['loss']:.3f} acc {base['acc']:.3f}", flush=True)
    rows = []
    for i in range(N):
        def _skip(module, margs, output, _i=i):
            hs = margs[0] if margs else output[0]
            return (hs,) + tuple(output[1:]) if isinstance(output, tuple) else hs
        h = layers[i].register_forward_hook(_skip)
        r = eval_heldout(model, texts, coll)
        h.remove()
        rows.append({"layer": i, "loss": r["loss"], "acc": r["acc"],
                     "dloss": r["loss"]-base["loss"], "dacc": r["acc"]-base["acc"]})
        print(f"skip L{i:2d}: loss {r['loss']:.3f} (Δ{r['loss']-base['loss']:+.3f}) "
              f"acc {r['acc']:.3f} (Δ{r['acc']-base['acc']:+.3f})", flush=True)
    rows.sort(key=lambda r: -r["dloss"])
    print("rank by importance: " + " ".join(f"L{r['layer']}({r['dloss']:+.2f})" for r in rows[:8]), flush=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"baseline": base, "rows": rows}, f, indent=1)
    print(f"saved {args.out}", flush=True)

if __name__ == "__main__": main()
