"""k-vs-difficulty probe: does per-token routing track per-token difficulty?
difficulty axis = dense per-token NLL on the SAME text (no corpus/word heuristics).
For each ours ckpt: Pearson r(k_total, dense_nll) + mean k by dense-NLL quartile +
low/high-hump mass fractions. A difficulty-blind router gives slope ~ 0 no matter
how wide the k histogram is (shape without function).

LoRA note (0911 fix): ours-family ckpts ship gate/LoRA keys in gate.pt; assemble only
wraps peft when CLI args say so. The first version of this probe (in /tmp/opencode)
called assemble() without args -> 224 LoRA keys silently dropped, measuring trained
gates on the raw base. Default here is --use_lora to match tools/probe_kdist.py.

Usage: kdiff_probe.py --ckpt CKPT [--ckpt ...] --out JSON (rest: model/data slice)."""
import argparse
import gc
import json
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from baselines.assemble import assemble
from data.sft import make_collate
from speaker.metrics import per_token_nll
from speaker.train_common import (
    build_model,
    build_tok,
    eval_slice,
    resolve_device,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_id", default="/home/hdd/model/Qwen2.5-7B-Instruct")
    p.add_argument("--data_path", default="./data/sft_t2t_mini.jsonl")
    p.add_argument("--ckpt", action="append", default=[])
    p.add_argument("--offset", type=int, default=5000)
    p.add_argument("--n", type=int, default=60)
    p.add_argument("--max_len", type=int, default=256)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out", default="/tmp/kdiff.json")
    p.add_argument("--use_lora", default=True, action=argparse.BooleanOptionalAction,
                   help="LoRA wrapping for ours ckpts (must match training)")
    p.add_argument("--lora_rank", type=int, default=8, help="same as the train.py default")
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_targets", default="q_proj,v_proj")
    p.add_argument("--save_nll", default="", help="phase 1 only: dump dense-NLL cache (pt file)")
    p.add_argument("--load_nll", default="", help="phase 2 only: read cache, skip dense load")
    return p.parse_args()


def main():
    args = parse_args()
    device = resolve_device(args.device)
    tok = build_tok(args.model_id)
    texts = eval_slice(args.data_path, args.offset, args.n)
    coll = make_collate(tok, device, args.max_len)

    if args.load_nll:
        cache = torch.load(args.load_nll, map_location="cpu", weights_only=True)
        saved = [(cache["ids"][i], cache["nll"][i]) for i in range(len(cache["ids"]))]
        dense = None
    else:
        # pass 1: dense per-token NLL (kept small: valid positions only)
        dense = build_model(args.model_id, device)
        dense.eval()
        saved = []
        with torch.no_grad():
            for t in texts:
                b = coll([t])
                nll = per_token_nll(dense(**b).logits.float(), b["labels"])[0]
                valid = (b["labels"] != -100)[0]
                saved.append((b["input_ids"][0][valid].cpu(),
                              nll[valid].cpu()))
        if args.save_nll:
            torch.save({"ids": [s[0] for s in saved],
                        "nll": [s[1] for s in saved]}, args.save_nll)
            print(f"saved NLL cache {args.save_nll} ({len(saved)} samples)", flush=True)
    if dense is not None:
        del dense
    # drop every reference buyers can't see (closures/lists pin models: AGENTS probe lesson)
    del texts, coll
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    lora_tag = f"lora r{args.lora_rank}/{args.lora_targets}" if args.use_lora else "NO-LORA"
    print(f"probe mode: {lora_tag}", flush=True)
    out = {"n": args.n, "offset": args.offset, "lora": lora_tag, "ckpts": {}}
    for ck in args.ckpt:
        asm = assemble(ck, args.model_id, args, device=device, skip_mode="hard")
        m = asm.model
        n_fixed = len((asm.cfg or {}).get("always_on_layers", []))
        m.eval()
        ks, ns = [], []
        with torch.no_grad():
            for ids, nll in saved:
                b = {"input_ids": ids.unsqueeze(0).to(device),
                     "attention_mask": torch.ones_like(ids).unsqueeze(0).to(device),
                     "labels": ids.unsqueeze(0).to(device)}
                mo = m(**b)
                getk = getattr(m, "get_active_counts", None)
                k = getk() if callable(getk) else None
                if k is None:  # non-wrapper families: constant fallback (not expected here)
                    continue
                kk = (k[0].float() + n_fixed).cpu()
                ks.append(kk)
                ns.append(nll)
        k_all = torch.cat(ks)
        n_all = torch.cat(ns)
        # Pearson r
        kc = k_all - k_all.mean()
        nc = n_all - n_all.mean()
        r = float((kc * nc).sum() / (kc.pow(2).sum() * nc.pow(2).sum()).sqrt().clamp_min(1e-12))
        # mean k by dense-NLL quartile
        qs = n_all.quantile(torch.tensor([0.25, 0.5, 0.75]))
        qk = []
        edges = [float("-inf")] + [float(q) for q in qs] + [float("inf")]
        for i in range(4):
            sel = (n_all > edges[i]) & (n_all <= edges[i + 1])
            qk.append(round(float(k_all[sel].mean()), 3) if sel.any() else None)
        out["ckpts"][ck] = {
            "r_k_nll": round(r, 4),
            "k_by_nll_quartile": qk,
            "k_mean": round(float(k_all.mean()), 3),
            "k_std": round(float(k_all.std()), 3),
            "low_hump_le6": round(float((k_all <= 6).float().mean()), 3),
            "high_hump_ge14": round(float((k_all >= 14).float().mean()), 3),
            "ntok": int(k_all.numel()),
        }
        print(f"[{ck}] r={r:.3f} qk={qk} k={k_all.mean():.1f}±{k_all.std():.1f} "
              f"low={out['ckpts'][ck]['low_hump_le6']:.2f} high={out['ckpts'][ck]['high_hump_ge14']:.2f}",
              flush=True)
        # bound methods / outputs pin the peft+speaker model graph (OOM between ckpts)
        getk = mo = k = kk = None
        del asm, m
        gc.collect()
        torch.cuda.empty_cache()
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
