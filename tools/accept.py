"""Acceptance gate for one-pass finetuning (ed9/D): PASS/FAIL a ckpt against the
deployment contract and write manifest.json into the ckpt dir (self-describing artifact).

Tier-1 (always): hard-mode held-out Δacc vs raw dense + total-k budget.
Tier-2 (opt-in --gen_n>0): repetition at the ckpt's own decode recipe (mod_config.json
decode block, ed9/C) vs dense under the SAME lever.
Exit code 0 = PASS, 1 = FAIL (pipeline-friendly).
Usage: accept.py --ckpt CKPT [--gen_n 10] (rest: model/data slice + thresholds)."""
import argparse
import gc
import hashlib
import json
import os
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from baselines.assemble import assemble
from data.sft import make_collate
from speaker.evaluate import eval_heldout
from speaker.train_common import (
    build_model,
    build_tok,
    eval_slice,
    resolve_device,
)


def md5_file(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_id", default="/home/hdd/model/Qwen2.5-7B-Instruct")
    p.add_argument("--data_path", default="./data/sft_t2t_mini.jsonl")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--offset", type=int, default=904000, help="acceptance slice (default: training heldout)")
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--max_len", type=int, default=1024)
    p.add_argument("--batch_size", type=int, default=1, help="bs1 = honest accounting (modd lesson)")
    p.add_argument("--acc_margin", type=float, default=0.0, help="PASS iff mod_acc >= dense_acc + margin")
    p.add_argument("--k_max", type=float, default=None, help="PASS iff total-k mean <= this (None = skip)")
    p.add_argument("--gen_n", type=int, default=0, help="tier-2 rep probe prompts (0 = skip)")
    p.add_argument("--gen_offset", type=int, default=5000, help="gen protocol slice (established)")
    p.add_argument("--gen_new", type=int, default=64)
    p.add_argument("--gen_temp", type=float, default=0.7)
    p.add_argument("--rep_max_mult", type=float, default=1.5, help="PASS iff ours_rep <= mult * dense_rep (same lever)")
    p.add_argument("--use_lora", default=True, action=argparse.BooleanOptionalAction)
    p.add_argument("--lora_rank", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_targets", default="q_proj,v_proj")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out", default="", help="manifest path (default: <ckpt>/manifest.json)")
    return p.parse_args()


def main():
    args = parse_args()
    device = resolve_device(args.device)
    tok = build_tok(args.model_id)
    texts = eval_slice(args.data_path, args.offset, args.n)
    coll = make_collate(tok, device, args.max_len)
    checks = []

    dense = build_model(args.model_id, device)
    dense.eval()
    d = eval_heldout(dense, texts, coll, args.batch_size)
    print(f"dense loss {d['loss']:.3f} acc {d['acc']:.3f}", flush=True)
    # tier-2 lever = the ckpt's own recipe (read from disk first, no model needed);
    # dense is generated under the SAME lever (fair comparison, ed9/D contract)
    dec = {}
    if args.gen_n > 0:
        try:
            with open(os.path.join(args.ckpt, "mod_config.json"), encoding="utf-8") as f:
                dec = json.load(f).get("decode") or {}
        except OSError:
            dec = {}
    rp = dec.get("repetition_penalty", 1.0)
    ng = dec.get("no_repeat_ngram_size", 0)
    reps, prompts = {}, None
    if args.gen_n > 0:
        import eval_gen as eg
        print(f"tier-2 rep probe under ckpt lever (rep_penalty={rp} no_repeat_ngram={ng}): "
              f"{'ckpt-shipped' if dec else 'neutral fallback'}", flush=True)
        gtexts = eval_slice(args.data_path, args.gen_offset, args.gen_n)
        enc = tok(gtexts, truncation=True, max_length=64 + 64,
                  padding=True, return_tensors="pt")
        L = enc["attention_mask"].sum(1)
        prompts = [tok.decode(enc["input_ids"][bi, :L[bi]][:64].tolist(),
                              skip_special_tokens=True)
                   for bi in range(len(gtexts))]
        outs, _, _, _, _, _, _ = eg.run_gen(
            dense, tok, prompts, args.gen_new, device, "accept-dense",
            args.gen_temp, rep_penalty=rp, no_repeat_ngram=ng)
        reps["dense"] = sum(eg.rep3_rate(o) for o in outs) / max(len(outs), 1)
        print(f"[accept-dense] rep3 {reps['dense']:.3f}", flush=True)
    del dense
    gc.collect()
    torch.cuda.empty_cache()

    asm = assemble(args.ckpt, args.model_id, args, device=device, skip_mode="hard")
    mod = asm.model
    mod.eval()
    m = eval_heldout(mod, texts, coll, args.batch_size)
    n_fixed = len((asm.cfg or {}).get("always_on_layers", []))
    k_total = (m["mean_k"] or 0.0) + n_fixed
    print(f"mod   loss {m['loss']:.3f} acc {m['acc']:.3f} "
          f"(Δloss {m['loss'] - d['loss']:+.3f} Δacc {m['acc'] - d['acc']:+.3f}) "
          f"| k_total {k_total:.1f} (gated {m['mean_k']:.1f}+fixed {n_fixed})", flush=True)
    checks.append({"name": "acc", "value": round(m["acc"] - d["acc"], 4),
                   "threshold": f">= dense+{args.acc_margin}",
                   "pass": bool(m["acc"] >= d["acc"] + args.acc_margin)})
    if args.k_max is not None:
        checks.append({"name": "k_total", "value": round(k_total, 3),
                       "threshold": f"<= {args.k_max}", "pass": bool(k_total <= args.k_max)})

    if args.gen_n > 0:
        outs, _, _, _, _, _, _ = eg.run_gen(
            mod, tok, prompts, args.gen_new, device, "accept-ours",
            args.gen_temp, rep_penalty=rp, no_repeat_ngram=ng)
        reps["ours"] = sum(eg.rep3_rate(o) for o in outs) / max(len(outs), 1)
        print(f"[accept-ours] rep3 {reps['ours']:.3f}", flush=True)
        checks.append({"name": "rep", "value": round(reps["ours"], 4),
                       "threshold": f"<= {args.rep_max_mult}x dense({reps['dense']:.3f})@{dec or 'neutral'}",
                       "pass": bool(reps["ours"] <= args.rep_max_mult * reps["dense"])})
    verdict = "PASS" if all(c["pass"] for c in checks) else "FAIL"
    manifest = {"ckpt": args.ckpt, "verdict": verdict, "checks": checks,
                "dense": {"loss": d["loss"], "acc": d["acc"]},
                "mod": {"loss": m["loss"], "acc": m["acc"],
                        "k_gated": m["mean_k"], "k_total": k_total},
                "provenance": {"slice": [args.offset, args.n], "max_len": args.max_len,
                               "gate_md5": md5_file(os.path.join(args.ckpt, "gate.pt")),
                               "has_decode": bool((asm.cfg or {}).get("decode"))}}
    out = args.out or os.path.join(args.ckpt, "manifest.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)
    print(f"[{verdict}] {args.ckpt} -> {out}", flush=True)
    for c in checks:
        print(f"  {c['name']}: {c['value']} ({c['threshold']}) "
              f"{'ok' if c['pass'] else 'VIOLATED'}", flush=True)
    sys.exit(0 if verdict == "PASS" else 1)


if __name__ == "__main__":
    main()
