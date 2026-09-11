"""Premise test for difficulty-conditional budgets (ed8 follow-up, 0911).

Question: does DEPTH (the layers our router may skip, i.e. the gated span 2..25)
differentially help HARD tokens? If not, per-token lambda shaping by difficulty
has no headroom regardless of lambda amplitude.

Method: dense raw model, bypass layers via forward hooks (probe_layers.py trick).
Difficulty axis = per-token NLL of the INTACT dense model on the same positions
(same 60-sample slice / valid-mask pipeline as the kdiff NLL cache).
  - single-layer sweep over the gated span: per-layer dNLL by difficulty quartile
  - depth dosage: random m in {4,8,12} mid layers, 2 seeds each
Premise holds iff dNLL concentrates in the top (hard) quartile, growing with dose.

Output JSON: per-config mean dNLL overall + per quartile + Q4/Q1 ratio + r(dnll, base_nll).
"""
import argparse
import gc
import json
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from data.sft import make_collate
from speaker.metrics import per_token_nll
from speaker.train_common import build_model, build_tok, eval_slice, resolve_device


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_id", default="/home/hdd/model/Qwen2.5-7B-Instruct")
    p.add_argument("--data_path", default="./data/sft_t2t_mini.jsonl")
    p.add_argument("--offset", type=int, default=5000)
    p.add_argument("--n", type=int, default=60)
    p.add_argument("--max_len", type=int, default=256)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--gated_span", default="2,25", help="inclusive layer range probed")
    p.add_argument("--doses", default="4,8,12", help="random-set sizes")
    p.add_argument("--seeds", type=int, default=2)
    p.add_argument("--nll_cache", default="/tmp/opencode/nll_cache.pt",
                   help="optional: sanity-check our baseline against this cache")
    p.add_argument("--out", default="/tmp/diff_ablation.json")
    return p.parse_args()


def find_layers(model):
    for path in (["model", "layers"], ["transformer", "h"], ["layers"]):
        cur = model
        try:
            for a in path:
                cur = getattr(cur, a)
            return cur
        except AttributeError:
            pass
    raise ValueError("layers not found")


def run_nll(model, batches):
    """Per-sample per-token NLL lists on valid positions (aligned with batch ids)."""
    outs = []
    with torch.no_grad():
        for b in batches:
            nll = per_token_nll(model(**b).logits.float(), b["labels"])[0]
            valid = (b["labels"] != -100)[0]
            outs.append((b["input_ids"][0][valid].cpu(), nll[valid].cpu()))
    return outs


def summarize(dnll, base, name):
    finite = torch.isfinite(dnll)
    frac_bad = float((~finite).float().mean())
    d, b = dnll[finite], base[finite]
    qs = base.quantile(torch.tensor([0.25, 0.5, 0.75]))
    edges = [float("-inf")] + [float(q) for q in qs] + [float("inf")]
    by_q = []
    for i in range(4):
        sel = (b > edges[i]) & (b <= edges[i + 1])
        by_q.append(round(float(d[sel].mean()), 4) if sel.any() else None)
    dc = d - d.mean()
    bc = b - b.mean()
    r = float((dc * bc).sum() / (dc.pow(2).sum() * bc.pow(2).sum()).sqrt().clamp_min(1e-12))
    ratio = (by_q[3] / by_q[0]) if by_q[0] and abs(by_q[0]) > 1e-6 else None
    row = {"config": name, "dnll_mean": round(float(d.mean()), 4),
           "dnll_by_nll_quartile_q1e_q4h": by_q,
           "q4_q1_ratio": round(ratio, 3) if ratio is not None else None,
           "r_dnll_basenll": round(r, 4), "frac_nonfinite": round(frac_bad, 5)}
    print(f"[{name}] dnll {row['dnll_mean']:+.4f} q={[f'{v:+.3f}' if v is not None else '-' for v in by_q]} "
          f"Q4/Q1 {row['q4_q1_ratio']} r {row['r_dnll_basenll']:+.3f}", flush=True)
    return row


def main():
    args = parse_args()
    device = resolve_device(args.device)
    tok = build_tok(args.model_id)
    texts = eval_slice(args.data_path, args.offset, args.n)
    coll = make_collate(tok, device, args.max_len)
    batches = [coll([t]) for t in texts]

    model = build_model(args.model_id, device)
    model.eval()
    layers = find_layers(model)
    N = len(layers)
    lo, hi = (int(x) for x in args.gated_span.split(","))
    span = list(range(lo, hi + 1))
    doses = [int(x) for x in args.doses.split(",")]

    base = run_nll(model, batches)
    ids = [s[0] for s in base]
    base_all = torch.cat([s[1] for s in base])
    print(f"baseline pooled nll mean {float(base_all.mean()):.4f}  ntok {base_all.numel()}  N={N}",
          flush=True)
    if args.nll_cache:
        cache = torch.load(args.nll_cache, map_location="cpu", weights_only=True)
        same = all(torch.equal(cache["ids"][i], ids[i]) for i in range(len(ids)))
        cb = torch.cat([c for c in cache["nll"]])
        print(f"cache ids aligned: {same}; cached nll mean {float(cb.mean()):.4f}", flush=True)

    rows = []

    def with_skip(skips, name):
        hooks = []
        for i in skips:
            def _skip(module, margs, output, _i=i):
                hs = margs[0] if margs else output[0]
                return (hs,) + tuple(output[1:]) if isinstance(output, tuple) else hs
            hooks.append(layers[i].register_forward_hook(_skip))
        abl = run_nll(model, batches)
        for h in hooks:
            h.remove()
        dnll = torch.cat([(a[1] - base[i][1]) for i, a in enumerate(abl)])
        rows.append(summarize(dnll, base_all, name))

    for i in span:  # single-layer sweep over the gated span
        with_skip([i], f"drop_L{i}")

    for m in doses:  # depth dosage: random mid-layer sets
        for seed in range(args.seeds):
            g = torch.Generator().manual_seed(1234 + m * 100 + seed)
            picks = sorted(torch.randperm(len(span), generator=g)[:m].tolist())
            picks = [span[j] for j in picks]
            with_skip(picks, f"dose_m{m}_s{seed}")

    with open(args.out, "w") as f:
        json.dump({"args": vars(args), "baseline_nll_mean": float(base_all.mean()),
                   "rows": rows}, f, indent=1)
    print(f"saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
