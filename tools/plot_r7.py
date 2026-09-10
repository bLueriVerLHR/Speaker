#!/usr/bin/env python3
"""Consolidate scattered eval_compare outputs into one acc-vs-depth figure.

Reads every ``<prefix>*.json`` produced by ``baselines/eval_compare.py`` (the
numeric stage writes one file per ckpt group), groups them into slices by
``meta.offset`` (falling back to the file timestamp when an old file has no
meta), and renders one panel per slice on a shared style:

- sweep families (MoDification alpha / MoD capacity / RT target) become curves,
- Speaker (ours) is a single red star, dense-ft / raw dense reference points,
- k uses mean_k, falling back to k_est (RT rows carry exec accounting only).

CLI:  python tools/plot_r7.py --log_dir ./.logs --prefix r7p2_num --out main.png
"""
import argparse
import glob
import json
import os
from collections import defaultdict

FAMILIES = [
    ("mdf", "MoDification (alpha sweep)", "o-", "#1f77b4"),
    ("modd", "MoD (capacity sweep)", "^-", "#ff7f0e"),
    ("rt", "Router-Tuning (target sweep)", "D-", "#2ca02c"),
    ("thr", "Speaker (ours)", "*", "red"),
    ("denseft", "dense-ft", "s", "black"),
]


def family(name: str):
    n = name.split(":")[-1]
    for pre, label, m, c in FAMILIES:
        if pre in n:
            return label
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--log_dir", default="./.logs")
    p.add_argument("--prefix", default="r7p2_num")
    p.add_argument("--out", default="")
    p.add_argument("--k_dense", type=int, default=28, help="total layer count (dense k)")
    args = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    slices = defaultdict(dict)  # slice_key -> {rowname -> row}
    for f in sorted(glob.glob(os.path.join(args.log_dir, f"*{args.prefix}*.json"))):
        if os.path.basename(f).startswith("mem_demand"):
            continue
        try:
            d = json.load(open(f))
        except Exception:
            continue
        meta = d.get("meta") or {}
        key = str(meta.get("offset") or "_".join(os.path.basename(f).split("_")[:2]))
        for r in [dict(name="dense", **d["dense"])] + d.get("rows", []):
            if r.get("name") == "dense" and "dense" in slices[key]:
                continue
            r = dict(r)
            r["k"] = r.get("mean_k") or r.get("k_est") or (args.k_dense if "dense" in r["name"] else 0)
            r["tag"] = r["name"].split(":")[-1]
            slices[key][r["name"]] = r

    if not slices:
        raise SystemExit("no eval_compare outputs found")
    keys = sorted(slices, key=lambda k: int(k) if str(k).isdigit() else 0)
    fig, axes = plt.subplots(1, len(keys), figsize=(6.2 * len(keys), 5.6), squeeze=False)
    for ax, key in zip(axes[0], keys):
        rows = slices[key]
        base = rows.get("dense", {}).get("acc")
        if base:
            ax.axhline(base, ls=":", c="gray", lw=1)
            ax.text(0.99, base + 0.004, "raw dense", fontsize=8, c="gray", ha="right",
                    transform=ax.get_yaxis_transform())
        groups = defaultdict(list)
        for r in rows.values():
            g = family(r["name"])
            if g and r["k"]:
                groups[g].append(r)
        for g in [x[1] for x in FAMILIES] + sorted(set(groups) - {x[1] for x in FAMILIES}):
            if g not in groups:
                continue
            rs = sorted(groups[g], key=lambda x: x["k"])
            m, c = next((x[2], x[3]) for x in FAMILIES if x[1] == g) or ("o", "purple")
            big = 18 if g == "Speaker (ours)" else 9
            ax.plot([r["k"] for r in rs], [r["acc"] for r in rs], m, color=c, ms=big,
                    label=g, lw=1.5, mfc=(c if g == "Speaker (ours)" else "none"), mec=c, mew=2)
            for r in rs:
                if len(rs) > 1 or g in ("Speaker (ours)", "dense-ft"):
                    ax.annotate(r["tag"], (r["k"], r["acc"]), textcoords="offset points",
                                xytext=(6, 4), fontsize=8)
        ax.set_xlabel(f"mean active layers k (of {args.k_dense} = dense)")
        ax.set_title(f"offset {key} (n={next(iter(slices[key].values())).get('n', '?')})",
                     fontsize=10)
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel("acc (eval_compare yardstick, see meta)")
    axes[0][-1].legend(loc="upper left", fontsize=9)
    fig.suptitle("accuracy vs depth — all families, one yardstick", y=0.995)
    fig.tight_layout()
    out = args.out or os.path.join(args.log_dir, f"{args.prefix}_main.png")
    fig.savefig(out, dpi=150)
    merged = {k: list(v.values()) for k, v in slices.items()}
    json.dump(merged, open(re.sub(r"\.png$", ".json", out), "w"), indent=1)
    print(f"-> {out} ({len(keys)} slices, {sum(len(v) for v in slices.values())} rows)")


if __name__ == "__main__":
    main()
