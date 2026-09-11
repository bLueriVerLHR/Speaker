#!/usr/bin/env python3
"""Consolidate scattered P2 outputs into ONE multi-metric figure.

Top row    — numeric metrics vs depth, one panel per (slice, metric): acc, loss.
Bottom row — generation metrics vs depth (decode-time k from probe_kdist):
             ROUGE-L, rep3 (repetition), stability, ms/token.

Inputs (all produced by the P2 driver; JSONs only, per-invocation PNGs are
suppressed with --no_png and consolidated here instead):

* ``*<num_prefix>*.json``   — eval_compare outputs (meta.offset keys the slice)
* ``*<gen_prefix>*.json``   — eval_gen outputs (per-ckpt, one method each)
* ``*<kdist_prefix>*.json`` — probe_kdist outputs (decode-time k per ckpt)

CLI:  python tools/plot_r7.py --log_dir ./.logs --gen_dir ... --kdist_dir ...
"""
import argparse
import glob
import json
import os
import re
from collections import defaultdict

FAMILIES = [
    ("mdf", "MoDification (alpha sweep)", "o-", "#1f77b4"),
    ("modd", "MoD (capacity sweep)", "^-", "#ff7f0e"),
    ("rt", "Router-Tuning (target sweep)", "D-", "#2ca02c"),
    ("thr", "Speaker (ours)", "*", "red"),
    ("denseft", "dense-ft", "s", "black"),
]


def fam_of(name: str):
    n = name.lower()  # match on the full name: the ckpt tag after ':' may not carry the family
    for pre, label, m, c in FAMILIES:
        if pre in n:
            return label
    return None


def style_of(label: str):
    for pre, lab, m, c in FAMILIES:
        if lab == label:
            return m, c
    return "o", "purple"


def load_numeric(d, prefix):
    slices = defaultdict(dict)
    for f in sorted(glob.glob(os.path.join(d, f"*{prefix}*.json"))):
        if os.path.basename(f).startswith("mem_demand"):
            continue
        try:
            data = json.load(open(f))
        except Exception:
            continue
        meta = data.get("meta") or {}
        key = str(meta.get("offset") or "_".join(os.path.basename(f).split("_")[:2]))
        for r in [dict(name="dense", **data["dense"])] + data.get("rows", []):
            if r["name"] == "dense" and "dense" in slices[key]:
                continue
            r = dict(r)
            r["k"] = r.get("mean_k") or r.get("k_est") or 0
            slices[key][r["name"]] = r
    return slices


def load_gen(d, prefix):
    out = {}
    for f in sorted(glob.glob(os.path.join(d, f"*{prefix}*.json"))):
        try:
            data = json.load(open(f))
        except Exception:
            continue
        tag = os.path.basename(f).split(f"{prefix}_")[-1].split(".json")[0]
        for name, m in (data.get("methods") or {}).items():
            if name == "dense":
                continue
            out[tag] = {"name": name, **m}
        if tag == "base":
            out.setdefault("rawdense", {"name": "dense", **data["methods"]["dense"]})
    return out


def load_kdist(d, prefix, k_dense):
    out = {"base": float(k_dense), "dense": float(k_dense), "rawdense": float(k_dense)}
    for f in sorted(glob.glob(os.path.join(d, f"*{prefix}*.json"))):
        try:
            data = json.load(open(f))
        except Exception:
            continue
        tag = os.path.basename(f).split(f"{prefix}_")[-1].split(".json")[0]
        if tag in ("base", "dense"):
            continue
        fam = [k for k in data if k not in ("meta", "dense")]
        if fam:
            kt = data[fam[0]].get("k_total") or {}
            if kt.get("mean"):
                out[tag] = kt["mean"]
    return out


def draw_series(ax, pts, label):
    m, c = style_of(label)
    big = 18 if label == "Speaker (ours)" else 9
    ax.plot([p[0] for p in pts], [p[1] for p in pts], m, color=c, ms=big, label=label,
            lw=1.5, mfc=(c if label == "Speaker (ours)" else "none"), mec=c, mew=2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--log_dir", default="./.logs")
    p.add_argument("--num_prefix", default="r7p2_num")
    p.add_argument("--gen_prefix", default="r7p2_gen")
    p.add_argument("--kdist_prefix", default="r7p2_kdist")
    p.add_argument("--gen_dir", default="")
    p.add_argument("--kdist_dir", default="")
    p.add_argument("--out", default="")
    p.add_argument("--k_dense", type=int, default=28)
    args = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    slices = load_numeric(args.log_dir, args.num_prefix)
    if not slices:
        raise SystemExit("no eval_compare outputs found")
    gen = load_gen(args.gen_dir or args.log_dir, args.gen_prefix)
    kdist = load_kdist(args.kdist_dir or args.log_dir, args.kdist_prefix, args.k_dense)
    # decode-k fallback for methods the probe does not cover (RT): use numeric k_est
    num_k = {}
    for rows in slices.values():
        for r in rows.values():
            num_k[r["name"].split(":")[-1].replace("r7_", "")] = r["k"]

    keys = sorted(slices, key=lambda k: int(k) if str(k).isdigit() else 0)
    num_metrics = [("acc", "acc"), ("loss", "loss (NLL)")]
    gen_metrics = [("rouge", "ROUGE-L"), ("rep", "rep3 (repetition)"),
                   ("stability", "stability"), ("ms_per_tok", "ms/token")]

    fig, axes = plt.subplots(2, max(len(keys) * len(num_metrics), len(gen_metrics)),
                             figsize=(4.2 * max(len(keys) * 2, 4), 10.5), squeeze=False)
    merged = {"numeric": {k: list(v.values()) for k, v in slices.items()}, "gen": {}}

    for si, key in enumerate(keys):
        rows = slices[key]
        base = rows.get("dense", {})
        groups = defaultdict(list)
        for r in rows.values():
            g = fam_of(r["name"])
            if g and r["k"]:
                groups[g].append(r)
        for mi, (metric, mlabel) in enumerate(num_metrics):
            ax = axes[0][si * len(num_metrics) + mi]
            if base.get(metric) is not None:
                ax.axhline(base[metric], ls=":", c="gray", lw=1)
                ax.text(0.99, base[metric], " raw dense", fontsize=8, c="gray", ha="right",
                        va="bottom", transform=ax.get_yaxis_transform())
            for g in [x[1] for x in FAMILIES]:
                if g not in groups:
                    continue
                rs = sorted(groups[g], key=lambda x: x["k"])
                draw_series(ax, [(r["k"], r[metric]) for r in rs], g)
                for r in rs:
                    if len(rs) > 1 or g in ("Speaker (ours)", "dense-ft"):
                        ax.annotate(r["name"].split(":")[-1].replace("r7_", ""),
                                    (r["k"], r[metric]), textcoords="offset points",
                                    xytext=(6, 4), fontsize=7)
            ax.set_title(f"offset {key} — {mlabel}", fontsize=10)
            ax.set_xlabel(f"k (of {args.k_dense} = dense)")
            ax.grid(alpha=0.3)
        axes[0][si * len(num_metrics)].set_ylabel("heldout metric")

    for mi, (metric, mlabel) in enumerate(gen_metrics):
        ax = axes[1][mi]
        groups = defaultdict(list)
        for tag, m in gen.items():
            g = "raw dense" if tag == "rawdense" else fam_of(m["name"])
            if g is None:
                continue
            k = kdist.get(tag) or num_k.get(tag)
            if k and m.get(metric) is not None:
                groups[g].append((k, m[metric], tag))
        for g in [x[1] for x in FAMILIES] + ["raw dense"]:
            if g not in groups:
                continue
            rs = sorted(groups[g])
            if g == "raw dense":
                ax.axhline(rs[0][1], ls=":", c="gray", lw=1)
                ax.text(0.99, rs[0][1], " raw dense", fontsize=8, c="gray", ha="right",
                        va="bottom", transform=ax.get_yaxis_transform())
                merged["gen"].setdefault("raw dense", {"k": rs[0][0], metric: rs[0][1]})
                continue
            draw_series(ax, [(k, v) for k, v, _ in rs], g)
            for k, v, tag in rs:
                if len(rs) > 1 or g in ("Speaker (ours)", "dense-ft"):
                    ax.annotate(tag, (k, v), textcoords="offset points", xytext=(6, 4),
                                fontsize=7)
                merged["gen"].setdefault(g, {})[metric] = v
            merged["gen"][g]["k"] = rs[-1][0]
        ax.set_title(f"generation (128 tok) — {mlabel}", fontsize=10)
        ax.set_xlabel("decode-time k (probe_kdist)")
        ax.grid(alpha=0.3)
    axes[1][0].set_ylabel("generation metric")

    axes[0][-1].legend(loc="best", fontsize=8)
    fig.suptitle("r7 — all families, one yardstick: numeric (top) + generation (bottom) vs depth",
                 y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = args.out or os.path.join(args.log_dir, f"{args.num_prefix}_main.png")
    fig.savefig(out, dpi=150)
    json.dump(merged, open(re.sub(r"\.png$", ".json", out), "w"), indent=1)
    n_gen = sum(1 for _ in gen)
    print(f"-> {out} ({len(keys)} numeric slices x {len(num_metrics)} metrics, "
          f"{n_gen} gen methods x {len(gen_metrics)} metrics)")


if __name__ == "__main__":
    main()
