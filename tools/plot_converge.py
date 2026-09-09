"""Convergence-curve comparison: reads each training run's converge.jsonl (plateau checkpoints),
plots held-out subset loss / k-or-exec over steps (for the ed3 all-methods convergence comparison).

Usage:
  python3 tools/plot_converge.py --logs ours=/path/ours_q05/converge.jsonl \
      --logs modd=/path/modd_q05/converge.jsonl --out /tmp/converge.png
JSONL row format: {"step","subset_loss","best","ema_lm","k"?"exec"?} (written by the training
script's plateau checkpoints).
"""
import argparse
import json
import os


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--logs", action="append", default=[],
                   help="tag=converge.jsonl path, repeatable")
    p.add_argument("--out", default="/tmp/converge.png")
    return p.parse_args()


def load(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    args = parse_args()
    assert args.logs, "at least one --logs tag=path is required"
    series = []
    for item in args.logs:
        tag, path = item.split("=", 1)
        assert os.path.exists(path), f"does not exist: {path}"
        series.append((tag, load(path)))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"ours": "#1f77b4", "modd": "#ff7f0e", "rt": "#2ca02c", "mdf": "#9467bd"}
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    ax = axes[0]
    for tag, rows in series:
        xs = [r["step"] for r in rows]
        ax.plot(xs, [r["subset_loss"] for r in rows], marker="o", ms=3,
                color=colors.get(tag, "#7f7f7f"), label=f"{tag} heldout-sub")
        if rows and rows[0].get("ema_lm") is not None:
            ax.plot(xs, [r["ema_lm"] for r in rows], ls="--", lw=1,
                    color=colors.get(tag, "#7f7f7f"), alpha=0.6, label=f"{tag} train-ema")
    ax.set_xlabel("step")
    ax.set_title("held-out subset loss (solid) + train ema (dashed)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax = axes[1]
    for tag, rows in series:
        xs = [r["step"] for r in rows]
        if rows and rows[0].get("k") is not None:
            ax.plot(xs, [r["k"] for r in rows], marker="o", ms=3,
                    color=colors.get(tag, "#7f7f7f"), label=f"{tag} k")
        elif rows and rows[0].get("exec") is not None:
            ax.plot(xs, [r["exec"] for r in rows], marker="o", ms=3,
                    color=colors.get(tag, "#7f7f7f"), label=f"{tag} exec")
    ax.set_xlabel("step")
    ax.set_title("k (or exec) over steps")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    plt.close(fig)
    print(f"saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
