"""Baseline comparison entry point: Dense vs Dense-ft vs ours (Speaker) ckpt(s) vs MoD vs RT vs MoDification, same slice, same settings.
The Dense baseline needs no training script (the native HF model is evaluated directly) and is included in the same arena here; Dense-ft (the fairly
fine-tuned version, same-spec LoRA pure SFT) is produced by baselines/train_dense.py.
Ours ckpt: --ours DIR (repeatable): mod_config.json + gate.pt (+LoRA, default r8/q,v)
Dense-ft ckpt: --dense_ft DIR (repeatable): denseft_config.json + lora.pt
MoD ckpt:  --modd DIR (repeatable): modd_config.json + routers.pt (the full-parameter version ships its own full base, the LoRA version is wrapped per config)
RT ckpt:   --rt DIR (repeatable): rt_config.json + routers.pt
MoDification ckpt: --mdf DIR (repeatable): mdf_config.json + routers.pt (same as MoD)
Metrics: loss/acc/Δ + k(mean±std); prints a comparison table + saves JSON + plots a comparison figure (PNG, needs matplotlib).
Usage: python3 baselines/eval_compare.py --ours /tmp/ours --modd /tmp/modd --rt /tmp/rt \
        --mdf ./ckpt/mdf_q05 --dense_ft /tmp/denseft \
        --data_path .../sft_t2t_mini.jsonl --offset 1000 --n 100 --out /tmp/compare.json
"""
import argparse
import gc
import json
import os
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from speaker import SpeakerConfig, convert_to_speaker
from speaker.checkpoint import load_gate
from data.sft import SFTDataset, make_collate
from speaker.evaluate import eval_heldout
from baselines.lib import (  # noqa: E402
    eval_heldout_mdf,
    eval_heldout_modd,
    eval_heldout_rt,
    patch_model_mdf,
    patch_model_modd,
    patch_model_rt,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_id", default="/home/hdd/model/Qwen2.5-7B-Instruct")
    p.add_argument("--data_path", default="./data/sft_t2t_mini.jsonl")
    p.add_argument("--offset", type=int, default=1000)
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--max_len", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--use_chat", default=True, action=argparse.BooleanOptionalAction)
    p.add_argument("--valid_mode", default="labels", choices=["labels", "attention_mask"],
                   help="token accounting for loss/acc: 'labels' = assistant tokens only (SFT standard, "
                        "unified across all families); 'attention_mask' = all tokens (legacy)")
    p.add_argument("--mask_user", default=True, action=argparse.BooleanOptionalAction)
    p.add_argument("--ours", action="append", default=[], help="ours ckpt directories, repeatable")
    p.add_argument("--dense_ft", action="append", default=[],
                   help="dense fine-tuned ckpt directories (denseft_config.json+lora.pt), repeatable")
    p.add_argument("--modd", action="append", default=[],
                   help="MoD baseline ckpt directories, repeatable (the full-parameter version ships its own full base, the LoRA version is wrapped per config)")
    p.add_argument("--rt", action="append", default=[], help="RT baseline ckpt directories, repeatable")
    p.add_argument("--mdf", action="append", default=[],
                   help="MoDification baseline ckpt directories (mdf_config.json+routers.pt), repeatable")
    p.add_argument("--use_lora", default=True, action=argparse.BooleanOptionalAction,
                   help="LoRA wrapping switch for ours ckpts (modd/mdf/dense_ft adapt per the ckpt config)")
    p.add_argument("--lora_rank", type=int, default=8, help="same as the train.py default")
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_targets", default="q_proj,v_proj")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out", default="/tmp/compare.json")
    p.add_argument("--no_png", action="store_true", help="skip the per-invocation figure (consolidate via tools/plot_r7.py instead)")
    return p.parse_args()


def load_base(model_id, device):
    m = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map=None,
        trust_remote_code=True, low_cpu_mem_usage=True).to(device)
    return m


def wrap_lora(m, cfg_json, args):
    """Wrap peft with the LoRA spec from the ckpt config (same params as training, consistent key layout); falls back to CLI args when missing."""
    from peft import LoraConfig, TaskType, get_peft_model
    targets = cfg_json.get("lora_targets")
    return get_peft_model(m, LoraConfig(
        r=cfg_json.get("lora_rank", args.lora_rank),
        lora_alpha=cfg_json.get("lora_alpha", args.lora_alpha),
        target_modules=targets or [t.strip() for t in args.lora_targets.split(",") if t.strip()],
        lora_dropout=0.05, bias="none", task_type=TaskType.CAUSAL_LM))


def main():
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        print(f"Using {device} free {torch.cuda.mem_get_info(device)[0] / 1024**3:.1f}GB",
              flush=True)
    tok = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    full = SFTDataset(args.data_path, args.offset + args.n,
                      tok=tok if args.use_chat else None, use_chat=args.use_chat)
    texts = full.samples[args.offset:args.offset + args.n]
    assert texts, f"empty eval slice ({len(full.samples)} samples total, offset {args.offset})"
    print(f"eval {len(texts)} texts from {args.data_path}@{args.offset} "
          f"max_len {args.max_len}", flush=True)

    rows = []

    dense = load_base(args.model_id, device)
    coll = make_collate(tok, device, args.max_len, args.use_chat, args.mask_user)
    d = eval_heldout(dense, texts, coll, args.batch_size, valid_mode=args.valid_mode)
    print(f"[dense] loss {d['loss']:.3f} acc {d['acc']:.3f}", flush=True)
    rows.append({"name": "dense", "loss": d["loss"], "acc": d["acc"]})
    del dense
    if device.type == "cuda":
        gc.collect()  # patch/peft bound-method reference cycles must be collected by gc, otherwise loading 7B models back-to-back OOMs
        torch.cuda.empty_cache()

    for ckpt in args.dense_ft:
        with open(os.path.join(ckpt, "denseft_config.json")) as f:
            dc = json.load(f)
        m2 = AutoModelForCausalLM.from_pretrained(
            args.model_id, dtype=torch.bfloat16, device_map=None,
            trust_remote_code=True, low_cpu_mem_usage=True)
        if dc.get("use_lora", True):
            m2 = wrap_lora(m2, dc, args)
        sd = torch.load(os.path.join(ckpt, "lora.pt"), map_location="cpu")
        missing, unexp = m2.load_state_dict(sd, strict=False)
        m2.to(device)
        n_lora_dropped = sum(1 for k in missing if "lora" in k)
        print(f"[{os.path.basename(ckpt)}] lora.pt missing {len(missing)} "
              f"(lora {n_lora_dropped}) unexpected {len(unexp)}", flush=True)
        assert n_lora_dropped == 0, "lora weights did not match any keys (rank/targets inconsistent with training?)"
        r = eval_heldout(m2, texts, coll, args.batch_size, valid_mode=args.valid_mode)
        k_fix = dc.get("n_layers", m2.config.num_hidden_layers)
        print(f"[{os.path.basename(ckpt)}] loss {r['loss']:.3f} acc {r['acc']:.3f} "
              f"(Δ {r['loss'] - d['loss']:+.3f}/{r['acc'] - d['acc']:+.3f}) "
              f"k {k_fix} (no sparsity)", flush=True)
        rows.append({"name": f"denseft:{os.path.basename(ckpt)}", "loss": r["loss"],
                     "acc": r["acc"], "dloss": r["loss"] - d["loss"],
                     "dacc": r["acc"] - d["acc"], "mean_k": float(k_fix), "std_k": 0.0})
        del m2
        if device.type == "cuda":
            gc.collect()
            torch.cuda.empty_cache()

    for ckpt in args.ours:
        # full-parameter joint-training ckpts (--save_full) ship their own full base (config.json), prefer loading from the ckpt
        base_src = ckpt if os.path.exists(os.path.join(ckpt, "config.json")) else args.model_id
        m2 = AutoModelForCausalLM.from_pretrained(
            base_src, dtype=torch.bfloat16, device_map=None,
            trust_remote_code=True, low_cpu_mem_usage=True)
        if args.use_lora:
            from peft import LoraConfig, TaskType, get_peft_model
            m2 = get_peft_model(m2, LoraConfig(
                r=args.lora_rank, lora_alpha=args.lora_alpha,
                target_modules=[t.strip() for t in args.lora_targets.split(",") if t.strip()],
                lora_dropout=0.05, bias="none", task_type=TaskType.CAUSAL_LM))
        cfg = SpeakerConfig.from_json(os.path.join(ckpt, "mod_config.json"))
        mod = convert_to_speaker(m2, cfg).to(device)
        missing, unexp = load_gate(mod, ckpt)
        print(f"[{os.path.basename(ckpt)}] gate.pt missing {len(missing)} "
              f"unexpected {len(unexp)}", flush=True)
        mod.set_skip_mode("hard")
        r = eval_heldout(mod, texts, coll, args.batch_size, valid_mode=args.valid_mode)
        print(f"[{os.path.basename(ckpt)}] loss {r['loss']:.3f} acc {r['acc']:.3f} "
              f"(Δ {r['loss'] - d['loss']:+.3f}/{r['acc'] - d['acc']:+.3f}) "
              f"k {r['mean_k']:.1f}±{r['std_k']:.1f}", flush=True)
        rows.append({"name": f"ours:{os.path.basename(ckpt)}", "loss": r["loss"],
                     "acc": r["acc"], "dloss": r["loss"] - d["loss"],
                     "dacc": r["acc"] - d["acc"], "mean_k": r["mean_k"],
                     "std_k": r["std_k"], "quartile_k": r["quartile_k"]})
        del mod, m2
        if device.type == "cuda":
            gc.collect()
            torch.cuda.empty_cache()

    for ckpt in args.modd:
        with open(os.path.join(ckpt, "modd_config.json")) as f:
            mc = json.load(f)
        # joint-training ckpts ship their own full base (config.json), prefer loading from the ckpt
        base_src = ckpt if os.path.exists(os.path.join(ckpt, "config.json")) else args.model_id
        m2 = AutoModelForCausalLM.from_pretrained(
            base_src, dtype=torch.bfloat16, device_map=None,
            trust_remote_code=True, low_cpu_mem_usage=True)
        routed = patch_model_modd(m2, mc["is_routed"], capacity=mc.get("capacity", 0.125))
        if mc.get("use_lora"):
            m2 = wrap_lora(m2, mc, args)  # same order as training (patch→peft), consistent key layout
        sd = torch.load(os.path.join(ckpt, "routers.pt"), map_location="cpu")
        missing, unexp = m2.load_state_dict(sd, strict=False)
        m2.to(device)
        n_drop = sum(1 for k in missing if "lora" in k or "router" in k)
        assert n_drop == 0, f"[{ckpt}] routers.pt keys did not match (LoRA spec inconsistent with training?)"
        print(f"[{os.path.basename(ckpt)}] routers.pt missing {len(missing)} "
              f"unexpected {len(unexp)} (base from {os.path.basename(base_src)})", flush=True)
        n_dense = len(mc["is_routed"]) - sum(mc["is_routed"])
        r = eval_heldout_modd(m2, routed, n_dense, texts, coll, args.batch_size)
        print(f"[{os.path.basename(ckpt)}] loss {r['loss']:.3f} acc {r['acc']:.3f} "
              f"(Δ {r['loss'] - d['loss']:+.3f}/{r['acc'] - d['acc']:+.3f}) "
              f"k {r['mean_k']:.1f}±{r['std_k']:.1f}", flush=True)
        rows.append({"name": f"modd:{os.path.basename(ckpt)}", "loss": r["loss"],
                     "acc": r["acc"], "dloss": r["loss"] - d["loss"],
                     "dacc": r["acc"] - d["acc"], "mean_k": r["mean_k"],
                     "std_k": r["std_k"]})
        del m2
        if device.type == "cuda":
            gc.collect()
            torch.cuda.empty_cache()

    for ckpt in args.mdf:
        with open(os.path.join(ckpt, "mdf_config.json")) as f:
            mc = json.load(f)
        # joint-training ckpts ship their own full base (config.json), prefer loading from the ckpt
        base_src = ckpt if os.path.exists(os.path.join(ckpt, "config.json")) else args.model_id
        m2 = AutoModelForCausalLM.from_pretrained(
            base_src, dtype=torch.bfloat16, device_map=None,
            trust_remote_code=True, low_cpu_mem_usage=True)
        routed = patch_model_mdf(m2, mc["is_routed"], p=mc.get("p", 0.5))
        if mc.get("use_lora"):
            m2 = wrap_lora(m2, mc, args)  # same order as training (patch→peft), consistent key layout
        sd = torch.load(os.path.join(ckpt, "routers.pt"), map_location="cpu")
        missing, unexp = m2.load_state_dict(sd, strict=False)
        m2.to(device)
        n_drop = sum(1 for k in missing if "lora" in k or "router" in k)
        assert n_drop == 0, f"[{ckpt}] routers.pt keys did not match (LoRA spec inconsistent with training?)"
        print(f"[{os.path.basename(ckpt)}] routers.pt missing {len(missing)} "
              f"unexpected {len(unexp)} (base from {os.path.basename(base_src)})", flush=True)
        n_dense = len(mc["is_routed"]) - sum(mc["is_routed"])
        r = eval_heldout_mdf(m2, routed, n_dense, texts, coll, args.batch_size)
        print(f"[{os.path.basename(ckpt)}] loss {r['loss']:.3f} acc {r['acc']:.3f} "
              f"(Δ {r['loss'] - d['loss']:+.3f}/{r['acc'] - d['acc']:+.3f}) "
              f"k {r['mean_k']:.1f}±{r['std_k']:.1f}", flush=True)
        rows.append({"name": f"mdf:{os.path.basename(ckpt)}", "loss": r["loss"],
                     "acc": r["acc"], "dloss": r["loss"] - d["loss"],
                     "dacc": r["acc"] - d["acc"], "mean_k": r["mean_k"],
                     "std_k": r["std_k"]})
        del m2
        if device.type == "cuda":
            gc.collect()
            torch.cuda.empty_cache()

    for ckpt in args.rt:
        with open(os.path.join(ckpt, "rt_config.json")) as f:
            rc = json.load(f)
        m2 = AutoModelForCausalLM.from_pretrained(
            args.model_id, dtype=torch.bfloat16, device_map=None,
            trust_remote_code=True, low_cpu_mem_usage=True)
        gated = patch_model_rt(m2, rc["is_mod"], rc.get("granularity", "block_token"),
                               rc.get("threshold", 0.5), rc.get("target"),
                               rc.get("scale", 0.0))
        sd = torch.load(os.path.join(ckpt, "routers.pt"), map_location="cpu")
        missing, unexp = m2.load_state_dict(sd, strict=False)
        m2.to(device)
        n_drop = sum(1 for k in missing if "lora" in k or "router" in k)
        assert n_drop == 0, f"[{ckpt}] routers.pt keys did not match (LoRA spec inconsistent with training?)"
        print(f"[{os.path.basename(ckpt)}] routers.pt missing {len(missing)} "
              f"unexpected {len(unexp)}", flush=True)
        n_mod = sum(rc["is_mod"])
        n_always = len(rc["is_mod"]) - n_mod
        r = eval_heldout_rt(m2, gated, texts, coll, args.batch_size, n_always=n_always)
        k_est = n_always + (r["exec_rate"] or 0) * n_mod
        print(f"[{os.path.basename(ckpt)}] loss {r['loss']:.3f} acc {r['acc']:.3f} "
              f"(Δ {r['loss'] - d['loss']:+.3f}/{r['acc'] - d['acc']:+.3f}) "
              f"exec {r['exec_rate']:.2f} k_est {k_est:.1f}", flush=True)
        rows.append({"name": f"rt:{os.path.basename(ckpt)}", "loss": r["loss"],
                     "acc": r["acc"], "dloss": r["loss"] - d["loss"],
                     "dacc": r["acc"] - d["acc"], "exec_rate": r["exec_rate"],
                     "k_est": k_est})
        del m2
        if device.type == "cuda":
            gc.collect()
            torch.cuda.empty_cache()

    print(f"{'ckpt':28s} {'loss':>6s} {'acc':>6s} {'Δloss':>7s} {'Δacc':>7s} {'k/exec'}",
          flush=True)
    for r in rows[1:]:
        kk = (f"{r['mean_k']:.1f}±{r['std_k']:.1f}" if "mean_k" in r
              else f"exec {r['exec_rate']:.2f}")
        print(f"{r['name']:28s} {r['loss']:6.3f} {r['acc']:6.3f} "
              f"{r['dloss']:+7.3f} {r['dacc']:+7.3f} {kk}", flush=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"meta": {"model_id": args.model_id, "offset": args.offset, "n": args.n,
                            "max_len": args.max_len, "use_chat": args.use_chat,
                            "valid_mode": args.valid_mode, "batch_size": args.batch_size},
                   "dense": {"loss": d["loss"], "acc": d["acc"]}, "rows": rows}, f,
                   indent=1, ensure_ascii=False)
    print(f"saved {args.out}", flush=True)
    try:
        png = plot_rows(rows, d, args)
        print(f"saved {png}", flush=True)
    except ImportError:
        print("matplotlib unavailable, skipping the plot (pip install matplotlib)", flush=True)


def plot_rows(rows, dense, args):
    """Four-panel comparison figure: loss / acc / Δacc / per-token k (mean±std). Returns the PNG path."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def style(name):
        for tag, color in (("ours", "#1f77b4"), ("modd", "#ff7f0e"),
                           ("rt", "#2ca02c"), ("mdf", "#9467bd")):
            if name.startswith(tag):
                return color
        return "#7f7f7f"

    names = [r["name"].split(":")[-1] for r in rows[1:]] or ["-"]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))
    fig.suptitle(f"{os.path.basename(args.model_id)}  |  held-out "
                 f"offset={args.offset} n={args.n} max_len={args.max_len}")

    def bars(ax, key, title, fmt="{:.3f}"):
        vals = [r[key] for r in rows[1:]] or [0.0]
        colors = [style(r["name"]) for r in rows[1:]] or ["#7f7f7f"]
        b = ax.bar(range(len(vals)), vals, color=colors, width=0.62)
        ref = {"loss": "loss", "acc": "acc"}.get(key)  # the reference line for dacc is 0
        if ref:
            ax.axhline(dense[ref], ls="--", c="k", lw=1, label=f"dense {dense[ref]:.3f}")
            ax.legend(fontsize=8)
        else:
            ax.axhline(0.0, ls="--", c="k", lw=1)
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(names, rotation=20, ha="right", fontsize=9)
        ax.set_title(title, fontsize=11)
        ax.grid(axis="y", alpha=0.3)
        for rect, v in zip(b, vals):
            ax.annotate(fmt.format(v), (rect.get_x() + rect.get_width() / 2, rect.get_height()),
                        ha="center", va="bottom", fontsize=8)

    bars(axes[0], "loss", "held-out loss (lower=better)")
    bars(axes[1], "acc", "held-out acc (higher=better)")
    bars(axes[2], "dacc", "Δacc vs dense", fmt="{:+.3f}")
    # per-token active layers k: mean±std errorbar (RT only has the k_est single point)
    ax = axes[3]
    xs, mus, sigmas, cols, offs = [], [], [], [], []
    for i, r in enumerate(rows[1:]):
        if "mean_k" in r:
            xs.append(i); mus.append(r["mean_k"]); sigmas.append(r["std_k"] or 0)
            cols.append(style(r["name"])); offs.append(None)
        elif "k_est" in r:
            xs.append(i); mus.append(r["k_est"]); sigmas.append(0)
            cols.append(style(r["name"])); offs.append(r.get("exec_rate"))
    if xs:
        ax.bar(xs, mus, color=cols, width=0.62,
               yerr=sigmas, capsize=4, error_kw={"lw": 1.2})
        for x, m, s, off in zip(xs, mus, sigmas, offs):
            lab = f"{m:.1f}±{s:.1f}" if off is None else f"{m:.1f} (exec {off:.2f})"
            ax.annotate(lab, (x, m + s), ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=9)
    ax.set_title("per-token active layers k (lower=faster)", fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    if getattr(args, "no_png", False):
        return
    png = os.path.splitext(args.out)[0] + ".png"
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(png, dpi=150)
    plt.close(fig)
    return png


if __name__ == "__main__":
    main()
