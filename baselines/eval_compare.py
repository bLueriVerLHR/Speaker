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
import gc
import os
import sys
from pathlib import Path
from typing import Annotated, Literal

import torch
import typer
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from baselines.assemble import assemble
from data.sft import SFTDataset, make_collate
from speaker.evaluate import eval_heldout
from speaker.log import logger
from tools.mem_demand import decoder_gb_from_model, demand, kv_layer_bytes_from_config
from baselines.lib import (  # noqa: E402
    eval_heldout_mdf,
    eval_heldout_modd,
    eval_heldout_rt,
)
from tools._common import collect_gc, dump_json  # noqa: E402


app = typer.Typer(add_completion=False)


def load_base(model_id, device):
    from speaker.train_common import build_model
    return build_model(model_id, device, dtype=torch.bfloat16)


@app.command()
def main(
    model_id: Annotated[str, typer.Option("--model_id")] = "/home/hdd/model/Qwen2.5-7B-Instruct",
    data_path: Annotated[str, typer.Option("--data_path")] = "./data/sft_t2t_mini.jsonl",
    offset: Annotated[int, typer.Option("--offset")] = 1000,
    n: Annotated[int, typer.Option("--n")] = 100,
    max_len: Annotated[int, typer.Option("--max_len")] = 256,
    batch_size: Annotated[int, typer.Option("--batch_size")] = 4,
    use_chat: Annotated[bool, typer.Option("--use_chat/--no-use_chat")] = True,
    valid_mode: Annotated[Literal["labels", "attention_mask"], typer.Option("--valid_mode", help="token accounting for loss/acc: 'labels' = assistant tokens only (SFT standard, unified across all families); 'attention_mask' = all tokens (legacy)")] = "labels",
    mask_user: Annotated[bool, typer.Option("--mask_user/--no-mask_user")] = True,
    ours: Annotated[list[str], typer.Option("--ours", help="ours ckpt directories, repeatable")] = [],
    dense_ft: Annotated[list[str], typer.Option("--dense_ft", help="dense fine-tuned ckpt directories (denseft_config.json+lora.pt), repeatable")] = [],
    modd: Annotated[list[str], typer.Option("--modd", help="MoD baseline ckpt directories, repeatable (the full-parameter version ships its own full base, the LoRA version is wrapped per config)")] = [],
    rt: Annotated[list[str], typer.Option("--rt", help="RT baseline ckpt directories, repeatable")] = [],
    mdf: Annotated[list[str], typer.Option("--mdf", help="MoDification baseline ckpt directories (mdf_config.json+routers.pt), repeatable")] = [],
    use_lora: Annotated[bool, typer.Option("--use_lora/--no-use_lora", help="LoRA wrapping switch (must match training; families adapt per ckpt config)")] = True,
    lora_rank: Annotated[int, typer.Option("--lora_rank", help="same as the train.py default")] = 8,
    lora_alpha: Annotated[int, typer.Option("--lora_alpha")] = 16,
    lora_targets: Annotated[str, typer.Option("--lora_targets")] = "q_proj,v_proj",
    device: Annotated[str, typer.Option("--device")] = "cuda:0",
    out: Annotated[str, typer.Option("--out")] = "/tmp/compare.json",
    no_png: Annotated[bool, typer.Option("--no_png", help="skip the per-invocation figure")] = False,
    kv_ctx: Annotated[int, typer.Option("--kv_ctx", help="context length for the KV term of the memory-demand columns")] = 1024,
) -> None:
    """Five-way comparison: dense / dense-ft / ours / MoD / RT / MoDification."""
    from speaker.terminal import setup_terminal
    setup_terminal()
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        logger.info(f"Using {device} free {torch.cuda.mem_get_info(device)[0] / 1024**3:.1f}GB")
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    full = SFTDataset(data_path, offset + n,
                      tok=tok if use_chat else None, use_chat=use_chat)
    texts = full.samples[offset:offset + n]
    assert texts, f"empty eval slice ({len(full.samples)} samples total, offset {offset})"
    logger.info(f"eval {len(texts)} texts from {data_path}@{offset} "
                f"max_len {max_len}")

    rows = []
    lora = (dict(rank=lora_rank, alpha=lora_alpha, targets=lora_targets)
            if use_lora else None)

    dense = load_base(model_id, device)
    coll = make_collate(tok, device, max_len, use_chat, mask_user)
    d = eval_heldout(dense, texts, coll, batch_size, valid_mode=valid_mode)
    logger.info(f"[dense] loss {d['loss']:.3f} acc {d['acc']:.3f}")
    # Memory-demand anchor (MoDification axes): measured once on dense, reused
    # for every row so all families share one source (tools/mem_demand.py).
    _n_layers = int(dense.config.num_hidden_layers)
    _dec_gb, _, _ = decoder_gb_from_model(dense)
    _kv_B = kv_layer_bytes_from_config(dense.config, 2)

    def _mem(k_total: float) -> dict:
        return demand(k_total, _n_layers, _dec_gb, _kv_B, kv_ctx)

    _dmem = _mem(float(_n_layers))
    rows.append({"name": "dense", "loss": d["loss"], "acc": d["acc"],
                 "mean_k": float(_n_layers), "std_k": 0.0,
                 "mem_active_gb": _dmem["active_w_gb"],
                 "kv_mb": _dmem["kv_mb_ctx"],
                 "flop_ratio": _dmem["flop_ratio"]})
    del dense
    if device.type == "cuda":
        gc.collect()  # patch/peft bound-method reference cycles must be collected by gc, otherwise loading 7B models back-to-back OOMs
        torch.cuda.empty_cache()

    for ckpt in dense_ft:
        asm = assemble(ckpt, model_id, lora, device)
        m2, dc = asm.model, asm.cfg
        r = eval_heldout(m2, texts, coll, batch_size, valid_mode=valid_mode)
        k_fix = dc.get("n_layers", m2.config.num_hidden_layers)
        logger.info(f"[{asm.name}] loss {r['loss']:.3f} acc {r['acc']:.3f} "
                    f"(Δ {r['loss'] - d['loss']:+.3f}/{r['acc'] - d['acc']:+.3f}) "
                    f"k {k_fix} (no sparsity)")
        _fm = _mem(float(k_fix))
        rows.append({"name": asm.name, "loss": r["loss"],
                     "acc": r["acc"], "dloss": r["loss"] - d["loss"],
                     "dacc": r["acc"] - d["acc"], "mean_k": float(k_fix), "std_k": 0.0,
                     "mem_active_gb": _fm["active_w_gb"], "kv_mb": _fm["kv_mb_ctx"],
                     "flop_ratio": _fm["flop_ratio"]})
        del m2, asm
        if device.type == "cuda":
            collect_gc()

    for ckpt in ours:
        asm = assemble(ckpt, model_id, lora, device)  # hard skip mode
        mod = asm.model
        r = eval_heldout(mod, texts, coll, batch_size, valid_mode=valid_mode)
        logger.info(f"[{asm.name}] loss {r['loss']:.3f} acc {r['acc']:.3f} "
                    f"(Δ {r['loss'] - d['loss']:+.3f}/{r['acc'] - d['acc']:+.3f}) "
                    f"k {r['mean_k']:.1f}±{r['std_k']:.1f}")
        _om = _mem(float(r["mean_k"]))
        rows.append({"name": asm.name, "loss": r["loss"],
                     "acc": r["acc"], "dloss": r["loss"] - d["loss"],
                     "dacc": r["acc"] - d["acc"], "mean_k": r["mean_k"],
                     "std_k": r["std_k"], "quartile_k": r["quartile_k"],
                     "mem_active_gb": _om["active_w_gb"], "kv_mb": _om["kv_mb_ctx"],
                     "flop_ratio": _om["flop_ratio"]})
        del mod, asm
        if device.type == "cuda":
            collect_gc()

    for ckpt in modd:
        asm = assemble(ckpt, model_id, lora, device)
        m2, routed, n_dense = asm.model, asm.routed, asm.n_dense
        r = eval_heldout_modd(m2, routed, n_dense, texts, coll, batch_size)
        logger.info(f"[{os.path.basename(ckpt)}] loss {r['loss']:.3f} acc {r['acc']:.3f} "
                    f"(Δ {r['loss'] - d['loss']:+.3f}/{r['acc'] - d['acc']:+.3f}) "
                    f"k {r['mean_k']:.1f}±{r['std_k']:.1f}")
        _mm = _mem(float(r["mean_k"]))
        rows.append({"name": asm.name, "loss": r["loss"],
                     "acc": r["acc"], "dloss": r["loss"] - d["loss"],
                     "dacc": r["acc"] - d["acc"], "mean_k": r["mean_k"],
                     "std_k": r["std_k"],
                     "mem_active_gb": _mm["active_w_gb"], "kv_mb": _mm["kv_mb_ctx"],
                     "flop_ratio": _mm["flop_ratio"]})
        del m2, asm
        if device.type == "cuda":
            collect_gc()

    for ckpt in mdf:
        asm = assemble(ckpt, model_id, lora, device)
        m2, routed, n_dense = asm.model, asm.routed, asm.n_dense
        r = eval_heldout_mdf(m2, routed, n_dense, texts, coll, batch_size)
        logger.info(f"[{os.path.basename(ckpt)}] loss {r['loss']:.3f} acc {r['acc']:.3f} "
                    f"(Δ {r['loss'] - d['loss']:+.3f}/{r['acc'] - d['acc']:+.3f}) "
                    f"k {r['mean_k']:.1f}±{r['std_k']:.1f}")
        _mf = _mem(float(r["mean_k"]))
        rows.append({"name": asm.name, "loss": r["loss"],
                     "acc": r["acc"], "dloss": r["loss"] - d["loss"],
                     "dacc": r["acc"] - d["acc"], "mean_k": r["mean_k"],
                     "std_k": r["std_k"],
                     "mem_active_gb": _mf["active_w_gb"], "kv_mb": _mf["kv_mb_ctx"],
                     "flop_ratio": _mf["flop_ratio"]})
        del m2, asm
        if device.type == "cuda":
            collect_gc()

    for ckpt in rt:
        asm = assemble(ckpt, model_id, lora, device)
        m2, gated, n_always = asm.model, asm.gated, asm.n_always
        r = eval_heldout_rt(m2, gated, texts, coll, batch_size, n_always=n_always)
        k_est = n_always + (r["exec_rate"] or 0) * asm.n_mod
        logger.info(f"[{os.path.basename(ckpt)}] loss {r['loss']:.3f} acc {r['acc']:.3f} "
                    f"(Δ {r['loss'] - d['loss']:+.3f}/{r['acc'] - d['acc']:+.3f}) "
                    f"exec {r['exec_rate']:.2f} k_est {k_est:.1f}")
        _rm = _mem(float(k_est))
        rows.append({"name": asm.name, "loss": r["loss"],
                     "acc": r["acc"], "dloss": r["loss"] - d["loss"],
                     "dacc": r["acc"] - d["acc"], "exec_rate": r["exec_rate"],
                     "k_est": k_est,
                     "mem_active_gb": _rm["active_w_gb"], "kv_mb": _rm["kv_mb_ctx"],
                     "flop_ratio": _rm["flop_ratio"]})
        del m2, asm
        if device.type == "cuda":
            collect_gc()

    from speaker.terminal import summary_table

    def _mem_cell(r):
        if "mem_active_gb" in r:
            return f"{r['mem_active_gb']:.2f}GB/{r['flop_ratio']:.2f}"
        return "-"
    summary_table("held-out comparison",
                  ["ckpt", "loss", "acc", "Δloss", "Δacc", "k/exec", "mem/FLOP"],
                  [[r["name"], f"{r['loss']:.3f}", f"{r['acc']:.3f}",
                    f"{r['dloss']:+.3f}" if "dloss" in r else "-",
                    f"{r['dacc']:+.3f}" if "dacc" in r else "-",
                    (f"{r['mean_k']:.1f}±{r['std_k']:.1f}" if "mean_k" in r
                     else f"exec {r['exec_rate']:.2f}"),
                    _mem_cell(r)]
                   for r in rows[1:]] or [["-", "-", "-", "-", "-", "-", "-"]])
    dump_json(out, {"meta": {"model_id": model_id, "offset": offset, "n": n,
                             "max_len": max_len, "use_chat": use_chat,
                             "valid_mode": valid_mode, "batch_size": batch_size,
                             "kv_ctx": kv_ctx, "decoder_gb": _dec_gb,
                             "kv_layer_B": _kv_B, "n_layers": _n_layers},
                    "dense": {"loss": d["loss"], "acc": d["acc"]}, "rows": rows})
    logger.info(f"saved {out}")
    try:
        png = plot_rows(rows, d, model_id=model_id, offset=offset, n=n,
                        max_len=max_len, out=out, no_png=no_png)
        logger.info(f"saved {png}")
    except ImportError:
        logger.warning("matplotlib unavailable, skipping the plot (pip install matplotlib)")


def plot_rows(rows, dense, *, model_id, offset, n, max_len, out, no_png):
    """Five-panel comparison figure: loss / acc / Δacc / per-token k / active memory (MoDification axes). Returns the PNG path."""
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
    fig, axes = plt.subplots(1, 5, figsize=(20, 4.2))
    fig.suptitle(f"{os.path.basename(model_id)}  |  held-out "
                 f"offset={offset} n={n} max_len={max_len}")

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
    # 5. active memory demand (MoDification axis: same loss -> lower GB wins)
    ax = axes[4]
    mems = [r.get("mem_active_gb", 0.0) for r in rows[1:]] or [0.0]
    flops = [r.get("flop_ratio", 0.0) for r in rows[1:]] or [0.0]
    b = ax.bar(range(len(mems)), mems,
               color=[style(r["name"]) for r in rows[1:]] or ["#7f7f7f"], width=0.62)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=9)
    ax.set_title("active weights GB/tok (lower=cheaper)", fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    for rect, v, f in zip(b, mems, flops):
        ax.annotate(f"{v:.2f}/{f:.2f}", (rect.get_x() + rect.get_width() / 2, rect.get_height()),
                    ha="center", va="bottom", fontsize=8)
    if no_png:
        return
    png = os.path.splitext(out)[0] + ".png"
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(png, dpi=150)
    plt.close(fig)
    return png


if __name__ == "__main__":
    app()
