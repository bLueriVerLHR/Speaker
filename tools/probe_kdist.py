"""Five-way k distribution / per-layer execution rate / peak GPU memory probe (for r1/r2
attribution, reproducible evidence).

Total formulation: ours uses ``get_total_counts`` (gated selections plus
always-on layers), matching deployment accounting and the other evaluation tools.
MoD's k already includes shared (fixed) layers.

Usage:
  python3 tools/probe_kdist.py --ours ./ckpt/ours_q05_full \
      --modd /tmp/modd_q05 --rt /tmp/rt_q05 --mdf ./ckpt/mdf_q05 \
      --n 100 --out .logs/0907_0000_kdist.json
Outputs JSON (+ same-named .png, needs matplotlib): per-token total-k distribution/quantiles/
histogram, per-layer execution rates, five-model forward peak GPU memory, optional generation
comparison (dense vs ours).
"""
import os
import pathlib
import sys
from typing import Annotated

import torch
import typer

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from transformers import AutoTokenizer

from data.sft import SFTDataset, make_collate
from speaker.evaluate import chunked_nll_correct
from baselines.assemble import assemble  # noqa: E402
from baselines.lib import collect_mdf_stats, collect_modd_stats, eval_heldout_rt
from tools._common import collect_gc, dump_json
from tools._common import module_gb as weights_gb  # noqa: E402
from speaker.log import logger  # noqa: E402
from speaker.terminal import setup_terminal, track

def load_base(src, device):
    from speaker.train_common import build_model
    return build_model(src, torch.device("cpu"), dtype=torch.bfloat16)


def fwd_eval(model, texts, coll, device, batch_size, k_fn=None, layer_fn=None,
             pos_bins=0):
    model.eval()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    sum_nll = n_tok = n_ok = 0.0
    ks, layer_sum, layer_n = [], {}, {}
    bin_sum = [0.0] * pos_bins
    bin_n = [0] * pos_bins
    with torch.no_grad():
        for i in track(range(0, len(texts), batch_size),
                       total=(len(texts) + batch_size - 1) // batch_size,
                       desc="fwd"):
            b = coll(texts[i:i + batch_size])
            out = model(**b)
            valid = b["attention_mask"].bool().to(out.logits.device)
            nll, correct = chunked_nll_correct(out.logits, b["labels"])
            sum_nll += nll[valid].sum().item()
            n_tok += valid.sum().item()
            n_ok += correct[valid].sum().item()
            del nll, correct
            if k_fn is not None:
                kv = k_fn(valid)
                if kv is not None:
                    ks.extend(kv.tolist())
                    if pos_bins:
                        # valid.nonzero() is row-major, same order as k_fn(valid)
                        pb = (valid.nonzero()[:, 1] * pos_bins
                              // valid.size(1)).clamp_max(pos_bins - 1).to(kv.device)
                        for bi in range(pos_bins):
                            m = pb == bi
                            bin_sum[bi] += kv[m].sum().item()
                            bin_n[bi] += m.sum().item()
            if layer_fn is not None:
                for idx, rate, n in layer_fn(valid):
                    layer_sum[idx] = layer_sum.get(idx, 0.0) + rate * n
                    layer_n[idx] = layer_n.get(idx, 0) + n
    res = {"loss": sum_nll / max(n_tok, 1), "acc": n_ok / max(n_tok, 1),
           "peak_gb": torch.cuda.max_memory_allocated(device) / 1e9}
    if ks:
        t = torch.tensor(ks)
        h = torch.histc(t, bins=30, min=0, max=30).tolist()
        s = [sum(h[max(i-1, 0):i+2])/len(h[max(i-1, 0):i+2]) for i in range(30)]  # 3-point smoothing
        mx = max(s) or 1.0
        n_peaks = sum(1 for i in range(1, 29)
                      if s[i] >= s[i-1] and s[i] >= s[i+1] and s[i] > 0.05 * mx)
        res["k_total"] = {"mean": t.mean().item(), "std": t.std().item(),
                          "min": t.min().item(), "max": t.max().item(),
                          "n_peaks": n_peaks,
                          "q01_10_25_50_75_90_100":
                              torch.quantile(t, torch.tensor([0, .1, .25, .5, .75, .9, 1])).tolist(),
                          "hist_total_k_0_30": h}
    if layer_sum:
        res["layer_exec"] = {str(i): layer_sum[i] / max(layer_n[i], 1) for i in sorted(layer_sum)}
    if pos_bins and any(bin_n):
        res["k_by_posbin"] = [s / max(n, 1) for s, n in zip(bin_sum, bin_n)]
    return res


def plot(out_dict, png_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def color(name):
        for tag, c in (("ours", "#1f77b4"), ("modd", "#ff7f0e"),
                       ("rt", "#2ca02c"), ("mdf", "#9467bd")):
            if name.startswith(tag):
                return c
        return "#7f7f7f"

    def short(name):
        return name.replace("ours:", "o:") if name.startswith("ours") else name

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.2))
    # 1. per-token total-k histogram (bin width = 1 layer, 0..30 covers 28 layers)
    ax = axes[0]
    for name, r in out_dict.items():
        if not isinstance(r, dict) or "k_total" not in r:
            continue
        h = r["k_total"]["hist_total_k_0_30"]
        ax.bar(range(30), h, width=1.0, alpha=0.55, color=color(name),
               label=f"{short(name)} μ={r['k_total']['mean']:.1f} σ={r['k_total']['std']:.1f}")
    ax.set_xlabel("per-token total active layers k")
    ax.set_title("k distribution (total, /28)")
    ax.legend(fontsize=7)
    ax.grid(axis="y", alpha=0.3)
    # 2. per-layer execution rate (grouped bars for all methods with layer_exec)
    ax = axes[1]
    series = [(nm, color(nm)) for nm, r in out_dict.items()
              if isinstance(r, dict) and r.get("layer_exec")]
    if series:
        idx = sorted({int(k2) for nm, _ in series for k2 in out_dict[nm]["layer_exec"]})
        w = 0.8 / len(series)
        for s, (nm, c) in enumerate(series):
            vals = [out_dict[nm]["layer_exec"].get(str(i), 0) for i in idx]
            off = (s - (len(series) - 1) / 2) * w
            ax.bar([i + off for i in idx], vals, width=w * 0.9, color=c, label=short(nm))
        ax.set_xticks(idx[::2])
    ax.set_xlabel("layer")
    ax.set_title("per-layer exec rate")
    ax.legend(fontsize=7)
    ax.grid(axis="y", alpha=0.3)
    # 3. peak GPU memory
    ax = axes[2]
    names = [nm for nm, r in out_dict.items()
             if isinstance(r, dict) and "peak_gb" in r]
    peaks = [out_dict[nm]["peak_gb"] for nm in names]
    wts = [out_dict[nm].get("weights_gb", 0) for nm in names]
    b1 = ax.bar([short(n) for n in names], wts, color="#999999", label="weights")
    ax.bar([short(n) for n in names], [p - wt for p, wt in zip(peaks, wts)], bottom=wts,
           color="#d62728", label="activations+tmp")
    ax.set_title("forward peak GB")
    ax.legend(fontsize=8)
    ax.tick_params(axis="x", rotation=20)
    for rect, v in zip(b1, peaks):
        ax.annotate(f"{v:.2f}", (rect.get_x() + rect.get_width() / 2, v),
                    ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(png_path, dpi=130)
    plt.close(fig)
    return png_path


def routed_k_from(collect_fn, routed, n_dense):
    """Per-token total-k getter closing over one baseline's stats collector."""
    def _k(valid):
        _, kk = collect_fn(routed, training=False)
        return (kk[valid] + n_dense).float() if kk is not None else None
    return _k


def routed_layers_from(routed, r_idx, is_routed):
    """Per-layer exec-rate rows closing over one baseline's patched layers."""
    def _layers(valid):
        nv = valid.sum().item()
        for j, p in enumerate(routed):
            sel = getattr(p, "_last_sel", None)
            if sel is not None:
                yield r_idx[j] if j < len(r_idx) else j, float(sel[valid].float().mean()), nv
        for i, r in enumerate(is_routed):
            if not r:
                yield i, 1.0, nv
    return _layers


app = typer.Typer(add_completion=False)


@app.command()
def main(
    model_id: Annotated[str, typer.Option("--model_id", help="same as the finetune-track default (7B)")] = "/home/hdd/model/Qwen2.5-7B-Instruct",
    data_path: Annotated[str, typer.Option("--data_path")] = "./data/sft_t2t_mini.jsonl",
    ours: Annotated[list[str], typer.Option("--ours", help="ours ckpt (mod_config.json+gate.pt, may carry a full base), repeatable, tag=ours:<basename>")] = [],
    use_lora: Annotated[bool, typer.Option("--use_lora/--no-use_lora", help="LoRA wrapping switch (must match training; families adapt per ckpt config)")] = True,
    lora_rank: Annotated[int, typer.Option("--lora_rank", help="same as the train.py default")] = 8,
    lora_alpha: Annotated[int, typer.Option("--lora_alpha")] = 16,
    lora_targets: Annotated[str, typer.Option("--lora_targets")] = "q_proj,v_proj",
    modd: Annotated[str, typer.Option("--modd", help="MoD ckpt (modd_config.json+routers.pt)")] = "",
    mdf: Annotated[str, typer.Option("--mdf", help="MoDification ckpt (mdf_config.json+routers.pt)")] = "",
    rt: Annotated[str, typer.Option("--rt", help="RT ckpt (rt_config.json+routers.pt)")] = "",
    offset: Annotated[int, typer.Option("--offset")] = 1000,
    n: Annotated[int, typer.Option("--n")] = 100,
    max_len: Annotated[int, typer.Option("--max_len")] = 256,
    batch_size: Annotated[int, typer.Option("--batch_size")] = 4,
    pos_bins: Annotated[int, typer.Option("--pos_bins", help="split the window into N position bins and report mean-k per bin (0=off)")] = 0,
    device: Annotated[str, typer.Option("--device")] = "cuda:0",
    out: Annotated[str, typer.Option("--out")] = ".logs/kdist.json",
    no_png: Annotated[bool, typer.Option("--no_png")] = False,
) -> None:
    """Five-way k distribution / per-layer execution rate / peak-memory probe."""
    setup_terminal()
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    full = SFTDataset(data_path, offset + n, tok=None, use_chat=False)
    texts = full.samples[offset:offset + n]
    assert texts, "the eval slice is empty"
    coll = make_collate(tok, device, max_len, False, True)
    res = {"meta": dict(model_id=model_id, data_path=data_path, offset=offset,
                        n=n, max_len=max_len, batch_size=batch_size)}
    lora = (dict(rank=lora_rank, alpha=lora_alpha, targets=lora_targets)
            if use_lora else None)

    m = load_base(model_id, device).to(device)
    res["dense"] = {"weights_gb": weights_gb(m)}
    res["dense"].update(fwd_eval(m, texts, coll, device, batch_size))
    logger.info(f"[dense] loss {res['dense']['loss']:.3f} acc {res['dense']['acc']:.3f} "
                f"peak {res['dense']['peak_gb']:.2f}GB")
    del m
    collect_gc()  # 7B: patch/peft reference cycles need gc, otherwise the next model load OOMs

    for ckpt in ours:
        tag = f"ours:{os.path.basename(ckpt.rstrip('/'))}"
        asm = assemble(ckpt, model_id, lora, device, skip_mode=None)
        w = asm.model
        m = w
        gp = sum(p.numel() for p in w.get_router_parameters())
        res[tag] = {"weights_gb": weights_gb(m), "gate_params": gp,
                    "gate_share": gp / sum(p.numel() for p in m.parameters())}

        def ours_k(valid, _w=w):
            # get_total_counts already includes the always-on layers
            c = _w.get_total_counts()
            return c[valid].float() if c is not None else None

        def ours_layers(valid, _w=w):
            nv = valid.sum().item()
            for lyr in _w.layers:
                if lyr.is_always_on:
                    yield lyr.layer_idx, 1.0, nv
                else:
                    o = lyr.last_gating_output
                    if o is not None:
                        hm = o.hard_mask.squeeze(-1)[valid].float()
                        yield lyr.layer_idx, hm.mean().item(), nv

        res[tag].update(fwd_eval(w, texts, coll, device, batch_size, ours_k, ours_layers,
                                 pos_bins=pos_bins))
        k = res[tag]["k_total"]
        logger.info(f"[{tag}] k_total {k['mean']:.1f}±{k['std']:.1f} "
                    f"q={['%.1f' % v for v in k['q01_10_25_50_75_90_100']]}")
        del ours_k, ours_layers, w, m  # the closure holds w via the _w default arg; without del, the next round's to(device) OOMs
        collect_gc()

    if modd:
        asm = assemble(modd, model_id, lora, device)
        m, routed, mc = asm.model, asm.routed, asm.cfg
        n_dense = asm.n_dense
        r_idx = [i for i, r in enumerate(mc["is_routed"]) if r]
        res["modd"] = {"weights_gb": weights_gb(m)}

        modd_k = routed_k_from(collect_modd_stats, routed, n_dense)
        modd_layers = routed_layers_from(routed, r_idx, mc["is_routed"])

        res["modd"].update(fwd_eval(m, texts, coll, device, batch_size, modd_k, modd_layers,
                                    pos_bins=pos_bins))
        k = res["modd"]["k_total"]
        logger.info(f"[modd] k_total {k['mean']:.1f}±{k['std']:.1f}")
        del modd_k, modd_layers, routed, m  # the closure holds the model via routed; without del, the next round OOMs
        collect_gc()

    if mdf:
        asm = assemble(mdf, model_id, lora, device)
        m, routed, mc = asm.model, asm.routed, asm.cfg
        n_dense = asm.n_dense
        r_idx = [i for i, r in enumerate(mc["is_routed"]) if r]
        res["mdf"] = {"weights_gb": weights_gb(m)}

        mdf_k = routed_k_from(collect_mdf_stats, routed, n_dense)
        mdf_layers = routed_layers_from(routed, r_idx, mc["is_routed"])

        res["mdf"].update(fwd_eval(m, texts, coll, device, batch_size, mdf_k, mdf_layers,
                                   pos_bins=pos_bins))
        k = res["mdf"]["k_total"]
        logger.info(f"[mdf] k_total {k['mean']:.1f}±{k['std']:.1f}")
        del mdf_k, mdf_layers, routed, m
        collect_gc()

    if rt:
        asm = assemble(rt, model_id, lora, device)
        m, gated = asm.model, asm.gated

        r = eval_heldout_rt(m, gated, texts, coll, batch_size, n_always=asm.n_always)
        torch.cuda.reset_peak_memory_stats(device)
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                m(**coll(texts[i:i + batch_size]))
        res["rt"] = {"loss": r["loss"], "acc": r["acc"], "exec_rate": r["exec_rate"],
                     "weights_gb": weights_gb(m),
                     "peak_gb": torch.cuda.max_memory_allocated(device) / 1e9}
        logger.info(f"[rt] exec {r['exec_rate']:.2f}")
        del gated, m
        collect_gc()

    dump_json(out, res)
    logger.info(f"saved {out}")
    try:
        png = None if no_png else plot(res, os.path.splitext(out)[0] + ".png")
        logger.info(f"saved {png}")
    except ImportError:
        logger.warning("matplotlib unavailable, skipping plot")


if __name__ == "__main__":
    app()
