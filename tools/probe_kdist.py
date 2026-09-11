"""Five-way k distribution / per-layer execution rate / peak GPU memory probe (for r1/r2
attribution, reproducible evidence).

total formulation: ours get_active_counts covers gated layers only, so total k = gated + n_always;
MoD's k already includes shared (fixed) layers. This script uniformly uses the total formulation
(differs from eval_compare's headline k, see the r1 attribution row in AGENTS.md).

Usage:
  python3 tools/probe_kdist.py --ours ./ckpt/ours_q05_full \
      --modd /tmp/modd_q05 --rt /tmp/rt_q05 --mdf ./ckpt/mdf_q05 \
      --n 100 --out .logs/0907_0000_kdist.json
Outputs JSON (+ same-named .png, needs matplotlib): per-token total-k distribution/quantiles/
histogram, per-layer execution rates, five-model forward peak GPU memory, optional generation
comparison (dense vs ours).
"""
import argparse
import gc
import json
import os
import pathlib
import sys
import time

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from transformers import AutoModelForCausalLM, AutoTokenizer

from data.sft import SFTDataset, make_collate
from speaker import SpeakerConfig, convert_to_speaker
from speaker.metrics import per_token_correct, per_token_nll
from baselines.assemble import assemble  # noqa: E402
from baselines.lib import collect_mdf_stats, collect_modd_stats, eval_heldout_rt


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_id", default="/home/hdd/model/Qwen2.5-7B-Instruct",
                   help="same as the finetune-track default (7B)")
    p.add_argument("--data_path", default="./data/sft_t2t_mini.jsonl")
    p.add_argument("--ours", action="append", default=[],
                   help="ours ckpt (mod_config.json+gate.pt, may carry a full base), repeatable, tag=ours:<basename>")
    p.add_argument("--use_lora", default=True, action=argparse.BooleanOptionalAction,
                   help="LoRA wrapping for ours ckpts (modd/mdf read use_lora from their config first)")
    p.add_argument("--lora_rank", type=int, default=8, help="same as the train.py default")
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_targets", default="q_proj,v_proj")
    p.add_argument("--modd", default="", help="MoD ckpt (modd_config.json+routers.pt)")
    p.add_argument("--mdf", default="", help="MoDification ckpt (mdf_config.json+routers.pt)")
    p.add_argument("--rt", default="", help="RT ckpt (rt_config.json+routers.pt)")
    p.add_argument("--offset", type=int, default=1000)
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--max_len", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--gen_tokens", type=int, default=0,
                   help=">0 additionally runs a dense-vs-ours generation comparison (new tokens per prompt)")
    p.add_argument("--out", default=".logs/kdist.json")
    p.add_argument("--no_png", action="store_true")
    return p.parse_args()


def load_base(src, device):
    m = AutoModelForCausalLM.from_pretrained(
        src, dtype=torch.bfloat16, device_map=None,
        trust_remote_code=True, low_cpu_mem_usage=True).to("cpu")
    return m


def weights_gb(model):
    return sum(p.numel() * p.element_size() for p in model.parameters()) / 1e9


def fwd_eval(model, texts, coll, device, batch_size, k_fn=None, layer_fn=None):
    model.eval()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    sum_nll = n_tok = n_ok = 0.0
    ks, layer_sum, layer_n = [], {}, {}
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            b = coll(texts[i:i + batch_size])
            out = model(**b)
            logits = out.logits.float()
            valid = b["attention_mask"].bool()
            sum_nll += per_token_nll(logits, b["labels"])[valid].sum().item()
            n_tok += valid.sum().item()
            n_ok += per_token_correct(logits, b["labels"])[valid].sum().item()
            if k_fn is not None:
                kv = k_fn(valid)
                if kv is not None:
                    ks.extend(kv.tolist())
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
    b2 = ax.bar([short(n) for n in names], [p - wt for p, wt in zip(peaks, wts)], bottom=wts,
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


def main():
    args = parse_args()
    device = args.device
    tok = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    full = SFTDataset(args.data_path, args.offset + args.n, tok=None, use_chat=False)
    texts = full.samples[args.offset:args.offset + args.n]
    assert texts, "the eval slice is empty"
    coll = make_collate(tok, device, args.max_len, False, True)
    out = {"meta": vars(args)}

    m = load_base(args.model_id, device).to(device)
    out["dense"] = {"weights_gb": weights_gb(m)}
    out["dense"].update(fwd_eval(m, texts, coll, device, args.batch_size))
    print(f"[dense] loss {out['dense']['loss']:.3f} acc {out['dense']['acc']:.3f} "
          f"peak {out['dense']['peak_gb']:.2f}GB", flush=True)
    del m
    gc.collect()  # 7B: patch/peft reference cycles need gc, otherwise the next model load OOMs
    torch.cuda.empty_cache()

    for ckpt in args.ours:
        tag = f"ours:{os.path.basename(ckpt.rstrip('/'))}"
        asm = assemble(ckpt, args.model_id, args, device, skip_mode=None)
        w = asm.model
        m = w
        oc = SpeakerConfig.from_json(os.path.join(ckpt, "mod_config.json"))
        n_always = len(oc.always_on_layers)
        gp = sum(p.numel() for p in w.get_router_parameters())
        out[tag] = {"weights_gb": weights_gb(m), "gate_params": gp,
                    "gate_share": gp / sum(p.numel() for p in m.parameters())}

        def ours_k(valid, _w=w):
            c = _w.get_active_counts()
            return (c[valid] + n_always).float() if c is not None else None

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

        out[tag].update(fwd_eval(w, texts, coll, device, args.batch_size, ours_k, ours_layers))
        k = out[tag]["k_total"]
        print(f"[{tag}] k_total {k['mean']:.1f}±{k['std']:.1f} "
              f"q={['%.1f' % v for v in k['q01_10_25_50_75_90_100']]}", flush=True)
        del ours_k, ours_layers, w, m  # the closure holds w via the _w default arg; without del, the next round's to(device) OOMs
        gc.collect()
        torch.cuda.empty_cache()

    if args.modd:
        asm = assemble(args.modd, args.model_id, args, device)
        m, routed, mc = asm.model, asm.routed, asm.cfg
        n_dense = asm.n_dense
        r_idx = [i for i, r in enumerate(mc["is_routed"]) if r]
        out["modd"] = {"weights_gb": weights_gb(m)}

        def modd_k(valid):
            _, kk = collect_modd_stats(routed, training=False)
            return (kk[valid] + n_dense).float() if kk is not None else None

        def modd_layers(valid):
            nv = valid.sum().item()
            for j, p in enumerate(routed):
                sel = getattr(p, "_last_sel", None)
                if sel is not None:
                    yield r_idx[j] if j < len(r_idx) else j, float(sel[valid].float().mean()), nv
            for i, r in enumerate(mc["is_routed"]):
                if not r:
                    yield i, 1.0, nv

        out["modd"].update(fwd_eval(m, texts, coll, device, args.batch_size, modd_k, modd_layers))
        k = out["modd"]["k_total"]
        print(f"[modd] k_total {k['mean']:.1f}±{k['std']:.1f}", flush=True)
        del modd_k, modd_layers, routed, m  # the closure holds the model via routed; without del, the next round OOMs
        gc.collect()
        torch.cuda.empty_cache()

    if args.mdf:
        asm = assemble(args.mdf, args.model_id, args, device)
        m, routed, mc = asm.model, asm.routed, asm.cfg
        n_dense = asm.n_dense
        r_idx = [i for i, r in enumerate(mc["is_routed"]) if r]
        out["mdf"] = {"weights_gb": weights_gb(m)}

        def mdf_k(valid):
            _, kk = collect_mdf_stats(routed, training=False)
            return (kk[valid] + n_dense).float() if kk is not None else None

        def mdf_layers(valid):
            nv = valid.sum().item()
            for j, p in enumerate(routed):
                sel = getattr(p, "_last_sel", None)
                if sel is not None:
                    yield r_idx[j] if j < len(r_idx) else j, float(sel[valid].float().mean()), nv
            for i, r in enumerate(mc["is_routed"]):
                if not r:
                    yield i, 1.0, nv

        out["mdf"].update(fwd_eval(m, texts, coll, device, args.batch_size, mdf_k, mdf_layers))
        k = out["mdf"]["k_total"]
        print(f"[mdf] k_total {k['mean']:.1f}±{k['std']:.1f}", flush=True)
        del mdf_k, mdf_layers, routed, m
        gc.collect()
        torch.cuda.empty_cache()

    if args.rt:
        asm = assemble(args.rt, args.model_id, args, device)
        m, gated = asm.model, asm.gated
        rc = asm.cfg
        r = eval_heldout_rt(m, gated, texts, coll, args.batch_size, n_always=asm.n_always)
        torch.cuda.reset_peak_memory_stats(device)
        with torch.no_grad():
            for i in range(0, len(texts), args.batch_size):
                m(**coll(texts[i:i + args.batch_size]))
        out["rt"] = {"loss": r["loss"], "acc": r["acc"], "exec_rate": r["exec_rate"],
                     "weights_gb": weights_gb(m),
                     "peak_gb": torch.cuda.max_memory_allocated(device) / 1e9}
        print(f"[rt] exec {r['exec_rate']:.2f}", flush=True)
        del gated, m
        gc.collect()
        torch.cuda.empty_cache()

    if args.gen_tokens > 0 and args.ours:
        prompts = ["The future development trends of artificial intelligence are",
                   "Explain what photosynthesis is:",
                   "Write a short poem about spring:", "The summation process of 1+2+...+100 is",
                   "What are Lu Xun's representative works? Please list them"]
        gen = {}
        for name, src, wrap in (("dense", args.model_id, False), ("ours", args.ours, True)):
            m = load_base(src, device)
            if wrap:
                oc = SpeakerConfig.from_json(os.path.join(args.ours, "mod_config.json"))
                w = convert_to_speaker(m, oc)
                w.load_state_dict(torch.load(os.path.join(args.ours, "gate.pt"),
                                             map_location="cpu"), strict=False)
                for mode in ("soft", "hard"):
                    w.set_skip_mode(mode)
                    w.get_skip_hits(reset=True)
                    mm = w.to(device).eval()
                    torch.cuda.reset_peak_memory_stats(device)
                    t0 = time.time()
                    n_new, outs = 0, []
                    with torch.no_grad():
                        for p in prompts:
                            ids = tok(p, return_tensors="pt").input_ids.to(device)
                            o = mm.generate(ids, max_new_tokens=args.gen_tokens, do_sample=False,
                                            use_cache=True, pad_token_id=tok.eos_token_id)
                            n_new += o.shape[1] - ids.shape[1]
                            outs.append(tok.decode(o[0, ids.shape[1]:]))
                    dt = time.time() - t0
                    gen[f"ours-{mode}"] = {"sec": dt, "tok_per_s": n_new / max(dt, 1e-6),
                                           "peak_gb": torch.cuda.max_memory_allocated(device) / 1e9,
                                           "skip_hits": mm.get_skip_hits(),
                                           "samples": [s[:150] for s in outs]}
                    print(f"[ours-{mode}] {gen[f'ours-{mode}']['tok_per_s']:.1f} tok/s "
                          f"peak {gen[f'ours-{mode}']['peak_gb']:.3f}GB "
                          f"skips {gen[f'ours-{mode}']['skip_hits']}", flush=True)
                del w, m
                gc.collect()
            else:
                m = m.to(device).eval()
                torch.cuda.reset_peak_memory_stats(device)
                t0 = time.time()
                n_new, outs = 0, []
                with torch.no_grad():
                    for p in prompts:
                        ids = tok(p, return_tensors="pt").input_ids.to(device)
                        o = m.generate(ids, max_new_tokens=args.gen_tokens, do_sample=False,
                                       use_cache=True, pad_token_id=tok.eos_token_id)
                        n_new += o.shape[1] - ids.shape[1]
                        outs.append(tok.decode(o[0, ids.shape[1]:]))
                dt = time.time() - t0
                gen["dense"] = {"sec": dt, "tok_per_s": n_new / max(dt, 1e-6),
                                "peak_gb": torch.cuda.max_memory_allocated(device) / 1e9,
                                "samples": [s[:150] for s in outs]}
                print(f"[dense] {gen['dense']['tok_per_s']:.1f} tok/s "
                      f"peak {gen['dense']['peak_gb']:.3f}GB", flush=True)
                del m
                gc.collect()
            torch.cuda.empty_cache()
        out["generate"] = gen

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print(f"saved {args.out}", flush=True)
    try:
        png = None if args.no_png else plot(out, os.path.splitext(args.out)[0] + ".png")
        print(f"saved {png}", flush=True)
    except ImportError:
        print("matplotlib unavailable, skipping plot", flush=True)


if __name__ == "__main__":
    main()
