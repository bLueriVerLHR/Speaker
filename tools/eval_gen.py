"""Generation-side deployment validation: dense vs five-way (ours hard-skip / MoD / RT /
MoDification), dataset continuation comparison.
prompt = first 64 tokens of the text, reference = the next 64 tokens, greedy generation.
Metrics (tools/gen_metrics.py, standard implementations only): ROUGE-L F1
(google rouge-score) / seq-rep-4 (Welleck Eq.10) / total time / peak GPU
memory / per-layer usage during generation.
Ours uses hard layer skipping (as defined by the method); baselines use their native forward
(dense execution + masking, same as training).
Usage:
  python3 tools/eval_gen.py --ours ./ckpt/ours_q05_full \
      --modd /tmp/modd_q05 --rt /tmp/rt_q05 --mdf ./ckpt/mdf_q05 --out /tmp/gen5.json
"""
import json, os, pathlib, sys, time
from typing import Annotated

import torch
import typer
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from transformers import AutoTokenizer
from baselines.assemble import assemble  # noqa: E402
from tools.gen_metrics import rouge_l, seq_rep  # noqa: E402 (standard metrics; single source)
from speaker.log import logger  # noqa: E402
from tools._common import collect_gc, dump_json, module_gb  # noqa: E402

app = typer.Typer(add_completion=False)

def load_texts(path, offset, n):
    texts = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i < offset: continue
            if len(texts) >= n: break
            line = line.strip()
            if not line: continue
            try: obj = json.loads(line)
            except json.JSONDecodeError: continue
            if "conversations" in obj:
                t = "\n".join(f"{m['role']}: {m['content']}" for m in obj["conversations"])
            elif "text" in obj: t = obj["text"]
            else: t = str(obj)
            if len(t) >= 100: texts.append(t)
    return texts


def run_gen(model, tok, prompts, max_new, device, tag,
            rep_penalty=1.0, no_repeat_ngram=0):
    """Single greedy pass (quality/latency formulation)."""
    model.eval()
    if hasattr(model, "set_skip_mode"): model.set_skip_mode("hard")
    if hasattr(model, "get_layer_usage"): model.get_layer_usage()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device); torch.cuda.synchronize(device)
    outs, n_new = [], 0
    mem_samples = []
    t0 = time.time()
    from speaker.terminal import track
    for p in track(prompts, total=len(prompts), desc=f"{tag} greedy"):
        enc = tok(p, return_tensors="pt", truncation=True, max_length=256).to(device)
        plen = enc["input_ids"].shape[1]
        with torch.no_grad():
            g = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                               repetition_penalty=rep_penalty,
                               no_repeat_ngram_size=no_repeat_ngram,
                               pad_token_id=tok.pad_token_id, use_cache=True)
        outs.append(tok.decode(g[0][plen:], skip_special_tokens=True))
        n_new += g.shape[1] - plen
        if device.type == "cuda":
            mem_samples.append(torch.cuda.memory_allocated(device) / 1e9)
    if device.type == "cuda": torch.cuda.synchronize(device)
    dt = time.time()-t0
    peak = torch.cuda.max_memory_allocated(device)/1e9 if device.type == "cuda" else 0.0
    avg = sum(mem_samples)/max(len(mem_samples), 1) if mem_samples else 0.0
    usage = model.get_layer_usage() if hasattr(model, "get_layer_usage") else None
    logger.info(f"[{tag}] {len(prompts)} prompts greedy {dt:.1f}s ({dt*1000/max(n_new,1):.0f}ms/tok) "
                f"peak {peak:.2f}GB avg {avg:.2f}GB")
    if hasattr(model, "layers"):
        hits = getattr(model, "get_skip_hits", lambda: sum(getattr(w, "_skip_hits", 0) for w in model.layers))()
        logger.info(f"[{tag}] sparse_skips {hits}")
    return outs, dt, peak, avg, usage, n_new

def load_base(model_id, device, device_map=None):
    from speaker.train_common import build_model
    m = build_model(model_id, torch.device("cpu"), dtype=torch.bfloat16,
                    device_map=device_map)
    if device_map is None:
        m = m.to("cpu")
    return m


def load_ours(model_id, tok, device, ckpt, lora, device_map="", resident=""):
    dm = device_map or None  # "auto" = base sharded across visible GPUs (pass device=None)
    asm = assemble(ckpt, model_id, lora, None if dm else device,
                   device_map=dm)
    mod = asm.model
    placed = "all-GPU"
    if resident.strip():
        if dm:
            raise ValueError("--resident (static manual placement) is incompatible with "
                             "--device_map auto (use tools/edge_bench.py for budgeted placement)")
        placed = mod.set_placement([int(x) for x in resident.split(",") if x.strip() != ""],
                                   gpu_device=device, cpu_device="cpu")
    logger.info(f"placement resident {placed} gpu-GB {mod.resident_gb('cuda'):.2f} "
                f"(full {module_gb(mod):.2f})")
    return mod


def load_family(model_id, device, ckpt, lora, device_map=""):
    dm = device_map or None
    return assemble(ckpt, model_id, lora, None if dm else device,
                    device_map=dm).model


def plot_gen(summary, png_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def color(m):
        for tag, c in (("ours", "#1f77b4"), ("modd", "#ff7f0e"),
                       ("rt", "#2ca02c"), ("mdf", "#9467bd")):
            if m.startswith(tag):
                return c
        return "#7f7f7f"

    methods = list(summary["methods"].keys())  # queue order, dense first
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.0))
    specs = [("rougeL", "ROUGE-L (higher=better)", "{:.3f}"),
             ("time", "gen time s (lower=better)", "{:.1f}"),
             ("peak_gb", "gen peak GB (lower=better)", "{:.2f}")]
    for ax, (key, title, fmt) in zip(axes, specs):
        vals = [summary["methods"][m][key] for m in methods]
        b = ax.bar(range(len(methods)), vals, color=[color(m) for m in methods], width=0.6)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels([m.replace("ours:", "o:") for m in methods],
                           rotation=20, ha="right", fontsize=9)
        ax.set_title(title, fontsize=11)
        ax.grid(axis="y", alpha=0.3)
        for rect, v in zip(b, vals):
            ax.annotate(fmt.format(v), (rect.get_x() + rect.get_width() / 2, rect.get_height()),
                        ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(png_path, dpi=130)
    plt.close(fig)
    return png_path


@app.command()
def main(
    model_id: Annotated[str, typer.Option("--model_id", help="same as the finetune-track default (7B)")] = "/home/hdd/model/Qwen2.5-7B-Instruct",
    ckpt: Annotated[str, typer.Option("--ckpt", help="ours ckpt (legacy flag name, synonym of --ours, empty to skip)")] = "",
    ours: Annotated[list[str], typer.Option("--ours", help="ours ckpt (mod_config.json+gate.pt), repeatable, tag=ours:<basename>")] = [],
    modd: Annotated[list[str], typer.Option("--modd", help="MoD ckpt (modd_config.json+routers.pt), repeatable")] = [],
    rt: Annotated[str, typer.Option("--rt", help="RT ckpt (rt_config.json+routers.pt)")] = "",
    mdf: Annotated[str, typer.Option("--mdf", help="MoDification ckpt (mdf_config.json+routers.pt)")] = "",
    dense_ft: Annotated[list[str], typer.Option("--dense_ft", help="dense fine-tuned ckpt (denseft_config.json+lora.pt), repeatable")] = [],
    data_path: Annotated[str, typer.Option("--data_path")] = "./data/sft_t2t_mini.jsonl",
    offset: Annotated[int, typer.Option("--offset", help="skip the first N rows already used by training/held-out")] = 5000,
    n: Annotated[int, typer.Option("--n")] = 30,
    prompt_len: Annotated[int, typer.Option("--prompt_len")] = 64,
    ref_len: Annotated[int, typer.Option("--ref_len")] = 64,
    max_new: Annotated[int, typer.Option("--max_new")] = 64,
    rep_penalty: Annotated[float, typer.Option("--rep_penalty", help="decode-time repetition penalty (>1 penalizes already-seen tokens, 1.0=off)")] = 1.0,
    no_repeat_ngram: Annotated[int, typer.Option("--no_repeat_ngram", help="decode-time n-gram hard ban (>0 forbids repeating any n-gram from prompt/output)")] = 0,
    use_ckpt_decode: Annotated[bool, typer.Option("--use_ckpt_decode", help="ours ckpts: use the ckpt-shipped decode recipe (mod_config.json decode block, ed9/C) instead of --rep_penalty/--no_repeat_ngram; other families and ckpts without a block keep CLI values (warns loudly on fallback)")] = False,
    device: Annotated[str, typer.Option("--device")] = "cuda:0",
    device_map: Annotated[str, typer.Option("--device_map", help="empty = whole-card (default); 'auto' = sharding across visible GPUs (backbones larger than one card; implies no --resident)")] = "",
    use_lora: Annotated[bool, typer.Option("--use_lora/--no-use_lora", help="LoRA wrapping switch (must match training; families adapt per ckpt config)")] = True,
    lora_rank: Annotated[int, typer.Option("--lora_rank", help="same as the train.py default")] = 8,
    lora_alpha: Annotated[int, typer.Option("--lora_alpha")] = 16,
    lora_targets: Annotated[str, typer.Option("--lora_targets")] = "q_proj,v_proj",
    resident: Annotated[str, typer.Option("--resident", help="GPU-resident layers, comma separated; empty = all layers on GPU. e.g. 0,1,15,16,17,19,22,23")] = "",
    out: Annotated[str, typer.Option("--out")] = "/tmp/gen_eval.json",
    no_png: Annotated[bool, typer.Option("--no_png")] = False,
) -> None:
    """Generation-side deployment validation: dense vs five-way continuation."""
    from speaker.terminal import setup_terminal
    setup_terminal()
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    texts = load_texts(data_path, offset, n)
    logger.info(f"texts {len(texts)}")
    enc = tok(texts, truncation=True, max_length=prompt_len + ref_len,
              padding=True, return_tensors="pt")
    L = enc["attention_mask"].sum(1)
    prompts, refs = [], []
    for bi in range(len(texts)):
        ids = enc["input_ids"][bi, :L[bi]].tolist()
        if len(ids) <= prompt_len + 8:
            continue
        prompts.append(tok.decode(ids[:prompt_len], skip_special_tokens=True))
        refs.append(tok.decode(ids[prompt_len:prompt_len + ref_len], skip_special_tokens=True))
    logger.info(f"pairs {len(prompts)}")
    lora = (dict(rank=lora_rank, alpha=lora_alpha, targets=lora_targets)
            if use_lora else None)

    queue = [("dense", None)]
    for c in (list(ours) + ([ckpt] if ckpt else [])):
        queue.append((f"ours:{os.path.basename(c.rstrip('/'))}", c))
    for c in modd:
        queue.append((f"modd:{os.path.basename(c.rstrip('/'))}", c))
    if rt:
        queue.append(("rt", rt))
    if mdf:
        queue.append(("mdf", mdf))
    for c in dense_ft:
        queue.append((f"denseft:{os.path.basename(c.rstrip('/'))}", c))
    loaders = {"ours": load_ours, "modd": load_family, "mdf": load_family, "rt": load_family,
               "denseft": load_family}

    methods, rows = {}, []
    for tag, ckpt in queue:
        rp, ng, dec_src = rep_penalty, no_repeat_ngram, "cli"
        if tag.startswith("ours") and use_ckpt_decode:
            try:
                with open(os.path.join(ckpt, "mod_config.json"), encoding="utf-8") as f:
                    dec = json.load(f).get("decode") or {}
            except OSError:
                dec = {}
            if dec:
                rp = dec.get("repetition_penalty", rp)
                ng = dec.get("no_repeat_ngram_size", ng)
                dec_src = f"ckpt:{dec}"
            else:
                logger.warning(f"[{tag}] --use_ckpt_decode but no decode block in "
                               f"mod_config.json — falling back to CLI")
        if dec_src != "cli":
            logger.info(f"[{tag}] decode from {dec_src}")
        if tag == "dense":
            m = load_base(model_id, device,
                          device_map=(device_map or None))
            if not device_map:
                m = m.to(device)
        elif tag.startswith("ours"):
            m = loaders["ours"](model_id, tok, device, ckpt, lora, device_map, resident)
        else:
            m = loaders[tag if ":" not in tag else tag.split(":")[0]](model_id, device, ckpt, lora, device_map)
        outs, dt, peak, avg, usage, n_new = run_gen(
            m, tok, prompts, max_new, device, tag,
            rep_penalty=rp, no_repeat_ngram=ng)
        methods[tag] = {"outs": outs, "time": dt, "peak_gb": peak, "avg_gb": avg,
                        "ms_per_tok": dt*1000/max(n_new, 1),
                        "decode_used": {"rep_penalty": rp, "no_repeat_ngram": ng,
                                        "src": dec_src},
                        "usage": ({k: round(v[0], 3) for k, v in sorted(usage.items())}
                                  if usage else None)}
        logger.info(f"[{tag}] done")
        del m
        if device.type == "cuda":
            collect_gc()  # 7B: patch/peft reference cycles need gc, otherwise the next load OOMs

    for i, ref in enumerate(refs):
        r = {"ref": ref}
        for tag in methods:
            o = methods[tag]["outs"][i]
            r[tag] = o
            r[f"rougeL_{tag}"] = rouge_l(ref, o)
            r[f"seq-rep-4_{tag}"] = seq_rep(o)
        rows.append(r)
    def avg(k): return sum(r[k] for r in rows)/max(len(rows), 1)
    for tag in methods:
        methods[tag]["rougeL"] = avg(f"rougeL_{tag}")
        methods[tag]["seq-rep-4"] = avg(f"seq-rep-4_{tag}")
        del methods[tag]["outs"]
    summary = {"n": len(rows),
                "rep_penalty": rep_penalty, "no_repeat_ngram": no_repeat_ngram,
                "methods": methods, "rows": rows}
    logger.info(f"{'method':22s} {'ROUGE-L':>7s} {'seq-rep-4':>9s} "
                f"{'ms/tok':>7s} {'time':>7s} {'peak':>6s}")
    for tag in methods:
        s = methods[tag]
        logger.info(f"{tag:22s} {s['rougeL']:7.3f} {s['seq-rep-4']:9.3f} "
                    f"{s['ms_per_tok']:7.0f} {s['time']:7.1f} "
                    f"{s['peak_gb']:6.2f}")
    for r in rows[:2]:
        logger.info(f"REF   {r['ref'][:100]}")
        for tag in methods:
            logger.info(f"{tag.upper()[:5]:5s} {r[tag][:100]}")
    dump_json(out, summary)
    logger.info(f"saved {out}")
    try:
        if not no_png:
            logger.info(f"saved {plot_gen(summary, os.path.splitext(out)[0] + '.png')}")
    except ImportError:
        logger.warning("matplotlib unavailable, skipping plot")

if __name__ == "__main__":
    app()
