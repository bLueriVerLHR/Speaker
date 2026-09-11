"""Generation-side deployment validation: dense vs five-way (ours hard-skip / MoD / RT /
MoDification), dataset continuation comparison.
prompt = first 64 tokens of the text, reference = the next 64 tokens, greedy generation of 64 tokens.
Metrics: ROUGE-L (F1, self-implemented, no deps) / repetition rate / total time / peak GPU
memory / per-layer usage during generation.
Ours uses hard layer skipping (as defined by the method); baselines use their native forward
(dense execution + masking, same as training).
Usage:
  python3 tools/eval_gen.py --ours ./ckpt/ours_q05_full \
      --modd /tmp/modd_q05 --rt /tmp/rt_q05 --mdf ./ckpt/mdf_q05 --out /tmp/gen5.json
"""
import argparse, gc, json, os, pathlib, sys, time, re
import torch
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from transformers import AutoTokenizer, AutoModelForCausalLM
from speaker import SpeakerConfig, convert_to_speaker
from baselines.assemble import assemble  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_id", default="/home/hdd/model/Qwen2.5-7B-Instruct",
                   help="same as the finetune-track default (7B)")
    p.add_argument("--ckpt", default="", help="ours ckpt (legacy flag name, synonym of --ours, empty to skip)")
    p.add_argument("--ours", action="append", default=[],
                   help="ours ckpt (mod_config.json+gate.pt), repeatable, tag=ours:<basename>")
    p.add_argument("--modd", action="append", default=[],
                   help="MoD ckpt (modd_config.json+routers.pt), repeatable")
    p.add_argument("--rt", default="", help="RT ckpt (rt_config.json+routers.pt)")
    p.add_argument("--mdf", default="", help="MoDification ckpt (mdf_config.json+routers.pt)")
    p.add_argument("--dense_ft", action="append", default=[],
                   help="dense fine-tuned ckpt (denseft_config.json+lora.pt), repeatable")
    p.add_argument("--data_path", default="./data/sft_t2t_mini.jsonl")
    p.add_argument("--offset", type=int, default=5000, help="skip the first N rows already used by training/held-out")
    p.add_argument("--n", type=int, default=30)
    p.add_argument("--prompt_len", type=int, default=64)
    p.add_argument("--ref_len", type=int, default=64)
    p.add_argument("--max_new", type=int, default=64)
    p.add_argument("--temp", type=float, default=0.7, help="sampling temperature for the stability formulation")
    p.add_argument("--rep_penalty", type=float, default=1.0,
                   help="decode-time repetition penalty (>1 penalizes already-seen tokens, 1.0=off)")
    p.add_argument("--no_repeat_ngram", type=int, default=0,
                   help="decode-time n-gram hard ban (>0 forbids repeating any n-gram from prompt/output)")
    p.add_argument("--use_ckpt_decode", action="store_true",
                   help="ours ckpts: use the ckpt-shipped decode recipe (mod_config.json decode block, "
                        "ed9/C) instead of --rep_penalty/--no_repeat_ngram; other families and "
                        "ckpts without a block keep CLI values (warns loudly on fallback)")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--use_lora", default=True, action=argparse.BooleanOptionalAction,
                   help="must be on when the ckpt contains LoRA, otherwise lora_ weights in gate.pt are silently dropped by strict=False")
    p.add_argument("--lora_rank", type=int, default=8, help="same as the train.py default")
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_targets", default="q_proj,v_proj")
    p.add_argument("--resident", default="", help="GPU-resident layers, comma separated; empty = all layers on GPU. e.g. 0,1,15,16,17,19,22,23")
    p.add_argument("--out", default="/tmp/gen_eval.json")
    p.add_argument("--no_png", action="store_true")
    return p.parse_args()

def load_texts(path, offset, n):
    texts = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i < offset: continue
            if len(texts) >= n: break
            line = line.strip()
            if not line: continue
            try: obj = json.loads(line)
            except: continue
            if "conversations" in obj:
                t = "\n".join(f"{m['role']}: {m['content']}" for m in obj["conversations"])
            elif "text" in obj: t = obj["text"]
            else: t = str(obj)
            if len(t) >= 100: texts.append(t)
    return texts

def toks(s):
    return re.findall(r"[A-Za-z0-9_]+|[^\sA-Za-z0-9_]", s)

def rouge_l_f1(ref, hyp, max_tok=128):
    r, h = toks(ref)[:max_tok], toks(hyp)[:max_tok]
    if not r or not h: return 0.0
    m, n = len(r), len(h)
    prev = [0]*(n+1)
    for i in range(1, m+1):
        cur = [0]*(n+1)
        ri = r[i-1]
        for j in range(1, n+1):
            cur[j] = prev[j-1]+1 if ri == h[j-1] else (prev[j] if prev[j] >= cur[j-1] else cur[j-1])
        prev = cur
    lcs = prev[n]
    p = lcs/max(len(h), 1); rr = lcs/max(len(r), 1)
    return 0.0 if p+rr == 0 else 2*p*rr/(p+rr)

def rep3_rate(s):
    g = [s[i:i+3] for i in range(max(len(s)-2, 0))]
    return 0.0 if not g else 1.0-len(set(g))/len(g)

def tok_f1(ref, hyp, max_tok=128):
    """token-level F1 (lexical split via toks(), Chinese by character) — the overlap between the
    generation and the reference answer, approximating "how much was answered correctly"."""
    from collections import Counter
    r, h = Counter(toks(ref)[:max_tok]), Counter(toks(hyp)[:max_tok])
    overlap = sum((r & h).values())
    if not r or not h or overlap == 0: return 0.0
    p, rc = overlap/max(sum(h.values()), 1), overlap/max(sum(r.values()), 1)
    return 2*p*rc/(p+rc)

def run_gen(model, tok, prompts, max_new, device, tag, temp=0.7,
            rep_penalty=1.0, no_repeat_ngram=0):
    """One greedy pass (quality/latency formulation) + one temperature-sampling pass (stability
    formulation: self-consistency ROUGE against greedy)."""
    model.eval()
    if hasattr(model, "set_skip_mode"): model.set_skip_mode("hard")
    if hasattr(model, "get_layer_usage"): model.get_layer_usage()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device); torch.cuda.synchronize(device)
    outs, n_new = [], 0
    mem_samples = []
    t0 = time.time()
    for p in prompts:
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
    outs_s = []
    for p in prompts:
        enc = tok(p, return_tensors="pt", truncation=True, max_length=256).to(device)
        plen = enc["input_ids"].shape[1]
        with torch.no_grad():
            g = model.generate(**enc, max_new_tokens=max_new, do_sample=True,
                               temperature=temp, top_p=0.95,
                               repetition_penalty=rep_penalty,
                               no_repeat_ngram_size=no_repeat_ngram,
                               pad_token_id=tok.pad_token_id, use_cache=True)
        outs_s.append(tok.decode(g[0][plen:], skip_special_tokens=True))
    usage = model.get_layer_usage() if hasattr(model, "get_layer_usage") else None
    print(f"[{tag}] {len(prompts)} prompts greedy {dt:.1f}s ({dt*1000/max(n_new,1):.0f}ms/tok) "
          f"peak {peak:.2f}GB avg {avg:.2f}GB", flush=True)
    if hasattr(model, "layers"):
        hits = getattr(model, "get_skip_hits", lambda: sum(getattr(w, "_skip_hits", 0) for w in model.layers))()
        print(f"[{tag}] sparse_skips {hits}", flush=True)
    return outs, outs_s, dt, peak, avg, usage, n_new

def load_base(model_id, device):
    m = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map=None,
        trust_remote_code=True, low_cpu_mem_usage=True).to("cpu")
    return m


def load_ours(args, tok, device, ckpt):
    asm = assemble(ckpt, args.model_id, args, device)
    mod = asm.model
    if args.resident.strip():
        resident = mod.set_placement([int(x) for x in args.resident.split(",") if x.strip() != ""],
                                     gpu_device=device, cpu_device="cpu")
        print(f"placement resident {resident} gpu-GB {mod.resident_gb('cuda'):.2f} "
              f"(full {sum(p.numel()*p.element_size() for p in mod.parameters())/1e9:.2f})", flush=True)
    return mod


def load_modd(args, device, ckpt):
    return assemble(ckpt, args.model_id, args, device).model


def load_mdf(args, device, ckpt):
    return assemble(ckpt, args.model_id, args, device).model


def load_dense_ft(args, device, ckpt):
    return assemble(ckpt, args.model_id, args, device).model


def load_rt(args, device, ckpt):
    return assemble(ckpt, args.model_id, args, device).model


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
    specs = [("rouge", "ROUGE-L (higher=better)", "{:.3f}"),
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


def main():
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda": torch.cuda.set_device(device)
    tok = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    texts = load_texts(args.data_path, args.offset, args.n)
    print(f"texts {len(texts)}", flush=True)
    enc = tok(texts, truncation=True, max_length=args.prompt_len+args.ref_len,
              padding=True, return_tensors="pt")
    L = enc["attention_mask"].sum(1)
    prompts, refs = [], []
    for bi in range(len(texts)):
        ids = enc["input_ids"][bi, :L[bi]].tolist()
        if len(ids) <= args.prompt_len+8: continue
        prompts.append(tok.decode(ids[:args.prompt_len], skip_special_tokens=True))
        refs.append(tok.decode(ids[args.prompt_len:args.prompt_len+args.ref_len], skip_special_tokens=True))
    print(f"pairs {len(prompts)}", flush=True)

    queue = [("dense", None)]
    for c in (list(args.ours) + ([args.ckpt] if args.ckpt else [])):
        queue.append((f"ours:{os.path.basename(c.rstrip('/'))}", c))
    for c in args.modd:
        queue.append((f"modd:{os.path.basename(c.rstrip('/'))}", c))
    if args.rt: queue.append(("rt", args.rt))
    if args.mdf: queue.append(("mdf", args.mdf))
    for c in args.dense_ft:
        queue.append((f"denseft:{os.path.basename(c.rstrip('/'))}", c))
    loaders = {"ours": load_ours, "modd": load_modd, "mdf": load_mdf, "rt": load_rt,
               "denseft": load_dense_ft}

    methods, rows = {}, []
    for tag, ckpt in queue:
        rp, ng, dec_src = args.rep_penalty, args.no_repeat_ngram, "cli"
        if tag.startswith("ours") and args.use_ckpt_decode:
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
                print(f"[{tag}] WARNING: --use_ckpt_decode but no decode block in "
                      f"mod_config.json — falling back to CLI", flush=True)
        if dec_src != "cli":
            print(f"[{tag}] decode from {dec_src}", flush=True)
        if tag == "dense":
            m = load_base(args.model_id, device).to(device)
        elif tag.startswith("ours"):
            m = loaders["ours"](args, tok, device, ckpt)
        else:
            m = loaders[tag if ":" not in tag else tag.split(":")[0]](args, device, ckpt)
        outs, outs_s, dt, peak, avg, usage, n_new = run_gen(
            m, tok, prompts, args.max_new, device, tag, args.temp,
            rep_penalty=rp, no_repeat_ngram=ng)
        stab = sum(rouge_l_f1(a, b) for a, b in zip(outs, outs_s))/max(len(outs), 1)
        methods[tag] = {"outs": outs, "time": dt, "peak_gb": peak, "avg_gb": avg,
                        "ms_per_tok": dt*1000/max(n_new, 1), "stability": stab,
                        "decode_used": {"rep_penalty": rp, "no_repeat_ngram": ng,
                                        "src": dec_src},
                        "usage": ({k: round(v[0], 3) for k, v in sorted(usage.items())}
                                  if usage else None)}
        print(f"[{tag}] done", flush=True)
        del m
        if device.type == "cuda":
            gc.collect()  # 7B: patch/peft reference cycles need gc, otherwise the next load OOMs
            torch.cuda.empty_cache()

    for i, ref in enumerate(refs):
        r = {"ref": ref}
        for tag in methods:
            o = methods[tag]["outs"][i]
            r[tag] = o
            r[f"rouge_{tag}"] = rouge_l_f1(ref, o)
            r[f"f1_{tag}"] = tok_f1(ref, o)
            r[f"rep_{tag}"] = rep3_rate(o)
        rows.append(r)
    def avg(k): return sum(r[k] for r in rows)/max(len(rows), 1)
    for tag in methods:
        methods[tag]["rouge"] = avg(f"rouge_{tag}")
        methods[tag]["f1"] = avg(f"f1_{tag}")
        methods[tag]["rep"] = avg(f"rep_{tag}")
        del methods[tag]["outs"]
    summary = {"n": len(rows), "temp": args.temp,
               "rep_penalty": args.rep_penalty, "no_repeat_ngram": args.no_repeat_ngram,
               "methods": methods, "rows": rows}
    print(f"{'method':22s} {'ROUGE':>6s} {'tokF1':>6s} {'rep3':>6s} {'stab':>6s} "
          f"{'ms/tok':>7s} {'time':>7s} {'peak':>6s}", flush=True)
    for tag in methods:
        s = methods[tag]
        print(f"{tag:22s} {s['rouge']:6.3f} {s['f1']:6.3f} {s['rep']:6.3f} "
              f"{s['stability']:6.3f} {s['ms_per_tok']:7.0f} {s['time']:7.1f} "
              f"{s['peak_gb']:6.2f}", flush=True)
    for r in rows[:2]:
        print(f"REF   {r['ref'][:100]}", flush=True)
        for tag in methods:
            print(f"{tag.upper()[:5]:5s} {r[tag][:100]}", flush=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    print(f"saved {args.out}", flush=True)
    try:
        if not args.no_png:
            print(f"saved {plot_gen(summary, os.path.splitext(args.out)[0] + '.png')}", flush=True)
    except ImportError:
        print("matplotlib unavailable, skipping plot", flush=True)

if __name__ == "__main__": main()
