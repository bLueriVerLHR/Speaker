"""Phase 0 probe: frozen MoD ckpt (hard mode), per-token k statistics by corpus type x word class.

Corpora (three-way split):
  syntax    ZhoBLiMP-good sentences + BLiMP-good sentences (grammatical sentences from minimal pairs)
  reasoning C-Eval HARD val question stems + GSM8K question (reasoning text)
  control   in-distribution SFT held-out slice
Word-class heuristic (lives only in this script, not in mod/):
  function  Chinese/English function words; punct gets its own bucket; everything else is content
headline: k_gap = mean_k(reasoning non-punct) - mean_k(syntax non-punct)
"""
import argparse
import glob
import json
import os
import pathlib
import sys
import unicodedata

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from speaker import SpeakerConfig, convert_to_speaker

MOD_SYN = "./data/mod_syntax"

# ---- word-class heuristic (script-local, unrelated to mod/) ----
_ZH_FUNC_CHARS = set(
    "的了着过地得在和与或把被对就才都也而但若如所之乎者也吗呢吧啊呀么每各该此其这那哪谁怎很太非最更比稍都全共连给让使由往向到自从当用以及跟同亦乃即则却且或"
)
_ZH_FUNC_WORDS = {
    "可以", "应该", "因为", "所以", "然后", "但是", "如果", "虽然", "尽管", "以及",
    "或者", "还是", "就是", "不是", "没有", "我们", "你们", "他们", "她们", "它们",
    "自己", "这个", "那个", "这些", "那些", "这里", "那里", "怎么", "什么", "为什么",
    "如何", "多少", "一样", "一起", "非常", "比较", "只有", "为了", "关于", "对于",
}
_EN_STOP = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being", "am",
    "to", "of", "and", "or", "in", "on", "at", "by", "for", "with", "as",
    "from", "that", "this", "it", "its", "he", "she", "they", "them", "his",
    "her", "their", "we", "you", "i", "my", "our", "your", "not", "no",
    "do", "does", "did", "done", "have", "has", "had", "having", "will",
    "would", "can", "could", "should", "may", "might", "must", "shall",
    "what", "which", "who", "whom", "whose", "when", "where", "how", "why",
    "there", "here", "than", "then", "so", "such", "only", "also", "very",
    "more", "most", "other", "some", "any", "each", "every", "all", "both",
    "either", "neither", "between", "into", "through", "during", "about",
    "against", "among", "off", "over", "under", "again", "once", "s",
    "t", "d", "ll", "m", "re", "ve",
}


def classify_token(surface: str) -> str:
    """token surface -> function/content/punct. Qwen BPE carries a Ġ prefix, strip it first."""
    s = surface.lstrip("Ġ").lstrip("▁").strip()
    if not s:
        return "punct"
    if all(unicodedata.category(ch).startswith("P") or ch.isspace() for ch in s):
        return "punct"
    if all("\u4e00" <= ch <= "\u9fff" for ch in s):
        if s in _ZH_FUNC_WORDS or (len(s) <= 2 and all(ch in _ZH_FUNC_CHARS for ch in s)):
            return "function"
        return "content"
    low = s.lower()
    if low in _EN_STOP:
        return "function"
    if len(s) == 1 and s in _ZH_FUNC_CHARS:
        return "function"
    return "content"


def load_syntax(n_per_file=2, max_files=30):
    texts = []
    files = sorted(glob.glob(f"{MOD_SYN}/ZhoBLiMP/data/ZhoBLiMP/*.jsonl"))[:max_files]
    files += sorted(glob.glob(f"{MOD_SYN}/BLiMP/data/*.jsonl"))[:max_files]
    for fp in files:
        got = 0
        with open(fp, encoding="utf-8") as f:
            for line in f:
                if got >= n_per_file:
                    break
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                s = obj.get("sentence_good", "")
                if len(s) >= 4:
                    texts.append(s)
                    got += 1
    return texts


def load_reason(n_ceval=60, n_gsm=60):
    import pyarrow.parquet as pq  # noqa: WPS433 (bundled with the cuda env, readable without pandas)

    texts = []
    for fp in sorted(glob.glob(f"{MOD_SYN}/C-Eval/val/*.parquet")):
        t = pq.read_table(fp).to_pylist()
        for r in t[: max(1, n_ceval // 8)]:
            q = f"{r['question']} A.{r['A']} B.{r['B']} C.{r['C']} D.{r['D']}"
            texts.append(q)
    with open(f"{MOD_SYN}/GSM8K/grade_school_math/data/train.jsonl", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "question" in obj:
                texts.append(obj["question"])
            if len(texts) >= n_ceval + n_gsm:
                break
    return texts


def load_control(path, offset, n):
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    from data.sft import SFTDataset

    full = SFTDataset(path, offset + n)
    return full.samples[offset:offset + n]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_id", default="/home/hdd/model/Qwen2.5-7B-Instruct")
    p.add_argument("--ckpt", default="/tmp/mod_ckpt_qwen_kl2")
    p.add_argument("--sft_path", default="./data/sft_t2t_mini.jsonl")
    p.add_argument("--sft_offset", type=int, default=2000)
    p.add_argument("--n_syntax", type=int, default=120, help="total number of syntax sentences (half Chinese, half English)")
    p.add_argument("--n_ceval", type=int, default=60)
    p.add_argument("--n_gsm", type=int, default=60)
    p.add_argument("--n_control", type=int, default=40)
    p.add_argument("--max_len", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out", default="/tmp/token_k_probe.json")
    return p.parse_args()


def run_corpus(model, tok, texts, max_len, batch_size, device):
    """Returns {bucket: [sum_k, n]}."""
    agg = {b: [0.0, 0] for b in ("function", "content", "punct")}
    n_texts = 0
    model.eval()
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc = tok(batch, truncation=True, max_length=max_len, padding=True,
                      return_tensors="pt")
            input_ids = enc["input_ids"].to(device)
            attn = enc["attention_mask"].to(device)
            out = model(input_ids=input_ids, attention_mask=attn)
            k = model.get_active_counts(hard=True)  # [B,T]
            if k is None:
                raise RuntimeError("get_active_counts() returned None")
            for bi in range(input_ids.shape[0]):
                L = int(attn[bi].sum())
                for t in range(L):
                    tid = int(input_ids[bi, t])
                    bucket = classify_token(tok.decode([tid]))
                    agg[bucket][0] += float(k[bi, t])
                    agg[bucket][1] += 1
            n_texts += len(batch)
    return agg, n_texts


def summarize(agg):
    return {b: {"mean_k": (s / n if n else 0.0), "n": n} for b, (s, n) in agg.items()}


def main():
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        print(f"Using {device} free {torch.cuda.mem_get_info(device)[0] / 1024**3:.1f}GB",
              flush=True)
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tok = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    syn = load_syntax()[:args.n_syntax]
    rea = load_reason(args.n_ceval, args.n_gsm)[:args.n_ceval + args.n_gsm]
    ctl = load_control(args.sft_path, args.sft_offset, args.n_control)
    print(f"syntax {len(syn)} reasoning {len(rea)} control {len(ctl)}", flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, dtype=torch.bfloat16, device_map=None,
        trust_remote_code=True, low_cpu_mem_usage=True)
    cfg = SpeakerConfig.from_json(os.path.join(args.ckpt, "mod_config.json"))
    model = convert_to_speaker(model, cfg).to(device)
    sd = torch.load(os.path.join(args.ckpt, "gate.pt"), map_location="cpu")
    missing, unexp = model.load_state_dict(sd, strict=False)
    print(f"gate.pt loaded, missing {len(missing)} unexpected {len(unexp)}", flush=True)
    model.set_skip_mode("hard")

    res = {}
    for tag, texts in (("syntax", syn), ("reasoning", rea), ("control", ctl)):
        agg, n = run_corpus(model, tok, texts, args.max_len, args.batch_size, device)
        res[tag] = summarize(agg)
        r = res[tag]
        print(f"[{tag}] n_texts={n} " + " ".join(
            f"{b} k={r[b]['mean_k']:.2f}(n={r[b]['n']})" for b in ("function", "content", "punct")),
            flush=True)

    def mean_nopunct(tag):
        f, c = res[tag]["function"], res[tag]["content"]
        tot = f["n"] + c["n"]
        return (f["mean_k"] * f["n"] + c["mean_k"] * c["n"]) / max(tot, 1)

    k_gap = mean_nopunct("reasoning") - mean_nopunct("syntax")
    res["k_gap"] = k_gap
    print(f"k_gap(reasoning-syntax, non-punct) = {k_gap:+.3f}", flush=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    print(f"saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
