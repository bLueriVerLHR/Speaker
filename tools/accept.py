"""Acceptance gate for one-pass finetuning (ed9/D): PASS/FAIL a ckpt against the
deployment contract and write manifest.json into the ckpt dir (self-describing artifact).

Tier-1 (always): hard-mode held-out Δacc vs raw dense + total-k budget.
Tier-2 (opt-in --gen_n>0): repetition at the ckpt's own decode recipe (mod_config.json
decode block, ed9/C) vs dense under the SAME lever.
Exit code 0 = PASS, 1 = FAIL (pipeline-friendly).
Usage: accept.py --ckpt CKPT [--gen_n 10] (rest: model/data slice + thresholds)."""
import hashlib
import json
import os
import pathlib
import sys
from typing import Annotated, Optional

import typer

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from data.sft import make_collate
from speaker.evaluate import eval_heldout
from speaker.log import logger
from tools.gen_metrics import seq_rep  # noqa: E402 (Welleck seq-rep-n)
from tools._common import collect_gc, dump_json
from speaker.train_common import (
    build_model,
    build_tok,
    eval_slice,
    resolve_device,
)

app = typer.Typer(add_completion=False)


def md5_file(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@app.command()
def main(
    model_id: Annotated[str, typer.Option("--model_id")] = "/home/hdd/model/Qwen2.5-7B-Instruct",
    data_path: Annotated[str, typer.Option("--data_path")] = "./data/sft_t2t_mini.jsonl",
    ckpt: Annotated[str, typer.Option("--ckpt")] = ...,
    offset: Annotated[int, typer.Option("--offset", help="acceptance slice (default: training heldout)")] = 904000,
    n: Annotated[int, typer.Option("--n")] = 100,
    max_len: Annotated[int, typer.Option("--max_len")] = 1024,
    batch_size: Annotated[int, typer.Option("--batch_size", help="bs1 = honest accounting (modd lesson)")] = 1,
    acc_margin: Annotated[float, typer.Option("--acc_margin", help="PASS iff mod_acc >= dense_acc + margin")] = 0.0,
    k_max: Annotated[Optional[float], typer.Option("--k_max", help="PASS iff total-k mean <= this (None = skip)")] = None,
    gen_n: Annotated[int, typer.Option("--gen_n", help="tier-2 rep probe prompts (0 = skip)")] = 0,
    gen_offset: Annotated[int, typer.Option("--gen_offset", help="gen protocol slice (established)")] = 5000,
    gen_new: Annotated[int, typer.Option("--gen_new")] = 64,
    rep_max_mult: Annotated[float, typer.Option("--rep_max_mult", help="PASS iff ours_rep <= mult * dense_rep (same lever)")] = 1.5,
    use_lora: Annotated[bool, typer.Option("--use_lora/--no-use_lora", help="LoRA wrapping switch (must match training; families adapt per ckpt config)")] = True,
    lora_rank: Annotated[int, typer.Option("--lora_rank", help="same as the train.py default")] = 8,
    lora_alpha: Annotated[int, typer.Option("--lora_alpha")] = 16,
    lora_targets: Annotated[str, typer.Option("--lora_targets")] = "q_proj,v_proj",
    device: Annotated[str, typer.Option("--device")] = "cuda:0",
    device_map: Annotated[str, typer.Option("--device_map", help="empty = whole-card (default); 'auto' = sharding across all visible GPUs (backbones larger than one card, e.g. 27B; batch_size 1 recommended)")] = "",
    out: Annotated[str, typer.Option("--out", help="manifest path (default: <ckpt>/manifest.json)")] = "",
) -> None:
    """PASS/FAIL a ckpt against the deployment contract; writes manifest.json."""
    from speaker.terminal import setup_terminal
    setup_terminal()
    device = resolve_device(device)
    tok = build_tok(model_id)
    texts = eval_slice(data_path, offset, n)
    coll = make_collate(tok, device, max_len)
    checks = []
    lora = (dict(rank=lora_rank, alpha=lora_alpha, targets=lora_targets)
            if use_lora else None)

    dense = build_model(model_id, device,
                        device_map=(device_map or None))
    dense.eval()
    d = eval_heldout(dense, texts, coll, batch_size)
    logger.info(f"dense loss {d['loss']:.3f} acc {d['acc']:.3f}")
    # tier-2 lever = the ckpt's own recipe (read from disk first, no model needed);
    # dense is generated under the SAME lever (fair comparison, ed9/D contract)
    dec = {}
    if gen_n > 0:
        try:
            with open(os.path.join(ckpt, "mod_config.json"), encoding="utf-8") as f:
                dec = json.load(f).get("decode") or {}
        except OSError:
            dec = {}
    rp = dec.get("repetition_penalty", 1.0)
    ng = dec.get("no_repeat_ngram_size", 0)
    reps, prompts = {}, None
    if gen_n > 0:
        import eval_gen as eg
        logger.info(f"tier-2 rep probe under ckpt lever (rep_penalty={rp} no_repeat_ngram={ng}): "
                    f"{'ckpt-shipped' if dec else 'neutral fallback'}")
        gtexts = eval_slice(data_path, gen_offset, gen_n)
        enc = tok(gtexts, truncation=True, max_length=64 + 64,
                  padding=True, return_tensors="pt")
        L = enc["attention_mask"].sum(1)
        prompts = [tok.decode(enc["input_ids"][bi, :L[bi]][:64].tolist(),
                              skip_special_tokens=True)
                   for bi in range(len(gtexts))]
        outs, _, _, _, _, _ = eg.run_gen(
            dense, tok, prompts, gen_new, device, "accept-dense",
            rep_penalty=rp, no_repeat_ngram=ng)
        reps["dense"] = sum(seq_rep(o) for o in outs) / max(len(outs), 1)
        logger.info(f"[accept-dense] seq-rep-4 {reps['dense']:.3f}")
    del dense
    collect_gc()

    from baselines.assemble import ModelBuilder
    asm = (ModelBuilder(model_id, lora, device_map=(device_map or None))
           .from_ckpt(ckpt).skip_mode("hard")
           .build(None if device_map else device))
    mod = asm.model
    mod.eval()
    m = eval_heldout(mod, texts, coll, batch_size)
    n_fixed = len((asm.cfg or {}).get("always_on_layers", []))
    k_total = (m["mean_k"] or 0.0) + n_fixed
    logger.info(f"mod   loss {m['loss']:.3f} acc {m['acc']:.3f} "
                f"(Δloss {m['loss'] - d['loss']:+.3f} Δacc {m['acc'] - d['acc']:+.3f}) "
                f"| k_total {k_total:.1f} (gated {m['mean_k']:.1f}+fixed {n_fixed})")
    checks.append({"name": "acc", "value": round(m["acc"] - d["acc"], 4),
                   "threshold": f">= dense+{acc_margin}",
                   "pass": bool(m["acc"] >= d["acc"] + acc_margin)})
    if k_max is not None:
        checks.append({"name": "k_total", "value": round(k_total, 3),
                       "threshold": f"<= {k_max}", "pass": bool(k_total <= k_max)})

    if gen_n > 0:
        outs, _, _, _, _, _ = eg.run_gen(
            mod, tok, prompts, gen_new, device, "accept-ours",
            rep_penalty=rp, no_repeat_ngram=ng)
        reps["ours"] = sum(seq_rep(o) for o in outs) / max(len(outs), 1)
        logger.info(f"[accept-ours] seq-rep-4 {reps['ours']:.3f}")
        checks.append({"name": "rep", "value": round(reps["ours"], 4),
                       "threshold": f"<= {rep_max_mult}x dense({reps['dense']:.3f})@{dec or 'neutral'}",
                       "pass": bool(reps["ours"] <= rep_max_mult * reps["dense"])})
    verdict = "PASS" if all(c["pass"] for c in checks) else "FAIL"
    manifest = {"ckpt": ckpt, "verdict": verdict, "checks": checks,
                "dense": {"loss": d["loss"], "acc": d["acc"]},
                "mod": {"loss": m["loss"], "acc": m["acc"],
                        "k_gated": m["mean_k"], "k_total": k_total},
                "provenance": {"slice": [offset, n], "max_len": max_len,
                               "gate_md5": md5_file(os.path.join(ckpt, "gate.pt")),
                               "has_decode": bool((asm.cfg or {}).get("decode"))}}
    out = out or os.path.join(ckpt, "manifest.json")
    dump_json(out, manifest)
    if verdict == "PASS":
        logger.info(f"[PASS] {ckpt} -> {out}")
    else:
        logger.warning(f"[FAIL] {ckpt} -> {out}")
    for c in checks:
        if c["pass"]:
            logger.info(f"  {c['name']}: {c['value']} ({c['threshold']}) ok")
        else:
            logger.warning(f"  {c['name']}: {c['value']} ({c['threshold']}) VIOLATED")
    sys.exit(0 if verdict == "PASS" else 1)


if __name__ == "__main__":
    app()
