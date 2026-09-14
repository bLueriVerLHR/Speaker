"""k-vs-difficulty probe: does per-token routing track per-token difficulty?
difficulty axis = dense per-token NLL on the SAME text (no corpus/word heuristics).
For each ours ckpt: Pearson r(k_total, dense_nll) + mean k by dense-NLL quartile +
low/high-hump mass fractions. A difficulty-blind router gives slope ~ 0 no matter
how wide the k histogram is (shape without function).

LoRA note (0911 fix): ours-family ckpts ship gate/LoRA keys in gate.pt; assemble only
wraps peft when CLI args say so. The first version of this probe (in /tmp/opencode)
called assemble() without args -> 224 LoRA keys silently dropped, measuring trained
gates on the raw base. Default here is --use_lora to match tools/probe_kdist.py.

Usage: kdiff_probe.py --ckpt CKPT [--ckpt ...] --out JSON (rest: model/data slice)."""
import gc
import pathlib
import sys
from typing import Annotated

import torch
import typer

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from baselines.assemble import assemble
from data.sft import make_collate
from speaker.evaluate import chunked_nll_correct
from speaker.log import logger
from speaker.train_common import (
    build_model,
    build_tok,
    eval_slice,
    resolve_device,
)
from tools._common import collect_gc, dump_json, pearson_r, quartile_means

app = typer.Typer(add_completion=False)


@app.command()
def main(
    model_id: Annotated[str, typer.Option("--model_id")] = "/home/hdd/model/Qwen2.5-7B-Instruct",
    data_path: Annotated[str, typer.Option("--data_path")] = "./data/sft_t2t_mini.jsonl",
    offset: Annotated[int, typer.Option("--offset")] = 5000,
    n: Annotated[int, typer.Option("--n")] = 60,
    max_len: Annotated[int, typer.Option("--max_len")] = 256,
    device: Annotated[str, typer.Option("--device")] = "cuda:0",
    out: Annotated[str, typer.Option("--out")] = "/tmp/kdiff.json",
    ckpt: Annotated[list[str], typer.Option("--ckpt")] = [],
    use_lora: Annotated[bool, typer.Option("--use_lora/--no-use_lora", help="LoRA wrapping switch (must match training; families adapt per ckpt config)")] = True,
    lora_rank: Annotated[int, typer.Option("--lora_rank", help="same as the train.py default")] = 8,
    lora_alpha: Annotated[int, typer.Option("--lora_alpha")] = 16,
    lora_targets: Annotated[str, typer.Option("--lora_targets")] = "q_proj,v_proj",
    save_nll: Annotated[str, typer.Option("--save_nll", help="phase 1 only: dump dense-NLL cache (pt file)")] = "",
    load_nll: Annotated[str, typer.Option("--load_nll", help="phase 2 only: read cache, skip dense load")] = "",
) -> None:
    """k-vs-difficulty probe: does per-token routing track dense NLL?"""
    from speaker.terminal import setup_terminal
    setup_terminal()
    device = resolve_device(device)
    tok = build_tok(model_id)
    texts = eval_slice(data_path, offset, n)
    coll = make_collate(tok, device, max_len)
    lora = (dict(rank=lora_rank, alpha=lora_alpha, targets=lora_targets)
            if use_lora else None)

    if load_nll:
        cache = torch.load(load_nll, map_location="cpu", weights_only=True)
        saved = [(cache["ids"][i], cache["nll"][i]) for i in range(len(cache["ids"]))]
        dense = None
    else:
        # pass 1: dense per-token NLL (kept small: valid positions only)
        dense = build_model(model_id, device)
        dense.eval()
        saved = []
        with torch.no_grad():
            for t in texts:
                b = coll([t])
                nll, _ = chunked_nll_correct(dense(**b).logits, b["labels"])
                valid = (b["labels"] != -100)[0]
                saved.append((b["input_ids"][0][valid].cpu(),
                              nll[0][valid].cpu()))
                del nll
        if save_nll:
            torch.save({"ids": [s[0] for s in saved],
                        "nll": [s[1] for s in saved]}, save_nll)
            logger.info(f"saved NLL cache {save_nll} ({len(saved)} samples)")
    if dense is not None:
        del dense
    # drop every reference buyers can't see (closures/lists pin models: AGENTS probe lesson)
    del texts, coll
    collect_gc()
    torch.cuda.synchronize()

    lora_tag = f"lora r{lora_rank}/{lora_targets}" if use_lora else "NO-LORA"
    logger.info(f"probe mode: {lora_tag}")
    res = {"n": n, "offset": offset, "lora": lora_tag, "ckpts": {}}
    for ck in ckpt:
        asm = assemble(ck, model_id, lora, device=device, skip_mode="hard")
        m = asm.model
        n_fixed = len((asm.cfg or {}).get("always_on_layers", []))
        m.eval()
        ks, ns = [], []
        with torch.no_grad():
            for ids, nll in saved:
                b = {"input_ids": ids.unsqueeze(0).to(device),
                     "attention_mask": torch.ones_like(ids).unsqueeze(0).to(device),
                     "labels": ids.unsqueeze(0).to(device)}
                mo = m(**b)
                getk = getattr(m, "get_active_counts", None)
                k = getk() if callable(getk) else None
                if k is None:  # non-wrapper families: constant fallback (not expected here)
                    continue
                kk = (k[0].float() + n_fixed).cpu()
                ks.append(kk)
                ns.append(nll)
        k_all = torch.cat(ks)
        n_all = torch.cat(ns)
        r = pearson_r(k_all, n_all)
        qk = quartile_means(k_all, n_all)
        res["ckpts"][ck] = {
            "r_k_nll": round(r, 4),
            "k_by_nll_quartile": qk,
            "k_mean": round(float(k_all.mean()), 3),
            "k_std": round(float(k_all.std()), 3),
            "low_hump_le6": round(float((k_all <= 6).float().mean()), 3),
            "high_hump_ge14": round(float((k_all >= 14).float().mean()), 3),
            "ntok": int(k_all.numel()),
        }
        logger.info(f"[{ck}] r={r:.3f} qk={qk} k={k_all.mean():.1f}±{k_all.std():.1f} "
                    f"low={res['ckpts'][ck]['low_hump_le6']:.2f} high={res['ckpts'][ck]['high_hump_ge14']:.2f}")
        # bound methods / outputs pin the peft+speaker model graph (OOM between ckpts)
        getk = mo = k = kk = None  # noqa: F841 (deliberately unpin bound methods for gc)
        del asm, m
        gc.collect()
        torch.cuda.empty_cache()
    dump_json(out, res)
    logger.info(f"saved {out}")


if __name__ == "__main__":
    app()
