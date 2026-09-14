"""Premise test for difficulty-conditional budgets (ed8 follow-up, 0911).

Question: does DEPTH (the layers our router may skip, i.e. the gated span 2..25)
differentially help HARD tokens? If not, per-token lambda shaping by difficulty
has no headroom regardless of lambda amplitude.

Method: dense raw model, bypass layers via forward hooks (probe_layers.py trick).
Difficulty axis = per-token NLL of the INTACT dense model on the same positions
(same 60-sample slice / valid-mask pipeline as the kdiff NLL cache).
  - single-layer sweep over the gated span: per-layer dNLL by difficulty quartile
  - depth dosage: random m in {4,8,12} mid layers, 2 seeds each
Premise holds iff dNLL concentrates in the top (hard) quartile, growing with dose.

Output JSON: per-config mean dNLL overall + per quartile + Q4/Q1 ratio + r(dnll, base_nll).
"""
import pathlib
import sys
from typing import Annotated

import torch
import typer

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from data.sft import make_collate
from speaker.evaluate import chunked_nll_correct
from speaker.log import logger
from speaker.train_common import build_model, build_tok, eval_slice, resolve_device
from tools._common import dump_json, find_layers, passthrough_hook, pearson_r, quartile_means

app = typer.Typer(add_completion=False)


def run_nll(model, batches):
    """Per-sample per-token NLL lists on valid positions (aligned with batch ids)."""
    outs = []
    with torch.no_grad():
        for b in batches:
            nll, _ = chunked_nll_correct(model(**b).logits, b["labels"])
            valid = (b["labels"] != -100)[0]
            outs.append((b["input_ids"][0][valid].cpu(), nll[0][valid].cpu()))
            del nll
    return outs


def summarize(dnll, base, name):
    finite = torch.isfinite(dnll)
    frac_bad = float((~finite).float().mean())
    d, b = dnll[finite], base[finite]
    by_q = quartile_means(d, b, ndigits=4)
    r = pearson_r(d, b)
    ratio = (by_q[3] / by_q[0]) if by_q[0] and abs(by_q[0]) > 1e-6 else None
    row = {"config": name, "dnll_mean": round(float(d.mean()), 4),
           "dnll_by_nll_quartile_q1e_q4h": by_q,
           "q4_q1_ratio": round(ratio, 3) if ratio is not None else None,
           "r_dnll_basenll": round(r, 4), "frac_nonfinite": round(frac_bad, 5)}
    logger.info(f"[{name}] dnll {row['dnll_mean']:+.4f} q={[f'{v:+.3f}' if v is not None else '-' for v in by_q]} "
                f"Q4/Q1 {row['q4_q1_ratio']} r {row['r_dnll_basenll']:+.3f}")
    return row


@app.command()
def main(
    model_id: Annotated[str, typer.Option("--model_id")] = "/home/hdd/model/Qwen2.5-7B-Instruct",
    data_path: Annotated[str, typer.Option("--data_path")] = "./data/sft_t2t_mini.jsonl",
    offset: Annotated[int, typer.Option("--offset")] = 5000,
    n: Annotated[int, typer.Option("--n")] = 60,
    max_len: Annotated[int, typer.Option("--max_len")] = 256,
    device: Annotated[str, typer.Option("--device")] = "cuda:0",
    gated_span: Annotated[str, typer.Option("--gated_span", help="inclusive layer range probed")] = "2,25",
    doses: Annotated[str, typer.Option("--doses", help="random-set sizes")] = "4,8,12",
    seeds: Annotated[int, typer.Option("--seeds")] = 2,
    nll_cache: Annotated[str, typer.Option("--nll_cache", help="optional: sanity-check our baseline against this cache")] = "/tmp/opencode/nll_cache.pt",
    out: Annotated[str, typer.Option("--out")] = "/tmp/diff_ablation.json",
) -> None:
    """Depth-dosage premise test: does depth differentially help hard tokens?"""
    from speaker.terminal import setup_terminal
    setup_terminal()
    device = resolve_device(device)
    tok = build_tok(model_id)
    texts = eval_slice(data_path, offset, n)
    coll = make_collate(tok, device, max_len)
    batches = [coll([t]) for t in texts]

    model = build_model(model_id, device)
    model.eval()
    layers = find_layers(model)
    N = len(layers)
    lo, hi = (int(x) for x in gated_span.split(","))
    span = list(range(lo, hi + 1))
    doses = [int(x) for x in doses.split(",")]

    base = run_nll(model, batches)
    ids = [s[0] for s in base]
    base_all = torch.cat([s[1] for s in base])
    logger.info(f"baseline pooled nll mean {float(base_all.mean()):.4f}  ntok {base_all.numel()}  N={N}")
    if nll_cache:
        cache = torch.load(nll_cache, map_location="cpu", weights_only=True)
        same = all(torch.equal(cache["ids"][i], ids[i]) for i in range(len(ids)))
        cb = torch.cat([c for c in cache["nll"]])
        logger.info(f"cache ids aligned: {same}; cached nll mean {float(cb.mean()):.4f}")

    rows = []

    def with_skip(skips, name):
        hooks = [layers[i].register_forward_hook(passthrough_hook) for i in skips]
        abl = run_nll(model, batches)
        for h in hooks:
            h.remove()
        dnll = torch.cat([(a[1] - base[i][1]) for i, a in enumerate(abl)])
        rows.append(summarize(dnll, base_all, name))

    for i in span:  # single-layer sweep over the gated span
        with_skip([i], f"drop_L{i}")

    for m in doses:  # depth dosage: random mid-layer sets
        for seed in range(seeds):
            g = torch.Generator().manual_seed(1234 + m * 100 + seed)
            picks = sorted(torch.randperm(len(span), generator=g)[:m].tolist())
            picks = [span[j] for j in picks]
            with_skip(picks, f"dose_m{m}_s{seed}")

    dump_json(out, {"meta": dict(model_id=model_id, offset=offset, n=n,
                                 gated_span=gated_span),
                    "baseline_nll_mean": float(base_all.mean()), "rows": rows})
    logger.info(f"saved {out}")


if __name__ == "__main__":
    app()
