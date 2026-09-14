"""Edge-side CPU-GPU collaborative inference benchmark (v0.2).

Simulates a VRAM-limited edge device against a Speaker checkpoint whose backbone
is larger than the card: the whole model lives in host RAM, the fixed layers +
strategy-chosen hot gated layers + IO modules (embeddings / final norm / lm_head)
reside inside a weights budget on the GPU, cold gated layers compute on the CPU
(the wrapper moves activations across the boundary), and the resident set is
re-planned between generations from the router's own activation counts.

Protocol mirrors tools/eval_gen.py (fresh-slice prompts, greedy decoding,
ROUGE-L F1 / word-trigram repetition) so numbers are comparable across the two tools.

Requirements on the ckpt structure (loud guards below):
- the LAST decoder layer must be always-on (tail >= 1): after the final layer the
  HF stack runs final-norm + lm_head, which we place on the GPU — the hidden
  state then has to arrive there. LayerScheduler forces all always-on layers
  onto the GPU, so a fixed tail satisfies this.
- run under taskset/numactl with the intended core count and pass --threads to
  pin the intra-op thread pools to the same count.

Usage (single-GPU worker, 16 cores):
  taskset -c 0-15 python3 tools/edge_bench.py --ckpt CKPT --model_id MODEL \
      --gpu_budget_gb 20 --reserve_gb 2 --strategy lfu --n 20 --max_new 128 \
      --data_path ... --offset 904300 --threads 16 --out report.json
"""
import json
import sys
import time
from pathlib import Path
from typing import Annotated, Literal

import torch
import typer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from baselines.assemble import ModelBuilder  # noqa: E402
from tools.gen_metrics import rouge_l, seq_rep  # noqa: E402 (standard metrics; single source)
from tools.eval_gen import load_texts  # noqa: E402
from tools._common import module_gb  # noqa: E402
from speaker.log import logger  # noqa: E402

app = typer.Typer(add_completion=False)


@app.command()
def main(
    ckpt: Annotated[str, typer.Option("--ckpt", help="speaker ckpt dir (mod_config.json + gate.pt)")] = ...,
    model_id: Annotated[str, typer.Option("--model_id", help="base backbone dir (original HF weights)")] = ...,
    use_lora: Annotated[bool, typer.Option("--use_lora/--no-use_lora", help="LoRA wrapping switch (must match training; families adapt per ckpt config)")] = True,
    lora_rank: Annotated[int, typer.Option("--lora_rank", help="same as the train.py default")] = 8,
    lora_alpha: Annotated[int, typer.Option("--lora_alpha")] = 16,
    lora_targets: Annotated[str, typer.Option("--lora_targets")] = "q_proj,v_proj",
    device: Annotated[str, typer.Option("--device")] = "cuda:0",
    gpu_budget_gb: Annotated[float, typer.Option("--gpu_budget_gb", help="simulated edge GPU allowance for WEIGHTS (decoder layers + IO modules)")] = 20.0,
    reserve_gb: Annotated[float, typer.Option("--reserve_gb", help="KV-cache / activation / logits reserve subtracted before weight packing")] = 2.0,
    strategy: Annotated[Literal["random", "lru", "lfu"], typer.Option("--strategy")] = "lfu",
    reschedule_every: Annotated[int, typer.Option("--reschedule_every", help="re-plan residency every K generations (0 = static placement)")] = 5,
    data_path: Annotated[str, typer.Option("--data_path")] = "/home/hdd/sanzo/minimind/dataset/sft_t2t_mini.jsonl",
    offset: Annotated[int, typer.Option("--offset", help="fresh (unseen) slice offset")] = 904300,
    n: Annotated[int, typer.Option("--n")] = 20,
    prompt_len: Annotated[int, typer.Option("--prompt_len")] = 64,
    ref_len: Annotated[int, typer.Option("--ref_len")] = 64,
    max_new: Annotated[int, typer.Option("--max_new")] = 128,
    temp: Annotated[float, typer.Option("--temp")] = 0.7,
    seed: Annotated[int, typer.Option("--seed")] = 42,
    threads: Annotated[int, typer.Option("--threads", help="torch intra-op threads (0 = leave torch defaults; use the taskset count)")] = 0,
    warmup: Annotated[int, typer.Option("--warmup")] = 1,
    out: Annotated[str, typer.Option("--out", help="JSON report path (default: <ckpt>/edge_report.json)")] = "",
) -> None:
    """Edge-bench: simulate a budgeted GPU, measure latency/quality."""
    from speaker.terminal import setup_terminal
    setup_terminal()
    if threads:
        torch.set_num_threads(threads)
    torch.manual_seed(seed)
    device = torch.device(device)
    logger.info(f"[edge] threads {torch.get_num_threads()} | budget {gpu_budget_gb}GB "
                f"(reserve {reserve_gb}GB) strategy {strategy}")
    lora = (dict(rank=lora_rank, alpha=lora_alpha, targets=lora_targets)
            if use_lora else None)

    # ---- model: whole backbone in host RAM, assembled from the ckpt, hard skip mode ----
    asm = (ModelBuilder(model_id, lora, dtype=torch.bfloat16)
           .from_ckpt(ckpt).skip_mode("hard").build(None))
    m = asm.model
    cfg = m.mod_config
    if cfg.gate_mode != "threshold" and cfg.gate_mode != "moe":
        raise ValueError(f"unexpected gate_mode {cfg.gate_mode}")
    if cfg.num_hidden_layers - 1 not in cfg.always_on_layers:
        raise ValueError(
            f"the last decoder layer ({cfg.num_hidden_layers - 1}) is NOT always-on; the "
            f"final-norm/lm_head epilogue is placed on the GPU and needs the tail fixed "
            f"(re-promote the ckpt with --always_tail >= 1)")
    tok_m = None
    from speaker.train_common import build_tok
    tok_m = build_tok(model_id)

    # ---- IO modules onto the GPU first; their bytes count against the budget ----
    embed, norm, lm_head = _io_modules(m)
    io = [embed, norm, lm_head]
    io_gb = module_gb(io, include_buffers=True)
    for mod in io:
        if mod is not None:
            mod.to(device)
    logger.info(f"[edge] IO modules on GPU: embed {type(embed).__name__}, norm "
                f"{type(norm).__name__ if norm is not None else None}, lm_head "
                f"{type(lm_head).__name__ if lm_head is not None else None} = {io_gb:.2f}GB")

    # ---- residency scheduling over the remaining budget ----
    sched = m.schedule_placement(
        strategy,
        gpu_total_gb=max(gpu_budget_gb - io_gb, 0.0),
        reserve_gb=reserve_gb, gpu_device=str(device))
    logger.info(f"[edge] {sched.describe()}")
    n_layers = cfg.num_hidden_layers
    resident_weights = sched and sum(
        sched.layer_bytes.get(i, 0) for i in (sched.resident or [])) / 1e9
    total_weights = sum(sched.layer_bytes.values()) / 1e9
    logger.info(f"[edge] resident decoder layers {len(sched.resident or [])}/{n_layers} "
                f"{resident_weights:.2f}GB of {total_weights:.2f}GB (+IO {io_gb:.2f}GB)")

    # ---- prompts (fresh slice, same construction as eval_gen) ----
    texts = load_texts(data_path, offset, n * 4)[:n * 4]
    enc = tok_m(texts, truncation=True, max_length=prompt_len + ref_len,
                add_special_tokens=False)
    prompts, refs = [], []
    for ids in enc["input_ids"]:
        if len(ids) <= prompt_len + 8:
            continue
        prompts.append(tok_m.decode(ids[:prompt_len], skip_special_tokens=True))
        refs.append(tok_m.decode(ids[prompt_len:prompt_len + ref_len],
                                 skip_special_tokens=True))
    prompts, refs = prompts[:n], refs[:n]
    logger.info(f"[edge] {len(prompts)} fresh-slice prompts (offset {offset})")

    def gen(prompt_txt):
        ids = tok_m(prompt_txt, return_tensors="pt", add_special_tokens=True)["input_ids"].to(device)
        t0 = time.time()
        torch.cuda.synchronize(device) if device.type == "cuda" else None
        g = m.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                       max_new_tokens=max_new, do_sample=temp > 0,
                       temperature=temp if temp > 0 else None,
                       top_k=0, top_p=1.0, pad_token_id=tok_m.pad_token_id,
                       use_cache=True)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        dt = time.time() - t0
        out_txt = tok_m.decode(g[0, ids.shape[1]:], skip_special_tokens=True)
        return out_txt, dt, ids.shape[1]

    # ---- warmup (lazy kernels / CUDA context) ----
    for i in range(max(warmup, 0)):
        gen(prompts[i % len(prompts)])
        if reschedule_every:
            sched.reschedule()

    # ---- measured run ----
    m.get_skip_hits(reset=True)
    torch.cuda.reset_peak_memory_stats(device) if device.type == "cuda" else None
    outs, times, prefill = [], [], []
    reschedule_events = 0
    for i, p in enumerate(prompts):
        o, dt, pl = gen(p)
        outs.append(o)
        times.append(dt)
        prefill.append(pl)
        if reschedule_every and (i + 1) % reschedule_every == 0:
            before = sched.resident
            sched.reschedule()
            reschedule_events += int(before != sched.resident)
    n_new = sum(len(tok_m(o, add_special_tokens=False)["input_ids"]) for o in outs)
    decode_ms = sum(t for t in times) / max(n_new, 1) * 1000
    rep = sum(seq_rep(o) for o in outs) / max(len(outs), 1)
    rouge = sum(rouge_l(r, o) for r, o in zip(refs, outs)) / max(len(outs), 1)

    free_b, total_b = (torch.cuda.mem_get_info(device) if device.type == "cuda"
                       else (0, 0))
    report = {
        "ckpt": ckpt, "model_id": model_id,
        "gate_mode": cfg.gate_mode, "layers": n_layers,
        "always_on": cfg.always_on_layers, "gated": len(cfg.gated_layers),
        "gpu_budget_gb": gpu_budget_gb, "reserve_gb": reserve_gb,
        "strategy": strategy, "reschedule_every": reschedule_every,
        "reschedule_events": reschedule_events,
        "resident_layers": len(sched.resident or []),
        "resident_decoder_gb": round(resident_weights, 3),
        "io_gb": round(io_gb, 3),
        "total_decoder_gb": round(total_weights, 3),
        "threads": torch.get_num_threads(),
        "n": len(prompts), "max_new": max_new, "temp": temp,
        "ms_per_tok": round(decode_ms, 2),
        "wall_s": round(sum(times), 2),
        "new_tokens": n_new,
        "gpu_process_gb": round((total_b - free_b) / 1e9, 3),
        "gpu_peak_alloc_gb": round(torch.cuda.max_memory_allocated(device) / 1e9, 3)
        if device.type == "cuda" else None,
        "skip_hits_decode": m.get_skip_hits(),
        "rougeL": round(rouge, 4), "seq-rep-4": round(rep, 4),
        "samples": outs[:3],
    }
    out_path = out or str(Path(ckpt) / "edge_report.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info(f"[edge] {len(prompts)} prompts x {max_new} tok: {decode_ms:.1f} ms/tok, "
                f"resident {report['resident_layers']}/{n_layers} layers "
                f"({report['resident_decoder_gb'] + report['io_gb']:.2f}GB weights on GPU), "
                f"ROUGE-L {rouge:.3f} seq-rep-4 {rep:.3f} skip_hits {report['skip_hits_decode']}")
    logger.info(f"[edge] report -> {out_path}")


def _io_modules(model):
    """(embed, norm, lm_head) of the decoder stack, wherever they live in the tree.
    embed/lm_head via the HF delegation API (peft-safe); the final norm is a sibling
    of the wrapped layer list — the wrapper records the parent exactly for this."""
    hf = model.hf_model
    embed = hf.get_input_embeddings()
    lm_head = hf.get_output_embeddings()
    norm = getattr(getattr(model, "_layers_parent", None), "norm", None)
    if norm is None:  # fallbacks for exotic stacks
        for path in ("model.language_model.norm", "model.norm", "transformer.ln_f"):
            cur = hf
            ok = True
            for a in path.split("."):
                cur = getattr(cur, a, None)
                if cur is None:
                    ok = False
                    break
            if ok:
                norm = cur
                break
    return embed, norm, lm_head


if __name__ == "__main__":
    app()
