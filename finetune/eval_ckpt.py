"""Final ckpt evaluation (can be rerun standalone, no dependency on the training tail):
dense baseline vs Speaker holdout comparison.
loss/acc/delta + mean_k±std + k quartiles, same protocol as the finetune/train.py
final eval.
Usage: python3 finetune/eval_ckpt.py --ckpt /tmp/mod_ckpt_mix1 --data_path ... --offset 3000 --n 100
"""
import gc
import json
import sys
from pathlib import Path
from typing import Annotated

import torch
import typer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from baselines.assemble import ModelBuilder
from speaker.evaluate import eval_heldout, format_k_quartile
from speaker.log import logger
from speaker.train_common import build_model, build_tok, eval_slice, resolve_device
from data.sft import make_collate


app = typer.Typer(add_completion=False)


@app.command()
def main(
    model_id: Annotated[str, typer.Option("--model_id")] = "/home/hdd/model/Qwen2.5-7B-Instruct",
    ckpt: Annotated[str, typer.Option("--ckpt")] = ...,
    data_path: Annotated[str, typer.Option("--data_path")] = "./data/sft_t2t_mini.jsonl",
    offset: Annotated[int, typer.Option("--offset")] = 0,
    n: Annotated[int, typer.Option("--n")] = 100,
    max_len: Annotated[int, typer.Option("--max_len")] = 512,
    batch_size: Annotated[int, typer.Option("--batch_size")] = 4,
    use_chat: Annotated[bool, typer.Option("--use_chat/--no-use_chat")] = True,
    mask_user: Annotated[bool, typer.Option("--mask_user/--no-mask_user")] = True,
    use_lora: Annotated[bool, typer.Option("--use_lora/--no-use_lora", help="must be on when the ckpt contains LoRA (same rank/targets as training), otherwise the lora_ weights are silently dropped")] = True,
    lora_rank: Annotated[int, typer.Option("--lora_rank", help="matches the train.py default")] = 8,
    lora_alpha: Annotated[int, typer.Option("--lora_alpha", help="matches training (train.py default 16)")] = 16,
    lora_targets: Annotated[str, typer.Option("--lora_targets")] = "q_proj,v_proj",
    device: Annotated[str, typer.Option("--device")] = "cuda:0",
    device_map: Annotated[str, typer.Option("--device_map", help="empty = whole-card (default); 'auto' = sharding across visible GPUs (backbones larger than one card)")] = "",
    out: Annotated[str, typer.Option("--out")] = "",
) -> None:
    """Final ckpt evaluation: dense baseline vs Speaker holdout comparison."""
    from speaker.terminal import setup_terminal
    setup_terminal()
    device = resolve_device(device)
    tok = build_tok(model_id)
    texts = eval_slice(data_path, offset, n,
                       tok=tok if use_chat else None, use_chat=use_chat)
    logger.info(f"eval {len(texts)} texts from {data_path}@{offset}")

    dense = build_model(model_id, device, device_map=(device_map or None))
    coll = make_collate(tok, device, max_len, use_chat, mask_user)
    d = eval_heldout(dense, texts, coll, batch_size)
    logger.info(f"dense loss {d['loss']:.3f} acc {d['acc']:.3f}")
    del dense
    if device.type == "cuda":
        gc.collect()  # 7B: gc to break reference cycles before empty_cache, otherwise loading the second model OOMs
        torch.cuda.empty_cache()

    lora = (dict(rank=lora_rank, alpha=lora_alpha, targets=lora_targets)
            if use_lora else None)
    mod = (ModelBuilder(model_id, lora=lora, dtype=torch.bfloat16,
                        device_map=(device_map or None))
           .from_ckpt(ckpt).skip_mode("hard").build(None if device_map else device).model)
    m = eval_heldout(mod, texts, coll, batch_size)
    logger.info(f"mod   loss {m['loss']:.3f} acc {m['acc']:.3f} "
                f"(Δloss {m['loss'] - d['loss']:+.3f} Δacc {m['acc'] - d['acc']:+.3f}) "
                f"| {format_k_quartile(m)}")
    if out:
        with open(out, "w", encoding="utf-8") as f:
            json.dump({"dense": d, "mod": m,
                       "dloss": m["loss"] - d["loss"], "dacc": m["acc"] - d["acc"]},
                      f, indent=1, ensure_ascii=False)
        logger.info(f"saved {out}")


if __name__ == "__main__":
    app()
