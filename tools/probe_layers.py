"""Per-layer importance probe: bypass one layer at a time in the dense model and see how much
held-out loss/acc drops.
Used to validate the "syntax at the ends, logic in the middle" hypothesis and to guide
shared (fixed) layer selection."""
import sys
from pathlib import Path as _P
from typing import Annotated

import torch
import typer
from transformers import AutoTokenizer, AutoModelForCausalLM
sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
from data.sft import SFTDataset, make_collate
from speaker.evaluate import eval_heldout
from speaker.log import logger
from tools._common import dump_json, find_layers, passthrough_hook

app = typer.Typer(add_completion=False)


@app.command()
def main(
    model_id: Annotated[str, typer.Option("--model_id")] = "/home/hdd/model/Qwen2.5-7B-Instruct",
    data_path: Annotated[str, typer.Option("--data_path")] = "./data/sft_t2t_mini.jsonl",
    offset: Annotated[int, typer.Option("--offset")] = 1000,
    n: Annotated[int, typer.Option("--n")] = 100,
    device: Annotated[str, typer.Option("--device")] = "cuda:0",
    out: Annotated[str, typer.Option("--out")] = "/tmp/layer_probe.json",
) -> None:
    """Bypass each dense layer in turn and rank them by held-out damage."""
    from speaker.terminal import setup_terminal
    setup_terminal()
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    full = SFTDataset(data_path, offset + n)
    texts = full.samples[offset:offset + n]
    logger.info(f"heldout {len(texts)}")
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16,
        device_map=None, trust_remote_code=True, low_cpu_mem_usage=True).to(device)
    layers = find_layers(model)
    N = len(layers)
    coll = make_collate(tok, device, 256)
    base = eval_heldout(model, texts, coll)
    logger.info(f"baseline loss {base['loss']:.3f} acc {base['acc']:.3f}")
    rows = []
    for i in range(N):
        h = layers[i].register_forward_hook(passthrough_hook)
        r = eval_heldout(model, texts, coll)
        h.remove()
        rows.append({"layer": i, "loss": r["loss"], "acc": r["acc"],
                     "dloss": r["loss"] - base["loss"], "dacc": r["acc"] - base["acc"]})
        logger.info(f"skip L{i:2d}: loss {r['loss']:.3f} (Δ{r['loss'] - base['loss']:+.3f}) "
                    f"acc {r['acc']:.3f} (Δ{r['acc'] - base['acc']:+.3f})")
    rows.sort(key=lambda r: -r["dloss"])
    logger.info("rank by importance: " + " ".join(f"L{r['layer']}({r['dloss']:+.2f})" for r in rows[:8]))
    dump_json(out, {"baseline": base, "rows": rows})
    logger.info(f"saved {out}")


if __name__ == "__main__":
    app()
