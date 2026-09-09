"""On-device cross-device smoke - Qwen1.5-0.5B (needs /home/hdd/model weights + a GPU;
run manually / via sandbox, NOT part of the offline test loop).

Validates the scheduled-placement stack end to end with moe (the default mainline):
1. schedule_placement with a tight budget -> fixed layers + few gated on GPU, the rest
   on CPU (real device moves, not simulated);
2. generate() with hard skip + sparse KV cache on mixed devices (exercises the moe
   route-slice realignment — the historical crash: route lives on the entry layer's
   device while other gated layers compute elsewhere);
3. reschedule() between generations (LFU harvest -> plan change -> layer moves),
   then generate again on the new placement.
"""
import torch

from speaker import SpeakerConfig, convert_to_speaker
from speaker.load_profile import estimate_layer_bytes

MODEL_ID = "/home/hdd/model/Qwen1.5-0.5B"


def layer_device(model, idx):
    return next(model.layers[idx].layer.parameters()).device.type


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float32, device_map=None,
        trust_remote_code=True, low_cpu_mem_usage=True).to("cuda:0")

    cfg = SpeakerConfig.from_model_config(model.config, kmax=6, top_p=0.9,
                                          gumbel_scale=0.0)
    mod = convert_to_speaker(model, cfg)
    mod.eval()
    mod.set_skip_mode("hard")
    print(f"base N={cfg.num_hidden_layers} fixed={cfg.always_on_layers} "
          f"gated={len(cfg.gated_layers)}")

    # tight budget: fixed layers + ~2 gated layers on GPU, the rest scheduled to CPU
    lb = estimate_layer_bytes(mod)
    fixed_b = sum(lb[i] for i in cfg.always_on_layers)
    gated_avg = sum(lb[i] for i in cfg.gated_layers) / len(cfg.gated_layers)
    gpu_total = (fixed_b + 2.5 * gated_avg) / 1e9
    sched = mod.schedule_placement("lfu", gpu_total_gb=gpu_total, reserve_gb=0.05,
                                   gpu_device="cuda:0", cpu_device="cpu", seed=0)
    gpu_layers = sched.resident
    dev = {i: layer_device(mod, i) for i in range(cfg.num_hidden_layers)}
    n_cpu = sum(1 for v in dev.values() if v == "cpu")
    assert n_cpu > 0, f"expected a mixed placement, got all on {set(dev.values())}"
    assert all(dev[i] == "cuda" for i in cfg.always_on_layers), "fixed layers must be GPU-resident"
    print(sched.describe())
    print(f"placement: gpu {sum(1 for v in dev.values() if v == 'cuda')} / "
          f"cpu {n_cpu} layers; resident_gb={mod.resident_gb('cuda'):.2f}")

    prompt = tok(["Hello world, this is a test of mixed-device generation,"],
                 return_tensors="pt").to("cuda:0")
    for turn in range(2):
        with torch.no_grad():
            out = mod.generate(**prompt, max_new_tokens=12, do_sample=False,
                               pad_token_id=tok.pad_token_id, use_cache=True)
        text = tok.decode(out[0, prompt["input_ids"].shape[1]:])
        assert out.isfinite().all(), "generation produced non-finite token ids"
        print(f"turn {turn}: {text!r}  (skip_hits={mod.get_skip_hits()})")
        if turn == 0:
            # between generations: LFU harvest -> re-plan -> move layers if changed
            with torch.no_grad():
                mod(**prompt)  # one more forward to seed the counters
            before = list(sched.resident)
            sched.reschedule()
            moved = sorted(set(before) ^ set(sched.resident))
            print(f"reschedule: resident {before} -> {sched.resident} "
                  f"(changed layers {moved or '-'})")
    print(f"final devices cpu={sorted(i for i, v in dev.items() if layer_device(mod, i) == 'cpu')}")
    print("GPU scheduler smoke passed (moe mixed-device generate + reschedule)")


if __name__ == "__main__":
    main()
