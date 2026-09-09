"""On-device smoke test - Qwen1.5-0.5B (finetune-track formulation, needs /home/hdd/model
weights, run manually).

Runs both schemes once each: moe (default mainline) and threshold (legacy scheme, validates
the old ckpt loading path).
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from speaker import SpeakerConfig, convert_to_speaker


def smoke(gate_mode: str):
    model_id = "/home/hdd/model/Qwen1.5-0.5B"
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32, device_map="cpu",
                                                 trust_remote_code=True, low_cpu_mem_usage=True)
    print(f"\n=== gate_mode={gate_mode} === base N={model.config.num_hidden_layers} "
          f"H={model.config.hidden_size}")
    inputs = tok(["Hello world, this is a test", "MoD gating test"], return_tensors="pt", padding=True)
    with torch.no_grad():
        out_base = model(**inputs, labels=inputs["input_ids"])
        print(f"base loss {out_base.loss.item():.4f}")

    if gate_mode == "moe":
        cfg = SpeakerConfig.from_model_config(model.config, kmax=6, top_p=0.9)
    else:
        cfg = SpeakerConfig.from_model_config(model.config, gate_mode="threshold",
                                              kmax=6, temp_affinity=1.0, gumbel_scale=1.0)
    print(cfg.summary())
    mod_model = convert_to_speaker(model, cfg)
    assert mod_model.layers[0].is_always_on and mod_model.layers[1].is_always_on \
        and mod_model.layers[-1].is_always_on
    n_gate = sum(p.numel() for p in mod_model.get_router_parameters())
    print(f"gated {len(cfg.gated_layers)}, gate params {n_gate}")
    if gate_mode == "moe":
        assert n_gate == model.config.hidden_size * len(cfg.gated_layers) + len(cfg.gated_layers)
    else:
        assert n_gate > model.config.hidden_size * len(cfg.gated_layers)  # per-layer router+tau+comp

    mod_model.train()
    out = mod_model(**inputs, labels=inputs["input_ids"])
    print(f"soft loss {out.loss.item():.4f}")
    aux = mod_model.get_aux_loss()
    print(f"aux {aux.item() if aux is not None else None}")
    print("counts(hard)", mod_model.get_active_counts())
    loss = out.loss + (aux or 0)
    loss.backward()
    ps = mod_model.get_router_parameters()
    ng = sum(1 for p in ps if p.grad is not None and p.grad.abs().sum().item() > 0)
    print(f"grad {ng}/{len(ps)}")
    assert ng > 0

    mod_model.eval()
    mod_model.set_skip_mode("hard")
    with torch.no_grad():
        out_h = mod_model(**inputs, labels=inputs["input_ids"])
        print(f"hard loss {out_h.loss.item():.4f} counts {mod_model.get_active_counts()}")
    print(f"Qwen smoke passed ({gate_mode})")


if __name__ == "__main__":
    smoke("moe")
    smoke("threshold")
