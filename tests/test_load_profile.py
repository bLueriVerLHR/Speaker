"""Load-profiling toolchain self-tests (no real weights needed): load -> shared-layer
promotion / greedy placement / profiling -> promotion -> reload loop (dual scheme)."""
import os
import pathlib
import sys
import tempfile

import torch
import torch.nn as nn

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from speaker import SpeakerConfig, convert_to_speaker
from speaker.load_profile import (
    estimate_layer_bytes,
    plan_placement,
    profile_layer_load,
    render_load_table,
    select_fixed_layers,
)


class FakeDecoderLayer(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.self_attn = nn.Identity()
        self.mlp = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.SiLU(),
                                 nn.Linear(hidden_size, hidden_size))

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        out = hidden_states + self.mlp(hidden_states) * 0.1
        return (out,)


class FakeHF(nn.Module):
    def __init__(self, n=6, h=32):
        super().__init__()
        self.config = type("Cfg", (), {"num_hidden_layers": n, "hidden_size": h})()
        self.model = type("M", (), {})()
        self.model.layers = nn.ModuleList([FakeDecoderLayer(h) for _ in range(n)])
        self.lm_head = nn.Linear(h, 100)

    def forward(self, hidden_states=None, input_ids=None, attention_mask=None, **kwargs):
        if hidden_states is None:
            hidden_states = torch.randn(2, 8, 32)
        for layer in self.model.layers:
            hidden_states = layer(hidden_states, attention_mask=attention_mask, **kwargs)[0]
        return {"logits": self.lm_head(hidden_states)}


def test_select_fixed_layers():
    load = {0: 1.0, 1: 1.0, 2: 0.96, 3: 0.5, 4: 0.91, 5: 1.0}
    # 0/1/5 are the original always-on layers (load always 1), 2/4 get promoted, 3 stays gated
    fixed, promoted = select_fixed_layers(load, threshold=0.9, always_on=[0, 1, 5])
    assert fixed == [0, 1, 2, 4, 5], fixed
    assert promoted == [2, 4], promoted
    # higher threshold: only 2 remains
    fixed2, promoted2 = select_fixed_layers(load, threshold=0.95, always_on=[0, 1, 5])
    assert promoted2 == [2], promoted2
    # even if every layer exceeds the threshold, keep the gated floor (a degenerate dense
    # model is pointless): with flat loads, give back the last one in layer order
    load_all = {i: 1.0 for i in range(6)}
    fixed3, promoted3 = select_fixed_layers(load_all, threshold=0.9,
                                            always_on=[0, 5], keep_gated_min=1)
    assert promoted3 == [1, 2, 3], promoted3
    assert 4 not in fixed3, "L4 should stay gated"
    try:
        select_fixed_layers(load, threshold=1.5)
        raise AssertionError("should reject an invalid threshold")
    except ValueError:
        pass
    print("[PASS] select_fixed_layers")


def test_plan_placement():
    load = {0: 1.0, 1: 0.9, 2: 0.2, 3: 0.1, 4: 0.8, 5: 1.0}
    layer_bytes = {i: 100 for i in range(6)}
    # budget 350: shared layers (0,5)=200 + layer 1 fits in load order -> 300, the rest do not fit
    plan = plan_placement(load, layer_bytes, 350, always_on=[0, 5])
    assert plan["gpu_layers"] == [0, 1, 5], plan
    assert plan["cpu_layers"] == [2, 3, 4], plan
    assert plan["gpu_bytes"] == 300
    # tiny budget: only the shared layers go to GPU (shared layers are forced in even over budget)
    plan2 = plan_placement(load, layer_bytes, 150, always_on=[0, 5])
    assert plan2["gpu_layers"] == [0, 5], plan2
    assert plan2["cpu_layers"] == [1, 2, 3, 4]
    # ample budget: all GPU
    plan3 = plan_placement(load, layer_bytes, 1e9, always_on=[0, 5])
    assert plan3["cpu_layers"] == []
    # skip big, fit small: a high-load layer too big to fit is skipped, smaller later layers still fit
    layer_bytes2 = {0: 100, 1: 200, 2: 100, 3: 40, 4: 100, 5: 100}
    plan4 = plan_placement(load, layer_bytes2, 350, always_on=[0, 5])
    assert plan4["gpu_layers"] == [0, 3, 4, 5], plan4  # 1 (0.9) does not fit and is skipped
    assert plan4["cpu_layers"] == [1, 2], plan4
    print("[PASS] plan_placement")


def test_profile_and_promote_threshold():
    """threshold: profile -> promote -> write a new ckpt -> reload under the new config
    (old gate.pt key compatible)."""
    torch.manual_seed(0)
    n, h = 6, 32
    fake = FakeHF(n=n, h=h)
    cfg = SpeakerConfig(num_hidden_layers=n, hidden_size=h, gate_mode="threshold",
                        gumbel_scale=0.0)
    mod = convert_to_speaker(fake, cfg)
    # artificially create load: gated layers are only [2,3]; L2 gates fully open (very negative tau), L3 gates fully closed
    for w in mod.layers:
        if not w.is_always_on:
            w.tau.data.fill_(-50.0 if w.layer_idx == 2 else 50.0)
    batches = [{"hidden_states": torch.randn(2, 4, h),
                "attention_mask": torch.ones(2, 4, dtype=torch.long)} for _ in range(3)]
    load = profile_layer_load(mod, batches)
    assert set(load) == set(range(n))
    assert all(load[i] == 1.0 for i in cfg.always_on_layers)
    assert load[2] == 1.0 and load[3] == 0.0, load
    tbl = render_load_table(load, always_on=cfg.always_on_layers)
    assert "L 2" in tbl and "fixed" in tbl and ">=90%" in tbl
    # promotion: only L2 passes the 90% threshold
    fixed, promoted = select_fixed_layers(load, threshold=0.9,
                                          always_on=cfg.always_on_layers)
    assert promoted == [2], promoted
    # old gate.pt (containing L2 gating keys) -> reload under the new config (L2 fixed), extra keys ignored
    with tempfile.TemporaryDirectory() as td:
        cfg.to_json(os.path.join(td, "mod_config.json"))
        sd = {k: v.cpu() for k, v in mod.state_dict().items()
              if any(s in k for s in ("router", "tau", "comp"))}
        torch.save(sd, os.path.join(td, "gate.pt"))
        cfg2 = SpeakerConfig.from_json(os.path.join(td, "mod_config.json"))
        assert cfg2.gate_mode == "threshold"
        cfg2.always_on_layers = fixed
        mod2 = convert_to_speaker(FakeHF(n=n, h=h), cfg2)
        # drop the promoted layer's gating keys: state_dict has dual hf_model.*/layers.*
        # registration paths, wildcard matched without a leading dot
        sd2 = {k: v for k, v in sd.items() if f"layers.{promoted[0]}." not in k}
        missing, unexp = mod2.load_state_dict(sd2, strict=False)
        # with strict=False, missing base keys are expected (gate.pt stores only gating);
        # gating keys must have zero missing and zero unexpected
        gate_missing = [m for m in missing
                        if any(s in m for s in ("router", "tau", "comp"))]
        assert not gate_missing and not unexp, (gate_missing, unexp)
        assert mod2.layers[promoted[0]].is_always_on
        assert mod2.layers[3].router is not None
    print("[PASS] profile+promote roundtrip (threshold)")


def test_profile_and_promote_moe():
    """moe: profile -> promote -> reload. Promotion changes the number of gated layers G ->
    joint_router dimension mismatch: following the profile_layers --out convention,
    joint_router is dropped entirely and re-initialized for continued training."""
    torch.manual_seed(0)
    n, h = 6, 32
    fake = FakeHF(n=n, h=h)
    cfg = SpeakerConfig(num_hidden_layers=n, hidden_size=h, gumbel_scale=0.0)
    mod = convert_to_speaker(fake, cfg)
    # artificially create load: gated layers [2,3], one-hot routing -> L2 (slot0) always selected, L3 (slot1) never selected
    mod.eval()
    mod.set_skip_mode("soft")
    mod.joint_router.net.weight.data.zero_()
    mod.joint_router.layer_bias.data = torch.tensor([50.0, -50.0])
    batches = [{"hidden_states": torch.randn(2, 4, h),
                "attention_mask": torch.ones(2, 4, dtype=torch.long)} for _ in range(3)]
    load = profile_layer_load(mod, batches)
    assert set(load) == set(range(n))
    assert all(load[i] == 1.0 for i in cfg.always_on_layers)
    assert load[2] == 1.0 and load[3] == 0.0, load
    fixed, promoted = select_fixed_layers(load, threshold=0.9,
                                          always_on=cfg.always_on_layers)
    assert promoted == [2], promoted
    with tempfile.TemporaryDirectory() as td:
        cfg.to_json(os.path.join(td, "mod_config.json"))
        sd = {k: v.cpu() for k, v in mod.state_dict().items()
              if any(s in k for s in ("router", "tau", "comp"))}
        assert all("joint_router" in k for k in sd), sd.keys()
        torch.save(sd, os.path.join(td, "gate.pt"))
        cfg2 = SpeakerConfig.from_json(os.path.join(td, "mod_config.json"))
        assert cfg2.gate_mode == "moe"
        cfg2.always_on_layers = fixed
        mod2 = convert_to_speaker(FakeHF(n=n, h=h), cfg2)
        assert mod2.layers[promoted[0]].is_always_on
        assert mod2.joint_router.net.weight.shape[0] == len(cfg2.gated_layers) == 1
        sd2 = {k: v for k, v in sd.items() if "joint_router" not in k}  # --out drop convention
        missing, unexp = mod2.load_state_dict(sd2, strict=False)
        assert not unexp, unexp
        assert any("joint_router" in m for m in missing), missing  # missing router keys = re-initialization
        # the re-initialized router works: the remaining gated layer L3 still participates in selection
        mod2.eval()
        mod2.set_skip_mode("soft")
        load2 = profile_layer_load(mod2, batches[:1])
        assert set(load2) == set(range(n))
    print("[PASS] profile+promote roundtrip (moe)")


def test_estimate_layer_bytes():
    # threshold: gated layers carry extra router/tau/comp -> more bytes; moe: gating params
    # live on the wrapper -> identical per layer
    mod_t = convert_to_speaker(FakeHF(), SpeakerConfig(num_hidden_layers=6, hidden_size=32,
                                                       gate_mode="threshold"))
    lb_t = estimate_layer_bytes(mod_t)
    assert set(lb_t) == set(range(6))
    assert lb_t[0] < lb_t[2], (lb_t[0], lb_t[2])
    mod_m = convert_to_speaker(FakeHF(), SpeakerConfig(num_hidden_layers=6, hidden_size=32))
    lb_m = estimate_layer_bytes(mod_m)
    assert lb_m[0] == lb_m[2] == lb_m[4], (lb_m[0], lb_m[2], lb_m[4])
    print("[PASS] estimate_layer_bytes (both modes)")


if __name__ == "__main__":
    test_select_fixed_layers()
    test_plan_placement()
    test_profile_and_promote_threshold()
    test_profile_and_promote_moe()
    test_estimate_layer_bytes()
    print("\nAll load-profile tests passed.")
