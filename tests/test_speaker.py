"""Speaker self-tests (no real HF weights needed, simulated layers): regression for the core
mechanics of shared (fixed) layers + gated layers (dual scheme).

- threshold (legacy scheme): per-layer Router+tau+comp, sigmoid+STE, budget λ·mean(k)+over-kmax penalty;
- moe (default): a single-point JointRouter(H->G) at the entry emits log p, top-p/top-k selection,
  selected-layer probabilities renormalized into a weighted residual, budget λ·mean(k) via soft-inclusive STE.
Checkpoints of the two schemes are incompatible with each other; each is regressed separately.
"""
import torch
import torch.nn as nn
from speaker import (SpeakerConfig, convert_to_speaker, Router, JointRouter,  # noqa: F401
                     RouteDecision, select_and_weight,
                     MoDConfig, MoDLayerWrapper, MoDModelWrapper, convert_to_mod)
from speaker.wrapper import SpeakerLayerWrapper, SpeakerModelWrapper

# Legacy-name aliases: they point to the same implementation (old code/scripts keep importing fine)
assert MoDConfig is SpeakerConfig and MoDLayerWrapper is SpeakerLayerWrapper
assert MoDModelWrapper is SpeakerModelWrapper and convert_to_mod is convert_to_speaker


class FakeDecoderLayer(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.self_attn = nn.Identity()
        self.mlp = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.SiLU(),
                                 nn.Linear(hidden_size, hidden_size))

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        out = hidden_states + self.mlp(hidden_states) * 0.1
        if kwargs.get("use_cache"):
            return (out, kwargs.get("past_key_value"))
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


def force_onehot(mod, slot: int):
    """moe: force the joint router into one-hot (zeroed net weights + a large prior for one
    slot -> only that gated layer is selected)."""
    mod.eval()
    mod.set_skip_mode("soft")
    mod.joint_router.net.weight.data.zero_()
    mod.joint_router.layer_bias.data.fill_(0.0)
    mod.joint_router.layer_bias.data[slot] = 50.0


def test_config():
    cfg = SpeakerConfig(num_hidden_layers=24, hidden_size=1024)  # default moe
    assert cfg.gate_mode == "moe"
    assert cfg.always_on_layers == [0, 1, 22, 23]
    assert cfg.gated_layers == list(range(2, 22))
    c2 = SpeakerConfig(num_hidden_layers=8, hidden_size=32, always_on_head=3, always_on_tail=2)
    assert c2.always_on_layers == [0, 1, 2, 6, 7], c2.always_on_layers
    assert c2.gated_layers == [3, 4, 5]
    # ed5: None derives from head/tail; an explicit [] stays empty (pure gating, r6) —
    # the old falsy check refilled [] from head/tail, silently resurrecting fixed layers
    c3 = SpeakerConfig(num_hidden_layers=8, hidden_size=32, always_on_layers=[],
                       always_on_head=2, always_on_tail=2)
    assert c3.always_on_layers == [] and c3.gated_layers == list(range(8)), c3.always_on_layers
    c4 = SpeakerConfig(num_hidden_layers=8, hidden_size=32, always_on_layers=None,
                       always_on_head=1, always_on_tail=1)
    assert c4.always_on_layers == [0, 7], c4.always_on_layers
    # from_model_config must fail fast on configs without layer info (not TypeError deep
    # inside __post_init__)
    try:
        SpeakerConfig.from_model_config({})
        raise AssertionError("from_model_config should reject a config without layer info")
    except ValueError:
        pass
    # threshold mode checks
    ct = SpeakerConfig(num_hidden_layers=6, hidden_size=32, gate_mode="threshold", tau_init=-1.0)
    assert ct.gate_mode == "threshold" and ct.tau_init == -1.0
    for bad in (dict(skip_mode="fast"), dict(gate_mode="sum"),  # invalid configs must fail fast
                dict(gate_mode="moe", top_p=0.0), dict(gate_mode="moe", min_layers=0),
                dict(gate_mode="moe", kmax=0, min_layers=1), dict(gate_mode="moe", top_k=0),
                dict(gate_mode="threshold", kmax=0), dict(gate_mode="threshold", sparsity_price=-0.1)):
        try:
            SpeakerConfig(num_hidden_layers=6, hidden_size=32, **bad)
            raise AssertionError(f"should reject invalid config {bad}")
        except ValueError:
            pass
    print("[PASS] config (both modes)", cfg.summary())


def test_config_from_json():
    """Old checkpoints (no gate_mode field) infer threshold; deprecated legacy fields are
    silently dropped; new checkpoints keep moe."""
    import json
    import os
    import tempfile
    cfg = SpeakerConfig(num_hidden_layers=6, hidden_size=32, top_p=0.8)
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "c.json")
        cfg.to_json(p)
        c2 = SpeakerConfig.from_json(p)
        assert c2.gate_mode == "moe" and c2.top_p == 0.8
        # old checkpoint: no gate_mode + legacy fields mixed in
        d = json.load(open(p))
        d.pop("gate_mode")
        d.update({"tau_init": -3.0, "use_ste": False, "router_hidden_dim": 8})
        json.dump(d, open(p, "w"))
        c3 = SpeakerConfig.from_json(p)
        assert c3.gate_mode == "threshold", "old checkpoints must be inferred as threshold"
        assert c3.tau_init == -3.0 and c3.use_ste is False, "legacy-scheme fields must parse as-is in threshold mode"
    print("[PASS] config from_json (mode inference + old-key filter)")


def test_router():
    """threshold: the per-layer Router has no bias; the threshold is carried solely by tau."""
    r = Router(hidden_size=32)
    assert r.net.bias is None, "Router must have no bias; the threshold is carried solely by tau"
    x = torch.randn(2, 8, 32)
    logits = r(x)
    assert logits.shape == (2, 8, 1)
    r2 = Router(hidden_size=32, hidden_dim=8)  # MLP variant
    assert r2(x).shape == (2, 8, 1)
    print("[PASS] router (threshold)")


def test_joint_router():
    """moe: single-point JointRouter, bias-free net + layer_bias prior."""
    r = JointRouter(hidden_size=32, n_gated=5)
    assert r.net.bias is None, "net carries no bias; the per-layer prior is carried solely by layer_bias"
    x = torch.randn(2, 8, 32)
    logits = r(x)
    assert logits.shape == (2, 8, 5) and logits.dtype == torch.float32
    print("[PASS] joint router (moe)")


def test_select_top_p():
    G = 4
    # peaked distribution: top probability >= p -> k=1, weight≈1
    logits = torch.tensor([[[10.0, 0.0, 0.0, 0.0]]])
    rd = select_and_weight(logits, top_p=0.9, kmax=10)
    assert rd.selected[0, 0].argmax().item() == 0 and rd.k[0, 0].item() == 1
    assert abs(rd.weights[0, 0, 0].item() - 1.0) < 1e-3
    # uniform distribution: p=0.9 -> the cumulative mass only reaches p at the 4th -> k=4, weights split evenly
    rd2 = select_and_weight(torch.zeros(1, 1, G), top_p=0.9, kmax=10)
    assert rd2.k[0, 0].item() == 4, rd2.k
    assert torch.allclose(rd2.weights[0, 0], torch.full((G,), 0.25), atol=1e-6)
    # kmax hard cap: uniform G=8 with p=0.9 would give k=8, truncated to 3
    rd3 = select_and_weight(torch.zeros(1, 1, 8), top_p=0.9, kmax=3)
    assert rd3.k[0, 0].item() == 3 and abs(rd3.weights.sum().item() - 1.0) < 1e-5
    # min_layers floor: peaked + min_layers=2 -> k=2
    rd4 = select_and_weight(logits, top_p=0.9, min_layers=2, kmax=10)
    assert rd4.k[0, 0].item() == 2
    # selected-layer weights always sum to 1, unselected are 0
    rd5 = select_and_weight(torch.randn(2, 5, 6), top_p=0.7, kmax=5)
    s = rd5.weights.sum(-1)
    assert torch.allclose(s, torch.ones_like(s), atol=1e-4)
    # padding: all zeros
    valid = torch.ones(2, 5)
    valid[1, 3:] = 0.0
    rd6 = select_and_weight(torch.randn(2, 5, 6), top_p=0.7, kmax=5, valid=valid)
    assert rd6.k[1, 3].item() == 0 and rd6.k[1, 4].item() == 0 and rd6.k[1, 2].item() > 0
    assert rd6.weights[1, 3].abs().sum().item() == 0
    # topk mode: fixed k
    rd7 = select_and_weight(torch.randn(2, 5, 6), select_mode="topk", top_k=3, kmax=5)
    assert (rd7.k == 3).all()
    # STE: the soft count is differentiable, gradient can flow back to logits
    lg = torch.randn(1, 1, 6, requires_grad=True)
    rd8 = select_and_weight(lg, top_p=0.9, kmax=6, count_temp=0.1)
    assert rd8.k_soft.requires_grad, "k_soft must be differentiable (gradient path for the budget λ·mean(k))"
    rd8.k_soft.sum().backward()
    assert lg.grad is not None and lg.grad.abs().sum().item() > 0, "soft-inclusive gradient should reach the router"
    print("[PASS] select top-p/top-k + STE (moe)")


def test_budget_threshold():
    """threshold: pure Lagrangian formulation; both router and tau should receive gradients."""
    n, h = 6, 32
    fake = FakeHF(n=n, h=h)
    cfg = SpeakerConfig(num_hidden_layers=n, hidden_size=h, gate_mode="threshold",
                        kmax=3, temp_affinity=1.0, gumbel_scale=1.0, use_ste=True,
                        sparsity_price=0.05)  # pure Lagrangian formulation: regularizer = λ·k
    assert cfg.gated_layers == [2, 3]
    mod = SpeakerModelWrapper(fake, cfg)
    assert mod.joint_router is None, "threshold mode must not create a JointRouter"
    gate_ps0 = mod.get_router_parameters()
    assert len(gate_ps0) > 0 and all(p.requires_grad for p in gate_ps0)
    am = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]])
    x = torch.randn(2, 4, h)
    mod.train()
    out = mod(hidden_states=x, attention_mask=am)
    assert "logits" in out
    counts = mod.get_active_counts()
    soft = mod.get_soft_counts()
    assert counts.shape == (2, 4) and soft.shape == (2, 4)
    assert counts[1, 3].item() == 0  # padding positions should be 0
    loss = mod.get_budget_loss(am)
    assert loss is not None and loss.requires_grad, "budget loss must be differentiable"
    mod.zero_grad()
    loss.backward()
    gate_ps = mod.get_router_parameters()
    ng = sum(1 for p in gate_ps if p.grad is not None and p.grad.abs().sum().item() > 0)
    assert ng > 0, "budget loss should give gradients to the gate"
    named = {nm: p for nm, p in mod.named_parameters()
             if p.grad is not None and p.grad.abs().sum().item() > 0}
    assert any("router" in nm for nm in named), "router should receive gradients"
    assert any(nm.endswith("tau") for nm in named), "tau should receive gradients"
    aux = mod.get_aux_loss()
    assert aux is not None and aux.requires_grad
    print(f"[PASS] budget grad (threshold) gate {ng}/{len(gate_ps)} loss {loss.item():.4f}")
    # eval: deterministic with noise off
    mod.eval()
    with torch.no_grad():
        o1 = mod(hidden_states=x, attention_mask=am)["logits"]
        o2 = mod(hidden_states=x, attention_mask=am)["logits"]
        assert torch.allclose(o1, o2), "eval should be deterministic"
    # hard skip: the first batch must not crash
    mod.set_skip_mode("hard")
    with torch.no_grad():
        o = mod(hidden_states=x, attention_mask=am)
        assert "logits" in o
    print("[PASS] eval deterministic + hard skip (threshold)")


def test_budget_moe():
    """moe: budget is differentiable and reaches joint_router; aux (z+balance+cos) is
    differentiable; padding/cap/floor are correct."""
    n, h = 6, 32
    fake = FakeHF(n=n, h=h)
    cfg = SpeakerConfig(num_hidden_layers=n, hidden_size=h, kmax=4, top_p=0.9,
                        sparsity_price=0.05, gumbel_scale=0.0)  # no noise: train/eval agree
    assert cfg.gated_layers == [2, 3]
    mod = SpeakerModelWrapper(fake, cfg)
    gate_ps = mod.get_router_parameters()
    assert len(gate_ps) == 2 and all(p.requires_grad for p in gate_ps), "net.weight + layer_bias"
    named = [nm for nm, _ in mod.named_parameters() if "joint_router" in nm]
    assert len(named) == 2, named  # single registration, no dual path
    am = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]])
    x = torch.randn(2, 4, h)
    mod.train()
    out = mod(hidden_states=x, attention_mask=am)
    counts = mod.get_active_counts()
    soft = mod.get_soft_counts()
    assert counts.shape == (2, 4) and soft.shape == (2, 4)
    assert counts[1, 3].item() == 0, "k at padding positions should be 0"
    assert (counts[am.bool()] >= 1).all(), "with min_layers=1 each valid token activates at least 1 gated layer"
    assert counts.max().item() <= cfg.kmax, "kmax hard cap"
    loss = mod.get_budget_loss(am)
    assert loss is not None and loss.requires_grad, "budget loss must be differentiable"
    mod.zero_grad()
    loss.backward()
    ng = sum(1 for p in gate_ps if p.grad is not None and p.grad.abs().sum().item() > 0)
    assert ng > 0, "budget loss should give gradients to the joint router"
    out = mod(hidden_states=x, attention_mask=am)
    aux = mod.get_aux_loss()
    assert aux is not None and aux.requires_grad
    mod.zero_grad()
    (out["logits"].sum() + aux).backward()
    ng2 = sum(1 for p in gate_ps if p.grad is not None and p.grad.abs().sum().item() > 0)
    assert ng2 == len(gate_ps), f"LM+aux should give gradients to all routing params, got {ng2}/{len(gate_ps)}"
    print(f"[PASS] budget grad (moe) {ng}/{len(gate_ps)} loss {loss.item():.4f} aux grad {ng2}")
    mod.eval()
    with torch.no_grad():
        o1 = mod(hidden_states=x, attention_mask=am)["logits"]
        o2 = mod(hidden_states=x, attention_mask=am)["logits"]
        assert torch.allclose(o1, o2), "eval should be deterministic"
    mod.set_skip_mode("hard")
    with torch.no_grad():
        o = mod(hidden_states=x, attention_mask=am)
        assert "logits" in o
    kh = mod.get_active_counts()
    assert torch.equal(kh, counts), "with no noise in eval, hard k should match the train formulation"
    print("[PASS] eval deterministic + hard forward (moe)")


def test_router_temp_calib():
    """moe: init-time router-temperature calibration (r7 fix): a peaked start (large-norm
    random projection, mimics 7B) is flattened until the top-p mean k reaches the target;
    router_temp=1.0 must stay bit-for-bit identity; threshold returns {}."""
    n, h = 12, 32
    torch.manual_seed(0)
    cfg = SpeakerConfig(num_hidden_layers=n, hidden_size=h, gate_mode="moe",
                        select_mode="topp", top_p=0.7, kmax=8, gumbel_scale=0.0)
    mod = SpeakerModelWrapper(FakeHF(n=n, h=h), cfg)
    assert cfg.gated_layers == list(range(2, 10)), cfg.gated_layers
    # threshold mirror: no-op
    mod2 = SpeakerModelWrapper(FakeHF(n=n, h=h),
                               SpeakerConfig(num_hidden_layers=n, hidden_size=h,
                                             gate_mode="threshold"))
    assert mod2.calibrate_router_temp([{"hidden_states": torch.randn(2, 4, h)}]) == {}
    # identity: router_temp=1.0 keeps the forward bit-for-bit unchanged; flattening changes it
    am = torch.ones(2, 8)
    x = torch.randn(2, 8, h)
    mod.eval()
    with torch.no_grad():
        o1 = mod(hidden_states=x, attention_mask=am)["logits"]
        cfg.router_temp = 1.0
        o2 = mod(hidden_states=x, attention_mask=am)["logits"]
        assert torch.equal(o1, o2), "router_temp=1.0 must be bit-for-bit identity"
        cfg.router_temp = 2.5
        o3 = mod(hidden_states=x, attention_mask=am)["logits"]
        assert not torch.equal(o1, o3), "a flattening temperature must change routing"
        cfg.router_temp = 1.0
    # peaked start -> calibration lifts mean k to the target
    mod.joint_router.net.weight.data *= 16.0
    batches = [{"hidden_states": torch.randn(2, 16, h),
                "attention_mask": torch.ones(2, 16)} for _ in range(3)]
    info = mod.calibrate_router_temp(batches, target_k=6)
    assert info, "moe calibration should return info"
    assert info["k_before"] < 4.5, f"peaked start expected, got k0 {info['k_before']}"
    assert abs(info["k_after"] - 6.0) <= 0.5, info
    assert cfg.router_temp > 1.0, info
    # persistence: config roundtrip keeps the calibrated temperature
    rt = float(cfg.router_temp)
    rt2 = SpeakerConfig(**{k: v for k, v in cfg.to_dict().items()
                           if k != "gated_layers"}).router_temp
    assert rt2 == rt, (rt, rt2)
    print(f"[PASS] router temp calibration (moe) temp {rt:.3f} "
          f"k0 {info['k_before']:.1f} -> {info['k_after']:.1f} (target 6)")


def test_weighted_residual_onehot():
    """moe: under one-hot routing only the selected gated layer executes and w=1 (full
    residual), the rest stay identity; soft weight mixing is correct."""
    n, h = 8, 32
    fake = FakeHF(n=n, h=h)
    inner = list(fake.model.layers)  # references to the original layers (the ModuleList is replaced after convert)
    cfg = SpeakerConfig(num_hidden_layers=n, hidden_size=h, kmax=4, gumbel_scale=0.0)
    mod = SpeakerModelWrapper(fake, cfg)
    mod.eval()
    mod.set_skip_mode("soft")  # soft forward still makes a hard selection, but skips nothing -> everything executes, weights take effect
    gated = cfg.gated_layers  # [2,3,4,5]
    for slot in range(len(gated)):
        force_onehot(mod, slot)
        x = torch.randn(1, 5, h)
        with torch.no_grad():
            out = mod(hidden_states=x)["logits"]
        # manual replay: shared layers pass through, among gated layers only the slot layer executes (w=1), the rest stay identity
        hs = x
        for i, layer in enumerate(inner):
            if cfg.is_always_on(i):
                hs = layer(hs)[0]
            elif i == gated[slot]:
                hs = hs + layer.mlp(hs) * 0.1  # FakeDecoderLayer: h + mlp(h)*0.1, w=1
        expect = fake.lm_head(hs)
        assert torch.allclose(out, expect, atol=1e-4), f"slot {slot} weighted residual mismatch"
    # soft selection with renormalized probabilities: medium bias, manual replay (top-p selection + renorm)
    mod.joint_router.net.weight.data.zero_()
    mod.joint_router.layer_bias.data = torch.tensor([2.0, 1.0, 0.0, 0.0])
    x = torch.randn(1, 3, h)
    with torch.no_grad():
        out = mod(hidden_states=x)["logits"]
        hs = x
        w = torch.softmax(mod.joint_router(torch.zeros(1, 3, h)), -1)[0, 0]  # p = softmax over the bias (zero input)
        sp, oi = w.sort(descending=True)
        cum = sp.cumsum(-1)
        cnt = int((cum < 0.9).sum().item()) + 1  # first position where cum>=p = prefix length
        wsel = torch.zeros_like(w)
        wsel[oi[:cnt]] = w[oi[:cnt]] / w[oi[:cnt]].sum()
        for i, layer in enumerate(inner):
            if cfg.is_always_on(i):
                hs = layer(hs)[0]
            else:
                fj = hs + layer.mlp(hs) * 0.1
                hs = hs + wsel[cfg.gated_layers.index(i)] * (fj - hs)
        expect = fake.lm_head(hs)
    assert torch.allclose(out, expect, atol=1e-4), "soft weight mixing mismatch"
    print("[PASS] weighted residual (moe, one-hot + soft)")


def test_sparse_cache_threshold():
    """threshold sparse KV cache: prefill must execute / decode with the gate closed neither
    executes nor writes K/V / legacy cache falls back to full compute."""
    H = 16

    class FakeLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, hs, attention_mask=None, **kw):
            self.calls += 1
            return (hs + 1.0,)

    class FakeCache:
        def __init__(self):
            self.store = {}

        def get_seq_length(self, i):
            return self.store[i][0].shape[2] if i in self.store else 0

        def update(self, k, v, layer_idx, cache_kwargs=None):
            if layer_idx in self.store:
                self.store[layer_idx] = (torch.cat([self.store[layer_idx][0], k], dim=2),
                                         torch.cat([self.store[layer_idx][1], v], dim=2))
            else:
                self.store[layer_idx] = (k, v)
            return self.store[layer_idx]

    cfg = SpeakerConfig(num_hidden_layers=6, hidden_size=H, gate_mode="threshold",
                        tau_init=10.0, gumbel_scale=0.0,
                        skip_mode="hard", use_ste=False)  # very large tau -> gate always closed
    inner = FakeLayer()
    w = SpeakerLayerWrapper(inner, layer_idx=3, config=cfg)
    w.eval()
    x = torch.randn(2, 4, H)
    pos = torch.arange(4).unsqueeze(0).expand(2, -1)
    # prefill: empty cache + use_cache -> must execute (to keep the cache filled)
    out = w(x, past_key_values=FakeCache(), use_cache=True, position_ids=pos)
    assert inner.calls == 1, "prefill must execute"
    # decode: this layer's cache already has content -> skip, no execution and no K/V write
    c = FakeCache()
    c.update(torch.randn(2, 4, 4, 4), torch.randn(2, 4, 4, 4), 3)
    n0 = inner.calls
    out2 = w(x[:, -1:, :], past_key_values=c, use_cache=True, position_ids=torch.tensor([[4, 4]]))
    assert inner.calls == n0, "decode skip must not execute the layer"
    assert c.get_seq_length(3) == 4, f"sparse cache must not write K/V, length should stay 4, got {c.get_seq_length(3)}"
    assert w._skip_hits == 1, "the sparse skip counter should be recorded"
    assert torch.equal(out2[0], x[:, -1:, :]), "skip should return the hidden states unchanged"
    # legacy tuple cache (no get_seq_length) -> conservative fallback to full compute
    w2 = SpeakerLayerWrapper(FakeLayer(), layer_idx=2, config=cfg)
    w2.eval()
    w2(x[:, -1:, :], past_key_values=([], []), use_cache=True, position_ids=torch.tensor([[4, 4]]))
    assert w2.layer.calls == 1, "legacy cache should fall back to full compute"
    # no cache (use_cache=False) -> skip directly
    w3 = SpeakerLayerWrapper(FakeLayer(), layer_idx=2, config=cfg)
    w3.eval()
    w3(x[:, -1:, :], use_cache=False)
    assert w3.layer.calls == 0, "no cache should skip directly"
    print("[PASS] sparse cache semantics (threshold)")


def test_sparse_cache_moe():
    """moe sparse KV cache: prefill executes everything to fill the cache / decode: unselected
    gated layers neither execute nor write K/V / shared layers always execute / legacy
    fallback / no cache skips directly."""

    class FakeCache:
        def __init__(self):
            self.store = {}

        def get_seq_length(self, i):
            return self.store[i][0].shape[2] if i in self.store else 0

        def update(self, k, v, layer_idx, cache_kwargs=None):
            if layer_idx in self.store:
                self.store[layer_idx] = (torch.cat([self.store[layer_idx][0], k], dim=2),
                                         torch.cat([self.store[layer_idx][1], v], dim=2))
            else:
                self.store[layer_idx] = (k, v)
            return self.store[layer_idx]

    n, h = 10, 32
    fake = FakeHF(n=n, h=h)
    cfg = SpeakerConfig(num_hidden_layers=n, hidden_size=h, kmax=4, gumbel_scale=0.0)
    mod = SpeakerModelWrapper(fake, cfg)
    force_onehot(mod, slot=0)  # only L2 (slot0) is selected; L3-L7 (gated) are unselected
    mod.eval()
    mod.set_skip_mode("hard")
    x = torch.randn(1, 2, h)
    pos = torch.arange(2).unsqueeze(0)
    calls = {i: 0 for i in range(n)}
    orig = {i: fake.model.layers[i] for i in range(n)}

    class Counting(nn.Module):
        def __init__(self, inner, idx):
            super().__init__()
            self.inner = inner
            self.idx = idx

        def forward(self, *a, **kw):
            calls[self.idx] += 1
            return self.inner(*a, **kw)

    for i in range(n):
        fake.model.layers[i] = Counting(orig[i], i)
    mod2 = SpeakerModelWrapper(fake, cfg)  # rebuild the wrapper (wrapping Counting)
    force_onehot(mod2, slot=0)
    mod2.eval()
    mod2.set_skip_mode("hard")
    c = FakeCache()
    with torch.no_grad():
        mod2(hidden_states=x, past_key_values=c, use_cache=True, position_ids=pos)
    assert all(calls[i] == 1 for i in range(n)), f"prefill executes all layers, got {calls}"
    # decode: L2 selected -> executes; L3-L7 unselected -> skipped, no K/V write; shared layers always execute
    c2 = FakeCache()
    for i in range(n):
        c2.update(torch.randn(1, 2, 2, 4), torch.randn(1, 2, 2, 4), i)
    calls.clear()
    calls.update({i: 0 for i in range(n)})
    with torch.no_grad():
        mod2(hidden_states=x[:, -1:, :], past_key_values=c2, use_cache=True,
             position_ids=torch.tensor([[2]]))
    assert calls[2] == 1, "the selected layer must execute at decode"
    assert all(calls[i] == 0 for i in range(3, 8)), f"unselected gated layers should be skipped at decode, got {calls}"
    assert all(calls[i] == 1 for i in (0, 1, 8, 9)), "shared layers always execute at decode"
    assert c2.get_seq_length(3) == 2, f"sparse cache must not write K/V, got {c2.get_seq_length(3)}"
    assert mod2.get_skip_hits() == 5, f"5 unselected gated layers each skip once, got {mod2.get_skip_hits()}"
    # legacy tuple cache (no get_seq_length) -> unselected layers conservatively fall back to full compute
    w3 = mod2.layers[3]
    with torch.no_grad():
        w3(x[:, -1:, :], past_key_values=([], []), use_cache=True,
           position_ids=torch.tensor([[2]]))
    assert calls[3] == 1, "legacy cache should fall back to full compute"
    # no cache (use_cache=False) -> unselected layers skip directly
    calls[3] = 0
    with torch.no_grad():
        w3(x[:, -1:, :], use_cache=False)
    assert calls[3] == 0, "no cache should skip directly"
    print("[PASS] sparse cache semantics (moe)")


def test_comp_and_cos_reg():
    """threshold trio: comp zero-init + skip compensation / cos router regularizer goes into
    aux / old checkpoints (without comp) stay loadable / tau calibration"""
    n, h = 6, 32
    fake = FakeHF(n=n, h=h)
    cfg = SpeakerConfig(num_hidden_layers=n, hidden_size=h, gate_mode="threshold",
                        cos_reg_coef=0.05)
    mod = SpeakerModelWrapper(fake, cfg)
    gated = [w for w in mod.layers if not w.is_always_on]
    assert gated and all(w.comp is not None for w in gated)
    assert all(w.comp.abs().sum().item() == 0 for w in gated), "comp must be zero-initialized (legacy behavior)"
    am = torch.ones(2, 4, dtype=torch.long)
    x = torch.randn(2, 4, h)
    mod.train()
    mod(hidden_states=x, attention_mask=am)
    aux = mod.get_aux_loss()
    assert aux is not None and aux.requires_grad, "aux (including the cos regularizer) must be differentiable"
    mod.zero_grad()
    aux.backward()
    ps = mod.get_router_parameters()
    assert any(p.grad is not None and p.grad.abs().sum().item() > 0 for p in ps), "the cos regularizer should give gradients to the gate"
    assert any("comp" in n_ for n_, p in mod.named_parameters()), "comp should be registered as a parameter"
    # old checkpoint (no comp): the strict-load diff should be exactly comp
    sd = {k: v.cpu() for k, v in mod.state_dict().items() if "comp" not in k}
    missing, unexp = mod.load_state_dict(sd, strict=False)
    assert not unexp and missing and all("comp" in m for m in missing), (missing, unexp)
    # hard with all gates closed + constant comp -> output h+comp
    w = gated[0]
    w.comp.data.fill_(0.5)
    w.tau.data.fill_(50.0)  # gate always closed
    w.eval()
    mod.eval()
    mod.set_skip_mode("hard")
    y = w(x[:1, -1:, :], use_cache=False)
    assert torch.allclose(y[0], x[:1, -1:, :] + 0.5), "skip should add the compensation"
    # tau calibration: run a batch in soft mode, every gated layer gets a new tau
    w.tau.data.fill_(0.0)
    mod.set_skip_mode("soft")
    taus = mod.calibrate_tau([{"hidden_states": x, "attention_mask": am}], spread=0.5)
    assert set(taus) == {g.layer_idx for g in gated}, taus.keys()
    # in moe mode tau calibration should be a no-op
    cfg_moe = SpeakerConfig(num_hidden_layers=n, hidden_size=h)
    mod_moe = SpeakerModelWrapper(FakeHF(n=n, h=h), cfg_moe)
    assert mod_moe.calibrate_tau([{"hidden_states": x, "attention_mask": am}]) == {}
    assert mod_moe.get_tau_params() == {}
    print(f"[PASS] comp+cos calib (threshold) taus [{min(taus.values()):+.2f},{max(taus.values()):+.2f}]")


def test_placement_and_usage():
    """Placement: the moe joint-router hub registers once; layers can live across devices
    (simulated with cpu); load stats reset on read."""
    n, h = 6, 32
    fake = FakeHF(n=n, h=h)
    cfg = SpeakerConfig(num_hidden_layers=n, hidden_size=h)
    mod = SpeakerModelWrapper(fake, cfg)
    am = torch.ones(1, 4, dtype=torch.long)
    mod.train()
    mod(hidden_states=torch.randn(1, 4, h), attention_mask=am)
    usage = mod.get_layer_usage()
    assert usage[0] == (1.0, 1.0) and usage[2][0] > 0, usage
    usage2 = mod.get_layer_usage()
    assert usage2[2] == (0.0, 0.0), "read once, cleared once"
    # CPU "placement" simulates cross-device: joint_router on the same device as the entry layer is enough to run
    mod.layers[3].to("cpu")
    mod.eval()
    with torch.no_grad():
        mod(hidden_states=torch.randn(1, 4, h), attention_mask=am)
    print("[PASS] placement/usage (moe)")


def test_resume_gate_pt():
    """gate.pt roundtrip: threshold stores per-layer router/tau/comp; moe stores joint_router.
    The key sets of the two schemes are mutually exclusive (the incompatibility)."""
    import os
    import tempfile
    n, h = 6, 32
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "gate.pt")
        # threshold
        cfg_t = SpeakerConfig(num_hidden_layers=n, hidden_size=h, gate_mode="threshold")
        mod_t = SpeakerModelWrapper(FakeHF(n=n, h=h), cfg_t)
        sd_t = {k: v.cpu() for k, v in mod_t.state_dict().items()
                if any(s in k for s in ("router", "tau", "comp"))}
        assert any(".tau" in k for k in sd_t) and all("joint_router" not in k for k in sd_t)
        torch.save(sd_t, p)
        mod_t2 = SpeakerModelWrapper(FakeHF(n=n, h=h), cfg_t)
        missing_t, unexp_t = mod_t2.load_state_dict(torch.load(p), strict=False)
        gate_missing_t = [m for m in missing_t if any(s in m for s in ("router", "tau", "comp"))]
        assert not gate_missing_t and not unexp_t, (gate_missing_t, unexp_t)
        # moe
        cfg_m = SpeakerConfig(num_hidden_layers=n, hidden_size=h)
        mod_m = SpeakerModelWrapper(FakeHF(n=n, h=h), cfg_m)
        sd_m = {k: v.cpu() for k, v in mod_m.state_dict().items()
                if any(s in k for s in ("router", "tau", "comp"))}
        assert all("joint_router" in k for k in sd_m), sd_m.keys()
        torch.save(sd_m, p)
        mod_m2 = SpeakerModelWrapper(FakeHF(n=n, h=h), cfg_m)
        missing_m, unexp_m = mod_m2.load_state_dict(torch.load(p), strict=False)
        gate_missing_m = [m for m in missing_m if "joint_router" in m]
        assert not gate_missing_m and not unexp_m, (gate_missing_m, unexp_m)
    print("[PASS] resume gate.pt roundtrip (both modes, key sets mutually exclusive)")


if __name__ == "__main__":
    test_config()
    test_config_from_json()
    test_router()
    test_joint_router()
    test_select_top_p()
    test_budget_threshold()
    test_budget_moe()
    test_router_temp_calib()
    test_weighted_residual_onehot()
    test_sparse_cache_threshold()
    test_sparse_cache_moe()
    test_comp_and_cos_reg()
    test_placement_and_usage()
    test_resume_gate_pt()
    print("\nAll speaker tests passed.")
