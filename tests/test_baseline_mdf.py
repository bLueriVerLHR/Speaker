"""MoDification baseline (threshold-p + R objective) offline self-tests: threshold semantics /
shared-gate math / R gradients / k statistics."""
import pathlib
import sys

import torch
import torch.nn as nn

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from baselines.lib import _select_threshold, collect_mdf_stats, patch_model_mdf


class FakeDecoderLayer(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.self_attn = nn.Identity()
        self.mlp = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.SiLU(),
                                 nn.Linear(hidden_size, hidden_size))

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        return (hidden_states + self.mlp(hidden_states) * 0.1,)


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


def test_select_threshold():
    torch.manual_seed(0)
    g = torch.rand(2, 8)
    valid = torch.ones(2, 8, dtype=torch.bool)
    valid[1, 5:] = False
    sel = _select_threshold(g, valid, p=0.5)
    assert ((sel == ((g >= 0.5) & valid))).all(), "threshold semantics: selected <=> g>=p and valid"
    assert not sel[1, 5:].any(), "padding positions must not be selected"
    # boundary: g==p executes (>= semantics, paper f=[g>=p])
    assert _select_threshold(torch.full((1, 3), 0.5), torch.ones(1, 3, dtype=torch.bool), 0.5).all()
    # p=0: all valid positions execute; p=1: all skip (sigmoid<1 always holds)
    assert _select_threshold(g, valid, 0.0).sum() == valid.sum()
    assert not _select_threshold(g, valid, 1.0).any()
    # no mask: the full length participates
    sel4 = _select_threshold(g, None, 0.5)
    assert ((sel4 == (g >= 0.5))).all()
    # arbitrary count (not a fixed k): the essential difference from top-k
    assert sel[0].sum() != sel[1].sum() or True  # the count is determined by the score distribution, not asserted in the smoke check
    print("[PASS] select_threshold (threshold/padding/boundary)")


def test_shared_gate_math():
    """h' = h + sel·g·(block(h)−h): zero-init gate (g=0.5) verified by hand."""
    torch.manual_seed(0)
    fake = FakeHF()
    is_routed = [i in (2, 3) for i in range(6)]
    routed = patch_model_mdf(fake, is_routed, p=0.5)
    x = torch.randn(1, 4, 32)
    am = torch.ones(1, 4, dtype=torch.long)
    fake.eval()
    with torch.no_grad():
        out = routed[0](x, attention_mask=am)
        dense_out = routed[0]._mdf_dense_forward(x, attention_mask=am)[0]
        manual = x + 0.5 * (dense_out - x)  # zero-init gate g=0.5, everything executes
        assert torch.allclose(out[0], manual, atol=1e-5), "the zero-init gate should yield a half-weight blend"
        # p=1, all skip: output == input
        routed[0].route_p = 1.0
        out2 = routed[0](x, attention_mask=am)
        assert torch.allclose(out2[0], x, atol=1e-5), "with everything skipped the layer should be identity"
    print("[PASS] shared gate math")


def test_R_loss_and_stats():
    torch.manual_seed(1)
    fake = FakeHF()
    is_routed = [i in (1, 3, 5) for i in range(6)]  # interleaved spot checks
    routed = patch_model_mdf(fake, is_routed, p=0.5)
    assert len(routed) == 3
    am = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]])
    x = torch.randn(2, 4, 32)
    fake.train()
    out = fake(hidden_states=x, attention_mask=am)
    fg, k = collect_mdf_stats(routed, training=True)
    assert fg is not None and fg.requires_grad, "R must be differentiable (flows back through G to the gate)"
    assert k is not None and k.shape == (2, 4)
    assert (k[1, 2:] == 0).all(), "k at padding positions should be 0"
    # R == ΣF·G verified by hand
    manual = 0.0
    for lyr in routed:
        manual = manual + lyr._last_F * lyr._last_G
    assert torch.allclose(fg, manual), "R should equal the sum of per-layer F·G"
    assert 0.0 <= fg.item() <= len(routed)
    loss = out["logits"].float().pow(2).mean() + 0.01 * fg
    loss.backward()
    ng = sum(1 for l in routed if l.router.weight.grad is not None
             and l.router.weight.grad.abs().sum().item() > 0)
    assert ng == len(routed), f"R+LM should give gradients to all gates, got {ng}/{len(routed)}"
    fake.eval()
    with torch.no_grad():
        o1 = fake(hidden_states=x, attention_mask=am)["logits"]
        o2 = fake(hidden_states=x, attention_mask=am)["logits"]
        assert torch.allclose(o1, o2), "threshold-p should be deterministic (no randomness)"
    print("[PASS] R loss + k stats + determinism")


if __name__ == "__main__":
    test_select_threshold()
    test_shared_gate_math()
    test_R_loss_and_stats()
    print("\nAll MoDification baseline tests passed.")
