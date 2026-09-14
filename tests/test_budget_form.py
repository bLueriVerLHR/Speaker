"""Unit tests for the budget-shape ablation (r8: mean|hinge|tail).

mean = legacy λ·mean(k) (bit-identical path); hinge parks mean at a setpoint;
tail adds SLO-style P(k>B) pressure. All forms keep ∂L/∂k ≥ 0 so the dual
lever stays connected (1/var-style shapes are excluded by design: batch var
rewards polarized collapse and disconnects adapt_price).
"""
from types import SimpleNamespace

import pytest
import torch

from speaker import SpeakerConfig, SpeakerModelWrapper
from speaker.hub import _budget_core
from tests._fakes import FakeHF


def _cfg(**over):
    base = dict(num_hidden_layers=6, hidden_size=32, always_on_head=1,
                always_on_tail=1, kmax=10, over_budget_coef=0.0,
                budget_form="mean", budget_target=0.0,
                tail_coef=1.0, tail_temp=0.5)
    base.update(over)
    return SimpleNamespace(**base)


@pytest.fixture
def kk_price():
    return torch.tensor([1.0, 2.0, 3.0, 4.0]), 0.1


def test_mean_is_legacy(kk_price):
    kk, price = kk_price
    out = _budget_core(kk, _cfg(budget_form="mean"), price)
    assert out.item() == pytest.approx(0.1 * 2.5)


def test_hinge_below_is_zero_above_is_excess(kk_price):
    kk, price = kk_price
    assert _budget_core(kk, _cfg(budget_form="hinge", budget_target=2.0),
                        price).item() == pytest.approx(0.1 * 0.5)
    assert _budget_core(kk, _cfg(budget_form="hinge", budget_target=9.0),
                        price).item() == pytest.approx(0.0)


def test_hinge_auto_target_is_kmax(kk_price):
    kk, price = kk_price
    out = _budget_core(kk, _cfg(budget_form="hinge", budget_target=0.0,
                                kmax=10), price)
    assert out.item() == 0.0


def test_tail_bounds_and_value(kk_price):
    kk, price = kk_price
    out = _budget_core(kk, _cfg(budget_form="tail", budget_target=2.5,
                                tail_coef=1.0, tail_temp=0.5), price)
    # frac = mean(sigmoid([-3,-1,1,3])) = 0.5 -> 0.1*(2.5+0.5)
    assert out.item() == pytest.approx(0.30, abs=1e-4)
    mean_part = price * kk.mean().item()
    assert out.item() >= mean_part
    assert out.item() <= price * (2.5 + 1.0)


def test_tail_monotone_in_k(kk_price):
    _, price = kk_price
    cfg = _cfg(budget_form="tail", budget_target=2.0)
    a = _budget_core(torch.tensor([1.0, 1.0]), cfg, price).item()
    b = _budget_core(torch.tensor([1.0, 5.0]), cfg, price).item()
    assert a < b


def test_bad_form_raises(kk_price):
    kk, price = kk_price
    with pytest.raises(ValueError):
        _budget_core(kk, _cfg(budget_form="var"), price)


def test_grad_flows(kk_price):
    _, price = kk_price
    for form in ("mean", "hinge", "tail"):
        kk = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        loss = _budget_core(kk, _cfg(budget_form=form, budget_target=1.0),
                            price)
        loss.backward()
        assert kk.grad is not None
        assert bool((kk.grad >= 0).all())  # dual lever: monotone


def _mod(gate_mode, **over):
    cfg = SpeakerConfig(num_hidden_layers=6, hidden_size=32, gate_mode=gate_mode,
                        always_on_head=1, always_on_tail=1, kmax=10,
                        over_budget_coef=0.0, **over)
    mod = SpeakerModelWrapper(FakeHF(n=6, h=32), cfg).train()
    torch.manual_seed(0)
    x = torch.randn(2, 8, 32)
    am = torch.ones(2, 8, dtype=torch.long)
    mod(hidden_states=x, attention_mask=am)
    return mod, am


def test_mean_matches_legacy_formula():
    # guards refactor drift: mean form must equal the historical λ·mean line
    for mode in ("threshold", "moe"):
        mod, am = _mod(mode, budget_form="mean")
        got = mod.get_budget_loss(am)
        assert got is not None
        if mode == "moe":
            k = mod.route.k_soft[am.bool()]
        else:
            ste, _, _ = mod._stack_masks()
            k = ste.sum(-1)[am.bool()]
        expect = mod.mod_config.sparsity_price * k.mean()
        assert got.item() == pytest.approx(expect.item())


def test_forms_run_and_gate_grads():
    for mode in ("threshold", "moe"):
        for form in ("mean", "hinge", "tail"):
            mod, am = _mod(mode, budget_form=form, budget_target=2.0)
            loss = mod.get_budget_loss(am)
            assert torch.isfinite(loss).item()
            mod.zero_grad()
            loss.backward()
            grads = [p.grad for p in mod.get_router_parameters()
                     if p.grad is not None]
            assert grads, f"{mode}/{form}: no gate grads"


def test_hinge_parks_below_target():
    # dense-start gates (wide open, k≈4) vs high setpoint -> zero pressure
    for mode in ("threshold", "moe"):
        mod, am = _mod(mode, budget_form="hinge", budget_target=100.0)
        assert mod.get_budget_loss(am).item() == 0.0


def test_config_rejects_bad_values():
    with pytest.raises(ValueError):
        SpeakerConfig(num_hidden_layers=6, hidden_size=32, budget_form="var")
    with pytest.raises(ValueError):
        SpeakerConfig(num_hidden_layers=6, hidden_size=32, budget_target=-1.0)
    with pytest.raises(ValueError):
        SpeakerConfig(num_hidden_layers=6, hidden_size=32, tail_coef=-0.5)
    with pytest.raises(ValueError):
        SpeakerConfig(num_hidden_layers=6, hidden_size=32, tail_temp=0.0)
