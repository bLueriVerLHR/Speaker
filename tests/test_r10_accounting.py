"""Layer-count and budget rules: K counts every layer that computes for a token.

Rules under test:
- K = gated selections + always-on layers, reported everywhere through
  get_total_counts (training log, held-out eval, compare table, probe script).
- Budget: lambda * mean((K - T)^2) with the target T counting every computing
  layer. With the target set to the always-on count this is exactly
  lambda * mean(gated layers): the always-on floor is free, and the price
  can never push the count below it (gated count can't go negative, so a
  collapse is impossible by construction).
- The over-maximum penalty is off by default; old checkpoints keep the value
  they were trained with.
"""
import torch

from speaker.config import SpeakerConfig, budget_pin_warnings
from speaker.hparams import FinetuneConfig
from speaker.wrapper import SpeakerModelWrapper
from tests._fakes import FakeHF


def _mod(**over):
    kw = dict(num_hidden_layers=6, hidden_size=32, gate_mode="threshold",
              always_on_head=1, always_on_tail=1, kmax=10, over_budget_coef=0.0)
    kw.update(over)
    cfg = SpeakerConfig(**kw)
    mod = SpeakerModelWrapper(FakeHF(n=6, h=32), cfg).train()
    torch.manual_seed(0)
    x = torch.randn(2, 8, 32)
    am = torch.ones(2, 8, dtype=torch.long)
    mod(hidden_states=x, attention_mask=am)
    return mod, am


def _gated_k(mod, am):
    ste, _, _ = mod._stack_masks()
    return ste.sum(-1)[am.bool()]


def test_total_counts_equal_gated_plus_always_on():
    mod, am = _mod()
    n_always = len(mod.mod_config.always_on_layers)
    assert n_always == 2
    gated = mod.get_active_counts()
    total = mod.get_total_counts()
    assert total is not None and gated is not None
    assert torch.equal(total, gated + n_always)


def test_total_counts_no_fixed_identity():
    mod, am = _mod(always_on_layers=[], always_on_head=0, always_on_tail=0)
    assert mod.mod_config.always_on_layers == []
    assert torch.equal(mod.get_total_counts(), mod.get_active_counts())


def test_budget_is_auto_kmin_floor_free():
    # With always-on layers present, the price charges gated selections only:
    # (total - floor)^2 with floor = always-on count is exactly gated^2
    mod, am = _mod()
    loss = mod.get_budget_loss(am)
    assert loss is not None
    expect = mod.mod_config.sparsity_price * _gated_k(mod, am).mean()
    assert loss.item() == pytest_approx(expect.item())


def test_budget_ignores_kmax_when_coef_off():
    # With the over-maximum penalty off (default), the budget is the price term
    # only, even when the gated count far exceeds kmax
    mod, am = _mod(over_budget_coef=0.0, kmax=1, tau_init=-10.0)  # gates wide open
    loss = mod.get_budget_loss(am)
    expect = mod.mod_config.sparsity_price * _gated_k(mod, am).mean()
    assert loss is not None and loss.item() == pytest_approx(expect.item())


def test_legacy_over_budget_coef_still_active():
    # Old checkpoints train with their recorded penalty; it must still work
    mod, am = _mod(over_budget_coef=0.05, kmax=1, tau_init=-10.0)
    kg = _gated_k(mod, am)  # STE basis, exactly what get_budget_loss prices
    over = torch.clamp(kg - mod.mod_config.kmax, min=0)
    expect = (mod.mod_config.sparsity_price * kg.mean()
              + mod.mod_config.over_budget_coef * over.pow(2).mean())
    loss = mod.get_budget_loss(am)
    assert loss is not None
    assert loss.item() == pytest_approx(expect.item())


def test_defaults_kmax_out_of_loss_and_no_fixed_finetune():
    assert SpeakerConfig().over_budget_coef == 0.0
    hp = FinetuneConfig()
    assert hp.always_head == 0 and hp.always_tail == 0


def test_soft_variant_consistent():
    mod, am = _mod()
    gated_s = mod.get_active_counts(hard=False)
    total_s = mod.get_total_counts(hard=False)
    assert torch.equal(total_s, gated_s + len(mod.mod_config.always_on_layers))


def test_sqdev_setpoint_on_total_basis():
    # The target T counts every computing layer, including always-on ones
    mod, am = _mod(budget_form="sqdev", budget_target=8.0)
    loss = mod.get_budget_loss(am)
    assert loss is not None
    n_always = len(mod.mod_config.always_on_layers)
    ste, _, _ = mod._stack_masks()
    k_total = (ste.sum(-1) + n_always)[am.bool()]
    expect = mod.mod_config.sparsity_price * (k_total - 8.0).pow(2).mean()
    assert loss.item() == pytest_approx(expect.item())


def pytest_approx(v):
    import pytest
    return pytest.approx(v)


def test_budget_pin_warnings_mean_without_cap():
    cfg = SpeakerConfig(num_hidden_layers=6, hidden_size=32, gate_mode="threshold",
                        always_on_head=1, always_on_tail=1, budget_form="mean",
                        over_budget_coef=0.0)
    warns = budget_pin_warnings(cfg)
    assert len(warns) == 1
    assert "drifts near-dense" in warns[0]
    assert "over_budget_coef" in warns[0]


def test_budget_pin_warnings_silent_when_pinned():
    pinned = SpeakerConfig(num_hidden_layers=6, hidden_size=32, gate_mode="threshold",
                           always_on_head=1, always_on_tail=1, budget_form="mean",
                           over_budget_coef=0.05)
    setpoint = SpeakerConfig(num_hidden_layers=6, hidden_size=32, gate_mode="threshold",
                             always_on_head=1, always_on_tail=1, budget_form="sqdev",
                             budget_target=10.0, over_budget_coef=0.0)
    moe = SpeakerConfig(num_hidden_layers=6, hidden_size=32, gate_mode="moe",
                        always_on_head=1, always_on_tail=1, budget_form="mean",
                        over_budget_coef=0.0)
    assert budget_pin_warnings(pinned) == []
    assert budget_pin_warnings(setpoint) == []
    assert budget_pin_warnings(moe) == []
