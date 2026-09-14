"""Structural tests for the registries (P1 hardening, no model loads).

Covers the transformers-Auto lesson: once the core is a registry of strings,
hand edits need automated guardrails — duplicate keys, broken references and
config/display drift are caught here instead of in a 2am eval run.
"""
import inspect

import pytest
import torch

from baselines.assemble import (
    DISPLAY_PREFIX,
    FAMILY_CONFIG,
    FAMILY_REGISTRY,
    list_families,
    register_family,
)
from speaker.strategies import (
    MoeStrategy,
    ThresholdStrategy,
    canonical_gate_mode,
    get_strategy,
    list_strategies,
)
from speaker.gating import select_and_weight


def test_family_keys_match_legacy_maps():
    assert set(FAMILY_REGISTRY) == set(FAMILY_CONFIG)
    for fam, entry in FAMILY_REGISTRY.items():
        assert entry["config"] == FAMILY_CONFIG[fam]
        assert entry["prefix"] == DISPLAY_PREFIX[fam]
        assert callable(entry["fn"])


def test_pipeline_fn_signature():
    for fam, entry in FAMILY_REGISTRY.items():
        sig = inspect.signature(entry["fn"])
        assert list(sig.parameters) == ["b", "asm"], \
            f"{fam} pipeline fn must be (builder, asm)"


def test_list_families_sorted():
    fams = list_families()
    assert fams == sorted(fams)
    assert set(fams) == set(FAMILY_CONFIG)


def test_duplicate_registration_raises():
    with pytest.raises(ValueError):
        register_family("ours", "mod_config.json", "ours")(lambda b, asm: asm)


def test_unknown_family_build_raises():
    from baselines.assemble import ModelBuilder
    b = ModelBuilder("some-model-id")
    b.ckpt_dir, b.family, b.cfg = "/none", "nope", {}
    with pytest.raises(ValueError):
        b.build()


def test_strategy_aliases_canonicalize_like_config():
    assert canonical_gate_mode("speaker") == "threshold"
    assert canonical_gate_mode("mol") == "moe"
    assert canonical_gate_mode("moe") == "moe"
    assert canonical_gate_mode("threshold") == "threshold"


def test_strategy_lookup_all_names():
    for name in ["moe", "mol", "threshold", "speaker"]:
        s = get_strategy(name)
        assert s.name in ("moe", "threshold")
    assert isinstance(get_strategy("moe"), MoeStrategy)
    assert isinstance(get_strategy("threshold"), ThresholdStrategy)


def test_unknown_strategy_raises():
    with pytest.raises(ValueError):
        get_strategy("topk-everything")


def test_list_strategies():
    assert list_strategies() == ["moe", "threshold"]


def test_moe_route_matches_select_and_weight():
    torch.manual_seed(0)
    logits = torch.randn(2, 5, 6)
    a = MoeStrategy.route(logits, select_mode="topp", top_p=0.9, min_layers=1,
                          kmax=6, count_temp=0.1)
    b = select_and_weight(logits, select_mode="topp", top_p=0.9, min_layers=1,
                          kmax=6, count_temp=0.1)
    for f in ("weights", "selected", "k_soft"):
        assert torch.equal(getattr(a, f), getattr(b, f))


def test_threshold_needs_no_joint_router():
    assert not ThresholdStrategy.needs_joint_router()
    assert MoeStrategy.needs_joint_router()
