"""Unit tests for the packaging seams (P2/P3, offline, no weights).

safetensors ckpt roundtrip + auto-detect load, torchmetrics accuracy
equivalence, pydantic config validation, sweep cartesian expansion, streaming
dataset equivalence with SFTDataset.
"""
import json
import os
import tempfile

import pytest
import torch

from tests._fakes import FakeHF
from data.sft import SFTDataset, SFTStreamDataset, iter_sft_texts
from speaker import SpeakerConfig, SpeakerModelWrapper
from speaker.checkpoint import load_gate, save_gate
from speaker.config_model import HAS_PYDANTIC, validate_dict
from speaker.metrics import per_token_correct
from speaker.ruler import batch_accuracy
from tools.sweep import expand_runs, load_sweep


def _gate_params(mod):
    return {k: v.cpu().clone() for k, v in mod.state_dict().items()
            if any(s in k for s in ("router", "tau", "comp"))}


def test_safetensors_roundtrip_and_autodetect():
    n, h = 6, 32
    with tempfile.TemporaryDirectory() as td:
        for mode in ("threshold", "moe"):
            cfg = SpeakerConfig(num_hidden_layers=n, hidden_size=h, gate_mode=mode)
            mod = SpeakerModelWrapper(FakeHF(n=n, h=h), cfg)
            ref = _gate_params(mod)
            sd = save_gate(mod, td, fmt="safetensors")
            assert os.path.exists(os.path.join(td, "gate.safetensors"))
            assert set(sd) == set(ref)
            mod2 = SpeakerModelWrapper(FakeHF(n=n, h=h), cfg)
            missing, unexp = load_gate(mod2, td)  # no filename: auto-detects .safetensors
            gate_missing = [m for m in missing
                            if any(s in m for s in ("router", "tau", "comp"))]
            assert not gate_missing
            assert not unexp
            for k, v in ref.items():
                assert torch.equal(mod2.state_dict()[k].cpu(), v)


def test_safetensors_pt_default_unchanged():
    n, h = 6, 32
    with tempfile.TemporaryDirectory() as td:
        cfg = SpeakerConfig(num_hidden_layers=n, hidden_size=h, gate_mode="threshold")
        mod = SpeakerModelWrapper(FakeHF(n=n, h=h), cfg)
        save_gate(mod, td)
        assert os.path.exists(os.path.join(td, "gate.pt"))
        assert not os.path.exists(os.path.join(td, "gate.safetensors"))


# Accuracy accounting pins the historical denominator quirks (why no torchmetrics
# here): the denominator is the FULL labels mask (position 0 counts although
# correctness is next-token, position T-1 has no prediction and always counts as
# incorrect). Every acc floor was tuned on this accounting — "fixing" it
# re-baselines all targets, so it is a research change, not a refactor. These
# tests stop future cleanups from silently "repairing" it.
def test_accuracy_last_position_always_wrong():
    torch.manual_seed(0)
    logits = torch.randn(2, 8, 50)
    labels = torch.randint(0, 50, (2, 8))
    correct = per_token_correct(logits.float(), labels)
    assert not correct[:, -1].any()


def test_accuracy_shift_and_padding():
    torch.manual_seed(1)
    logits = torch.randn(2, 8, 50)
    labels = torch.randint(0, 50, (2, 8))
    labels[0, :2] = -100
    correct = per_token_correct(logits.float(), labels)
    pred = logits[:, :-1].argmax(-1)
    tgt = labels[:, 1:]
    expect = torch.zeros(2, 8, dtype=torch.bool)
    expect[:, :-1] = (pred == tgt) & (tgt != -100)
    assert torch.equal(correct, expect)


def test_ruler_uses_same_formula():
    torch.manual_seed(2)
    logits = torch.randn(2, 8, 50)
    batch = {"labels": torch.randint(0, 50, (2, 8)),
             "attention_mask": torch.ones(2, 8, dtype=torch.long)}
    correct = per_token_correct(logits.float(), batch["labels"])
    valid = batch["labels"] != -100
    assert batch_accuracy(logits, batch, "labels") == \
        pytest.approx(correct[valid].float().mean().item())


def test_config_valid_dict_passes():
    cfg = SpeakerConfig(num_hidden_layers=6, hidden_size=32)
    out = validate_dict(cfg.to_dict())
    assert out["gate_mode"] == "moe"


def test_config_alias_and_bad_value():
    cfg = SpeakerConfig(num_hidden_layers=6, hidden_size=32, gate_mode="mol")
    assert validate_dict(cfg.to_dict())["gate_mode"] == "moe"
    bad = cfg.to_dict()
    bad["gate_mode"] = "bogus"
    with pytest.raises(ValueError):
        validate_dict(bad)


def test_config_flag():
    assert isinstance(HAS_PYDANTIC, bool)


def test_sweep_cartesian_expansion():
    spec = {"command": ["finetune/train.py", "--gate_mode", "moe"],
            "axes": {"kmax": [12, 16], "ul_coef": [0.3, 0.6]}}
    runs = expand_runs(spec)
    assert len(runs) == 4
    names = sorted(t for t, _ in runs)
    assert names == ["kmax12_ul_coef0.3", "kmax12_ul_coef0.6",
                     "kmax16_ul_coef0.3", "kmax16_ul_coef0.6"]
    for _, argv in runs:
        assert "--kmax" in argv
        assert "--ul_coef" in argv


def test_sweep_no_axes_single():
    assert expand_runs({"command": ["a"]}) == [("single", ["a"])]


def test_sweep_example_configs_load():
    for fn in ("configs/moe_kmax.yaml", "configs/speaker_repair.yaml"):
        spec = load_sweep(fn)
        assert expand_runs(spec)


def test_typer_app_builds_config():
    """The Typer entry packs flags into a FinetuneConfig (no weights:
    run_finetune is stubbed, only the wiring is exercised)."""
    from unittest.mock import patch
    from typer.testing import CliRunner
    from finetune.cli import app
    from speaker.hparams import FinetuneConfig
    argv = ["--gate_mode", "mol", "--no-use_lora", "--lr", "3e-5",
            "--seed", "42", "--kmax", "12"]
    with patch("finetune.pipeline.run_finetune") as m:
        r = CliRunner().invoke(app, argv)
        assert r.exit_code == 0, r.output
        cfg = m.call_args[0][0]
    assert isinstance(cfg, FinetuneConfig)
    assert cfg.gate_mode == "mol"
    assert not cfg.use_lora
    assert cfg.lr == 3e-5
    assert cfg.seed == 42
    assert cfg.kmax == 12
    # untouched defaults still match the historical argparse values
    assert cfg.budget_form == "mean"
    assert cfg.save_dir == "/tmp/mod_ckpt"
    assert cfg.lora_targets == "q_proj,v_proj"


def test_train_py_entry_uses_typer():
    import ast
    with open("finetune/train.py", encoding="utf-8") as f:
        tree = ast.parse(f.read())
    mains = [n for n in ast.walk(tree)
             if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)]
    calls = [ast.unparse(n.value.func) for n in mains]
    assert "app" in calls


def test_all_apps_help():
    from typer.testing import CliRunner
    import importlib
    mods = ["finetune.cli", "pretrain.train", "finetune.eval_ckpt",
            "finetune.profile_layers", "baselines.train_mod",
            "baselines.train_mdf", "baselines.train_rt",
            "baselines.train_dense", "baselines.eval_compare",
            "tools.probe_kdist", "tools.probe_nll_ablation",
            "tools.kdiff_probe", "tools.probe_layers",
            "tools.plot_converge", "tools.eval_gen", "tools.accept",
            "tools.edge_bench", "tools.sweep", "tools.mem_demand"]
    for name in mods:
        mod = importlib.import_module(name)
        assert hasattr(mod, "app"), name
        r = CliRunner().invoke(mod.app, ["--help"])
        assert r.exit_code == 0, f"{name}: {r.output}"


def _write_rows(td, rows):
    p = os.path.join(td, "s.jsonl")
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return p


def test_stream_dataset_equivalence():
    rows = [{"text": f"hello world sample {i} with enough length"} for i in range(10)]
    rows.append({"text": "short"})
    rows.append("not a dict line")
    with tempfile.TemporaryDirectory() as td:
        p = _write_rows(td, rows)
        ref = SFTDataset(p, 0)
        stream = list(SFTStreamDataset(p, 0))
        assert stream == ref.samples
        assert list(iter_sft_texts(p, 4)) == ref.samples[:4]
