"""Unit tests for baselines.assemble (family detection + display naming, no model loads)."""
import os

import pytest

from baselines.assemble import FAMILY_CONFIG, Assembly, ModelBuilder, detect_family


@pytest.fixture
def ckpt_dir(tmp_path):
    d = tmp_path / "r7_x"
    d.mkdir()
    return str(d)


def _write(ckpt, fn, cfg=None):
    with open(os.path.join(ckpt, fn), "w") as f:
        f.write('{"is_routed": [false]}' if cfg is None else cfg)


def test_detect_every_family(ckpt_dir):
    for fam, fn in FAMILY_CONFIG.items():
        _write(ckpt_dir, fn)
        assert detect_family(ckpt_dir) == fam
        os.remove(os.path.join(ckpt_dir, fn))


def test_detect_rejects_empty_dir(ckpt_dir):
    with pytest.raises(FileNotFoundError):
        detect_family(ckpt_dir)


def test_display_names_match_history():
    # rows JSON names must stay identical to the pre-refactor formats
    for fam, prefix in [("ours", "ours"), ("dense_ft", "denseft"), ("modd", "modd"),
                        ("mdf", "mdf"), ("rt", "rt")]:
        asm = Assembly(model=None, family=fam, ckpt="/x/r7_x", cfg={})
        assert asm.name == f"{prefix}:r7_x"


def test_builder_requires_ckpt_before_build():
    b = ModelBuilder("some-model-id")
    with pytest.raises(RuntimeError):
        b.build()
