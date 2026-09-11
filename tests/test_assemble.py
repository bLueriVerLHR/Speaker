"""Unit tests for baselines.assemble (family detection + display naming, no model loads)."""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baselines.assemble import FAMILY_CONFIG, Assembly, ModelBuilder, detect_family  # noqa: E402


class TestAssemble(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.ckpt = os.path.join(self.dir.name, "r7_x")

    def tearDown(self):
        self.dir.cleanup()

    def _write(self, fn, cfg=None):
        os.makedirs(self.ckpt, exist_ok=True)
        with open(os.path.join(self.ckpt, fn), "w") as f:
            f.write('{"is_routed": [false]}' if cfg is None else cfg)

    def test_detect_every_family(self):
        for fam, fn in FAMILY_CONFIG.items():
            self._write(fn)
            self.assertEqual(detect_family(self.ckpt), fam)
            os.remove(os.path.join(self.ckpt, fn))

    def test_detect_rejects_empty_dir(self):
        os.makedirs(self.ckpt, exist_ok=True)
        with self.assertRaises(FileNotFoundError):
            detect_family(self.ckpt)

    def test_display_names_match_history(self):
        # rows JSON names must stay identical to the pre-refactor formats
        for fam, prefix in [("ours", "ours"), ("dense_ft", "denseft"), ("modd", "modd"),
                            ("mdf", "mdf"), ("rt", "rt")]:
            asm = Assembly(model=None, family=fam, ckpt="/x/r7_x", cfg={})
            self.assertEqual(asm.name, f"{prefix}:r7_x")

    def test_builder_requires_ckpt_before_build(self):
        b = ModelBuilder("some-model-id")
        with self.assertRaises(RuntimeError):
            b.build()


if __name__ == "__main__":
    unittest.main()
