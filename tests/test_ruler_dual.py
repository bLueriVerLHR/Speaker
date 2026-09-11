"""Offline unit tests: accuracy ruler (valid masks / target derivation, P0) +
dual controller cadence (P0.5)."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import torch

from speaker.evaluate import ema_update
from speaker.metrics import per_token_correct
from speaker.ruler import AccRuler, batch_accuracy, parse_target, valid_positions
from speaker.dual import DualController, difficulty_mult


def test_valid_positions():
    b = {"labels": torch.tensor([[3, 5, -100, 7]]),
         "attention_mask": torch.tensor([[1, 1, 0, 1]])}
    assert valid_positions(b, "labels").tolist() == [[True, True, False, True]]
    assert valid_positions(b, "attention_mask").tolist() == [[True, True, False, True]]
    try:
        valid_positions(b, "junk")
        raise AssertionError("unknown mode must raise")
    except ValueError:
        pass
    print("[PASS] valid_positions (labels/attention_mask equivalent-error)")


def test_batch_accuracy_matches_history():
    torch.manual_seed(0)
    logits = torch.randn(2, 6, 11)
    labels = torch.randint(0, 11, (2, 6))
    labels[0, 0] = -100
    labels[1, 3] = -100
    # non-degenerate: force a couple of hits and misses
    labels[0, 1] = int(logits[0, 1].argmax())
    labels[0, 2] = (int(logits[0, 2].argmax()) + 1) % 11
    b = {"labels": labels, "attention_mask": (labels != -100).long()}
    # historical inline block
    correct = per_token_correct(logits.float(), b["labels"])
    valid_tok = (b["labels"] != -100)
    expected = correct[valid_tok].float().mean().item() if valid_tok.any() else 0.0
    assert 0.0 < expected < 1.0, expected
    assert batch_accuracy(logits, b) == expected
    assert batch_accuracy(logits, b) == batch_accuracy(logits, b, mode="attention_mask")
    print(f"[PASS] batch_accuracy == historical inline block ({expected:.4f})")


def test_target_policy():
    assert parse_target("auto") == ("auto", None)
    assert parse_target("none") == ("none", None)
    assert parse_target("0.55") == ("fixed", 0.55)
    try:
        parse_target("junk")
        raise AssertionError("garbage must raise")
    except ValueError:
        pass
    r = AccRuler.from_cli("auto", margin=0.03)
    assert abs(r.resolve(0.52) - 0.49) < 1e-9, r.resolve(0.52)  # dense - margin
    assert r.resolve(None) == 0.55  # legacy fallback keeps the dual active
    assert AccRuler.from_cli("0.61").resolve(0.9) == 0.61
    assert AccRuler.from_cli("none").resolve(0.9) is None
    r2 = AccRuler.from_cli("auto", margin=0.99)
    assert r2.resolve(0.5) == 0.0  # clamped at 0
    print("[PASS] target policy (auto/fixed/none/fallback/clamp)")


class _Cfg:
    price_warmup_steps = 3
    acc_target = 0.5
    price_adapt = True
    adapt_rate = 0.01
    price_min = 1e-4
    price_max = 0.5
    sparsity_price = 0.03


class _Model:
    def __init__(self, cfg):
        self.calls = []
        self._cfg = cfg

    def adapt_price(self, ema_acc):
        self.calls.append(ema_acc)
        self._cfg.sparsity_price *= 1.0 + 0.01  # mirrors wrapper.adapt_price above-target branch


def test_dual_cadence():
    m, cfg = _Model(_Cfg()), _Cfg()
    m._cfg = cfg
    dual = DualController(m, cfg)
    ema = None
    for step in range(1, 6):
        ema = ema_update(ema, 0.6)
        assert dual.observe(step, 0.6) == ema
    # warmup gate: adapt fires only for step > 3
    assert m.calls == [ema, ema], m.calls
    assert abs(cfg.sparsity_price - 0.03 * 1.01 * 1.01) < 1e-12
    print("[PASS] dual cadence (ema identical, warmup gate, adapt math)")


def test_difficulty_mult():
    nll = torch.tensor([[0.2, 0.5, 1.5, 2.5, 5.0]])
    m = difficulty_mult(nll, 0.5, 2.5, 2.0, 0.5)
    # strict inequalities: boundaries belong to mid (1.0)
    assert m[0].tolist() == [2.0, 1.0, 1.0, 1.0, 0.5], m
    assert m.dtype == nll.dtype
    print("[PASS] difficulty_mult (easy/mid/hard tiers, boundary ownership)")


if __name__ == "__main__":
    test_valid_positions()
    test_batch_accuracy_matches_history()
    test_target_policy()
    test_dual_cadence()
    test_difficulty_mult()
    print("All ruler/dual tests passed.")
