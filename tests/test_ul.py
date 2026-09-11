"""Offline unit tests: n-gram unlikelihood trigger/loss (repetition regularizer, r4)."""
import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import torch

from finetune.train import ngram_repeat_trigger, unlikelihood_loss  # shim guard: the historical import path keeps working
from speaker.ul import gt_repeat_rate, rep3_rate


def test_trigger():
    ids = torch.tensor([[1, 2, 3, 4, 1, 2, 3, 9]])
    valid = torch.ones_like(ids, dtype=torch.bool)
    trg = ngram_repeat_trigger(ids, 3, valid)
    # only t6 = (1,2,3) recurs and triggers; all others are new n-grams
    assert trg[0, 6] and trg.sum() == 1, trg
    # n=1 = immediate token-level loop
    ids2 = torch.tensor([[5, 5, 5, 5]])
    trg2 = ngram_repeat_trigger(ids2, 1, torch.ones_like(ids2, dtype=torch.bool))
    assert trg2[0].tolist() == [False, True, True, True]
    # loop body (n=2): every looping token triggers
    ids3 = torch.tensor([[1, 2, 1, 2, 1]])
    trg3 = ngram_repeat_trigger(ids3, 2, torch.ones_like(ids3, dtype=torch.bool))
    assert trg3[0].tolist() == [False, False, False, True, True], trg3
    # valid=False blocks triggering (but the context still accumulates)
    v = torch.ones_like(ids, dtype=torch.bool)
    v[0, 6] = False
    assert ngram_repeat_trigger(ids, 3, v).sum() == 0
    print("[PASS] trigger (recurrence/immediate loop/loop body/valid blocking)")


def test_loss_value():
    V = 10
    logits = torch.zeros(1, 4, V, requires_grad=True)
    targets = torch.tensor([[1, 2, 3, 4]])
    trg = torch.tensor([[False, True, False, True]])
    loss = unlikelihood_loss(logits, targets, trg)
    # uniform distribution p=1/V=0.1 -> -log(1-0.1)
    assert abs(loss.item() - (-math.log(0.9))) < 1e-5, loss.item()
    assert torch.isfinite(loss)
    # empty trigger -> 0 without crashing
    z = unlikelihood_loss(logits.detach(), targets, torch.zeros_like(trg))
    assert z.item() == 0.0
    print("[PASS] loss value (uniform-distribution analytic value / empty trigger)")


def test_pushdown():
    """Behavioral: repeatedly descend on a p≈1 repetition token; its probability should be
    pushed down."""
    V = 5
    logits = torch.nn.Parameter(torch.tensor([[[0., 8., 0., 0., 0.]]]))  # target=1, p≈0.9997
    targets = torch.tensor([[1]])
    trg = torch.tensor([[True]])
    opt = torch.optim.SGD([logits], lr=1.0)
    for _ in range(10):
        opt.zero_grad()
        loss = unlikelihood_loss(logits, targets, trg)
        loss.backward()
        opt.step()
    p_after = logits.softmax(-1)[0, 0, 1].item()
    assert p_after < 0.9, p_after
    print(f"[PASS] pushdown (p 0.9997 -> {p_after:.4f})")


def test_rep3_rate():
    # distinct trigrams -> 0; exact loop "aaaaaaa": 5 grams all "aaa" -> 1 - 1/5 = 0.8
    assert rep3_rate("abcdefgh") == 0.0
    assert abs(rep3_rate("aaaaaaa") - 0.8) < 1e-6
    # "abcabc": grams [abc,bca,cab,abc] -> 1 recurring of 4
    assert abs(rep3_rate("abcabc") - 0.25) < 1e-6
    print("[PASS] rep3_rate (distinct 0 / loop 0.8 / mixed 0.25)")


def test_gt_repeat_rate():
    ids = torch.tensor([[1, 2, 3, 1, 2, 3, 9]])
    valid = torch.ones_like(ids, dtype=torch.bool)
    # only position 5 (second (1,2,3)) triggers -> 1/7
    assert abs(gt_repeat_rate(ids, valid, 3) - 1 / 7) < 1e-6
    # invalid positions never trigger but stay out of the denominator
    v2 = valid.clone()
    v2[0, 5] = False
    assert gt_repeat_rate(ids, v2, 3) == 0.0
    print("[PASS] gt_repeat_rate (density / valid blocking)")


if __name__ == "__main__":
    test_trigger()
    test_loss_value()
    test_pushdown()
    test_rep3_rate()
    test_gt_repeat_rate()
    print("All UL tests passed.")
