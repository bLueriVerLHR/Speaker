"""Unit tests for tools/gen_metrics.py (offline, no weights).

ROUGE-L runs through google rouge-score's public API
(RougeScorer + a Tokenizer-ABC subclass); the repetition formulas are the
Holtzman rep-n / Li distinct-n family on CJK-safe word tokens.
"""
import pytest

from tools.gen_metrics import CjkTokenizer, distinct_bi, rouge_l, seq_rep, std_toks


def test_std_toks_cjk_safe():
    assert std_toks("hello world") == ["hello", "world"]
    assert std_toks("完全相同") == ["完", "全", "相", "同"]
    assert std_toks("a,b") == ["a", ",", "b"]


def test_tokenizer_subclasses_package_abc():
    from rouge_score.tokenizers import Tokenizer
    assert isinstance(CjkTokenizer(), Tokenizer)


def test_rouge_l_english_matches_default_path():
    # no CJK involved: custom and default tokenizers must agree exactly
    from rouge_score import rouge_scorer
    ref, hyp = "the cat sat on the mat", "the cat sat on the rug"
    std = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    assert rouge_l(ref, hyp) == pytest.approx(
        std.score(ref, hyp)["rougeL"].fmeasure)


def test_rouge_l_pinned_values():
    assert rouge_l("the cat sat on the mat",
                   "the cat sat on the rug") == pytest.approx(0.833333, rel=1e-5)
    assert rouge_l("完全相同的句子", "完全相同的句子") == pytest.approx(1.0)
    assert rouge_l("", "abc") == 0.0
    assert rouge_l("abc", "") == 0.0
    assert 0.0 <= rouge_l("abc def", "ghi jkl") <= 1.0


def test_seq_rep_identities():
    # Welleck Eq. 10: 1 - |unique n-grams| / |n-grams|, canonical n = 4
    assert seq_rep("a b c") == 0.0  # fewer than 4 words: no 4-gram
    assert seq_rep("a b a b a b a b") == pytest.approx(0.6)
    assert seq_rep("a b a b", n=3) == pytest.approx(0.0)
    assert seq_rep("a b a b a b", n=3) == pytest.approx(0.5)
    assert seq_rep("") == 0.0


def test_distinct_bi_identities():
    assert distinct_bi("a b a b") == pytest.approx(0.5)
    assert distinct_bi("a b c d") == pytest.approx(3 / 4)
    assert distinct_bi("") == 0.0
