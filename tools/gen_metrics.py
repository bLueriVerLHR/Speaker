"""Standard generation metrics (industry-vetted implementations, no home wheels).

Two metrics only (the project's entire generation acceptance runs on these):

- Continuation score — ROUGE-L F1 (Lin, WAS 2004) via Google's
  ``rouge-score`` public API: ``RougeScorer(['rougeL'],
  tokenizer=CjkTokenizer)``. The custom tokenizer subclasses the package's
  public ``tokenizers.Tokenizer`` ABC (the documented extension point) and
  returns :func:`std_toks` tokens. This is required, not optional: the
  built-in tokenizer lowercases and drops every token outside ``[a-z0-9]``
  (verified: identical CJK sentences score 0.0 through the default path),
  and the Chinese community recipe (jieba/char-split then join, cf.
  ``rouge-chinese``) is exactly "segment first, then score".
  ``use_stemmer=False`` (Porter is English-only).
- Repetition — seq-rep-n (Welleck et al. 2020, "Neural Text Generation
  with Unlikelihood Training", Eq. 10), the identical formula reused as
  rep-n by Li et al. 2023 ("Repetition In Repetition Out", NeurIPS, Eq. 4)::

      seq-rep-n = 1.0 − |unique n-grams(continuation)| / |n-grams|,

  averaged over continuations; 0 = no repeating n-grams. Canonical n = 4
  (the papers report seq-rep-4 / Rep-4). Computed on the generated
  continuation only, with :func:`std_toks` word tokens.
- Diversity companion — Distinct-2 (Li et al. 2016, "A Diversity-Promoting
  Objective Function for Neural Conversation Models"): unique word bigrams
  / total word tokens. Available as :func:`distinct_bi`; not part of the
  acceptance gate (the project gates on ROUGE-L + seq-rep-4 only).

Training-time machinery (``speaker/ul.py``: token-id UL triggers,
``rep3_rate`` char probe, rollout) is untouched — this module is eval only.
"""
import re

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[^\sA-Za-z0-9_]")


def std_toks(s: str) -> list[str]:
    """CJK-safe tokenization: ASCII words stay whole, CJK/marks split by char."""
    return _TOKEN_RE.findall(s)


_scorer = None


try:
    from rouge_score.tokenizers import Tokenizer as _TokBase
except Exception:  # pragma: no cover - rouge-score is a hard dependency
    _TokBase = object


class CjkTokenizer(_TokBase):
    """CJK-safe tokenizer for rouge-score (registered via the public
    ``tokenizer=`` extension point). English words stay whole, CJK/marks
    split by character — the same segmentation the repetition metrics below
    use, so both scores see the same tokens."""

    def tokenize(self, text):
        return std_toks(text)


def _rouge():
    global _scorer
    if _scorer is None:
        from rouge_score import rouge_scorer
        _scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False,
                                           tokenizer=CjkTokenizer())
    return _scorer


def rouge_l(ref: str, hyp: str, max_tok: int = 128) -> float:
    """Sentence-level ROUGE-L F1 via google rouge-score (public API)."""
    if not ref or not hyp:
        return 0.0
    _r = std_toks(ref)[:max_tok]
    _h = std_toks(hyp)[:max_tok]
    if not _r or not _h:
        return 0.0
    return _rouge().score(" ".join(_r), " ".join(_h))["rougeL"].fmeasure


def _word_ngrams(text: str, n: int) -> list[tuple[str, ...]]:
    w = std_toks(text)
    return [tuple(w[i:i + n]) for i in range(max(len(w) - n + 1, 0))]


def seq_rep(text: str, n: int = 4) -> float:
    """seq-rep-n (Welleck et al. 2020, Eq. 10): portion of duplicate n-grams
    in the continuation; 0 = clean. Canonical n = 4."""
    g = _word_ngrams(text, n)
    return 0.0 if not g else 1.0 - len(set(g)) / len(g)


def distinct_bi(text: str) -> float:
    """Distinct-2 (Li et al. 2016): unique word bigrams / total word tokens."""
    w = std_toks(text)
    if not w:
        return 0.0
    return len(set(_word_ngrams(text, 2))) / len(w)
