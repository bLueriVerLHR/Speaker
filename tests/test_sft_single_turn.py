"""--single_turn (P1): first-round truncation keeps the first user+assistant round,
is opt-in, and leaves the default path bit-identical."""
import json

from data.sft import SFTDataset, _text_of, first_turn, iter_sft_texts

SAMPLE = {
    "conversations": [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "Q1"},
        {"role": "assistant", "content": "A1"},
        {"role": "user", "content": "Q2"},
        {"role": "assistant", "content": "A2"},
    ],
    "ref": "unused",
}


def test_first_turn_keeps_first_round_only():
    out = first_turn(SAMPLE)
    roles = [m["role"] for m in out["conversations"]]
    assert roles == ["system", "user", "assistant"]
    assert out["conversations"][-1]["content"] == "A1"
    assert "Q2" not in json.dumps(out) and "A2" not in json.dumps(out)
    assert out["ref"] == "unused"  # non-conversation fields preserved


def test_first_turn_no_assistant_or_plain_obj_unchanged():
    user_only = {"conversations": [{"role": "user", "content": "Q1"}]}
    assert first_turn(user_only) is user_only
    plain = {"text": "hello world"}
    assert first_turn(plain) is plain
    assert first_turn(SAMPLE) is not SAMPLE  # never mutates the input


def test_text_of_default_path_bit_identical():
    assert _text_of(SAMPLE) == _text_of(SAMPLE, single_turn=False)
    assert "Q2" in _text_of(SAMPLE) and "A2" in _text_of(SAMPLE)


def test_text_of_single_turn_drops_later_rounds():
    text = _text_of(SAMPLE, single_turn=True)
    assert "A1" in text and "Q1" in text
    assert "Q2" not in text and "A2" not in text


def test_dataset_and_stream_truncate_consistently(tmp_path):
    p = tmp_path / "sft.jsonl"
    p.write_text(json.dumps(SAMPLE) + "\n", encoding="utf-8")
    ds = SFTDataset(str(p), single_turn=True)
    assert len(ds.samples) == 1
    assert "A2" not in ds.samples[0]
    streamed = list(iter_sft_texts(str(p), single_turn=True))
    assert streamed == ds.samples
    full = SFTDataset(str(p))
    assert "A2" in full.samples[0]  # default untouched
