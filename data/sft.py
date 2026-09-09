"""SFT dataset and collate: jsonl -> text -> batch dict.

- format_sft_text: uses the chat template when chat_template is available, otherwise
  falls back to role: content concatenation (legacy behavior);
- mask_non_assistant: loss/acc are computed only on assistant reply segments
  (user questions serve only as context);
- SFTDataset: filters len>=10 at build time (keeps line count == sample count, no
  misalignment).
"""
import json
import weakref

import torch
from torch.utils.data import Dataset

CHAT_ROLES = ("system", "user", "assistant")


def format_sft_text(obj, tok=None):
    """jsonl record -> training text. When tok has a chat_template, use the template
    (only role/content is taken; intermediate states such as reasoning_content never
    enter the text); otherwise fall back to the legacy role: content concatenation."""
    conv = obj.get("conversations")
    if not isinstance(conv, list) or not conv:
        if "text" in obj:
            return obj["text"]
        return str(obj)
    msgs = [{"role": m["role"], "content": m["content"]} for m in conv
            if m.get("role") in CHAT_ROLES and isinstance(m.get("content"), str)]
    if tok is not None and getattr(tok, "chat_template", None) and msgs:
        try:
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        except Exception:
            pass
    if "text" in obj and not msgs:
        return obj["text"]
    if msgs:
        return "\n".join(f"{m['role']}: {m['content']}" for m in msgs)
    return str(obj)


_HEADER_CACHE: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()


def _role_header_ids(tok):
    """Token ids of each role's start marker: {role: [ids...]}. Covers both the Qwen
    ChatML and the legacy role: concatenation candidates.
    WeakKey cache: invalidated as soon as the tokenizer is garbage collected, so there
    is no risk of id reuse colliding with a stale cache."""
    if tok not in _HEADER_CACHE:
        vocab = set()
        try:
            vocab = set(tok.get_vocab())
        except Exception:
            pass
        heads: dict = {}
        if "<|im_start|>" in vocab:
            for r in CHAT_ROLES:
                try:
                    e = tok.encode(f"<|im_start|>{r}\n", add_special_tokens=False)
                except Exception:
                    continue
                if e:
                    heads[r] = [e]
        else:
            for r in CHAT_ROLES:
                try:
                    e = tok.encode(f"{r}:", add_special_tokens=False)
                except Exception:
                    continue
                if e:
                    heads[r] = [e]
        _HEADER_CACHE[tok] = heads
    return _HEADER_CACHE[tok]


def mask_non_assistant(input_ids: torch.Tensor, tok) -> torch.Tensor:
    """labels mask: keep only assistant reply segments (assistant header -> the next
    role header of any kind), everything else -100. Falls back to full supervision
    when no header is found (legacy behavior: only slower, never wrong)."""
    labels = input_ids.clone()
    heads = _role_header_ids(tok)
    if not heads or "assistant" not in heads:
        return labels
    pats = [(role, h) for role, hs in heads.items() for h in hs]
    for bi in range(input_ids.shape[0]):
        row = input_ids[bi].tolist()
        bounds = []  # (pos, role)
        for role, h in pats:
            Lh = len(h)
            for s in range(len(row) - Lh + 1):
                if row[s:s + Lh] == h:
                    bounds.append((s, role))
        if not any(r == "assistant" for _, r in bounds):
            continue  # no assistant header: keep everything (fallback)
        bounds.sort()
        bounds.append((len(row), None))  # sentinel
        keep = torch.zeros(len(row), dtype=torch.bool)
        for j, (pos, role) in enumerate(bounds[:-1]):
            if role == "assistant":
                Lh = len(heads["assistant"][0])
                keep[pos + Lh:bounds[j + 1][0]] = True
        if keep.any():
            labels[bi][~keep] = -100
    return labels


class SFTDataset(Dataset):
    """jsonl -> plain text list. Tokenization/truncation is done per batch in collate."""

    def __init__(self, path, max_samples=0, tok=None, use_chat=False):
        self.samples = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                if max_samples and len(self.samples) >= max_samples:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if use_chat:
                    text = format_sft_text(obj, tok)
                elif "conversations" in obj:
                    text = "\n".join(f"{t['role']}: {t['content']}" for t in obj["conversations"])
                elif "text" in obj:
                    text = obj["text"]
                else:
                    text = str(obj)
                if len(text) >= 10:
                    self.samples.append(text)
        print(f"Loaded {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate(batch, tok, device, max_len):
    enc = tok(batch, truncation=True, max_length=max_len, padding=True, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def collate_chat(batch, tok, device, max_len, mask_user=True):
    """chat-mode collate: same protocol as collate, but additionally sets labels to
    -100 for non-assistant segments. mask_user=False degrades to collate (legacy
    behavior)."""
    out = collate(batch, tok, device, max_len)
    if mask_user:
        out["labels"] = mask_non_assistant(out["input_ids"].cpu(), tok).to(device)
        out["labels"][out["attention_mask"] == 0] = -100
    return out


def make_collate(tok, device, max_len, use_chat=False, mask_user=True):
    if use_chat:
        return lambda b: collate_chat(b, tok, device, max_len, mask_user)
    return lambda b: collate(b, tok, device, max_len)
