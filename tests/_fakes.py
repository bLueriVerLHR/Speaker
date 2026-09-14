"""Shared tiny fake backbones for offline unit tests (no weights required).

Superset of the five copy-pasted variants: the use_cache branch is a no-op
unless the caller passes use_cache=True, so outputs are identical for all
existing non-cache callers.
"""
import torch
from torch import nn


class FakeDecoderLayer(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.self_attn = nn.Identity()
        self.mlp = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.SiLU(),
                                 nn.Linear(hidden_size, hidden_size))

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        out = hidden_states + self.mlp(hidden_states) * 0.1
        if kwargs.get("use_cache"):
            return (out, kwargs.get("past_key_value"))
        return (out,)


class FakeHF(nn.Module):
    def __init__(self, n=6, h=32):
        super().__init__()
        self.config = type("Cfg", (), {"num_hidden_layers": n, "hidden_size": h})()
        self.model = type("M", (), {})()
        self.model.layers = nn.ModuleList([FakeDecoderLayer(h) for _ in range(n)])
        self.lm_head = nn.Linear(h, 100)

    def forward(self, hidden_states=None, input_ids=None, attention_mask=None, **kwargs):
        if hidden_states is None:
            hidden_states = torch.randn(2, 8, 32)
        for layer in self.model.layers:
            hidden_states = layer(hidden_states, attention_mask=attention_mask, **kwargs)[0]
        return {"logits": self.lm_head(hidden_states)}


def force_onehot(mod, slot: int):
    """moe: force the joint router into one-hot (zeroed net weights + a large prior for one
    slot -> only that gated layer is selected)."""
    mod.eval()
    mod.set_skip_mode("soft")
    mod.joint_router.net.weight.data.zero_()
    mod.joint_router.layer_bias.data.fill_(0.0)
    mod.joint_router.layer_bias.data[slot] = 50.0
