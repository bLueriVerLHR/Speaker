"""edge6g deployment parity: non-ours families (dense-ft / mdf / ...) ride the speaker
wrapper with every layer always-on so all methods share one forward + placement path.

Two seams: apply_placement(force_always_gpu=False) and wrap math-neutrality."""
import torch

from speaker import SpeakerConfig, convert_to_speaker
from speaker.placement import apply_placement
from tests._fakes import FakeHF


def _dense_cfg(n=6, h=32):
    return SpeakerConfig(num_hidden_layers=n, hidden_size=h, gate_mode="threshold",
                         always_on_layers=list(range(n)))


def test_placement_pins_always_on_by_default():
    w = convert_to_speaker(FakeHF(n=6, h=32), _dense_cfg())
    resident = apply_placement(w, [2], gpu_device="meta", cpu_device="cpu")
    assert resident == list(range(6))
    devs = {i: next(w.layers[i].parameters()).device.type for i in range(6)}
    assert all(d == "meta" for d in devs.values()), "default must pin always_on on the gpu device"


def test_placement_flag_frees_always_on():
    w = convert_to_speaker(FakeHF(n=6, h=32), _dense_cfg())
    resident = apply_placement(w, [0, 2], gpu_device="meta", cpu_device="cpu",
                               force_always_gpu=False)
    assert resident == [0, 2]
    devs = {i: next(w.layers[i].parameters()).device.type for i in range(6)}
    assert devs[0] == "meta" and devs[2] == "meta"
    assert all(devs[i] == "cpu" for i in (1, 3, 4, 5)), "non-resident must stay on cpu"


def test_all_always_on_wrap_is_math_neutral():
    torch.manual_seed(0)
    fake = FakeHF(n=6, h=32)
    am = torch.ones(2, 8)
    x = torch.randn(2, 8, 32)
    a = fake(hidden_states=x, attention_mask=am)["logits"].clone()
    w = convert_to_speaker(fake, _dense_cfg())
    w.eval()
    with torch.no_grad():
        b = w(hidden_states=x, attention_mask=am)["logits"]
    assert torch.equal(a, b), "all-always-on wrap must not change the forward math"
