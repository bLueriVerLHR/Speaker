"""Speaker: model with shared (fixed/syntax) layers + gated (reasoning) layers (dual scheme,
gate_mode switch).

Core idea: when a speaker talks, fewer layers handle syntactic structure, leaving more layers
for complex logical reasoning.
- Fixed (shared) layers: every token passes through; in theory the layers with load >= 90%~95%,
  holding the syntactic/representational foundation;
- gate_mode="moe" (default, mainline): gated layers are the experts; the gated-region entry
  single-point JointRouter(H->G) produces log p, top-p/top-k selection, and the selected layers'
  probabilities are renormalized into weighted residuals (per-token adaptive depth);
- gate_mode="threshold" (legacy scheme): each layer's independent Router decides whether the
  token executes that layer in full.

Checkpoints of the two schemes are not interchangeable (user decision, for same-arena
comparison); from_json infers threshold for old ckpts (no gate_mode field). The old names
MoDConfig/MoDLayerWrapper/MoDModelWrapper/convert_to_mod are import aliases.
"""
from .config import SpeakerConfig
from .gating import Router, JointRouter, RouteDecision, select_and_weight, GatingOutput
from .wrapper import SpeakerLayerWrapper, SpeakerModelWrapper, convert_to_speaker
from .metrics import per_token_nll, per_token_correct, estimate_act_mb, distill_kl_loss

# ---- Legacy-name aliases (class names from the historical ckpt era; new code uses
# Speaker* / convert_to_speaker) ----
MoDConfig = SpeakerConfig
MoDLayerWrapper = SpeakerLayerWrapper
MoDModelWrapper = SpeakerModelWrapper
convert_to_mod = convert_to_speaker

__all__ = [
    "SpeakerConfig", "Router", "JointRouter", "RouteDecision", "select_and_weight",
    "GatingOutput",
    "SpeakerLayerWrapper", "SpeakerModelWrapper", "convert_to_speaker",
    # compatibility aliases
    "MoDConfig", "MoDLayerWrapper", "MoDModelWrapper", "convert_to_mod",
    "per_token_nll", "per_token_correct", "estimate_act_mb", "distill_kl_loss",
]
