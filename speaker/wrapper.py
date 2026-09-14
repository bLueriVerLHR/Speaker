"""Backward-compatible facade (P0 structure).

The implementation moved: SpeakerLayerWrapper -> speaker/layer.py,
SpeakerModelWrapper (+ convert_to_speaker / apply_decode_config) ->
speaker/hub.py, placement -> speaker/placement.py, calibration ->
speaker/calibrate.py. This module re-exports the historical names so
``from speaker.wrapper import ...`` (tests, baselines/assemble.py) keeps
working bit-identically.
"""
from __future__ import annotations

from .hub import (
    SpeakerModelWrapper,
    _rd_to,
    _to_common,
    apply_decode_config,
    convert_to_speaker,
)
from .layer import SpeakerLayerWrapper

__all__ = [
    "SpeakerLayerWrapper",
    "SpeakerModelWrapper",
    "convert_to_speaker",
    "apply_decode_config",
    "_to_common",
    "_rd_to",
]
