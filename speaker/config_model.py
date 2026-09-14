"""Optional Pydantic-v2 validation model for SpeakerConfig (P0 structure).

SpeakerConfig (dataclass in speaker/config.py) stays canonical and this module
is strictly opt-in: ``HAS_PYDANTIC`` is False without the dependency and every
helper falls back to the dataclass validation (``__post_init__``). With
pydantic>=2 (a main pyproject dependency since the Typer migration) you get:

- ``SpeakerConfigModel``: same fields/constraints as SpeakerConfig, with
  field_validator error messages suited for config-file (YAML/JSON) loading;
- ``validate_dict(d)``: validate a raw mod_config.json dict before
  ``SpeakerConfig.from_json`` (catches schema drift with clear errors);
- ``from_speaker_config`` / ``to_speaker_config``: lossless round-trip.

No training numerics depend on this module.
"""
from __future__ import annotations

try:
    import pydantic  # noqa: F401
    from pydantic import BaseModel, Field, field_validator
    HAS_PYDANTIC = True
except ImportError:
    HAS_PYDANTIC = False


if HAS_PYDANTIC:
    class SpeakerConfigModel(BaseModel):
        """Pydantic mirror of SpeakerConfig (validation only, never on hot paths)."""

        model_config = {"extra": "ignore"}  # old ckpts carry unknown keys (from_json drops them)

        num_hidden_layers: int = Field(default=24, ge=1)
        hidden_size: int = Field(default=1024, ge=1)
        gate_mode: str = "moe"
        always_on_layers: list | None = None
        always_on_head: int = Field(default=2, ge=0)
        always_on_tail: int = Field(default=2, ge=0)
        select_mode: str = "topp"
        top_p: float = Field(default=0.9, gt=0.0, le=1.0)
        top_k: int = Field(default=6, ge=1)
        min_layers: int = Field(default=1, ge=1)
        weight_mode: str = "pmax"
        kmax: int = Field(default=10, ge=1)
        count_temp: float = Field(default=0.1, gt=0.0)
        router_temp: float = Field(default=1.0, gt=0.0)
        sparsity_price: float = Field(default=0.03, ge=0.0)
        budget_form: str = "mean"
        budget_target: float = Field(default=0.0, ge=0.0)
        tail_coef: float = Field(default=1.0, ge=0.0)
        tail_temp: float = Field(default=0.5, gt=0.0)
        price_min: float = Field(default=1e-4, gt=0.0)
        price_max: float = Field(default=0.5, gt=0.0)
        price_adapt: bool = True
        acc_target: float | None = 0.55
        adapt_rate: float = Field(default=0.01, gt=0.0, lt=1.0)
        price_warmup_steps: int = Field(default=100, ge=0)
        skip_mode: str = "soft"
        decode: dict | None = None

        @field_validator("gate_mode")
        @classmethod
        def _canon_gate(cls, v: str) -> str:
            v = {"speaker": "threshold", "mol": "moe"}.get(v, v)
            if v not in ("moe", "threshold"):
                raise ValueError(f"gate_mode must be moe|threshold, got {v!r}")
            return v

        @field_validator("select_mode")
        @classmethod
        def _sel(cls, v: str) -> str:
            if v not in ("topp", "topk"):
                raise ValueError(f"select_mode must be topp|topk, got {v!r}")
            return v

        @field_validator("weight_mode")
        @classmethod
        def _wm(cls, v: str) -> str:
            if v not in ("pmax", "renorm"):
                raise ValueError(f"weight_mode must be pmax|renorm, got {v!r}")
            return v

        @field_validator("skip_mode")
        @classmethod
        def _sm(cls, v: str) -> str:
            if v not in ("soft", "hard"):
                raise ValueError(f"skip_mode must be soft|hard, got {v!r}")
            return v

        @field_validator("budget_form")
        @classmethod
        def _bf(cls, v: str) -> str:
            if v not in ("mean", "hinge", "tail"):
                raise ValueError(f"budget_form must be mean|hinge|tail, got {v!r}")
            return v

    def validate_dict(d: dict) -> dict:
        """Validate a raw config dict; returns the normalized dict (raises on error)."""
        return SpeakerConfigModel(**d).model_dump()

    def from_speaker_config(cfg) -> "SpeakerConfigModel":
        return SpeakerConfigModel(**cfg.to_dict())

    def to_speaker_config(m: "SpeakerConfigModel"):
        from .config import SpeakerConfig
        return SpeakerConfig(**m.model_dump())

else:  # pragma: no cover — fallback when pydantic is absent (cuda env)

    def validate_dict(d: dict) -> dict:
        from .config import SpeakerConfig
        import inspect
        valid = set(inspect.signature(SpeakerConfig).parameters)
        SpeakerConfig(**{k: v for k, v in d.items() if k in valid})
        return d

    def from_speaker_config(cfg):
        return cfg.to_dict()

    def to_speaker_config(m):
        from .config import SpeakerConfig
        import inspect
        d = m if isinstance(m, dict) else m.to_dict()
        valid = set(inspect.signature(SpeakerConfig).parameters)
        return SpeakerConfig(**{k: v for k, v in d.items() if k in valid})


__all__ = [
    "HAS_PYDANTIC", "validate_dict", "from_speaker_config", "to_speaker_config",
] + (["SpeakerConfigModel"] if HAS_PYDANTIC else [])
