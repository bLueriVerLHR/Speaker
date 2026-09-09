"""Speaker configuration: all hyperparameters for shared (fixed) layers / gating structure /
budget and dual (price) adjustment.

Dual scheme coexists (gate_mode switch; the two schemes' checkpoints are not interchangeable,
so they can be compared in the same arena):
- gate_mode="moe" (default, new scheme, mainline): hierarchical MoE joint routing — the gated-region
  entry point uses a single JointRouter(H->G) to produce log p, top-p (default)/top-k selection,
  selected layers' probabilities are renormalized into weighted residuals; the budget λ·mean(k) is
  differentiable via soft-inclusive STE; price accounting = actual demand per inference
  (per-token average activation memory, avg per-inference demand), only avg, never peak/residency.
- gate_mode="threshold" (legacy scheme): per-layer linear Router(H->1, no bias) + scalar threshold tau,
  logit=(router_logit - tau_l)/Ta (+Gumbel), STE for differentiability; budget loss = LM
  + λ·mean(k) + over-kmax penalty.
The shared-layer idea is identical across both schemes: always_on fixed layers (shared layers,
promoted when load >= 90%~95%) + the profiling pipeline.
from_json: unknown keys are dropped; an old ckpt's mod_config.json without a gate_mode field is
inferred as threshold (legacy semantics unchanged); new ckpts write gate_mode explicitly.
"""
from __future__ import annotations

import inspect
import json
from dataclasses import asdict, dataclass
from typing import List, Optional

SKIP_MODES = ("soft", "hard")
GATE_MODES = ("moe", "threshold")
SELECT_MODES = ("topp", "topk")


@dataclass
class SpeakerConfig:
    """Shared (fixed) layers + dual-scheme gating (moe = hierarchical MoE joint routing /
    threshold = legacy per-layer threshold).

    Shared: always_on fixed layers, elastic budget λ·mean(k) + accuracy dual adjustment,
    Ta/gumbel annealing, skip_mode soft (training) / hard (deployment layer-skipping).
    """

    num_hidden_layers: int = 24
    hidden_size: int = 1024
    gate_mode: str = "moe"  # moe (hierarchical MoE joint routing, default mainline) | threshold (legacy per-layer threshold gating)
    always_on_layers: Optional[List[int]] = None  # None = derive from head/tail; [] = explicitly no fixed layers (pure gating); explicit list overrides head/tail
    always_on_head: int = 2  # first m layers fixed/shared (foundation: syntax/representation)
    always_on_tail: int = 2  # last n layers fixed/shared (output: speaking/prediction)
    # ---- Selection (moe only): in log p space ----
    select_mode: str = "topp"      # topp | topk
    top_p: float = 0.9             # top-p: cumulative probability threshold
    top_k: int = 6                 # topk mode: fixed k
    min_layers: int = 1            # minimum gated layers activated per token
    kmax: int = 10                 # moe: hard cap during selection; threshold: safety cap
    count_temp: float = 0.1        # soft-inclusion sigmoid temperature (STE backward; smaller = closer to hard)
    # ---- Budget (shared; price accounting = actual demand per inference, avg only, no peak/residency) ----
    sparsity_price: float = 0.03   # λ, loss += λ·mean(k)
    price_min: float = 1e-4        # >0: multiplicative dual adjustment can climb back after hitting the floor (0 forbidden)
    price_max: float = 0.5
    price_adapt: bool = True       # automatically loosen/tighten λ based on the accuracy-target gap
    acc_target: Optional[float] = 0.55  # per-token accuracy floor; below it, relax sparsity
    adapt_rate: float = 0.01       # at most 1% per step, slow dual adjustment to avoid exponential blowup
    price_warmup_steps: int = 100  # λ frozen during warmup, waiting for the EMA to stabilize
    mem_bytes_per_hidden: float = 28.0  # activation byte coefficient per token·layer·hidden (bf16 calibrated empirically)
    # ---- Gating structure (threshold only) ----
    router_hidden_dim: Optional[int] = None  # None = single linear; set an int for H->int->1
    tau_init: float = 0.0          # initial threshold logit; negative = dense start (gates fully open first, then sparsify)
    use_ste: bool = True           # forward uses hard counts, backward goes through soft
    over_budget_coef: float = 0.05  # penalty for exceeding kmax (threshold safety cap)
    # ---- Temperature + randomness (shared; moe = softmax temperature, threshold = sigmoid temperature) ----
    temp_affinity: float = 1.0     # Ta, annealed down to ~0.3 to harden
    gumbel_scale: float = 1.0      # training noise strength, disabled automatically at eval
    # ---- Auxiliary (shared) ----
    balance_loss_coef: float = 0.01  # moe: variance of the mean routing distribution; threshold: which-uniformity across layers
    cos_reg_coef: float = 0.01       # StableSkip-style: tokens whose representation is unchanged should be skipped
    z_loss_coef: float = 0.001       # squared penalty on router logits
    # ---- Training ----
    skip_mode: str = "soft"  # soft for training (hard selection in forward), hard for deployment (layer skipping saves memory)

    def __post_init__(self):
        if self.always_on_layers is None:
            # None = unset: first m + last n always resident, middle optional (m/n
            # configurable; the default 2/2 is the ablation-validated configuration).
            # An explicit empty list stays empty (pure-gating startup, r6) — only None
            # derives from head/tail, so a saved [] reloads as [].
            N = self.num_hidden_layers
            self.always_on_layers = list(range(max(self.always_on_head, 0))) + \
                list(range(max(N - max(self.always_on_tail, 0), 0), N))
        self.always_on_layers = sorted(
            set(i for i in self.always_on_layers if 0 <= i < self.num_hidden_layers))
        if self.skip_mode not in SKIP_MODES:
            raise ValueError(f"skip_mode must be one of {SKIP_MODES}, got {self.skip_mode!r}")
        if self.gate_mode not in GATE_MODES:
            raise ValueError(f"gate_mode must be one of {GATE_MODES}, got {self.gate_mode!r}")
        if self.sparsity_price < 0:
            raise ValueError(f"sparsity_price must be >= 0, got {self.sparsity_price}")
        if self.gate_mode == "moe":
            if self.select_mode not in SELECT_MODES:
                raise ValueError(f"select_mode must be one of {SELECT_MODES}, got {self.select_mode!r}")
            if not 0.0 < self.top_p <= 1.0:
                raise ValueError(f"top_p must be in (0,1], got {self.top_p}")
            if self.min_layers < 1:
                raise ValueError(f"min_layers must be >= 1, got {self.min_layers}")
            if self.kmax < self.min_layers:
                raise ValueError(f"kmax({self.kmax}) must be >= min_layers({self.min_layers})")
            if self.top_k < 1:
                raise ValueError(f"top_k must be >= 1, got {self.top_k}")
        else:  # threshold
            if self.kmax < 1:
                raise ValueError(f"kmax must be >= 1, got {self.kmax}")

    @property
    def gated_layers(self) -> List[int]:
        return [i for i in range(self.num_hidden_layers) if i not in self.always_on_layers]

    def is_always_on(self, idx: int) -> bool:
        return idx in self.always_on_layers

    def to_dict(self) -> dict:
        return {**asdict(self), "gated_layers": self.gated_layers}

    def to_json(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def from_json(cls, path: str) -> "SpeakerConfig":
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        d.pop("gated_layers", None)
        # Old ckpts (before 0909) lack the gate_mode field -> infer the legacy scheme;
        # new ckpts carry gate_mode explicitly
        if "gate_mode" not in d:
            d["gate_mode"] = "threshold"
        # Filter unknown keys; deprecated fields are dropped automatically
        valid = set(inspect.signature(cls).parameters)
        d = {k: v for k, v in d.items() if k in valid}
        return cls(**d)

    @classmethod
    def from_model_config(cls, hf_config, **overrides) -> "SpeakerConfig":
        if isinstance(hf_config, dict):
            n = hf_config.get("num_hidden_layers")
            h = hf_config.get("hidden_size")
        else:
            n = getattr(hf_config, "num_hidden_layers", None)
            h = getattr(hf_config, "hidden_size", None)
        if n is None or h is None:
            raise ValueError(
                f"cannot infer num_hidden_layers/hidden_size from the given config "
                f"(got N={n!r}, H={h!r}); pass them explicitly via overrides")
        base = dict(num_hidden_layers=n, hidden_size=h)
        base.update(overrides)
        return cls(**base)

    def summary(self) -> str:
        if self.gate_mode == "moe":
            sel = (f"p={self.top_p}" if self.select_mode == "topp" else f"k={self.top_k}")
            gate = (f"moe sel={self.select_mode}({sel},kmax={self.kmax})")
        else:
            gate = f"threshold tau_init={self.tau_init} kmax={self.kmax} ste={self.use_ste}"
        return (f"Speaker[N={self.num_hidden_layers},H={self.hidden_size}] "
                f"shared={self.always_on_layers} gated={len(self.gated_layers)} {gate} "
                f"Ta={self.temp_affinity} price={self.sparsity_price}(adapt={self.price_adapt}) "
                f"acc_target={self.acc_target})")
