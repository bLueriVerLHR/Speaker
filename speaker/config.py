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
Naming (0910 pivot): the per-layer scheme is called "Speaker" (CLI alias --gate_mode speaker),
the joint-router scheme "MoL / Mixture of Layers" (CLI alias mol); both map onto the legacy
canonical values threshold/moe before anything is stored, so ckpt compatibility is unchanged.
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
    weight_mode: str = "pmax"      # moe only: residual weighting — pmax (w=p/p_max, gain
                                   # scales with k, dual lever connected; ed7 default) |
                                   # renorm (legacy Σw=1, gain fixed at 1.0, collapses)
    kmax: int = 10                 # moe: hard cap during selection; threshold: safety cap
    count_temp: float = 0.1        # soft-inclusion sigmoid temperature (STE backward; smaller = closer to hard)
    router_temp: float = 1.0       # moe only: persistent JointRouter logit temperature, calibrated at init so
                                   # the top-p mean k starts near the target (1.0 = off/legacy; r7 evidence:
                                   # uncalibrated random projection over large-norm hiddens starts peaked and
                                   # collapses during the budget ramp)
    # ---- Budget (shared; price accounting = actual demand per inference, avg only, no peak/residency) ----
    sparsity_price: float = 0.03   # λ, loss += λ·mean(k)
    price_min: float = 1e-4        # >0: multiplicative dual adjustment can climb back after hitting the floor (0 forbidden)
    price_max: float = 0.5
    price_adapt: bool = True       # automatically loosen/tighten λ based on the accuracy-target gap
    acc_target: Optional[float] = 0.55  # per-token accuracy floor; below it, relax sparsity
    adapt_rate: float = 0.01       # at most 1% per step, slow dual adjustment to avoid exponential blowup
    price_warmup_steps: int = 100  # λ frozen during warmup, waiting for the EMA to stabilize
    mem_bytes_per_hidden: float = 28.0  # activation byte coefficient per token·layer·hidden (bf16 calibrated empirically)
    # ---- Difficulty-conditioned budget (ed8, both schemes): per-token λ shaping ----
    diff_easy_nll: float = 0.5   # frozen-teacher NLL below this = easy token (grounded: p33 on the SFT slice)
    diff_hard_nll: float = 2.5   # frozen-teacher NLL above this = hard token (grounded: p67)
    diff_easy_mult: float = 2.0  # λ multiplier on easy tokens (press harder: solve cheaply)
    diff_hard_mult: float = 0.5  # λ multiplier on hard tokens (allow depth: buy accuracy)
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
    # ---- Deployment (inference recipe baked into the ckpt, ed9/C) ----
    decode: Optional[dict] = None  # None = legacy (no shipped recipe); e.g.
        # {"repetition_penalty": 1.15, "no_repeat_ngram_size": 3} (validated 0911:
        # 128tok/temp0.7 rep 0.448->0.042 at -0.015 ROUGE vs pen1.0; applied by
        # assemble onto generation_config, explicit generate() kwargs still win)

    def __post_init__(self):
        # public aliases (0910 pivot): speaker -> threshold, mol -> moe; canonical values are
        # normalized here so ckpts always store one of threshold/moe (bit-compatible with history)
        if self.gate_mode == "speaker":
            self.gate_mode = "threshold"
        elif self.gate_mode == "mol":
            self.gate_mode = "moe"
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
        if not 0.0 <= self.diff_easy_nll <= self.diff_hard_nll:
            raise ValueError(f"need 0 <= diff_easy_nll <= diff_hard_nll, got "
                             f"{self.diff_easy_nll}/{self.diff_hard_nll}")
        if self.diff_easy_mult < 0 or self.diff_hard_mult < 0:
            raise ValueError("diff multipliers must be >= 0")
        if self.gate_mode == "moe":
            if self.select_mode not in SELECT_MODES:
                raise ValueError(f"select_mode must be one of {SELECT_MODES}, got {self.select_mode!r}")
            if self.weight_mode not in ("pmax", "renorm"):
                raise ValueError(f"weight_mode must be pmax|renorm, got {self.weight_mode!r}")
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
        if self.decode is not None:
            if not isinstance(self.decode, dict):
                raise ValueError(f"decode must be a dict, got {type(self.decode).__name__}")
            unknown = set(self.decode) - {"repetition_penalty", "no_repeat_ngram_size"}
            if unknown:
                raise ValueError(f"decode has unknown GenerationConfig keys {sorted(unknown)}")
            rp = self.decode.get("repetition_penalty", 1.0)
            if not isinstance(rp, (int, float)) or not rp >= 1.0:
                raise ValueError(f"decode repetition_penalty must be >= 1.0, got {rp!r}")
            ng = self.decode.get("no_repeat_ngram_size", 0)
            if not isinstance(ng, int) or not ng >= 0:
                raise ValueError(f"decode no_repeat_ngram_size must be a non-negative int, got {ng!r}")

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
                f"acc_target={self.acc_target}"
                f"{' decode=' + str(self.decode) if self.decode else ''})")
