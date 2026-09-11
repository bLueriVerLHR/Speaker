# Gating Mechanism (dual-mode, one switch)

`SpeakerConfig.gate_mode` selects one of two mutually incompatible gating schemes.
Both share the same outer idea — fixed (shared) layers + gated layers with
per-token adaptive depth — but they differ in *who decides* and *how the budget
enters the loss*. Checkpoints of one mode do not load in the other (`gate.pt` key
sets are disjoint); `mod_config.json` records the mode, and a missing `gate_mode`
field (old checkpoints) is inferred as `threshold`.

**Scheme names (and CLI aliases)**:

| Scheme | Canonical `gate_mode` | CLI alias | Decides |
|---|---|---|---|
| **Speaker** | `threshold` | `--gate_mode speaker` | each layer independently (per-layer gate) |
| **MoL** (Mixture of Layers) | `moe` | `--gate_mode mol` | one joint router for the whole gated region |

Aliases are normalized to the canonical values in `SpeakerConfig.__post_init__`,
so checkpoints stay bit-compatible with the historical `threshold`/`moe` formats.

---

## moe — joint routing over layers (default mainline)

One decision per token, made **once**, before the gated region is walked.

### Router

A single `JointRouter` (one linear map + bias, e.g. 1024×20+20 ≈ 20.5K params on a
24-layer 0.5B model) sits at the **entry of the gated region** — the lowest gated
layer. It reads the token state `h` at that point and produces a log-probability
over all `G` gated layers:

```
log p = JointRouter(h)          p over gated layers, sums to 1
```

The entry layer computes the route once per forward pass and stores it in the
model hub; every other gated layer consumes its own slice (`_consume_route`),
re-aligned to that layer's device in mixed CPU/GPU placement.

### Init calibration (router temperature)

A randomly initialized `JointRouter` over large-norm hidden states (7B) starts
**peaked**: the initial top-p mean k lands far below the near-dense start the
threshold scheme gets from `calibrate_tau`. Training from such a start collapses
during the budget ramp (r7: k sank 4.9→1.7, accuracy never took off — the STE
soft-inclusion saturates and the dual price cannot reopen gates it already
stopped pressing). `calibrate_router_temp` (finetune `--router_calib_batches`,
0 = legacy off) caches entry hiddens from a few batches and bisects a persistent
logit temperature (`router_temp`, stored in `mod_config.json`, never in
`gate.pt` — key sets unchanged; 1.0 = bit-for-bit identity) until the initial
mean k reaches the target (`--router_start_k`, default ≈ 0.6·G). Only flattening
is applied: a start that is already dense enough is left untouched.

### Selection: top-p (k adapts per token)

Layers are sorted by probability; the smallest prefix whose cumulative probability
reaches `top_p` (default 0.9) is selected:

- easy tokens: probability concentrated → few layers selected, small k;
- hard tokens: probability spread → more layers selected, larger k.

A `top-k` variant (fixed k per token) is also available; in that mode the budget
term is disabled (`budget=None`) since k is not something the loss can steer.
`kmax` caps k in either mode.

### Weighted residual

Selected layer probabilities become residual weights `w` (`weight_mode`, ed7),
and each selected layer is applied as a weighted residual:

```
h ← h + w_l · (F_l(h) − h)        for l in S(t)
```

- `pmax` (default): `w = p / p_max` — the top layer carries 1.0, the rest
  proportionally. Total gain scales *with* k (each opened layer adds up to a
  full-strength residual, as in the threshold scheme), the all-selected limit is
  exactly the dense forward, and the dual's "buy layers back" lever works.
- `renorm` (legacy): `w = p / Σ_selected p` (Σw = 1). Total gain is fixed at
  1.0 whatever k is: opening a layer dilutes the good layers (the LM loss itself
  pushes k down), and relaxing λ cannot buy accuracy back — the r7 collapse
  (init lm 13.2/acc 0.01, 4000 steps only back to acc 0.24 at λ floor).

Skipped layers cost nothing: pass-through, and at decode they write no K/V.

### Forward hard, backward soft

The forward pass uses the hard subset S(t). Gradients flow through a soft
inclusion: each layer's soft mask is `sigmoid((p_sorted − cum) / T)` around the
cut point (a straight-through-friendly relaxation), so the router learns *which*
layers to keep. The mean depth `mean(k)` is differentiable through `k_soft`
(expected number of included layers) — this is what the budget price attaches to.

### Loss terms

```
loss  = LM
      + λ · k_soft.mean()          budget: dual price on depth (see docs/training.md)
      + aux                        z-loss (router logit magnitude)
      + routing balance            variance of the mean routing distribution
                                    (anti-collapse: not every token the same subset)
      + cos_reg · mean(soft·sim)   skip layers that would barely change the state
                                    (skipped when cos_reg_coef = 0 in moe mode)
```

### Price accounting

The "price" the dual controls is the **per-inference actual demand**: the average
per-token active-layer memory (avg, not peak, not resident set). This matches what
a decoding loop actually needs, rather than worst-case occupancy.

### Fixed-layer promotion interplay

Promoting a gated layer to fixed shrinks the routed set `G` (the joint router no
longer scores it). `finetune/profile_layers.py --out` therefore **excludes**
`joint_router.*` from the new checkpoint: resuming re-initializes the (now smaller)
router over the remaining gated layers.

---

## threshold — legacy per-layer gate

The original scheme, kept verbatim for old checkpoints and for `calibrate_tau`
diagnostics. Each gated layer owns an independent binary decision:

```
logit = Router_l(h)                    per-layer linear map H→1
m     = σ((logit − τ_l) / T_a)         τ_l: per-layer threshold (learnable, clipped)
              + Gumbel noise (train)   STE: hard forward, soft backward
m = 1:  h ← h + (F_l(h) − h)           full execution
m = 0:  h ← h + (1 − m)·c_l            skip + compensation c_l (0-init, DASH-style)
```

Budget enters as `λ · mean(k) + over-kmax penalty` (k = Σ m_l per token, plus the
fixed layers), and `cos_reg` is always computed in this mode (it feeds τ
calibration).

### Parameter accounting (0.5B, 24 layers, 20 gated)

- moe: ~20.5K params (joint router only)
- threshold: ~41K params (20 × (router + τ + comp))

Both are negligible relative to the base model; the difference that matters is the
decision structure — one global, per-token, k-adaptive choice (moe) vs G local,
k-capped choices (threshold).

---

## Configuration quick reference

| Key | Mode | Meaning |
|---|---|---|
| `gate_mode` | both | `"moe"` (default) / `"threshold"` |
| `top_p` | moe | cumulative-probability cut (default 0.9) |
| `kmax` | both | hard cap on per-token k |
| `gumbel_scale` | threshold | Gumbel noise scale while training the gate |
| `always_on_layers` | both | `None` = derive (head/tail prior); `[]` = pure gating (gating-first mainline); list = explicit fixed set |
| `sparsity_price` (λ), `acc_target` | both | dual-budget knobs: `acc_target` = `auto` (default, dense-referenced) / float / `none` (docs/training.md) |

Checkpoint formats: moe `gate.pt` stores `joint_router.*` only; threshold stores
per-layer `router/tau/comp` (+ LoRA if used). `speaker/checkpoint.py` filters
promoted-layer keys on promotion and strips wrapper prefixes for clean-base saves.
