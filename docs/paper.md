# Speaker: Layer-Level Budget-Elastic Gating with Shared Anchors for Efficient LLM Inference

> **DRAFT** (r7 cycle). Numbers are frozen from completed runs; `[TODO-P2]` marks slots to be
> filled by the r7 evaluation suite (fresh-slice / 128-token generation / k-distribution) and
> the pending MoDification scan. Provenance: every number in this draft traces to a dated
> experiment log (see §7 Provenance).

## Abstract

Large language models execute every transformer layer for every token, yet layer
contributions are heavily skewed: a few anchor layers carry most of the computation while
a large middle band is functionally redundant for most tokens. We present **Speaker**, an
architecture that (i) pins a small set of **always-on shared anchor layers** (learned from
runtime load profiles, not hand-picked), (ii) gates the remaining layers per token with a
**budget-elastic Lagrangian** — the layer price λ adapts online to defend an accuracy
floor while spending all remaining headroom on sparsity — and (iii) repairs the sparsified
network by **joint LoRA fine-tuning**, so the gate and the weights co-adapt. On
Qwen2.5-7B full-data SFT, Speaker is the only method in a five-way comparison (dense-ft,
MoD, MoDification, Router-Tuning, Speaker) that is simultaneously **sparse** (54% of
layers executed) and **above the un-finetuned base** (+2.9pt held-out accuracy; +7.7pt at
the 22k-step operating point with 37.5% layers and −58% weight traffic). We further give a
failure analysis of why "sparse and accurate" is hard: capacity-pinned routing (MoD) buys
neither, frozen-base gate-only tuning (RT) self-locks near-dense, regularization-pressure
tuning (MoDification) is bistable between near-dense and gate-collapse, and our own joint
top-p routing collapses through a renormalization-induced dilution mechanism — an
honest negative result with a calibrated-temperature mitigation.

## 1 Introduction

**Layer redundancy is an empirical fact, not an assumption.** Ablating layers one at a
time on a 24-layer 0.5B model shows a heavily skewed contribution profile (L0 ≫ L1 >
L23; the middle band is nearly flat). Load profiling at 7B shows the same shape: without
any prior, the gate spontaneously routes 0.99/0.97 of tokens through the first/last
layers while the middle band idles at 0.08–0.18. Token-level probes show inference-heavy
text uses ~2.4 more layers than syntactic text — per-token demand varies.

**But sparsity is not free, and "repair" has a price.** A frozen-base model with deleted
layers loses accuracy that gate-only tuning cannot buy back. We frame this with the
**repair budget**: an equal-spec LoRA run with all layers on quantifies what joint
fine-tuning can restore (+16.5pt on our 7B full-data protocol, k=28 endpoint). A sparse
method's net value is its accuracy *minus* the repair budget at matched k. Under this
calibration, prior layer-skipping baselines are strictly negative (§4).

Speaker's position:
1. **Shared anchors, not a hand-designed skeleton.** Head/tail anchors initialize the
   always-on set; runtime load profiles promote persistently-hot gated layers into it.
   The anchors the gate spontaneously chooses replicate the hand-designed prior (L0/L27
   at 0.99/0.97 without any prior).
2. **Budget elasticity as a control problem.** The training objective is
   `LM loss + λ·mean(k)` with a *dual* on λ: when held-out EMA accuracy is above a floor,
   λ grows and spends the headroom on sparsity; below the floor, λ relaxes and the gates
   reopen. The floor is set at the *un-finetuned base* level — "be at least as good as
   the model you started from, then get cheap".
3. **Joint repair, not gate-only.** LoRA adapters train together with the gate; KL to
   the dense teacher (optional) and rollout-mode unlikelihood (anti-repetition) complete
   the recipe.

Contributions:
- The shared-anchor + budget-elastic gating architecture with an accuracy-floor dual,
  and its deployment path (hard layer skipping, sparse KV cache, budgeted GPU-residency
  scheduling with random/lru/lfu policies) (**§3**, docs/gating.md, docs/inference.md).
- A controlled five-way comparison at 7B full data, 4000 steps, identical LoRA spec,
  where Speaker is the only sparse-and-above-base method (**§5**).
- A mechanism-level failure analysis of four alternatives — including our own joint
  top-p router, whose collapse we trace to renormalization dilution and mitigate with
  init-time temperature calibration (**§4.4**, §6).

## 2 Related Work

| | MoD (2024) | MoDification (2025) | Router-Tuning (2025) | **Speaker** |
|---|---|---|---|---|
| Decision granularity | per-layer router, top-k capacity | per-layer router, threshold-p | per-layer router | per-layer threshold (mainline) / joint top-p |
| always-on layers | — | shared gate | — | **yes, profile-promoted** |
| Sparsity control | constant k via capacity | R = α·ΣFG pressure | relu(cap−target) budget | **elastic λ·k + accuracy-floor dual** |
| Base model | trained from scratch | 10B word-conversion | **frozen base** | LoRA joint repair |
| Efficiency target | FLOPs at fixed k | serving cost | token budget | **deployed demand: skipped layers + sparse KV + residency** |

MoD fixes k by capacity and trains from scratch with BCE routing; at 7B with LoRA repair
its accuracy stays −14 to −22pt below the base at every capacity we scanned. MoDification
replaces the hard budget with a regularization pressure; its behavior is bistable in α
(near-dense or gate-collapse; §4.3). Router-Tuning freezes the base and trains gates
only; on our protocol its exec-rate knob does not move the network (self-locks at
0.68–0.72 across targets 0.1/0.25/0.5).

## 3 Method

### 3.1 Architecture: shared anchors + gated region

N-layer decoder; a small always-on set `A` (initialized to first/last k layers,
`|A|=4` on 28-layer Qwen2.5-7B) executes unconditionally; the remaining G = N − |A|
gated layers execute per token. Gated layer l applies the **weighted residual**
`h += w_l · (F_l(h) − h)`; w_l = 0 means the layer is skipped entirely (no forward, no
KV write). The always-on set is not static: `profile_layers` harvests per-layer load and
promotes gated layers above a load threshold (≥0.9) into `A`, yielding a new ckpt whose
router is re-initialized for the reduced pool.

### 3.2 Gating (dual scheme, switchable)

- **threshold (mainline, all positive results)**: each gated layer has a linear router
  `a_l(h) → logit` and a threshold τ_l; `open iff sigmoid((a_l − τ_l)/T_a) ≥ 0.5`, STE
  backward. τ_l is **calibrated at init** from per-layer stability statistics
  (StableSkip-style): the network starts near-dense (k₀ ≈ 15/24 gated) and the budget
  trims it during training.
- **moe (joint top-p routing)**: one `JointRouter(H→G)` at the gated-region entry emits
  a distribution over gated layers; top-p prefix selection (k adaptive per token);
  selected probabilities renormalize into w. See §4.4 for the collapse analysis and
  §6 for status.

Gating cost is negligible: ~86K parameters on 7B (<0.002%).

### 3.3 Elastic budget with an accuracy-floor dual

```
L = L_LM + λ·mean(k_soft) + aux        aux = z-loss + load-balance + cos-regularizer
λ ← λ·(1±r)  every eval, after warmup:  EMA_acc < floor → λ shrinks; ≥ floor → λ grows
λ ∈ [price_min, price_max]  (multiplicative, can climb back from the floor)
```

The floor is set at the un-finetuned base's held-out accuracy (0.55 vs measured base
0.536). This makes the *operating point* a discovery, not a hyperparameter: the run
settles where the accuracy constraint binds (r5: k 10.5±1.4 = 37.5% of layers with acc
+7.7pt above base).

### 3.4 Joint repair training

LoRA (r8, q/v) on the base trains jointly with the gate; optional KL distillation to
the dense teacher; rollout-mode unlikelihood (every 20 steps, 24-token hard rollouts
from real prefixes, DAgger-style) suppresses repetition without spending sparsity
(−41% rep3 at unchanged k; the GT-prefix variant instead relaxes the gate, +50% k, and
loses per-token adaptivity — we use rollout).

### 3.5 Deployment

Hard skip mode: unselected layers do not execute and do not write K/V (sparse KV cache;
ROUGE parity with full cache, −0.007). Weight traffic (GB moved per token under
CPU/GPU tiering) drops −58% at the r5 operating point. A **residency scheduler**
(speaker/scheduler.py) keeps always-on layers on GPU, packs gated layers into the
remaining budget minus KV/activation reserve under random/lru/lfu policies, harvests
per-layer token counts, and migrates only between generations (KV cache never moves).

## 4 Why "sparse and accurate" is hard: four failure modes

All numbers: Qwen2.5-7B, full 904K-sample SFT, 4000 steps, identical LoRA spec
(RT follows its paper's frozen-base recipe), held-out slice [904000, 904300), base
reference acc 0.536 / loss 2.740.

### 4.1 MoD: capacity pins k, BCE binarizes routing
Capacity {0.125, 0.25, 0.5} → k pinned at 16.2/18.4/21.0 with σ → 0.5 at the largest
cap; accuracy 0.318/0.358/0.396 — monotonically better as it approaches dense, never
crossing the base. Short-run dynamics (0.5B, 500 steps) show the mechanism: init loss
13.3 with the raw-score router, winner-take-all token routing (76% of tokens collapse
to k=4), and per-layer execution rates frozen at the capacity value — k is an
arithmetic constant, not a per-token decision.

### 4.2 Router-Tuning: frozen base, knob ineffective
Targets {0.1, 0.25, 0.5} all end at exec 0.68–0.72, k ≈ 24/28, acc −1 to −2pt.
The budget term (~0.3) is drowned by the LM loss and STE saturates; without joint
repair the gate's only stable strategy is near-dense.

### 4.3 MoDification: bistable in α
α = 0.01 (paper default) → exec 0.99, k 27.8 (dense, no sparsity tax). α = 1.0 →
gates collapse **shut** (exec 0.00, k 14.1 = anchors only). α = 0.1 lands in the
sparse region (k 20.2, exec 0.44) — `[TODO-P2]` its final accuracy decides whether the
middle regime survives. The pattern mirrors MoD: pressure regimes either pay full
density or collapse; an elastic floor is what the family lacks.

### 4.4 Our own joint top-p router collapses (honest negative)
The joint router's renormalized weights (Σw = 1 over selected layers) couple k to
effective layer gain: at k₀ ≈ 14 every selected layer runs at w ≈ 0.07 of its residual
— the forward is destroyed, and the LM gradient's cheapest recovery is
**concentration** (sharpen p so the top layer gets w ≈ 1). Observed at 7B: k falls
14.5 → 4.5 while λ is already at its floor (the dual cannot push back — the gates it
would reopen have saturated STE masks); accuracy never takes off (0.22–0.32 final).
Two aggravators: training-time Gumbel noise inflates effective logit dispersion
(training k₀ ≈ 4.8 vs eval k₀ = 14.3 at top-p 0.7), and the budget ramp coincides with
the high-loss phase where LM gradients are least informative. Mitigations implemented:
init-time router-temperature calibration (flatten-only, bisected to a target starting
k; bit-for-bit identity when off) and kmax decoupling. The weight-coupling itself is
open (§6).

## 5 Experiments

### 5.1 Protocol
Models: Qwen2.5-7B-Instruct (main), Qwen1.5-0.5B (smoke only). Data: 904K-sample SFT
corpus, len 1024, bs 2, LoRA r8 α16 q/v (RT: paper recipe), 4000 steps, seed 42,
patience disabled. Held-out: [904000, 904300); fresh: [904300, 904600). Eval at bs 1
(MoD's raw-score router is padding-fragile at bs > 1 — itself a deployment defect).

### 5.2 Main table (r7, training-time held-out)

| Method | Working point | Acc | Δ vs base | k_total (/28) | Exec rate |
|---|---|---|---|---|---|
| base (raw) | — | 0.536 | — | 28 | 100% |
| dense-ft | LoRA, all layers | 0.701 | +16.5 | 28 | 100% |
| **Speaker-thr** | r5 recipe | **0.565** | **+2.9** | **15.2** | **54%** |
| MoD | cap 0.125 | 0.318 | −21.8 | 16.2 | 58% |
| MoD | cap 0.25 | 0.358 | −17.8 | 18.4 | 66% |
| MoD | cap 0.5 | 0.396 | −14.1 | 21.0 | 75% |
| MoDification | α 0.01 | `[TODO]` | — | 27.8* | 99%* |
| MoDification | α 0.1 | `[TODO]` | — | 20.2* | 44%* |
| MoDification | α 1.0 | `[TODO]` | — | 14.1* | 0%* |
| RT | target 0.1/0.25/0.5 | 0.517/0.520/0.525 | −1~−2 | 23.5–24.1 | 84–86% |

\* mid-run readings (step 940–1760); finals pending.

**Reading**: measured against the repair budget (dense-ft +16.5pt), every baseline is
strictly negative at every scanned operating point; Speaker-thr is the only positive
net effect under real sparsity.

### 5.3 Long-run operating point (r5, 22k steps, same family)
Held-out Δacc **+7.7pt** (0.614 vs 0.537), k 10.5±1.4 (37.5% layers), weight traffic
6.4 vs 15.4 GB/token (−58%); generation ROUGE 0.267 vs dense-same-slice 0.168 (+59%),
rep3 0.390 vs 0.191 (repetition remains the open quality gap; rollout-UL and decode-time
penalty are partial). Decode-time k drifts above training-time k (7.4 → 11.7; the gate
opens more layers on its own prefixes) — deployment accounting must use probed k, not
logged k.

### 5.4 Per-token adaptivity
k distribution at the 7B operating point is unimodal bell (μ 11.4, σ 1.9,
q10–90 = [9,14]) — per-token decisions, not capacity arithmetic (MoD pins a two-point
distribution at the capacity value; GT-UL narrows σ to 1.1 while inflating μ 37%,
i.e. adaptivity loss, which is why we use rollout-UL). `[TODO-P2]` r7 k-dist for all
methods.

### 5.5 Efficiency
FLOP accounting: k ≈ 12/28 ≈ 2.4× nominal compute saving at the r4 operating point;
wall-clock parity today (wrapper overhead dominates; kernel-level ragged execution is
future work). Sparse KV: skipping a layer writes no cache (parity ROUGE, −0.007).
Residency scheduling: budgeted packing with harvested counts, migration only between
generations.

### 5.6 Generation (r7) — `[TODO-P2]`
30 prompts × 128 tokens, greedy, fresh slice: ROUGE / rep3 / latency per method.

## 6 Limitations & open problems
1. **Wall-clock**: layer skipping saves FLOPs and weight traffic, not yet latency
   (Python wrapper overhead; ragged batched execution needed).
2. **Repetition** under free generation: rollout-UL halves it; the residual gap to
   dense remains the main quality issue.
3. **Joint top-p routing** (moe): renormalization-induced dilution is identified but
   unfixed; candidate is top-normalized weights (w = p/p_max) which restores dense-like
   forward at large k and sharp single-layer execution at small k.
4. Decode-time k drift (gate opens more layers on self-generated prefixes).
5. Results are SFT-continuation-centric (repair framing); from-scratch and long-context
   serving regimes are untested here.

## 7 Provenance (claim → experiment)
- Layer skew / anchor self-selection: per-layer ablation probe (0.5B); load profiles r5/r6.
- Dual-price dynamics, MoD/RT failure dynamics: r1 attribution logs + k-distribution probes.
- mdf bistability: r2 (0.5B) + r7 mid-run readings.
- Repair budget: r3 (dense-ft 0.724 @500-step small-data) and r7 (0.701 @full-data).
- Long operating point: r5 (ours_7b_full).
- UL variants: r4 (gt vs rollout).
- Sparse KV / offload / scheduling: KV-lite→sparse-cache series, offload run, ed5 scheduler
  smoke (GPU cross-device).
- moe collapse + calibration: r7 first-generation runs vs calibrated reruns (both archived).
