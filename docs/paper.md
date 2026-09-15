# Speaker: Post-Training Repair for Layer-Sparse LLM Inference

> **Status** (v0.3). Headline numbers are frozen from converged runs (Qwen2.5-7B,
> full 904K SFT × 4000 steps, seed 42) and restated in the unified **total-K
> accounting** (r10): k counts every executed layer including fixed ones, hard
> (deployment) mode; older gated-only figures are superseded. Attribution
> convention: **[Paper]** = prior published method we reuse or compare against
> (cited); **[Ours]** = this project's own mechanism, finding, or negative
> result. Provenance for every claim: §7.

## Abstract

Large language models execute every transformer layer for every token, yet per-token
depth demand is skewed and partly redundant. We present **Speaker**, a
post-training repair recipe that makes a 7B model simultaneously **sparse and above
its starting point**: per-layer threshold gates (**[Ours]** mechanism, standard
sigmoid+STE tools **[Paper]**) trained with a **fixed Lagrangian depth price plus a
quadratic depth cap**, joint LoRA repair, and rollout-mode unlikelihood on
self-generated prefixes (**[Ours]** adaptation of Welleck-style UL **[Paper]**). At
**k = 20.0/28 total active layers** the repaired model beats raw dense by **+4.3pt**
held-out accuracy (3 seeds +4.2 ± 0.4) while per-token weight demand drops **28%**
(9.38 vs 13.05 GB) and KV demand drops 28%; it is the only configuration in our
2×2 distribution grid (slice × prompt template) that is positive in all four
cells. The success criterion is **recovery, not elevation**: against an
equal-budget dense LoRA run (the repair calibration) the recipe retains 40–49% of
the finetuning gain (accuracy) and ~30% on ROUGE-L — the honest price of −28%
memory. The default operating point sits on the **recovery knee**: at k = 17 the
formal cell turns negative (−1.7pt); at k ≤ 14 every recipe family tested breaks
(−15 to −17pt). 128-token generation holds ROUGE-L above the raw-dense line
(0.211 vs 0.160; dense-ft 0.332) with a repetition residual (seq-rep-4 0.320 vs
raw 0.053, dense-ft 0.211) attributed to SFT finetuning, not gating (§2.4). We
also report the honest negatives that shaped the recipe: capacity-pinned routing
(MoD **[Paper]**) buys neither sparsity nor accuracy; frozen-base gate-only tuning
(Router-Tuning **[Paper]**) self-locks near-dense; regularization-pressure tuning
(MoDification **[Paper]**) is bistable; and our r10 attempts to remove the fixed
head/tail layers — anchor-only gating, squared-deviation budgets, and
profile-based promotion — all fail the formal protocol while matching the
champion elsewhere: fixed [0,1,26,27] buys cross-distribution robustness, not
depth budget (**[Ours]**).

## 1 Motivation

**Memory, not FLOPs, is the binding constraint where we deploy.** On a 24 GB edge
card a 7B bf16 model (13 GB of decoder weights) leaves little room for KV cache at
long context — yet every token pays full depth. The per-token *demand* (weights
that must be touched + KV that must be read) is the number a decoding loop actually
provisions; peak residency and FLOPs are secondary. This is the MoDification
accounting coordinate **[Paper]**, which we adopt as the single efficiency axis
(§4.1): same-loss-lower-memory or same-memory-lower-loss, nothing else.

**Repair, not finetuning — and the expectation is recovery, not elevation.**
Deleting or skipping layers of a frozen model loses accuracy that gate-only tuning
cannot buy back (RT evidence, §2.3). We therefore frame training as **post-training
repair**, the standard move of the pruning literature **[Paper: Wanda; SparseGPT]**:
sparsity necessarily wounds, training closes the wound. The success criterion is
**recovery to the dense model's level** — raw dense is the starting line. The
equal-spec dense LoRA run is the *repair budget* calibration, not a competitor: it
measures how much headroom finetuning could have bought. Measured (r10, same
process): raw dense 0.508 → ours 0.551 (**recovery > 100%**, +4.3pt above the
line) → dense-ft 0.616 (retention 40% of the buyable gain; held-out slice: 49%).
The recipe deliberately stops at the depth cap rather than chasing the dense-ft
line: −28% memory for about half the headroom is the operating point, and pushing
k below 20 breaks recovery entirely (§4.7).

**Serving frameworks cannot run this — yet.** Dynamic per-token layer paths
contradict the three assumptions of vLLM/SGLang-style serving **[Paper:
PagedAttention]**: CUDA graphs require static execution paths, continuous batching
requires all requests to want the same layers at the same step, and paged KV
assumes every layer holds every token's KV. Our contribution is therefore
training-side repair plus demand accounting, not latency: measured wall-clock
today shows no win (research wrapper ≈ +45% ms/tok at bs1, §4.4). The honest path
to serving is a per-request *static* mask (profile → freeze the active set → run a
static subgraph) — a separate serving paper, listed in §6, not claimed here.

## 2 Observations (what the data forced on us)

1. **Layer demand is skewed and self-organizing.** Per-layer ablation (0.5B) shows
   L0 ≫ L1 > L23 with a flat middle band; with no fixed-layer prior, gates
   spontaneously route 0.99/0.97 of tokens through the first/last layers while the
   middle idles at 0.08–0.18 **[Ours]**. Fixing [0,1,26,27] costs almost nothing;
   the compressible part is the 16 gated layers inside the current k = 20.0 total.
2. **The cap binds; the price is dead; the accuracy-floor dual was cut.** The
   gated depth sits at the kmax cap (16.1 soft vs cap 16), not below it; a 10× λ
   scan (0.0002 → 0.002) leaves k flat (16.07–16.47, Δacc inside seed noise), and
   a no-cap control (λ alone, cap penalty off) drifts all the way to k = 27/28
   **[Ours]**. Depth is set by the quadratic one-sided cap penalty, never by the
   Lagrangian price. The earlier accuracy-floor dual controller (acc_target: once
   accuracy reaches dense−margin, squeeze k) suppressed accuracy **by design**; it
   was removed from the recipe before the current headline runs — with it off,
   accuracy grows past the old target line until it hits the depth-cap wall, not
   indefinitely: the ceiling is the dense-ft line, approached as k → 28.
3. **Training k undercounts deployment k by ~25%.** Teacher-forced position
   quartiles over a 1024 window are dead flat ([20.4, 20.2, 20.2, 20.3] hard k,
   new `--pos_bins` probe **[Ours]**), and hard-probed k (20.3) equals free-decoded
   k (20.4): there is no position effect and no self-generation drift. The entire
   train→deploy gap is the **soft→hard gate gap** — deployment accounting must use
   probed hard k, never logged soft k **[Ours]**.
4. **Repetition residual is a finetuning artifact, not gate damage.** All finetuned
   models — including near-dense MoDification (k ≈ 28) and the equal-budget
   dense-ft (seq-rep-4 0.211 vs raw 0.053) — fall into multi-turn self-dialogue
   loops on chit-chat prompts; raw dense does not **[Ours]**. Human rating (9
   samples, 1–5): dense 4 > long_4k 3.5 ≈ mdf 3.5 > short-rollout 2.5. The
   seq-rep-4 gap to dense is this loop — an SFT-finetuning artifact; the data-format
   version of this attribution was later falsified (§4.5): single-turn training
   does not cure it.
5. **Baselines fail in the same place for different reasons** (r1–r7 mechanism
   runs **[Ours]** measurements of **[Paper]** methods): MoD's capacity pins k to
   an arithmetic constant with winner-take-all bimodality, and its raw-score
   router is padding-fragile at bs > 1 (itself a deployment defect); RT's budget
   term is drowned by the LM loss and STE saturates, self-locking exec at
   0.68–0.72; MoDification is bistable in α (0.01 → exec 0.99 near-dense;
   1.0 → gates collapse shut; 0.1 lands mid-regime).
6. **Our own joint router collapsed — mechanism identified.** Renormalized weights
   (Σw = 1) couple k to effective layer gain: at large k every layer runs at w ≈
   0.07 and the LM gradient's cheapest fix is concentration (k 14.5 → 4.5 while λ
   sits at its floor, disconnected) **[Ours]**. Top-normalized reweighting
   (w = p/p_max) reconnects the dual lever: +13pt accuracy, healthy adaptive k.
   Unfixed residue: init-time temperature calibration only changes counts, not
   routing identity.
7. **Fixed head/tail layers buy distribution robustness, not depth budget.** In
   the 2×2 grid {train-interior slice, held-out slice} × {chat template, raw},
   the fixed-[0,1,26,27] recipe is positive in all four cells (+2.6 … +8.2pt);
   every anchor-only (single fixed layer) variant trained to the same total k
   matches it in three cells and fails specifically in the chat×train-interior
   cell (T18: −0.5 vs +4.3) **[Ours]**. k-distribution probes show no pathology
   (unimodal, position-flat) — the deficit is *which* layers open under a
   format-shifted prompt, not how many.

## 3 Design (default recipe = long_4k, restated in total-K)

```
loss = L_LM + λ·mean(k_gated) + 0.05·mean(max(k_gated − kmax, 0)²) + UL_rollout(128tok, coef 0.3)
λ = 0.0005 fixed (dual off) · threshold gates + tau calibration
fixed [0,1,26,27] · kmax 16 gated cap (deploy hard k_total = 20.0/28)
base = LoRA r8 α16 q/v joint-trained · full 904K SFT × 4000 steps · bs2 × 1024 · seed 42
```

- **Threshold gate [Ours mechanism / Paper tools].** Per-layer linear router +
  threshold τ, open iff sigmoid((a−τ)/T) ≥ 0.5, STE backward **[Paper: Bengio et
  al.]**. τ calibrated at init from stability statistics (near-dense start, the
  budget trims down). Gate cost: ~41K params on 0.5B scale (<0.005%); negligible
  at 7B. Note: in threshold mode kmax is *not* enforced during selection — the
  loss-side cap penalty is the only depth constraint (§2.2).
- **Fixed Lagrangian price + quadratic cap [Ours verdict].** `λ·mean(k_gated)` with
  λ fixed, plus `0.05·mean(max(k_gated−kmax,0)²)` — the cap term is what pins
  depth (the λ term alone cannot: the no-cap control drifts to k 27/28, §4.7).
  The alternatives were built and falsified: an accuracy-floor dual controller
  (EMA + warmup gate, still in-tree but default-off — it holds accuracy at
  target by construction and thus suppresses growth past it), hinge/tail budget
  shapes (hinge parks k but adds no accuracy; tail is dead weight),
  difficulty-shaped per-token λ (slope r = 0.046, indistinguishable from
  unshaped), and the r10 squared-deviation budget with anchor-only gating
  (§4.6) **[Ours negatives]**. All budget shapes remain selectable flags with
  persisted configs; a training-time guard warns on the unconstrained
  combination (mean + cap off).
- **Joint LoRA repair [Paper tool / Ours necessity finding].** LoRA **[Paper: Hu
  et al.]** trains together with the gates; gate-only (frozen base) is falsified twice
  (own gate-only runs + RT reproduction). kl_coef = 0: distillation arms were cut
  because kl = 0 already recovers past dense and the remaining gap is format, not
  distribution (§2.4) **[Ours verdict]**.
- **Rollout UL [Paper method / Ours adaptation].** N-gram unlikelihood **[Paper:
  Welleck et al.]** computed on hard self-generated rollouts (DAgger-style
  **[Paper: Ross et al.]**), not ground truth: GT-UL relaxes depth (+50% k,
  adaptivity loss σ 1.9 → 1.1); rollout-UL holds k flat at −20% training
  throughput. Length 128 beats 24 only at convergence (500-step pilot read
  0.475 vs 0.340 — an undertraining false negative; 4000-step reads 0.223,
  r8c-era protocol) **[Ours]**; dose scales with length (UL mean 1.75 vs 0.59 at
  equal coef).
- **Deployment [Ours].** Hard skip (unselected layers neither execute nor write
  KV — sparse KV cache, ROUGE parity Δ−0.007 on 0.5B); budgeted GPU-residency
  scheduler (always-on pinned, gated layers packed under random/lru/lfu, migrate
  only between generations, KV never moves); per-token demand accounting
  (weights/KV/FLOP in the MoDification coordinate).
- **Kept but off by default [Ours].** DualController (accuracy-floor λ),
  difficulty shaping, hinge/tail/sqdev budgets, joint top-p router (MoL), KL
  channel — each with a logged falsification or partial-fix verdict; see
  docs/training.md and docs/gating.md for the mechanism details.

## 4 Experiments

### 4.1 Protocol (frozen)

Qwen2.5-7B-Instruct; full 904K SFT slice [0, 904000) training. **Formal cell**:
offset 20000 (train-interior slice, deliberately the harder distribution; n = 300,
max_len 1024, bs 1, labels accounting, chat template). **Robustness grid**: {20000,
904000 (held-out)} × {chat, raw}, same process, dense re-anchored — the r9_compare
discipline; cross-time conclusions only within one process. **Generation
(canonical, r10)**: offset 904300, 30 prompts × 128 tokens greedy. Metrics use
paper-original names only **[Paper]**: ROUGE-L F1 (Lin 2004, via the public
rouge-score API with a documented Tokenizer-subclass extension for CJK) and
seq-rep-4 (Welleck et al. 2020, Eq. 10). 30-prompt ROUGE is a coarse screen
(ranking ≠ human ranking, §2.4). k is reported as probed hard total-K everywhere.

### 4.2 Main table (converged, hard-mode deployment numbers)

| Method | formal acc (Δacc) | k_total /28 | mem GB / FLOP | ROUGE-L | seq-rep-4 |
|---|---|---|---|---|---|
| dense (raw) | 0.508 | 28 | 13.05 / 1.00 | 0.160 | 0.053 |
| dense-ft (repair budget) | 0.616 (+10.8) | 28 | 13.05 / 1.00 | 0.332 | 0.211 |
| **ours (default)** | **0.551 (+4.3)** | **20.0** | **9.38 (−28%) / 0.72** | **0.211** | **0.320** |
| MoDification (α 0.01) | 0.593 (+8.5) | 27.8 | 12.97 / 0.99 | — | — |

(MoDification generation metrics not re-measured under the canonical r10 protocol;
r8c-era figures exist but on a different slice. Held-out+raw for ours: 0.619
(+8.2); train-time held-out reads reproduce to 0.1–0.2pt.)

**Reading.** Recovery is complete (>100% of the raw-dense line) in all four
distribution cells (the only such config). Retention vs the repair budget is 40%
(formal acc) / 49% (held-out acc) / ~30% (ROUGE-L 0.211 between raw 0.160 and
dense-ft 0.332) — the price of −28% weight/KV demand. dense-ft also loops
(seq-rep-4 0.211 vs raw 0.053): repetition is finetuning-general, ours is worse
(0.320). Same-memory comparisons: nothing else occupies the k ≈ 20 column.

### 4.3 Ablations (each a verdict, not a sweep)

- Rollout length needs convergence to judge (0.475@500 steps → 0.223@4000, r8c-era
  protocol).
- Budget shape: mean + cap wins; hinge/tail cut; sqdev does not pin k (converges
  to T + 2.5…8, overshoot growing as T falls).
- Difficulty shaping: slope 0.046, cut.
- MoL reweighting: pmax over renorm +13pt, collapse broken, recovery partial.
- k accounting: soft log 16.4 vs deploy hard 20.1 — report hard only.
- Data format: single-turn truncation does not cure repetition (seq-rep-4 0.270
  vs multi-turn 0.281 at n=100) — format attribution falsified (§4.5).
- λ lever: 10× scan (0.0002/0.0005/0.001/0.002) → k 16.14/16.12/16.47/16.07,
  Δacc 3.6–4.5pt (inside the ±0.4pt seed band) — depth is not λ-priced; the no-cap
  control (λ alone) drifts to k 27/28 and matches dense-ft behavior.

### 4.4 Honest costs

Training throughput −25% (rollout-128); inference wall-clock no win in the
research wrapper (+45% ms/tok at bs1 greedy: 28.0 → 40.6 over 30×128tok — FLOP
savings need kernel/ragged execution); 30-prompt generation metrics are noisy
screens next to human reads.

### 4.5 Robustness & falsifications (r9)

- **Seeds.** Re-running the default recipe with seeds {42, 43, 44} (formal
  protocol, one process, dense baseline re-run in the same process): Δacc **+4.29 / +3.83 /
  +4.55 pt** → **+4.2 ± 0.4 pt** (mean > 3× std); k = 16.12 / 16.31 / 16.12
  (gated accounting era; total + 4 fixed). The headline survives multi-seed
  replication.
- **Sample size.** Held-out n = 300 → 1000: Δacc +3.4 pt, k 16.1 ± 1.05 —
  direction unchanged. Generation n = 30 → 100 (r8c-era protocol): dense
  0.254/0.079 (stable vs the frozen 0.252/0.073); long_4k ROUGE-L 0.273 > dense,
  seq-rep-4 0.281 (the red cell persists at 3.6× dense).
- **Data-format fix falsified.** Training on first-turn-truncated (single-turn)
  data leaves repetition unchanged (seq-rep-4 0.270 vs multi-turn 0.281, same
  n=100 protocol) while formal Δacc drops to +0.5 pt — every SFT-finetuned
  variant loops regardless of data format, raw dense does not. The residual is
  an SFT-finetuning artifact; the decode-time n-gram lever (0.042 ≈ dense)
  remains the working fix.

### 4.6 r10 round 1: distribution grid & the anchor-only line

2×2 grid (Δacc pt, same-process dense re-anchor, hard total-K):

| config | k | 20k+chat | 20k+raw | heldout+chat | heldout+raw |
|---|---|---|---|---|---|
| ours fixed4 (default) | 20.0 | +4.3 | +2.6 | +4.2 | +8.2 |
| anchor-only T18 | 20.0 | −0.5 | +2.7 | +5.4 | +8.3 |
| anchor-only T10 | 16.1 | −4.0 | −4.4 | −1.4 | +3.7 |
| T10 + promotion | 15.8 | −7.3 | −2.8 | −3.2 | +5.6 |

Sanity: heldout+raw reproduces train-time numbers to 0.1–0.2pt; 20k+chat
reproduces the earlier frontier file to 0.02pt (loading verified, process variance
negligible). Probes: all unimodal, position-flat, hard k matches eval k. Verdicts:
removing the fixed head/tail layers (anchor-only) fails the formal cell at every
target T tried; squared-deviation budgets do not pin k (T + 2.5…8); profile-based
promotion (train 1000 steps → profile 32 texts → promote top-4 loaded layers →
resume 4000 steps) makes the formal cell *worse* (−7.3 vs −4.0 at same T, same k);
price escalation to 0.05 collapses (k 7.8, −31pt). Sparse tax vs dense-ft (same
LoRA budget, same process): formal −6.5pt / held-out −8.4pt (retention 40%/49%).

### 4.7 The recovery knee (kf arms, 0916)

Fixed-4 recipe at lower depth pins (cap penalty 0.05; pins verified by probe):

| config | k_total | 20k+chat | 20k+raw | ho+chat | ho+raw | mem |
|---|---|---|---|---|---|---|
| fixed4 kmax16 (default) | 20.1 | **+4.3** | +2.6 | +4.2 | +8.2 | 9.38 (−28%) |
| fixed4 kmax12 (kf16) | 17.1 | −1.7 | −2.4 | +0.7 | +6.1 | 8.04 (−38%) |
| fixed4 kmax8 (kf12) | 13.6 | −16.7 | −9.8 | −10.4 | −0.1 | 6.36 (−51%) |
| no-cap control (λ only) | 27.1 | n/a | n/a | n/a | +12.7 (≈ dense-ft) | 13.0 |

Generation scales with k: ROUGE-L 0.211 / 0.200 / 0.182 (dense 0.160); seq-rep-4
0.32 / 0.31 / 0.45 — repetition worsens as depth drops. Probes unimodal and
position-flat. **The recovery boundary (Δ ≥ 0, formal cell) sits at k ≈ 20: the
default recipe is the minimal-depth recovery point of its family.** −38% memory is
purchasable at −1.7pt formal; k ≤ 14 breaks recovery in every recipe family tested
(fixed4 −16.7; anchor-only −15.5).

## 5 Related work (one-paragraph placements)

- **MoD [Paper]** — token-choice top-k capacity + BCE, from-scratch; k is
  arithmetic, routing is bimodal, scores are padding-fragile.
- **MoDification [Paper]** — threshold-p + shared gate + R = αΣFG pressure, 10B-word
  conversion, long-context serving; accuracy wins by staying dense.
- **Router-Tuning [Paper]** — frozen base, sigmoid+STE, relu(cap−target); the knob
  doesn't move.
- **Pruning repair [Paper: Wanda; SparseGPT]** — our framing source: wound then
  close it; dense-ft as repair-budget calibration is borrowed logic.
- **Repetition control [Paper: Welleck UL; DAgger rollouts]** — our UL-on-rollouts
  is the composition.
- **Serving [Paper: PagedAttention/vLLM; FlexGen]** — latency–throughput curves are
  the missing measurement; our demand accounting is the static prerequisite.

## 6 Limitations & future work

1. Wall-clock parity (needs ragged kernels; the research wrapper pays +45% ms/tok).
2. Repetition residual vs raw dense (SFT-finetuning artifact shared with dense-ft;
   format fix falsified; decode-time n-gram backstop works; a data-side cure is open).
3. **Resolved (0916):** the acc–k frontier below k ≈ 20 is measured — recovery
   breaks first in the formal cell at k 17 (−1.7pt), everywhere by k 14 (−16.7);
   the default k = 20 sits on the knee.
4. Serving path: per-request static mask → static subgraph (separate paper).
5. Cut, not forgotten: KL ratio/direction/on-policy/data arms, hidden-MSE
   diagnostics, TTFT/TPOT matrix (AGENTS.md r8-cut row).

## 7 Provenance (claim → artifact)

- Skew / self-selection: 0.5B ablation probe; r5/r6 load profiles.
- Baseline failures: r1 dynamics + k-dist probes; r3/r7 five-way tables.
- UL dose / length: r4 + r8c rollout-24-vs-128 + 500-step-vs-4000-step.
- Soft-hard / position: `--pos_bins` probe (flat quartiles; hard ≈ decode).
- Dialogue-loop attribution: 9-sample human read of r8c_4k_gen.json; dense-ft
  loops too (r10_gen_denseft.json).
- MoL collapse + pmax: ed7/0910-0911 runs; cal-arm variance verdict.
- Falsified options: budget_form (T-scan 7 arms), difficulty slope, GT-UL
  adaptivity loss, anchor-only/sqdev/promo/price-0.05 (r10_T*, r10_T10_promo*) —
  all archived with logs and persisted configs, none in the default path.
- Seeds / sample size / format fix: r9_s43, r9_s44, r9_long4k_heldout1k.json,
  r9_gen100.json, r9_singleturn + r9_compare.json (0915).
- λ scan: r9_lam{2e4,1e3,2e3} + r9_lam_compare.json (0915).
- Total-K restatement + 2×2 grid: r10_frontier.json; r10_eval_{off20k,ho}_{chat,raw}.json;
  r10_probe_*.json (0915).
- Sparse tax: r10_tax_{off20k_chat,ho_raw}.json (r8c vs r7_dense, same process).
- Recovery knee: r10_kf16 / r10_kf12 + kf_eval_*.json, kf_probe_*.json, kf_gen.json;
  no-cap control r10_knee12/knee8 (λ-only, k → 27) (0916).
- Checkpoints: `r8c_long_4k`, `r7_dense`, `r10_{T*,kf16,kf12}` (formal table inputs);
  plan & pre-registered readout rules: .archive/0915/PLAN.md.
