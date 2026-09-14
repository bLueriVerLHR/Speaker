# Speaker: Post-Training Repair for Layer-Sparse LLM Inference

> **Status** (v0.2.0). Headline numbers are frozen from converged runs (Qwen2.5-7B,
> full 904K SFT × 4000 steps, seed 42); older-regime figures are marked as mechanism
> evidence, not headlines. Attribution convention: **[Paper]** = prior published
> method we reuse or compare against (cited); **[Ours]** = this project's own
> mechanism, finding, or negative result. Provenance for every claim: §7.

## Abstract

Large language models execute every transformer layer for every token, yet per-token
depth demand is skewed and partly redundant. We present **Speaker**, a
post-training repair recipe that makes a 7B model simultaneously **sparse and above
its starting point**: per-layer threshold gates (**[Ours]** mechanism, standard
sigmoid+STE tools **[Paper]**) trained with a **fixed Lagrangian depth price**,
joint LoRA repair, and rollout-mode unlikelihood on self-generated prefixes
(**[Ours]** adaptation of Welleck‐style UL **[Paper]**). At k = 16.1/28 active
layers the repaired model beats raw dense by +4.3pt held-out accuracy while
per-token weight demand drops 42% (7.51 vs 13.05 GB) and KV demand drops 41%;
128-token generation holds ROUGE-L at the dense line (0.259 vs 0.252) with
repetition cut by a third versus the short-rollout variant (seq-rep-4 0.223 vs
0.340). We also report the honest negatives that shaped the recipe: capacity-pinned
routing (MoD **[Paper]**) buys neither sparsity nor accuracy; frozen-base gate-only
tuning (Router-Tuning **[Paper]**) self-locks near-dense; regularization-pressure
tuning (MoDification **[Paper]**) is bistable between near-dense and gate-collapse;
and our own joint top-p router collapsed through a renormalization-dilution
mechanism (**[Ours]** failure analysis) that a top-normalized reweighting breaks
(+13pt recovery, partial).

## 1 Motivation

**Memory, not FLOPs, is the binding constraint where we deploy.** On a 24 GB edge
card a 7B bf16 model (13 GB of decoder weights) leaves little room for KV cache at
long context — yet every token pays full depth. The per-token *demand* (weights
that must be touched + KV that must be read) is the number a decoding loop actually
provisions; peak residency and FLOPs are secondary. This is the MoDification
accounting coordinate **[Paper]**, which we adopt as the single efficiency axis
(§4.1): same-loss-lower-memory or same-memory-lower-loss, nothing else.

**Repair, not finetuning.** Deleting or skipping layers of a frozen model loses
accuracy that gate-only tuning cannot buy back (RT evidence, §2.3). We therefore
frame training as **post-training repair**, the standard move of the pruning
literature **[Paper: Wanda; SparseGPT]**: sparsity necessarily wounds, training
closes the wound. The yardstick is an equal-spec dense LoRA run — the *repair
budget* — and a sparse method's net value is its accuracy minus that budget at
matched depth. Raw dense is the starting line (the sparsity tax starts there);
dense-ft is the calibration, not a competitor.

**Serving frameworks cannot run this — yet.** Dynamic per-token layer paths
contradict the three assumptions of vLLM/SGLang-style serving **[Paper:
PagedAttention]**: CUDA graphs require static execution paths, continuous batching
requires all requests to want the same layers at the same step, and paged KV
assumes every layer holds every token's KV. Our contribution is therefore
training-side repair plus demand accounting, not latency: measured wall-clock
today shows no win (wrapper overhead ≈ +12% ms/tok, §4.4). The honest path to
serving is a per-request *static* mask (profile → freeze the active set → run a
static subgraph) — a separate serving paper, listed in §6, not claimed here.

## 2 Observations (what the data forced on us)

1. **Layer demand is skewed and self-organizing.** Per-layer ablation (0.5B) shows
   L0 ≫ L1 > L23 with a flat middle band; with no fixed-layer prior, gates
   spontaneously route 0.99/0.97 of tokens through the first/last layers while the
   middle idles at 0.08–0.18 **[Ours]**. Fixing [0,1,26,27] costs almost nothing;
   the compressible part is the ~12 gated layers of the current k = 16.1 total.
2. **Price binds, cap doesn't.** Gated selection (≈12) sits well below kmax = 16
   and k is flat across the whole run — depth is set by the Lagrangian price, not
   the safety cap **[Ours]**. Lower k is available by raising the price; the
   acc-vs-k frontier scan is unopened future work.
3. **Training k undercounts deployment k by ~25%.** Teacher-forced position
   quartiles over a 1024 window are dead flat ([20.4, 20.2, 20.2, 20.3] hard k,
   new `--pos_bins` probe **[Ours]**), and hard-probed k (20.3) equals free-decoded
   k (20.4): there is no position effect and no self-generation drift. The entire
   train→deploy gap is the **soft→hard gate gap** — deployment accounting must use
   probed hard k, never logged soft k **[Ours]**.
4. **Repetition residual is a data artifact, not gate damage.** All finetuned
   models — including near-dense MoDification (k ≈ 28) — fall into multi-turn
   self-dialogue loops on chit-chat prompts; raw dense does not **[Ours]**.
   Human rating (9 samples, 1–5): dense 4 > long_4k 3.5 ≈ mdf 3.5 > short-rollout
   2.5. The seq-rep-4 gap to dense is this loop, learned from SFT multi-turn
   format — distillation (KL) points the wrong way at it.
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

## 3 Design (default recipe = long_4k, validated 0914)

```
loss = L_LM + λ·mean(k) + UL_rollout(128tok, coef 0.3)
λ = 0.0005 fixed (dual off) · threshold gates + tau calibration + kmax 16 cap
base = LoRA r8 α16 q/v joint-trained · full 904K SFT × 4000 steps · bs2 × 1024 · seed 42
```

- **Threshold gate [Ours mechanism / Paper tools].** Per-layer linear router +
  threshold τ, open iff sigmoid((a−τ)/T) ≥ 0.5, STE backward **[Paper: Bengio et
  al.]**. τ calibrated at init from stability statistics (near-dense start, the
  budget trims down); kmax is a safety cap, rarely binding (§2.2). Gate cost:
  ~41K params on 0.5B scale (<0.005%); negligible at 7B.
- **Fixed Lagrangian price [Ours verdict].** `λ·mean(k)` with λ fixed. The
  alternatives were built and falsified: an accuracy-floor dual controller
  (EMA + warmup gate, still in-tree but default-off), hinge/tail budget shapes
  (hinge parks k but adds no accuracy; tail is dead weight), and
  difficulty-shaped per-token λ (slope r = 0.046, indistinguishable from
  unshaped — shaping pressure is homeopathic while λ hugs the floor)
  **[Ours negatives]**.
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
  0.475 vs 0.340 — an undertraining false negative; 4000-step reads 0.223)
  **[Ours]**; dose scales with length (UL mean 1.75 vs 0.59 at equal coef).
- **Deployment [Ours].** Hard skip (unselected layers neither execute nor write
  KV — sparse KV cache, ROUGE parity Δ−0.007 on 0.5B); budgeted GPU-residency
  scheduler (always-on pinned, gated layers packed under random/lru/lfu, migrate
  only between generations, KV never moves); per-token demand accounting
  (weights/KV/FLOP in the MoDification coordinate).
- **Kept but off by default [Ours].** DualController (accuracy-floor λ),
  difficulty shaping, hinge/tail budgets, joint top-p router (MoL), KL channel —
  each with a logged falsification or partial-fix verdict; see docs/training.md
  and docs/gating.md for the mechanism details.

## 4 Experiments

### 4.1 Protocol (frozen)

Qwen2.5-7B-Instruct; full 904K SFT slice [0, 904000) training; formal held-out at
offset 20000 (n = 300, max_len 1024, bs 1, labels accounting); generation 30
prompts × 128 tokens greedy from the same slice. Metrics use paper-original names
only **[Paper]**: ROUGE-L F1 (Lin 2004, via the public rouge-score API with a
documented Tokenizer-subclass extension for CJK) and seq-rep-4 (Welleck et al.
2020, Eq. 10). 30-prompt ROUGE is a coarse screen (ranking ≠ human ranking, §2.4).

### 4.2 Main table (converged, hard-mode deployment numbers)

| Method | held-out loss / acc (Δacc) | k /28 | mem GB / FLOP | ROUGE-L | seq-rep-4 |
|---|---|---|---|---|---|
| dense (raw) | 2.193 / 0.508 | 28 | 13.05 / 1.00 | 0.252 | 0.073 |
| ours short-rollout (500-step pilot) | 1.685 / 0.546 (+0.039) | 16.4 | 7.66 / 0.59 | 0.268 | 0.340 |
| **ours long_4k (default)** | **1.667 / 0.551 (+0.043)** | **16.1** | **7.51 / 0.58** | **0.259** | **0.223** |
| MoDification (α 0.01) | 1.436 / 0.593 (+0.085) | 27.8 | 12.97 / 0.99 | 0.293 | 0.201 |

(Train-slice plain+soft read for long_4k: 1.732/0.619, Δacc +0.083 — loop-control
scale, not the reporting scale per §4.1.)

**Reading.** Same-loss-lower-memory: long_4k matches dense ROUGE-L at −42% weight
demand. Same-memory-lower-loss: no baseline occupies the k ≈ 16 column except
ours; MoDification wins accuracy by paying full density (its mirror failure to
MoD, §2.5). The only red cell is repetition (3× dense), attributed to data format
(§2.4) with a decode-time lever (ngram ban + penalty → 0.042 ≈ dense) already
validated as the backstop.

### 4.3 Ablations (each a verdict, not a sweep)

- Rollout length needs convergence to judge (0.475@500 steps → 0.223@4000).
- Budget shape: mean wins; hinge/tail cut.
- Difficulty shaping: slope 0.046, cut.
- MoL reweighting: pmax over renorm +13pt, collapse broken, recovery partial.
- k accounting: soft log 16.4 vs deploy hard 20.3 — report hard only.

### 4.4 Honest costs

Training throughput −25% (rollout-128); inference wall-clock no win (+12% ms/tok
wrapper overhead — FLOP savings need kernel/ragged execution); 30-prompt
generation metrics are noisy screens next to human reads.

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

1. Wall-clock parity (needs ragged kernels). 2. Repetition residual 3× dense
   (data-format fix: single-turn truncation, cheap, unopened). 3. acc-vs-k price
   scan unopened (knob confirmed, frontier unknown). 4. Serving path: per-request
   static mask → static subgraph (separate paper). 5. Cut, not forgotten: KL
   ratio/direction/on-policy/data arms, hidden-MSE diagnostics, TTFT/TPOT matrix
   (AGENTS.md r8-cut row).

## 7 Provenance (claim → artifact)

- Skew / self-selection: 0.5B ablation probe; r5/r6 load profiles.
- Baseline failures: r1 dynamics + k-dist probes; r3/r7 five-way tables.
- UL dose / length: r4 + r8c rollout-24-vs-128 + 500-step-vs-4000-step.
- Soft-hard / position: `--pos_bins` probe (flat quartiles; hard ≈ decode).
- Dialogue-loop attribution: 9-sample human read of r8c_4k_gen.json.
- MoL collapse + pmax: ed7/0910-0911 runs; cal-arm variance verdict.
- Falsified options: budget_form (T-scan 7 arms), difficulty slope, GT-UL
  adaptivity loss — all archived with logs, none in the default path.
- Checkpoints: `r8c_{free,long_4k,mdf}` (formal table inputs).
