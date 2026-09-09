# Related work & positioning

Lineage: **MoD** (Raposo et al., arXiv 2404.02258) → **MoDification**
(Zhang et al., arXiv 2410.14268) → **Router-Tuning** (arXiv 2410.13184,
EMNLP'25). All three are reproduced in `baselines/` and compared on identical
slices via `baselines/eval_compare.py`.

## What each method does

### MoD — token-choice top-k per layer

Each (gated) layer contains a router; at every layer the top-k *tokens* by
router score execute that layer's attention+MLP, the rest pass through the
residual. k is fixed a priori per layer — chosen precisely so the compute graph
stays static (known tensor sizes, hardware-friendly). Trained from scratch with
BCE aux on the routing decision. Payoff: isoFLOP parity or ~50% cheaper forward
passes / faster sampling. Limits (also argued by MoDification): k constant per
layer saves the same amount at unimportant layers as at critical ones; top-k is
not cheap; converting an existing checkpoint is costly; every layer stays live
(all weights must remain resident).

### MoDification — threshold-p conversion of existing LLMs

Replaces top-k with a threshold-p operator (any number of retained tokens;
p≈0.5, interpreted as a two-expert MoE where one expert is NoOp), multiplies
the gate value into the attention module too, interleaves gated layers
(every other layer), and adds a layer-load-reducing objective
`R = α·Σ F_j·G_j` (expert-load-balancing inspired) to induce sparsity.
Conversion takes ~10B tokens of diverse data; reported ~1.2× latency speedup
and ~1.8× memory reduction in long-context serving at 3B–70B scales.

### Router-Tuning — gates on a frozen base

Trains only lightweight per-layer gates (linear + sigmoid + STE) on a fully
frozen base, with a budget `relu(cap − target)` term. EMNLP'25.

## Speaker's differences

| | MoD | MoDification | Router-Tuning | **Speaker** |
|---|---|---|---|---|
| routing unit | tokens per layer | tokens per layer | tokens per layer | **layers per token** |
| decision count | G decisions/token | G decisions/token | G decisions/token | **1 joint decision/token** |
| k semantics | constant, fixed a priori | free count, p pinned 0.5 | free, budget-pushed | **top-p, per-token adaptive, dual-trained** |
| always-on layers | none | none (interleaved) | none | **shared layers promoted by measured load** |
| base model | from scratch | ~10B-token conversion | frozen base | **LoRA joint finetune (router learns with the model)** |
| efficiency locus | training FLOPs | serving latency/memory | FLOPs | **edge VRAM: residency + sparse KV + scheduling** |

Three structural claims:

1. **One joint decision per token** (a single router at the gated-region entry)
   lets k adapt per token under a *budget dual* — the loss directly trades mean
   depth against an accuracy target. Per-layer independent gates diffuse this
   pressure (our MoDification reproduction: the R objective was drowned by the
   LM gradient and the model drifted to near-dense, k 23.9/24; MoD's capacity
   pinned k at the arithmetic value with zero adaptivity).
2. **Shared layers selected by measured load** — not by position and not
   interleaved — concentrate per-token demand: the fixed spine plus few hot
   gated layers cover most activations. This is what makes a *small resident
   set* meaningful on edge hardware; MoD-lineage methods keep every layer live.
3. **Sparsity with accuracy recovered**: KL self-distillation from the frozen
   dense teacher + the dual budget + unlikelihood anti-repetition let the
   sparse model match or exceed dense finetuning accuracy at a fraction of the
   active depth.

## Experimental evidence (same-slice, identical protocol)

| run | setup | dense | Speaker | MoD | MoDification | RT |
|---|---|---|---|---|---|---|
| r1 | Qwen1.5-0.5B, 500 steps | ref | **+10.4pt**, k 14.7/24 — only method above dense | −18.3pt, k 6.2 | — | −42.6pt, exec frozen 0.43 |
| r2 | 0.5B, MoDification added | ref | sparse, k 14.7/24 | — | +21.4pt but k 23.9/24 (≈ dense, no sparsity) | — |
| r3 | Qwen2.5-7B, LoRA, 500 steps | +21.6pt (dense-ft) | +2.0pt @ **k 7.4/28 (26%)** — only true-sparse | −3.2pt, k 15.7 | +19.9pt @ k 24.9/28 (≈ dense) | −0.1pt (frozen base stalls) |
| r5 | 7B full-data finetune (22K steps) | ref | **+7.7pt @ k 10.5/28 (37.5%)**, weight traffic −58% | — | — | — |

Read: MoD-lineage methods either stay sparse and lose accuracy (MoD, RT) or buy
accuracy by collapsing to dense (MoDification in our regime). Speaker is the
only family that stays genuinely sparse *and* above dense after full finetuning.

## Edge-side positioning

MoD optimizes the forward pass; MoDification optimizes serving throughput;
Speaker targets the **edge device**, where VRAM — not FLOPs — is the binding
constraint. On top of the same per-token depth sparsity we stack:

1. **Sparse KV cache** — decode memory scales with executed layers;
2. **Heterogeneous placement** — fixed + hot layers GPU-resident, cold layers
   on the CPU (peak memory 0.97 → 0.54 GB at unchanged quality);
3. **Residency scheduling** (`speaker/scheduler.py`) — the resident set is
   re-decided at runtime from the router's own activation counts
   (random/lru/lfu), turning "which layers sit in scarce memory" from a static
   partition into a schedulable, load-following decision.

The goal: reduce the memory a large model needs on-device, widen the space of
models a given device can schedule, and keep intelligence (accuracy at parity
or better) while doing so.
