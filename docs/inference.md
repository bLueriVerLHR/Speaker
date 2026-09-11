# Inference: skipping, placement, scheduling

Three composable modes, from "enough memory" to "edge-side budgeted".

## Primary efficiency metric — per-token memory demand

Wall-clock at batch size 1 does not reflect the savings (wrapper overhead ~12%
eats the FLOP gap; r4/r7 both re-confirmed). The honest currency is **demand**:
what a token's execution actually requires,

- **weights**: `k × bytes-per-layer` (GB/token; 7B reference: dense 13.05 vs
  Speaker 7.11 at k=15.3/28 → −46%),
- **KV cache**: `k × bytes-per-layer-KV` at the serving context length
  (MB/token; 31.2 vs 57.3 at 1024 ctx → −46%).

`tools/mem_demand.py` aggregates this from `probe_kdist` JSONs. Process-level
`peak_gb/avg_gb` from sequential single-process evals is allocator-polluted and
stays diagnostic-only. Demand is also what the scheduler below budgets against.

## Mode 1 — hard skipping + sparse KV cache (speed)

```python
model.eval()
model.set_skip_mode("hard")
model.generate(**tok("hello", return_tensors="pt").to("cuda:0"), max_new_tokens=64)
```

- Training uses soft gates; hard mode rounds them: a gated layer whose mask is
  all-zero for the current position is **skipped entirely** (no forward, no
  kernel launch).
- **Sparse KV cache**: skipped layers write no K/V at decode — cache memory and
  bandwidth scale with *executed* layers, not model depth. (For a skipped
  position the attention of later layers simply does not see it from that layer,
  mirroring the skip that happened in training.)
- **Prefill stays dense**: the prompt is processed by everything, filling each
  layer's cache for the positions it executed; only decode steps skip. Wrong is
  never an option — at worst it is slower.
- Expected saving at k≈12/28: ~2.4× fewer layer-FLOPs per decoded token; turning
  that into wall-clock needs kernel/batched-ragged execution (current wrapper
  overhead eats most of it — measured wall-clock parity with dense on 7B).

Honest caveats measured so far: decode-time k drifts higher than training-time k
(7.4 → ~11.7 on self-generated prefixes — the gate opens more layers on its own
rollouts); free generation quality needs the rollout-UL training remedy.

## Mode 2 — static CPU/GPU placement (memory)

```python
from speaker.load_profile import profile_layer_load
load = profile_layer_load(model, batches)          # batches: list of collated dicts
model.set_placement(resident_ids, gpu_device="cuda", cpu_device="cpu")
model.resident_gb("cuda")                          # resident param bytes
```

Fixed + high-load layers resident on the GPU, the rest on the CPU. Layer
inputs/outputs (hidden states, masks, rotary embeddings) are moved across the
device boundary automatically, per layer, per step. Measured: 8 resident GPU
layers cut peak memory 0.97 → 0.54 GB with ROUGE unchanged.

## Mode 3 — scheduled residency (memory + adaptivity)

Static placement freezes one partition; scheduling re-decides it from **what the
router actually activates**:

```python
sched = model.schedule_placement("lfu", gpu_total_gb=18, reserve_gb=4)
for req in requests:
    out = model.generate(**req)
    sched.reschedule()      # between generations only
```

### Budget accounting

```
weights budget = gpu_total_gb − reserve          (reserve = KV cache + activations)
                  reserve_gb > 0, else reserve_fraction (default 0.25)
                  gpu_total_gb=None → auto-detect free GPU memory
```

- **Fixed layers are forced GPU-resident** — they run for every token; if they
  do not fit, the scheduler raises (a configuration error, not something to
  silently degrade on).
- Gated layers are packed into the leftover budget **greedily in strategy
  ranking order**: an oversized candidate is skipped, a smaller later one may
  still fill the gap.
- Everything not selected computes on the CPU (Mode-2 machinery).

### Strategies (`speaker/scheduler.py`)

| strategy | ranking | notes |
|---|---|---|
| `random` | seeded shuffle | baseline: "some K layers in the leftover space" |
| `lru` | most recently activated first | window granularity = one observe tick |
| `lfu` | cumulative activation count, top first | the primitive version: count tokens each layer executed, keep the hottest resident |

With no observed data yet, all strategies fall back to index order —
deterministic, no thrashing on a cold start.

### The reschedule contract

- Activation statistics are harvested from the wrapper's own accumulators
  (`pop_layer_counts`: per gated layer, the token counts it executed — a
  **read-and-reset**; whoever reads first wins the window).
- `reschedule()` = observe → plan → apply. **If the plan is unchanged, nothing
  migrates** (layer moves are expensive; hysteresis is free).
- Call it **between generations only**: per-layer KV caches are *not* migrated
  with weights. Mid-generation rescheduling is out of contract.
- Mixed-device moe routing: the joint route is computed once (entry layer's
  device); each gated layer realigns its slice to its own device
  (`_consume_route(dev)`) — routing works across the CPU/GPU boundary.

Measured smoke (0.5B, fp32, 0.28 GB weights budget): 5 GPU / 19 CPU layers;
LFU reschedule moved residency L2 → L12 (the layer the router actually used
most) and the next generation ran on the new placement unchanged.

### Why sparsity makes this work

MoD-style per-layer token routing keeps every layer "live" (any layer may fire
on any token), so all weights must stay resident. Per-token layer routing with
promoted shared layers concentrates demand: the fixed spine + a handful of hot
gated layers cover most activations (observed profiles: fixed 100%, mid-stack
as low as 0.16) — a small resident set suffices, and the *same* activation
statistics that drive training sparsity drive the runtime schedule.
