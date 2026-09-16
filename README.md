# Speaker: Fixed (Shared) Layers + Gated Layers for Edge-Side Efficient Inference

> A **speaker** finishes syntactic reasoning with **fewer layers** and reserves the rest
> for complex logical reasoning — the layers a token does *not* need is memory it does
> not have to hold.

Two gating schemes, one switch (`--gate_mode`), one shared fixed/gated layer design:

| Scheme | `--gate_mode` | Idea |
|---|---|---|
| **Speaker** | `speaker` (canonical `threshold`) | per-layer gate — each layer decides whether this token needs to pass through it |
| **MoL** (Mixture of Layers) | `mol` (canonical `moe`) | one joint router at the gated-region entry picks the layer subset, MoE-style |

Both target **edge-side memory demand**: a smaller GPU footprint (selective weights
residency + sparse KV cache; the validated 7B recipe runs k = 20.0/28 total layers
for −28% weight demand and −28% KV demand), a **schedulable** layer set driven by the
router's own activation statistics, and accuracy recovered via a unified
**post-training repair** pipeline (joint LoRA repair + a fixed depth price + rollout
unlikelihood against repetition; KL self-distillation and the accuracy-floor dual stay
in-tree as optional arms, **off in the validated default**). Measured under a
memory-constrained budget (single GPU capped at 6–8 GB VRAM + 16 CPU cores, CPU
offload, no retraining): **1.34×/1.76×** wall-clock over an equal-budget dense-LoRA
baseline, while a masking baseline (MoDification) that executes every layer gains
nothing — only hard layer skipping accelerates in this regime
([docs/paper.md §4.8](docs/paper.md)). The roadmap: repair on
traditional self-attention first, then adapt to linear-attention backbones.

## 1. Core Idea

| Concept | Definition |
|---|---|
| **Fixed (shared) layers** | Layers **every** token executes. Chosen by **measured load** (execution rate >= 90~95%), *not* by position — head/tail layers typically self-promote, but a hot middle layer joins the same way. They carry grammar / foundational representations |
| **Gated layers** | The remaining layers. **One joint router** at the gated-region entry scores all of them per token; each token recruits the subset it needs (easy tokens few, hard tokens many). They carry reasoning |

```
                         token t
                            │
                            ▼
      L0    ████████████████   execute   fixed
      L1    ████████████████   execute   fixed
      L2    ▓▓▓▓▓▓ 0.31 ▓▓▓▓   execute   gated ──┐
      L3    ░░░░░░░░░░░░░░░░   pass      gated    │
      L4    ▓▓▓▓▓▓ 0.44 ▓▓▓▓   execute   gated    │
      L5    ░░░░░░░░░░░░░░░░   pass      gated    │
       ⋮           ⋮                              │   ONE joint router at the
      L11   ▓▓▓▓▓▓ 0.25 ▓▓▓▓   execute   gated    │   gated-region entry: a
      L12   ░░░░░░░░░░░░░░░░   pass      gated    │   single linear map scores
       ⋮           ⋮                              │   ALL gated layers p(L|h)
      L21   ░░░░░░░░░░░░░░░░   pass      gated    │   for this token; top-p
      L22   ████████████████   execute   fixed    │   picks the subset S(t) —
      L23   ████████████████   execute   fixed    │   k adapts per token — and
      L24   ▓▓▓▓▓▓ 0.39 ▓▓▓▓   execute   gated    │   kept probs renormalize
      L25   ░░░░░░░░░░░░░░░░   pass      gated ──┘   into weights w
      L26   ████████████████   execute   fixed
      L27   ████████████████   execute   fixed
                            │
                            ▼
                         logits

      gated execute :  h ← h + w·(F_l(h) − h)   weighted residual (w from the router)
      gated pass    :  h unchanged; at decode the layer writes no K/V
      fixed         :  every token executes — promoted by MEASURED LOAD, not position
                       (L22/L23 above are promoted middle layers)
      k(t)          :  #fixed + |S(t)|   per-token adaptive: low mean, large std
```

The next token gets a different subset — same weights, different depth.

Training turns two knobs on this picture:

```
      loss  = LM + λ·mean(k) + UL_rollout(128 tok, coef 0.3)   ← validated default (v0.2.0)
                λ = 0.0005 fixed · threshold gates + τ calibration · kmax 16 safety cap
      dual  :  ema_acc < acc_target  ⇒  λ relaxes (buys layers back)
               ema_acc ≥ acc_target  ⇒  λ tightens (pushes sparsity)
               optional arm: acc_target = none (default) holds λ fixed; 'auto' re-anchors
               the floor to measured dense acc − margin; KL self-distillation likewise
               opt-in via --kl_coef — both cut from the default after falsification runs
               (details: docs/training.md, docs/paper.md §3)
```

Gating exists in two modes behind one switch (`gate_mode`): **threshold** (default,
per-layer gate) and **moe** (joint router above) — details in [docs/gating.md](docs/gating.md).

## 2. Positioning vs the MoD lineage

| | MoD (2404.02258) | MoDification (2410.14268) | **Speaker** |
|---|---|---|---|
| decision | per layer: top-k **tokens**, k fixed a priori | per layer: threshold-p **tokens** (p≈0.5) | per token: per-layer **threshold gates** (k adapts, priced by λ; a joint top-p router exists as `gate_mode=moe`) |
| always-on layers | none | none (interleaved gating) | **shared layers selected by measured load** |
| conversion | from-scratch training | ~10B-token finetune | gating-first finetune (LoRA joint + fixed price + rollout UL) → profile → promote → resume |
| efficiency target | training FLOPs / sampling speed | long-context serving latency/memory | **edge-side VRAM: residency + sparse KV + scheduling** |
| accuracy machinery | weighted residual + BCE | gate scaling + load objective R | joint LoRA repair + fixed depth price + rollout unlikelihood |

MoD and MoDification make the *forward pass* cheaper; Speaker additionally makes the
*memory* smarter: with per-token demand concentrated on few layers, only fixed + hot
gated layers need to be GPU-resident, cold layers compute on the CPU, and the resident
set is re-decided at runtime from activation counts. Full comparison with experimental
evidence: [docs/related-work.md](docs/related-work.md).

## 3. Repository Layout

```
Speaker/
  speaker/            # core library: fixed+gated layer model (shared by both tracks)
    config.py         #   SpeakerConfig: fixed set / gate_mode / budget & dual hyperparams
    gating.py         #   JointRouter (moe) + Router (threshold) + GatingOutput
    layer.py          #   SpeakerLayerWrapper (gated residual + sparse KV + cross-device)
    hub.py            #   SpeakerModelWrapper (routing hub / stats / budget) + convert_to_speaker
    placement.py      #   hierarchical placement primitives (delegated from the hub)
    calibrate.py      #   init-time tau / router-temp calibration (delegated from the hub)
    wrapper.py        #   facade re-exporting layer.py + hub.py (old imports keep working)
    strategies.py     #   gating-strategy seam: ThresholdStrategy / MoeStrategy registry
    hparams.py        #   FinetuneConfig dataclass (typed training knobs, no argparse)
    terminal.py       #   rich console / tracebacks / tty-aware progress
    config_model.py   #   optional pydantic-v2 validation mirror of SpeakerConfig
    accelerate_backend.py  # opt-in Accelerate/Fabric wrapper (default none = legacy)
    scheduler.py      #   GPU-residency scheduling: random/lru/lfu over a weights budget
    load_profile.py   #   layer-load profiling: load -> fixed-layer promotion / GPU-CPU plan
    metrics.py        #   per-token NLL/acc / activation-memory estimate / KL distillation
    evaluate.py       #   eval_heldout + EMA (collate injected, shared with baselines)
    ruler.py          #   accuracy scale: valid masks + acc_target policy (auto = dense − margin)
    dual.py           #   DualController: accuracy EMA + warmup gate + λ adaptation
    ul.py             #   anti-repetition: n-gram unlikelihood terms + rep3 probes
    checkpoint.py     #   ckpt I/O: gate-key filtering / prefix stripping / clean base (+safetensors)
    converge.py       #   StopOnPlateau: shared early-stop rule across the five training scripts
    train_common.py   #   shared training boilerplate (device/tokenizer/model/LoRA/groups)
    log.py            #   primary logger (loguru): console + run.log + JSONL sinks, no tee
  docs/               # per-mechanism documentation: gating / training / inference / related work
  data/               # data pipeline (shared): SFTDataset (+streaming) / chat template / collate
  configs/            # sweep YAMLs (cartesian multirun specs for tools/sweep.py)
  pretrain/           # from-scratch training (design-first)
  finetune/           # finetune track: cli.py (Typer) / pipeline.py (loop) / train.py (entry)
                      #   + profile+promote / resume / eval
  baselines/          # MoD + MoDification + Router-Tuning + dense-ft + same-slice comparison
                      #   train_loop.py: one shared loop (BaselineRecipe per family)
                      #   assemble.py: ModelBuilder with decorator family registry
  tests/              # offline unit suites + on-device smokes (incl. ruler/dual + UL +
                      #   registry + tooling seams)
  tools/              # experiment tooling: eval_gen / probe_kdist / sweep / plotting + ci.sh
```

Legacy aliases: `from speaker import MoDConfig, convert_to_mod, ...` still work; old
checkpoints (`mod_config.json` + `gate.pt`) load as-is.

## 4. Model Menu (`models/`)

| Path | Layers | Hidden | Size | Note |
|---|---|---|---|---|
| `Qwen1.5-0.5B` | 24 | 1024 | 1.2G single | ⭐ prototype, full-param `bf16` fits in `24GB` |
| `Llama-2-7b-hf` | 32 | 4096 | 12G | paper-scale comparison |
| `Qwen2.5-7B-Instruct` | 28 | 3584 | 15G | scale validation, needs LoRA |
| `gpt-oss-20b` / `Qwen3-30B` MoE | - | - | 71G | ❌ |

## 5. Quick Start

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from speaker import SpeakerConfig, convert_to_speaker

model_id = "./models/Qwen1.5-0.5B"
base = AutoModelForCausalLM.from_pretrained(model_id, dtype="auto", device_map="auto",
                                            trust_remote_code=True)
tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

cfg = SpeakerConfig.from_model_config(base.config, gate_mode="threshold",
                                      kmax=16, sparsity_price=0.0005, acc_target=None)
model = convert_to_speaker(base, cfg)  # wraps the layer list in place

out = model(input_ids=input_ids, attention_mask=mask, labels=labels)
loss = out.loss + (model.get_aux_loss() or 0) + (model.get_budget_loss(mask) or 0)
loss.backward()

model.get_layer_sparsity()     # {0: 0.0, 1: 0.0, 2: 0.8, ...} (fixed layers are always 0)
model.get_router_parameters()  # router params, for a separate lr group
```

## 6. Training Tracks (details: [docs/training.md](docs/training.md))

```bash
# Track A — from scratch (design-first; head/tail structural prior, middle gated)
python3 pretrain/train.py --device cuda:0 --max_steps 500 --num_layers 12 --hidden_size 512 \
    --shared_head 2 --shared_tail 2 --save_dir ./ckpt/speaker_pretrain

# Track B — finetune (mainline): gating-first, no prior fixed layers
# CLI defaults = the validated v0.2.0 long_4k recipe (threshold, kmax 16, λ 0.0005,
# dual off, rollout-UL 128 tok × 0.3); flags shown explicitly for clarity
python3 finetune/train.py --model_id ./models/Qwen2.5-7B-Instruct --use_lora \
    --use_chat_template --always_head 0 --always_tail 0 \
    --gate_mode threshold --kmax 16 --sparsity_price 0.0005 --acc_target none \
    --ul_mode rollout --rollout_tokens 128 --ul_coef 0.3 \
    --eval_every 2000 --patience 10 --anneal_steps 2000 --max_steps 4000 \
    --device cuda:0 --save_dir ./ckpt/run
# converged headline (Qwen2.5-7B, full 904K SFT): held-out Δacc +4.3pt at k 20.0/28
# total layers (hard deployment accounting), weight demand −28% (9.38 vs 13.05 GB),
# KV −28%, 128-tok ROUGE-L 0.211 > raw dense 0.160
# then: profile measured load -> promote fixed layers (>=90%) -> resume on new structure
python3 finetune/profile_layers.py --ckpt ./ckpt/run --threshold 0.9 --out ./ckpt/run_fixed
python3 finetune/train.py --resume_dir ./ckpt/run_fixed ...
```

## 7. Inference: Three Modes (details: [docs/inference.md](docs/inference.md))

```python
# Mode 1 — enough memory: skip layers for speed (sparse KV at decode; prefill stays dense)
model.eval(); model.set_skip_mode("hard")
model.generate(**tok("hello", return_tensors="pt").to("cuda:0"), max_new_tokens=64)

# Mode 2 — tight memory: static placement by profiled load (fixed+hot on GPU, rest on CPU)
model.set_placement(resident_ids, gpu_device="cuda", cpu_device="cpu")

# Mode 3 — tight memory, scheduled: residency re-decided from activation statistics
sched = model.schedule_placement("lfu", gpu_total_gb=18, reserve_gb=4)
for req in requests:
    out = model.generate(**req)
    sched.reschedule()      # between generations only (KV caches are not migrated)
```

```
        GPU (budget = allowance − KV/activation reserve)     CPU (host memory)
       ┌────────────────────────────────────────┐          ┌──────────────────┐
       │ fixed layers      — forced resident    │          │ cold gated       │
       │ hot gated layers  — picked by strategy │ ◄─swap─► │ layers compute   │
       │ KV cache + activations                 │  between │ here             │
       └────────────────────────────────────────┘    gens  └──────────────────┘
         strategies: random | lru (recently-active first) | lfu (most-activated first)
```

Measured under a constrained budget (Qwen2.5-7B, single GPU capped via
`torch.cuda.set_per_process_memory_fraction`, 16 pinned CPU cores, bf16, zero
retraining; 20 prompts × 128 tok greedy; `tools/edge_bench.py --vram_cap_gb`):

| VRAM cap | ours (hard skip) | dense-ft (equal-budget LoRA) | MoDification α0.01 (masking) |
|---|---|---|---|
| 6 GB | **182 ms/tok** (5.5 tok/s), 4/28 layers resident | 247 ms/tok (4.0) | 251 ms/tok (4.0) |
| 8 GB | **124 ms/tok** (8.1 tok/s), 10/28 resident | 219 ms/tok (4.6) | 262 ms/tok (3.8) |

Skipping converts to wall-clock exactly where memory binds (**1.34×/1.76×**; measured
skips 8.2 layers/token match the probed hard k = 20/28); masking executes all 28
layers and buys none; enlarging the budget favors sparsity (6→8 GB: ours 1.49× vs
dense 1.13×) because the gates' skipped layers stay on the CPU unpaid. Full analysis
and caveats (LFU rescheduling null result, IO-module tax, variance bands):
[docs/paper.md §4.8](docs/paper.md).

## 8. Baselines (details: [docs/related-work.md](docs/related-work.md))

**Primary baseline = MoD and its follow-ups; secondary = dense (upper reference).**

```bash
python3 baselines/train_mod.py --device cuda:0 --capacity 0.1 --max_steps 500 --save_dir ./ckpt/modd
python3 baselines/train_mdf.py --device cuda:0 --save_dir ./ckpt/mdf
python3 baselines/train_rt.py  --device cuda:0 --rt_target 0.1 --save_dir ./ckpt/rt
python3 baselines/eval_compare.py --ours ./ckpt/ours --modd ./ckpt/modd --mdf ./ckpt/mdf \
    --rt ./ckpt/rt --model_id ./models/Qwen1.5-0.5B --offset 1000 --n 100
```

Paper PDFs live in `baselines/papers/` (gitignored). The RT baseline additionally needs
the paper's third-party `model_patch.py` under `baselines/router-tuning/utils/model/`
(torch-only single file); `baselines/lib.py` raises a clear error when it is missing.

## 9. Roadmap

Architecture status: dual-mode gating behind `gate_mode` — **threshold** mainline
(per-layer router + τ + STE + kmax cap; the v0.2.0 validated default recipe and the
finetune CLI default) and **moe** (MoL joint router + top-p + pmax-weighted residual;
kept in-tree, off the mainline after the renorm-collapse diagnosis — pmax reweighting
recovers +13pt, partial). Note: `SpeakerConfig`'s library-level default is still
`"moe"`; training entry points pin `threshold`. Gating overhead ~41K params on 0.5B
(threshold) / ~20.5K (moe) is negligible.

- **P1 from-scratch scale-up**: smoke passed; pending scale-up validation of the
  "syntax in fixed layers / reasoning in middle layers" emergence.
- **P2 profile-promote-resume loop**: works (fixed layers 100%, mid layers 20-86% stay
  gated); pending end-to-end resume accuracy validation.
- **P3 edge-side deployment**: ✅ constrained-budget benchmark done (6/8 GB VRAM cap +
  16 cores, CPU offload, `tools/edge_bench.py`): 1.34×/1.76× over equal-budget
  dense-ft, masking baselines no speedup, LFU rescheduling null (flat activation
  frequencies); between-generation scheduling kept as machinery. Remaining: kernel /
  batched-ragged execution to convert FLOP savings into wall-clock in the
  whole-card-resident regime, and long-context runs where the −28% KV term binds.
- **T1 (deferred)**: difficulty head decides k, router decides which; or decide masks
  at prefill and reuse at decode (composable with sparse KV).
- **T2 (threshold mode only)**: make τ actually learn (own lr group).
- **T3 one-shot affinity routing**: ✅ done — landed as `gate_mode="moe"`; off the
  validated mainline (renorm collapse diagnosed, pmax reweighting recovers +13pt, partial).
- **T4 dense self-distillation**: implemented (`--kl_coef`); cut from the default —
  kl = 0 already recovers past dense, the remaining gap is data format (docs/paper.md §2.4).
- **T5 SOTA baseline comparison**: ✅ done (`baselines/eval_compare.py`, one command).
