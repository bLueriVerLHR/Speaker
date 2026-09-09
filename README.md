# Speaker: Fixed (Shared) Layers + Gated Layers for Efficient Inference

> Core idea: when a **speaker** talks, it completes syntactic-structure reasoning with **fewer layers**
> and reserves **more layers** for complex logical reasoning.

## 1. Core Idea and Design Goals

| Concept | Definition |
|---|---|
| **Fixed (shared) layers** | Layers every token must pass through. In theory these are the layers with **load >= 90%~95%**; they store grammar / foundational representations |
| **Gated layers** | Middle layers where an **independent per-layer Router** decides whether the token executes the layer fully; they store reasoning ability. Easy tokens pass fewer layers, hard tokens recruit more |

1. **Division of labor**: syntax lives in the fixed layers (first k + last k), reasoning lives in the
   middle gated layers; during fluent narration the model finishes syntactic reasoning quickly, and
   the saved layers serve actual knowledge reasoning.
2. **Optimization target**: per-token active-layer count `k` with a **low mean and a large std**
   (the pure Lagrangian `loss = LM + λ·mean(k)` realizes this naturally: λ pushes the mean down,
   the LM gradient "buys back" layers for hard tokens).
3. **Determining the fixed layers**:
   - **From-scratch training**: structural prior — fix the first/last k layers directly (`pretrain/`);
   - **Finetuning**: the distribution differs, so first train the gating, then **promote fixed
     layers by measured load**, keeping the rest gated (`finetune/`).
4. **Inference targets (decode-focused)**:
   - Enough memory: hard gating skips layers + sparse KV cache, fewer active layers → lower latency;
   - Tight memory: fixed + high-load layers resident on GPU, low-load layers on CPU → lower latency
     and memory;
   - Prefill degrades to dense exactly like MoE (execute everything, fill the cache) — only slower,
     never wrong.

## 2. Repository Layout

```
Speaker/
  speaker/            # core library: fixed+gated layer model (shared by both tracks)
    config.py         #   SpeakerConfig: fixed layers / gating / budget & dual hyperparams
    gating.py         #   Router + GatingOutput (linear gating)
    wrapper.py        #   SpeakerLayerWrapper (gated residual + sparse KV cache + cross-device)
                      #   + SpeakerModelWrapper + convert_to_speaker
    metrics.py        #   per-token NLL/acc / activation-memory estimate / KL distillation
    evaluate.py       #   eval_heldout + EMA (collate injected, decoupled from data)
    log.py            #   RunLogger: metrics.jsonl (per-interval values) + layers.jsonl (per-layer load)
                      #   + one-line stdout status
    load_profile.py   #   layer-load profiling: load -> fixed-layer promotion (>90/95%) / greedy GPU-CPU plan
    checkpoint.py     #   ckpt I/O: gate-key filtering / wrapper-prefix stripping / clean-base saving
    train_common.py   #   shared training boilerplate (device/tokenizer/model/slice/LoRA/param groups)
  data/               # data pipeline (shared, keeps everyone on the same protocol)
    sft.py            #   SFTDataset / chat template / assistant-only masking / collate
  pretrain/           # from-scratch training (design-first): syntax in fixed layers, reasoning in the middle
    train.py          #   random init + fixed head/tail shared layers + gated middle + budget training;
                      #   self-contained ckpt
  finetune/           # finetune track (fast validation on existing models)
    train.py          #   gating training: LoRA joint + KL self-distillation + budget dual
    eval_ckpt.py      #   standalone ckpt final eval (dense vs ckpt; must wrap LoRA)
    profile_layers.py #   profiling: load -> promote fixed layers -> new ckpt + suggested command
  baselines/          # baselines: Dense + MoD family
    lib.py            #   shared baseline mechanism (layer patch / stats / eval adapters)
    train_mod.py      #   MoD original reproduction (token-choice top-k capacity + weighted residual + BCE)
    train_mdf.py      #   MoDification reproduction (threshold-p + shared gate + R target)
    train_rt.py       #   Router-Tuning (EMNLP'25) baseline
    train_dense.py    #   dense-ft (same-spec LoRA pure SFT, fair control)
    eval_compare.py   #   same-slice comparison (dense / ours / MoD / RT / MoDification)
  tests/              # unit tests only (offline suites + on-device smoke)
  tools/              # experiment tooling: eval_gen (generation comparison) / probe_layers (per-layer
                      #   ablation) / probe_token_k (k probe) / probe_kdist (k distribution) / plot_converge
```

Legacy aliases: `from speaker import MoDConfig, convert_to_mod, ...` still work;
old checkpoints (`mod_config.json` + `gate.pt`) load as-is (field/param names `router/tau/comp`
unchanged).

## 3. Model Menu (`models/`)

| Path | Layers | Hidden | Size | Note |
|---|---|---|---|---|
| `Qwen1.5-0.5B` | 24 | 1024 | 1.2G single | ⭐ prototype, full-param `bf16` fits in `24GB` |
| `Llama-2-7b-hf` | 32 | 4096 | 12G | paper-scale comparison |
| `Qwen2.5-7B-Instruct` | 28 | 3584 | 15G | scale validation, needs LoRA |
| `gpt-oss-20b` / `Qwen3-30B` MoE | - | - | 71G | ❌ |

## 4. Core Logic

```
                        token t
                           │
                           ▼
        ┌──────────────────────────────────────┐
        │  fixed head   L0 … Lk-1              │
        │  every token executes                │
        │  syntax / representation foundation  │
        └──────────────────┬───────────────────┘
                           │
                           ▼
        ┌─────────────────────────────────────────────────┐
        │  gated middle   Lk … LN-1-k'   (G layers)       │
        │                                                 │
        │      h_l ──►  Router(H→1)                       │
        │                    │                            │
        │                    ▼                            │
        │      (logit − τ_l)/Ta  ──►  σ  (+Gumbel, train) │
        │                    │                            │
        │                    ▼                            │
        │      m ∈ {0, 1}   (STE: hard fwd, soft bwd)     │
        │                    │                            │
        │      ┌─────────────┴─────────────┐              │
        │      │ m = 1                     │ m = 0        │
        │      ▼                           ▼              │
        │  h + m·(F_l(h) − h)      h + (1−m)·c_l          │
        │  full execution          skip + compensation    │
        │                           (DASH-style, 0-init)  │
        └──────────────────┬──────────────────────────────┘
                           │
                           ▼
        ┌──────────────────────────────────────┐
        │  fixed tail   LN-k' … LN-1           │
        │  every token executes                │
        │  output: speaking / prediction       │
        └──────────────────┬───────────────────┘
                           ▼
                        logits

     k(t)      = fixed + Σ_l m_l(t)          target: mean low, std high
     budget    = LM + λ·mean(k) + over-kmax penalty + β·KL(sparse ‖ frozen dense)
     dual      = ema_acc < acc_target ⇒ λ relaxes, else tightens
     regular.  = cos_reg·mean(soft·sim)   (near-unchanged reps should skip)
     fixed set = pretrain: first-k/last-k prior
                finetune: load ≥ 90/95% promotion (profile → promote → resume)
     inference = hard mode skips all-zero layers; sparse KV at decode
                 (skipped layers write no K/V); prefill stays dense
```

### 4.1 Exploration direction: one-shot affinity routing (analogy, untested)

```
                        token t
                           │
                           ▼
   ┌───────────────────────────────────────────────────┐
   │  ONE affinity pass, up front            (trained) │
   │                                                   │
   │       h(t)  ·  φ_L        for every layer L       │
   │   (token state)  (learned layer features)         │
   │                    │                              │
   │                    ▼                              │
   │       a_L(t)  for all L                           │
   │                    │                              │
   │                    ▼                              │
   │       select S(t):  top-k  ⇄  top-p (cum log p)   │
   │                       switchable                  │
   └───────────────────────┬───────────────────────────┘
                           │  token walks the stack;
                           │  only layers in S(t) compute
                           ▼
      L0     ████████  executes   ← always in S(t)  *
      L1     ████████  executes   ← always in S(t)  *
      L2     ████████  executes   ← in S(t)
      L3     ░░░░░░░░  pass       ← zero compute
      L4     ░░░░░░░░  pass
      L5     ████████  executes   ← in S(t)
       ⋮         ⋮        ⋮
      LN-2   ████████  executes   ← always in S(t)  *
      LN-1   ████████  executes   ← always in S(t)  *
                           │
                           ▼
                        logits

     * fixed-layer policy unchanged:
         pretrain  = first-k / last-k structural prior
         finetune  = promoted by measured load (profile → promote → resume)

     trained   = affinity scores + layer features φ_L
     selector  = top-k ⇄ top-p (cumulative log p), switchable
     k(t)      = |S(t)|                (fixed layers are simply always in S(t))
     contrast  = per-layer gating: G local decisions per token
                 one-shot routing:  ONE global decision, made before the stack
```

One global decision per token instead of G local ones: the token–layer affinity scores
and the per-layer features φ_L are both trained; selection is top-k / top-p
(cumulative log p) switchable; the fixed-layer policy is unchanged (pretrain:
first-k/last-k structural prior; finetune: promoted by measured load). This is a
MoE-inspired analogy and a research direction, not the current implementation —
see T3 in the Roadmap.

## 5. Quick Start

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from speaker import SpeakerConfig, convert_to_speaker

model_id = "./models/Qwen1.5-0.5B"
base = AutoModelForCausalLM.from_pretrained(model_id, dtype="auto", device_map="auto",
                                            trust_remote_code=True)
tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

cfg = SpeakerConfig.from_model_config(base.config, kmax=6, sparsity_price=0.03, acc_target=0.55)
model = convert_to_speaker(base, cfg)  # wraps the layer list in place

out = model(input_ids=input_ids, attention_mask=mask, labels=labels)
loss = out.loss + (model.get_aux_loss() or 0) + (model.get_budget_loss(mask) or 0)
loss.backward()

model.get_layer_sparsity()     # {0: 0.0, 1: 0.0, 2: 0.8, ...} (fixed layers are always 0)
model.get_router_parameters()  # router + tau + comp, for a separate lr group
```

## 6. Track A: From-scratch Training (pretrain, design-first)

> No large-scale training budget yet, so **start from a smaller parameter size**
> (architecture overridable, tokenizer/vocabulary inherited):

```bash
python3 -u pretrain/train.py --device cuda:0 --steps 500 --max_samples 1000 \
    --num_layers 12 --hidden_size 512 --num_heads 8 --intermediate_size 1408 \
    --shared_head 2 --shared_tail 2 --save_dir /tmp/speaker_pretrain
# ~40M params: head/tail fixed layers hold syntax, middle gated layers hold reasoning;
# ckpt is self-contained (clean base + tokenizer + mod_config.json + gate.pt).
# Reuse: from_pretrained -> convert_to_speaker -> load_state_dict(gate.pt, strict=False)
```

## 7. Track B: Finetune Validation (finetune, fast testing)

```bash
# B1 train gating (gating-first: no prior fixed layers; LoRA joint + budget dual
#     + rollout unlikelihood against repetition)
python3 finetune/train.py --model_id ./models/Qwen2.5-7B-Instruct \
    --use_lora --use_chat_template --always_head 0 --always_tail 0 \
    --ul_mode rollout --ul_coef 0.3 --eval_every 2000 --patience 10 \
    --device cuda:0 --steps 2000 --max_samples 20000 --save_dir /tmp/mod_ckpt_run
# B2 load profiling -> promote fixed layers (>=90%) -> new ckpt + suggested command
python3 finetune/profile_layers.py --ckpt /tmp/mod_ckpt_run --model_id ./models/Qwen2.5-7B-Instruct \
    --n 32 --threshold 0.9 --out /tmp/mod_ckpt_run_fixed --gpu_budget_gb 4
# B3 resume training (structure = fixed layers + remaining gated layers)
python3 finetune/train.py --resume_dir /tmp/mod_ckpt_run_fixed ...
# standalone final eval
python3 finetune/eval_ckpt.py --ckpt /tmp/mod_ckpt_run --offset 3000 --n 100
```

## 8. Inference

```python
# Enough memory: skip layers for speed (sparse KV at decode, prefill stays dense)
model.eval()
model.set_skip_mode("hard")
model.generate(**tok("hello", return_tensors="pt").to("cuda:0"), max_new_tokens=64)

# Tight memory: profile -> fixed + high-load layers on GPU, low-load on CPU
# (layer inputs/outputs are moved across the device boundary automatically)
from speaker.load_profile import profile_layer_load
load = profile_layer_load(model, batches)   # batches: list of collated dicts
model.set_placement(resident_ids, gpu_device="cuda", cpu_device="cpu")
model.resident_gb("cuda")                   # resident param bytes on the GPU
```

## 9. Baselines (baselines/)

**Primary baseline = MoD and its follow-up SOTA (we aim to beat them); secondary = dense
(the upper reference with controlled accuracy loss under finetuning).**

```bash
# MoD original (token-choice top-k capacity + weighted residual + BCE aux, paper-faithful)
python3 baselines/train_mod.py --device cuda:0 --capacity 0.1 \
    --steps 500 --save_dir /tmp/modd_q05
# RT (MoD follow-up SOTA, EMNLP'25)
python3 baselines/train_rt.py --device cuda:0 --rt_target 0.1 --save_dir /tmp/rt_q05
# same-slice comparison: dense vs ours vs MoD vs RT
python3 baselines/eval_compare.py --ours /tmp/ours_q05 --modd /tmp/modd_q05 --rt /tmp/rt_q05 \
    --model_id ./models/Qwen1.5-0.5B --no-use_lora --offset 1000 --n 100
```

Lineage: MoD (arXiv 2404.02258) -> MoDification (arXiv 2410.14268, threshold-p adaptation for
existing LLMs, same thresholding lineage as ours) -> Router-Tuning (EMNLP'25).
Paper PDFs live in `baselines/papers/` (gitignored).

## 10. Roadmap

Architecture status: the gated residual is binary (`m=0` pass-through / `m=1` full execution),
training uses `soft`/STE and inference uses `hard`; the decision maker is a per-layer linear
Router(H→1) + scalar threshold τ; total gating overhead 21K/464M (0.5B) is negligible.

- **P1 from-scratch scale-up**: `pretrain/train.py` works (0.5B smoke passed); pending scale-up
  validation of whether the "syntax in fixed layers / reasoning in middle layers" division
  emerges (check load_profile layering + probe_token_k k_gap).
- **P2 finetune profile-promote-resume loop**: `finetune/profile_layers.py` works (measured on the
  kl2 ckpt: fixed layers 100%, L2/L21/L26 >=92% promoted, middle layers 20-86% stay gated);
  pending end-to-end resume validation of accuracy.
- **P3 heterogeneous placement measurement**: `load_profile` plan + `set_placement` application
  work; pending decode latency/memory measurement (historical offload run: 8 resident GPU layers,
  peak memory 0.97→0.54GB, ROUGE unchanged).
- **T1 (deferred) fast/slow dual path without a complexity head**: a lightweight difficulty head
  only decides k (budget count), Router only decides which; or decide the mask at prefill and
  reuse it at decode (composable with the sparse KV cache).
- **T2 make τ actually learn**: check whether tau `[min,max]` spreads; if not, give τ its own
  larger-lr param group.
- **T3 one-shot affinity routing (MoE-inspired analogy, untested)**: keep the Speaker design but
  drop the per-layer Router entirely — score token–layer affinity against **learned per-layer
  features** for all layers once, up front; select with **top-k / top-p (switchable)**; the token
  then computes only in the selected layers. Fixed-layer policy unchanged (pretrain first-k/last-k
  prior; finetune measured-load promotion). Figure: §4.1.
- **T4 dense self-distillation**: ✅ done (`--kl_coef`, the biggest accuracy lever;
  kl2 keeper Δacc +1.4pt).
- **T5 SOTA baseline comparison**: ✅ done (RT lost both runs: frozen-base gate-only training
  makes the pressure/accuracy knobs oppose each other in this regime; one-command comparison via
  `baselines/eval_compare.py`).
