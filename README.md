# Speaker: Fixed (Shared) Layers + Gated Layers for Edge-Side Efficient Inference

> A **speaker** finishes syntactic reasoning with **fewer layers** and reserves the rest
> for complex logical reasoning — the layers a token does *not* need are memory it does
> not have to hold.

Speaker turns per-token layer sparsity into edge-side gains: a smaller GPU memory
footprint (selective weights residency + sparse KV cache), a **schedulable** layer set
driven by the router's own activation statistics, and accuracy recovered above dense
via KL self-distillation and a dual budget.

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
      loss  = LM + λ·mean(k) + β·KL(sparse ‖ frozen dense)  (+ aux regularizers)
      dual  :  ema_acc < acc_target  ⇒  λ relaxes (buys layers back)
               ema_acc ≥ acc_target  ⇒  λ tightens (pushes sparsity)
```

Gating exists in two modes behind one switch (`gate_mode`): **moe** (default mainline,
the joint router above) and **threshold** (legacy per-layer gate, kept for old
checkpoints) — details in [docs/gating.md](docs/gating.md).

## 2. Positioning vs the MoD lineage

| | MoD (2404.02258) | MoDification (2410.14268) | **Speaker** |
|---|---|---|---|
| decision | per layer: top-k **tokens**, k fixed a priori | per layer: threshold-p **tokens** (p≈0.5) | per token: ONE joint router over **layers** (top-p, k adapts) |
| always-on layers | none | none (interleaved gating) | **shared layers selected by measured load** |
| conversion | from-scratch training | ~10B-token finetune | gating-first finetune (LoRA + KL + dual) → profile → promote → resume |
| efficiency target | training FLOPs / sampling speed | long-context serving latency/memory | **edge-side VRAM: residency + sparse KV + scheduling** |
| accuracy machinery | weighted residual + BCE | gate scaling + load objective R | distillation + budget dual + unlikelihood |

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
    wrapper.py        #   SpeakerLayerWrapper (gated residual + sparse KV + cross-device)
                      #   + SpeakerModelWrapper + convert_to_speaker
    scheduler.py      #   GPU-residency scheduling: random/lru/lfu over a weights budget
    load_profile.py   #   layer-load profiling: load -> fixed-layer promotion / GPU-CPU plan
    metrics.py        #   per-token NLL/acc / activation-memory estimate / KL distillation
    evaluate.py       #   eval_heldout + EMA (collate injected, shared with baselines)
    checkpoint.py     #   ckpt I/O: gate-key filtering / prefix stripping / clean base
    train_common.py   #   shared training boilerplate (device/tokenizer/model/LoRA/groups)
    log.py            #   RunLogger: metrics.jsonl + layers.jsonl + one-line stdout
  docs/               # per-mechanism documentation: gating / training / inference / related work
  data/               # data pipeline (shared): SFTDataset / chat template / collate
  pretrain/           # from-scratch training (design-first)
  finetune/           # finetune track: gating-first train / profile+promote / resume / eval
  baselines/          # MoD + MoDification + Router-Tuning + dense-ft + same-slice comparison
  tests/              # offline unit suites + on-device smokes
  tools/              # experiment tooling: eval_gen / probe_layers / probe_kdist / plotting
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

cfg = SpeakerConfig.from_model_config(base.config, kmax=6, sparsity_price=0.03, acc_target=0.55)
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
python3 finetune/train.py --model_id ./models/Qwen2.5-7B-Instruct --use_lora \
    --use_chat_template --always_head 0 --always_tail 0 --ul_mode rollout --ul_coef 0.3 \
    --eval_every 2000 --patience 10 --device cuda:0 --anneal_steps 2000 \
    --max_samples 20000 --save_dir ./ckpt/run
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

Architecture status: dual-mode gating behind `gate_mode` — **moe** mainline (joint
router + top-p + weighted residual; differentiable budget via soft inclusion) and
**threshold** legacy (per-layer router + τ, kept for old checkpoints). Gating overhead
~20K params on 0.5B is negligible.

- **P1 from-scratch scale-up**: smoke passed; pending scale-up validation of the
  "syntax in fixed layers / reasoning in middle layers" emergence.
- **P2 profile-promote-resume loop**: works (fixed layers 100%, mid layers 20-86% stay
  gated); pending end-to-end resume accuracy validation.
- **P3 edge-side deployment**: placement + residency scheduling work (mixed-device
  generation smoke passed); pending decode latency/memory benchmarks on real edge
  hardware, and kernel/batched-ragged execution to convert FLOP savings (~2.4× at
  k≈12/28) into wall-clock.
- **T1 (deferred)**: difficulty head decides k, router decides which; or decide masks
  at prefill and reuse at decode (composable with sparse KV).
- **T2 (threshold mode only)**: make τ actually learn (own lr group).
- **T3 one-shot affinity routing**: ✅ done — landed as `gate_mode="moe"` mainline.
- **T4 dense self-distillation**: ✅ done (`--kl_coef`, biggest accuracy lever).
- **T5 SOTA baseline comparison**: ✅ done (`baselines/eval_compare.py`, one command).
