# Training (both tracks)

Two tracks share the core library and the RunLogger logging stack; they differ in
how the fixed-layer set is born.

- **Track A — pretrain (from scratch, design-first)**: fixed layers come from a
  structural prior (first/last k), because no trained model exists to profile.
- **Track B — finetune (mainline)**: **gating-first** — start with *no* prior fixed
  layers (`--always_head 0 --always_tail 0`), train the router, then **promote**
  fixed layers by measured load and resume. The fixed set is data-decided and may
  include middle layers (observed: L0/L27 self-promote at ~0.99/0.97; a mid-stack
  L2/L21/L26 also cleared 92% in one profile).

## The budget dual (price of depth)

The core training objective (moe mode; threshold differs only in k accounting):

```
loss = LM + λ · k_soft.mean() + β · KL(sparse ‖ frozen dense) + aux
```

λ is not a constant — it is a dual variable steering mean depth against an
accuracy target:

```
ema_acc < acc_target  ⇒  λ ← λ · relax      # buy layers back for hard tokens
ema_acc ≥ acc_target  ⇒  λ ← λ · tighten    # push sparsity while accuracy holds
```

`price_warmup` delays the price's start; `budget_ramp` ramps it in. The price
semantic is the **per-inference actual demand** (per-token average active-layer
memory, avg not peak) — the number a decoding loop actually has to provision.

Empirically this produces the target k-distribution: low mean, large std, unimodal
per-token (no winner-take-all bimodality), e.g. μ≈11.4±1.9 over 28 layers on a 7B
finetune.

## KL self-distillation

The frozen dense model is the teacher (`--kl_coef`, β): the sparse student matches
its logits while paying the depth price. This is the single biggest accuracy lever
(full-finetune run: held-out acc **above** dense, +7.7pt at k=10.5/28 layers).

## Unlikelihood vs repetition (optional)

Sparse decode has a repetition attractor; two optional remedies in
`finetune/train.py`:

- `--ul_mode gt`: n-gram unlikelihood on ground-truth repeats (rep3 −57%, but
  depth relaxes k 7.4→11.1 — sparsity pays for it);
- `--ul_mode rollout`: periodic hard rollouts + DAgger-style unlikelihood on
  self-generated repeats (rep3 −41% at unchanged k; costs ~20% training throughput).

## Track B loop

```
B1  gating-first finetune      (LoRA joint train + KL + dual + optional UL)
B2  profile measured load      finetune/profile_layers.py
      -> promote layers with load >= threshold (default 0.9) to fixed
      -> new ckpt (promoted keys dropped from gate.pt, router re-init on resume)
B3  resume on the new structure (fixed + remaining gated)
```

CLI semantics (note: `--anneal_steps` is the *annealing horizon*, NOT a stop
condition — stopping is `--max_steps` only, enforced via StopOnPlateau):

```bash
python3 finetune/train.py --model_id ./models/Qwen2.5-7B-Instruct --use_lora \
    --use_chat_template --always_head 0 --always_tail 0 \
    --ul_mode rollout --ul_coef 0.3 \
    --eval_every 2000 --patience 10 --anneal_steps 2000 --max_steps 3000 \
    --device cuda:0 --save_dir ./ckpt/run
python3 finetune/profile_layers.py --ckpt ./ckpt/run --n 32 --threshold 0.9 \
    --model_id ./models/Qwen2.5-7B-Instruct --out ./ckpt/run_fixed
python3 finetune/train.py --resume_dir ./ckpt/run_fixed ...
python3 finetune/eval_ckpt.py --ckpt ./ckpt/run --offset 3000 --n 100   # standalone eval
```

## Track A loop

Random-init small architecture (tokenizer inherited), structural prior for fixed
layers, budget training from step 0. Checkpoints are self-contained: clean base
(weights stripped of wrapper prefixes) + tokenizer + `mod_config.json` + `gate.pt`.

```bash
python3 pretrain/train.py --device cuda:0 --max_steps 500 --max_samples 1000 \
    --num_layers 12 --hidden_size 512 --num_heads 8 --intermediate_size 1408 \
    --shared_head 2 --shared_tail 2 --save_dir ./ckpt/speaker_pretrain
```

(`--max_steps` is a true stop in pretrain, as in all scripts.)

## Logging (RunLogger)

Every training run writes, into the checkpoint directory:

- `metrics.jsonl` — all numeric state per `log_interval` (lm/acc/k±std/sparsity/
  price/λ/memory/throughput …) — plot and analyze straight from this file;
- `layers.jsonl` — per-layer execution rates at plateau checkpoints (the profile
  input for promotion);
- stdout — one-line status per interval.

`converge.jsonl` (eval history) is written by the train scripts themselves.
