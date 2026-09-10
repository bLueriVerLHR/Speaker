"""Finetune route: validate the Speaker design (shared + gated layers) on an existing
model for quick effect testing.

- Dual scheme (gate_mode): moe = hierarchical MoE joint routing (default mainline; a
  single JointRouter at the entry emits log p, top-p/top-k selection, renormalized
  weighted residual over the selected layers) / threshold = legacy per-layer threshold
  gating; the two schemes' checkpoints are mutually incompatible -- run one line each
  on the same data/slice for a same-arena comparison;
- Pipeline: train the gating first (this script) -> finetune/profile_layers.py promotes
  layers with load >90/95% to shared (fixed) layers -> continue training with
  --always_layers / --resume_dir (the remaining layers stay gated);
- Pure Lagrangian: loss = LM + lambda*mean(k) (+threshold over-kmax penalty); accuracy
  is held by the acc dual adjustment; price basis = actual demand per inference
  (per-token average activation memory); peak/residency are ignored;
- Target shape: low mean and high variance of the per-token active layer count k
  (see the k X±Y column in the logs).
"""
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from speaker import SpeakerConfig, convert_to_speaker  # noqa: E402
from speaker.metrics import distill_kl_loss, estimate_act_mb, per_token_correct  # noqa: E402
from speaker.evaluate import ema_update, eval_heldout  # noqa: E402
from speaker.converge import StopOnPlateau  # noqa: E402 (ed3 unified convergence rule)
from speaker.log import RunLogger  # noqa: E402
from speaker.checkpoint import clean_base_state_dict, load_gate, save_gate  # noqa: E402
from speaker.train_common import (  # noqa: E402
    build_model,
    build_param_groups,
    build_tok,
    dtype_of,
    resolve_device,
    split_train_eval,
    wrap_lora,
)
from data.sft import make_collate  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_id", default="/home/hdd/model/Qwen2.5-7B-Instruct",
                   help="finetune route defaults to 7B; only the pretrain route uses small models")
    p.add_argument("--data_path", default="./data/sft_t2t_mini.jsonl")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16", "float16"])
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--anneal_steps", type=int, default=100,
                   help="annealing horizon (denominator of the Ta/gumbel annealing schedule, "
                        "part of our method's definition, keep fixed); stopping is governed "
                        "solely by --max_steps (ed5: the --steps legacy alias was removed after "
                        "two smoke incidents)")
    p.add_argument("--max_steps", type=int, default=3000,
                   help="hard step cap; actual stopping is decided by the ed3 unified convergence rule (StopOnPlateau)")
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--router_lr", type=float, default=1e-4)
    p.add_argument("--kmax", type=int, default=10)
    p.add_argument("--gate_mode", default="moe", choices=["moe", "threshold"],
                   help="moe=hierarchical MoE joint routing (default mainline) / threshold=legacy per-layer threshold gating; "
                        "on resume, the ckpt's mod_config.json takes precedence")
    p.add_argument("--select_mode", default="topp", choices=["topp", "topk"],
                   help="moe: log p selection rule, topp=stop once the cumulative probability reaches p (adaptive k) / topk=fixed k")
    p.add_argument("--top_p", type=float, default=0.9, help="moe topp mode cumulative-probability threshold")
    p.add_argument("--top_k", type=int, default=6, help="moe topk mode fixed k")
    p.add_argument("--min_layers", type=int, default=1, help="moe: minimum number of gated layers activated per token")
    p.add_argument("--always_head", type=int, default=2, help="first m layers fixed (shared)")
    p.add_argument("--always_tail", type=int, default=2, help="last n layers fixed (shared)")
    p.add_argument("--always_layers", default="",
                   help="explicit list of fixed layers (comma-separated), overrides head/tail; used after profile promotion, e.g. '0,1,2,22,23'")
    p.add_argument("--temp_affinity", type=float, default=1.0)
    p.add_argument("--ta_end", type=float, default=0.3)
    p.add_argument("--gumbel_scale", type=float, default=1.0)
    p.add_argument("--tau_init", type=float, default=0.0,
                   help="threshold: initial gating threshold; negative = dense start (all gates open first, then sparsify)")
    p.add_argument("--sparsity_price", type=float, default=0.03)
    p.add_argument("--price_adapt", default=True, action=argparse.BooleanOptionalAction)
    p.add_argument("--acc_target", type=float, default=0.55)
    p.add_argument("--price_warmup", type=int, default=100)
    p.add_argument("--budget_ramp", type=int, default=100,
                   help="budget-loss ramp steps: task*=min(1,step/ramp); let the gating learn utility before adding sparsity pressure")
    p.add_argument("--eval_samples", type=int, default=100,
                   help="number of same-distribution held-out eval samples (taken right after the training slice)")
    p.add_argument("--max_samples", type=int, default=500)
    p.add_argument("--log_interval", type=int, default=10)
    p.add_argument("--save_dir", default="/tmp/mod_ckpt")
    p.add_argument("--seed", type=int, default=None, help="random seed (unset by default, preserving legacy behavior)")
    p.add_argument("--freeze_base", action="store_true", help="freeze the base model, finetune only the router gating")
    p.add_argument("--freeze_gate", action="store_true", help="freeze the gating, train only the base (ablation)")
    p.add_argument("--use_lora", default=True, action=argparse.BooleanOptionalAction,
                   help="add LoRA to the base and joint-train with the gating (7B full-parameter does not fit in 24GB, default on; small models may use --no-use_lora)")
    p.add_argument("--lora_rank", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_targets", default="q_proj,v_proj")
    p.add_argument("--gradient_checkpointing", default=True, action=argparse.BooleanOptionalAction,
                   help="must enable for long sequences/large batches (with LoRA, enable_input_require_grads is applied automatically to fix the frozen-embed backward pitfall)")
    p.add_argument("--use_chat_template", default=False, action=argparse.BooleanOptionalAction,
                   help="assemble multi-turn conversations with the tokenizer chat_template (required for Instruct models)")
    p.add_argument("--mask_user_tokens", default=True, action=argparse.BooleanOptionalAction,
                   help="in chat mode, compute loss/acc only on assistant replies (user questions serve only as context)")
    p.add_argument("--kl_coef", type=float, default=0.0,
                   help="dense self-distillation weight: loss += kl_coef*KL(sparse||frozen dense), 0=off (legacy behavior)")
    p.add_argument("--kl_temp", type=float, default=1.0, help="distillation temperature")
    p.add_argument("--teacher_model_id", default="",
                   help="distillation teacher model path; empty = same as model_id")
    p.add_argument("--teacher_device", default="cuda:0",
                   help="device hosting the frozen teacher (separate from the training device to save memory); in single-GPU worker scenarios it shares the training device")
    p.add_argument("--cos_reg_coef", type=float, default=0.01,
                   help="StableSkip-style routing regularization weight (0=off)")
    p.add_argument("--tau_calib_batches", type=int, default=8,
                   help="threshold: number of model-aware tau calibration batches before training (0=off, uses the tau_init constant)")
    p.add_argument("--tau_spread", type=float, default=0.5,
                   help="threshold: tau calibration spread (by per-layer stability z-score)")
    p.add_argument("--router_calib_batches", type=int, default=0,
                   help="moe: number of router-temperature calibration batches before training (0=off/legacy); "
                        "calibrates the JointRouter logit temperature so the initial top-p mean k starts near "
                        "--router_start_k (near-dense healthy start, mirrors threshold's tau calibration)")
    p.add_argument("--router_start_k", type=int, default=0,
                   help="moe: target initial mean k for router calibration (0 = auto, ~0.6*gated)")
    p.add_argument("--resume_dir", default="",
                   help="resume from a ckpt: structure follows mod_config.json (including gate_mode), gating params are overlaid, "
                        "annealing/price are re-scheduled from the CLI (temp/gumbel/price reset to initial values)")
    p.add_argument("--eval_every", type=int, default=200,
                   help="plateau check cadence (steps); scale up for full long runs (e.g. 2000)")
    p.add_argument("--patience", type=int, default=3,
                   help="plateau tolerance count; scale up for full long runs (e.g. 10)")
    p.add_argument("--save_full", action="store_true",
                   help="required for the joint (full-parameter) version: additionally saves the full base (gating keys stripped) + tokenizer, "
                        "otherwise gate.pt paired with the original base will mismatch (non-LoRA only)")
    p.add_argument("--ul_mode", default="none", choices=["none", "gt", "rollout"],
                   help="anti-repetition regularizer (r4): gt=n-gram unlikelihood on the GT side; "
                        "rollout=unlikelihood on self-generated rollouts (also gives the gating off-policy prefix gradients)")
    p.add_argument("--ul_coef", type=float, default=0.3, help="unlikelihood term weight (0=off)")
    p.add_argument("--ul_n", type=int, default=3, help="n-gram length that triggers UL")
    p.add_argument("--rollout_every", type=int, default=20,
                   help="rollout mode: self-generate a rollout every K steps")
    p.add_argument("--rollout_tokens", type=int, default=24, help="number of tokens generated per rollout")
    p.add_argument("--rollout_prompt", type=int, default=32, help="length of the real prefix taken from the batch for rollout")
    return p.parse_args()


def ngram_repeat_trigger(ids: torch.Tensor, n: int, valid: torch.Tensor) -> torch.Tensor:
    """[B,T] bool: position t triggers when the n-gram ending at t (including x_t)
    appeared earlier in the sequence and valid[t].
    Trigger positions are where "repetition is forming" (n-gram-triggered variant of
    Welleck unlikelihood);
    positions with valid=False (padding/unsupervised segments) never trigger, but
    their tokens still enter the context as usual."""
    B, T = ids.shape
    out = torch.zeros(B, T, dtype=torch.bool)
    for b in range(B):
        seq = ids[b].tolist()
        seen: set = set()
        for t in range(T):
            if t + 1 >= n:
                gram = tuple(seq[t + 1 - n:t + 1])
                if valid[b, t] and gram in seen:
                    out[b, t] = True
                seen.add(gram)
    return out


def unlikelihood_loss(logits, targets, trigger) -> torch.Tensor:
    """-log(1 - p(target)) at trigger positions (suppress the probability of repeated
    tokens); softmax is computed only on the triggered rows to save compute.
    logits [B,T-1,V] align with targets [B,T-1] (HF shift: logits[:,t] predicts x_{t+1})."""
    idx = trigger.nonzero(as_tuple=False)
    if idx.numel() == 0:
        return logits.new_zeros(())
    rows = logits[idx[:, 0], idx[:, 1]].float()
    tgt = targets[idx[:, 0], idx[:, 1]]
    p = rows.softmax(-1).gather(-1, tgt[:, None]).squeeze(-1)
    return -(1 - p).clamp_min(1e-6).log().mean()


def rollout_unlikelihood(mod_model, batch, args, tok, device):
    """Scheme 2: hard greedy rollout with the current model -> trigger n-gram UL on the
    self-generated segment.
    Side effect: the UL forward pass with gradients gives the gating/LoRA gradients on
    drifted prefixes (DAgger style)."""
    prompt = batch["input_ids"][:1, :args.rollout_prompt]
    mod_model.eval()
    mod_model.set_skip_mode("hard")  # rollout takes the deployment path (hard layer skipping + sparse KV)
    with torch.no_grad():
        g = mod_model.generate(input_ids=prompt, attention_mask=torch.ones_like(prompt),
                               max_new_tokens=args.rollout_tokens, do_sample=False,
                               pad_token_id=tok.pad_token_id, use_cache=True)
    mod_model.set_skip_mode("soft")
    mod_model.train()
    seq = g[:, :prompt.shape[1] + args.rollout_tokens]  # [1, Lp+G] (shorter if eos truncates early)
    valid = torch.zeros_like(seq, dtype=torch.bool)
    valid[0, prompt.shape[1]:] = True  # trigger only on the self-generated segment; the prompt segment only enters the context
    trg = ngram_repeat_trigger(seq, args.ul_n, valid)
    out = mod_model(input_ids=seq, attention_mask=torch.ones_like(seq))
    return unlikelihood_loss(out.logits[:, :-1], seq[:, 1:], trg[:, 1:])


def main():
    args = parse_args()
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
    device = resolve_device(args.device)

    tok = build_tok(args.model_id)

    model = build_model(args.model_id, device, dtype_of(args.dtype))

    if args.use_lora:
        model = wrap_lora(model, args.lora_rank, args.lora_alpha, args.lora_targets)
        tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"lora enabled: rank {args.lora_rank} trainable {tr / 1e6:.1f}M", flush=True)

    full, eval_texts = split_train_eval(
        args.data_path, args.max_samples, args.eval_samples,
        tok=tok if args.use_chat_template else None, use_chat=args.use_chat_template)
    if not eval_texts:
        print(f"WARNING: eval slice is empty ({len(full.samples)} lines in file, max_samples {args.max_samples}); "
              f"final eval skipped! Check that the build-time filter matches the SFTDataset protocol", flush=True)
    ds = full
    coll_fn = make_collate(tok, device, args.max_length,
                           args.use_chat_template, args.mask_user_tokens)

    dense_res = None
    if eval_texts:
        dense_res = eval_heldout(model, eval_texts, coll_fn)
        print(f"heldout dense baseline: loss {dense_res['loss']:.3f} acc {dense_res['acc']:.3f} "
              f"({len(eval_texts)} samples)", flush=True)

    teacher = None
    t_device = None
    if args.kl_coef > 0:
        t_device = torch.device(args.teacher_device)
        teacher = AutoModelForCausalLM.from_pretrained(
            args.teacher_model_id or args.model_id, dtype=dtype_of(args.dtype), device_map=None,
            trust_remote_code=True, low_cpu_mem_usage=True)
        teacher.to(t_device)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        print(f"teacher ready on {t_device} kl_coef {args.kl_coef} temp {args.kl_temp}", flush=True)

    if args.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        if args.use_lora:
            # peft: with a frozen base, checkpointing requires inputs to carry gradients, otherwise backward breaks (frozen-embed pitfall)
            model.enable_input_require_grads()

    if args.resume_dir:
        cfg = SpeakerConfig.from_json(os.path.join(args.resume_dir, "mod_config.json"))
        # Reset end-state values to the training initial state (annealing/price rescheduling);
        # structure (always-on layers/gating/kmax) follows the ckpt to keep weights aligned
        cfg.temp_affinity = args.temp_affinity
        cfg.gumbel_scale = args.gumbel_scale
        cfg.sparsity_price = args.sparsity_price
        cfg.acc_target = args.acc_target
        cfg.price_warmup_steps = args.price_warmup
        cfg.cos_reg_coef = args.cos_reg_coef
        print(f"resumed config from {args.resume_dir}", flush=True)
    else:
        overrides = dict(
            gate_mode=args.gate_mode,
            kmax=args.kmax, select_mode=args.select_mode, top_p=args.top_p,
            top_k=args.top_k, min_layers=args.min_layers,
            always_on_head=args.always_head,
            always_on_tail=args.always_tail, temp_affinity=args.temp_affinity,
            gumbel_scale=args.gumbel_scale, sparsity_price=args.sparsity_price,
            price_adapt=args.price_adapt, acc_target=args.acc_target,
            price_warmup_steps=args.price_warmup, tau_init=args.tau_init,
            cos_reg_coef=args.cos_reg_coef)
        if args.always_layers.strip():
            overrides["always_on_layers"] = [int(x) for x in args.always_layers.split(",")
                                             if x.strip() != ""]
        cfg = SpeakerConfig.from_model_config(model.config, **overrides)
    print(cfg.summary(), flush=True)
    mod_model = convert_to_speaker(model, cfg).to(device)
    if args.resume_dir:
        missing, unexp = load_gate(mod_model, args.resume_dir)
        print(f"resumed gate.pt, missing {len(missing)} unexpected {len(unexp)}", flush=True)

    if args.tau_calib_batches > 0 and not args.resume_dir and cfg.gate_mode == "threshold":
        mod_model.set_skip_mode("soft")
        calib = []
        for b in DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=coll_fn):
            calib.append(b)
            if len(calib) >= args.tau_calib_batches:
                break
        new_taus = mod_model.calibrate_tau(calib, spread=args.tau_spread)
        if new_taus:
            vals = list(new_taus.values())
            print(f"tau calibrated: [{min(vals):+.2f},{max(vals):+.2f}] "
                  f"(init {args.tau_init:+.2f} spread {args.tau_spread})", flush=True)
        mod_model.train()

    if args.router_calib_batches > 0 and not args.resume_dir and cfg.gate_mode == "moe":
        calib = []
        for b in DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=coll_fn):
            calib.append(b)
            if len(calib) >= args.router_calib_batches:
                break
        info = mod_model.calibrate_router_temp(
            calib, target_k=(args.router_start_k or None))
        if info:
            print(f"router temp calibrated: {info['router_temp']:.3f} "
                  f"(k0 {info['k_before']:.1f} -> {info['k_after']:.1f}, "
                  f"target {info['target_k']})", flush=True)
        mod_model.train()

    groups, base_params, gate_params = build_param_groups(
        mod_model, args.lr, args.router_lr, args.freeze_base, args.freeze_gate)
    n_base_train = sum(1 for p in base_params if p.requires_grad)
    n_gate_train = sum(1 for p in gate_params if p.requires_grad)
    print(f"base {len(base_params)}(train {n_base_train}) gate {len(gate_params)}(train {n_gate_train}) "
          f"freeze_base={args.freeze_base} freeze_gate={args.freeze_gate}", flush=True)

    opt = torch.optim.AdamW(groups, weight_decay=0.01)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=coll_fn)

    os.makedirs(args.save_dir, exist_ok=True)
    rl = RunLogger(args.save_dir)
    step = 0
    ema_lm = None
    ema_acc = None
    window_t0 = time.time()
    window_tokens = 0
    stopper = StopOnPlateau(eval_every=args.eval_every, patience=args.patience,
                            max_steps=args.max_steps)  # ed3 unified convergence rule (cadence/cap adjustable)
    stop_subset = eval_texts[:40] if eval_texts else []  # subset for plateau checks; the final eval still uses the full set
    mod_model.train()
    for epoch in range(1000):
        for b in dl:
            step += 1
            if stopper.capped(step):
                break
            mod_model.anneal(step, args.anneal_steps, ta_end=args.ta_end, g_end=0.0)
            out = mod_model(**b)
            lm_loss = out.loss
            aux = mod_model.get_aux_loss()
            kl = None
            if teacher is not None:
                with torch.no_grad():
                    tb = {"input_ids": b["input_ids"].to(t_device),
                          "attention_mask": b["attention_mask"].to(t_device)}
                    t_out = teacher(**tb)
                kl = args.kl_coef * distill_kl_loss(
                    out.logits, t_out.logits.to(device), b["labels"], args.kl_temp)
            with torch.no_grad():
                correct = per_token_correct(out.logits.float(), b["labels"])
                valid_tok = (b["labels"] != -100)  # under the chat mask only the assistant segment counts (equivalent to attention_mask in the legacy protocol)
                acc_item = correct[valid_tok].float().mean().item() if valid_tok.any() else 0.0
            # Pure Lagrangian: regularizer = lambda*mean(k), lower is better; accuracy is held
            # by the acc dual adjustment; no easy/hard split
            task = mod_model.get_budget_loss(b["attention_mask"])
            if task is not None and args.budget_ramp > 0 and step < args.budget_ramp:
                task = task * (step / args.budget_ramp)  # gradual pressure: protect accuracy and let the gating learn first, then go sparse
            loss = lm_loss + (aux or 0) + (task or 0) + (kl or 0)
            ema_lm = ema_update(ema_lm, lm_loss.item())
            ema_acc = ema_update(ema_acc, acc_item)
            if step > cfg.price_warmup_steps:
                mod_model.adapt_price(ema_acc)  # acc dual adjustment: relax lambda below target, apply more pressure above it
            # stats (pure k basis)
            with torch.no_grad():
                counts = mod_model.get_active_counts()
                valid = b["attention_mask"].bool()
                if counts is not None and valid.any():
                    ks = counts[valid].float()
                    mean_k = ks.mean().item()
                    std_k = ks.std().item() if ks.numel() > 1 else 0.0
                    est_mb = estimate_act_mb(mean_k, *b["input_ids"].shape,
                                             cfg.hidden_size, cfg.mem_bytes_per_hidden)
                else:
                    mean_k = std_k = est_mb = float("nan")
                window_tokens += int(valid.sum())
            # UL must come after the stats block: the rollout forward overwrites last_gating_output,
            # so aux/budget/k stats must read the training pass state first (known smoke pitfall: index 256 vs 56 mismatch)
            ul = None
            if args.ul_mode == "gt":
                # GT-side UL: positions where the target token continues an n-gram already seen
                # earlier in the context (repetition forming points); suppress their probability
                trg = ngram_repeat_trigger(b["input_ids"], args.ul_n, b["labels"] != -100)
                ul = unlikelihood_loss(out.logits[:, :-1], b["input_ids"][:, 1:], trg[:, 1:])
            elif args.ul_mode == "rollout" and step % args.rollout_every == 0:
                ul = rollout_unlikelihood(mod_model, b, args, tok, device)
            if ul is not None:
                loss = loss + args.ul_coef * ul
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(mod_model.parameters(), 1.0)
            opt.step()
            if step % args.log_interval == 0:
                rate = window_tokens / max(time.time() - window_t0, 1e-6)
                sp = mod_model.get_layer_sparsity()
                spars = sum(sp[i] for i in cfg.gated_layers) / max(len(cfg.gated_layers), 1)
                mem_gb = torch.cuda.memory_allocated(device) / 1024**3
                rl.metrics(step=step, lm=round(lm_loss.item(), 4), ema_lm=round(ema_lm, 4),
                            acc=round(acc_item, 4), ema_acc=round(ema_acc, 4),
                            k=round(mean_k, 3), k_std=round(std_k, 3), spars=round(spars, 4),
                            aux=round(float(aux.detach()), 5) if aux is not None else 0.0,
                            task=round(float(task.detach()), 4) if task is not None else 0.0,
                            tot=round(float(loss.detach()), 4),
                            kl=round(float(kl.detach()), 4) if kl is not None else None,
                            ul=round(float(ul.detach()), 4) if ul is not None else None,
                            price=round(cfg.sparsity_price, 5), ta=round(cfg.temp_affinity, 3),
                            gumbel=round(cfg.gumbel_scale, 3), tok_s=round(rate, 1),
                            mem_gb=round(mem_gb, 2), est_mb=round(est_mb, 2))
                RunLogger.status(
                    f"[{time.strftime('%H:%M:%S')}] step {step} lm {ema_lm:.3f} "
                    f"acc {ema_acc:.2f} k {mean_k:.1f}±{std_k:.1f} spars {spars:.0%} "
                    f"price {cfg.sparsity_price:.4f} {rate:.0f}tok/s mem {mem_gb:.1f}GB"
                    + (f" ul {ul.item():.2f}" if ul is not None else ""))
                window_t0 = time.time()
                window_tokens = 0
            if step % stopper.eval_every == 0 and stop_subset:
                # plateau check (eval_heldout restores train mode itself); per-layer load written to layers.jsonl at low frequency
                chk = eval_heldout(mod_model, stop_subset, coll_fn)
                rl.layers(step, mod_model.get_layer_usage(), cfg.always_on_layers)
                with open(os.path.join(args.save_dir, "converge.jsonl"), "a", encoding="utf-8") as f:
                    f.write(json.dumps({"step": step, "subset_loss": chk["loss"],
                                        "best": stopper.best, "ema_lm": ema_lm,
                                        "k": chk.get("mean_k")}) + "\n")
                # long-run insurance: overwrite a small ckpt at every plateau beat (recoverable on crash, no waiting for the final state)
                save_gate(mod_model, args.save_dir, extra_marks=("lora_",))
                cfg.to_json(os.path.join(args.save_dir, "mod_config.json"))
                mk = chk.get("mean_k")
                RunLogger.event(f"step {step} subset_loss {chk['loss']:.4f} "
                                f"best {stopper.best} k {mk if mk is not None else '-'}, ckpt saved")
                if stopper.check(step, chk["loss"]):
                    RunLogger.event(f"converged at step {step}, "
                                    f"best heldout-subset loss {stopper.best:.3f}")
                    break
            if stopper.capped(step):
                break
        if stopper.capped(step):
            break

    cfg.to_json(os.path.join(args.save_dir, "mod_config.json"))
    save_gate(mod_model, args.save_dir, extra_marks=("lora_",))
    if args.save_full:
        assert not args.use_lora, "--save_full only supports non-LoRA (peft requires adapter save)"
        mod_model.hf_model.save_pretrained(args.save_dir,
                                           state_dict=clean_base_state_dict(mod_model))
        tok.save_pretrained(args.save_dir)
    rl.close()
    print(f"saved to {args.save_dir}", flush=True)
    # Final eval: same-distribution held-out; accuracy delta = Speaker - dense baseline, at a glance
    if eval_texts and dense_res is not None:
        mod_res = eval_heldout(mod_model, eval_texts, coll_fn)
        print(f"heldout | dense loss {dense_res['loss']:.3f} acc {dense_res['acc']:.3f} "
              f"| mod loss {mod_res['loss']:.3f} acc {mod_res['acc']:.3f} "
              f"(Δloss {mod_res['loss'] - dense_res['loss']:+.3f} Δacc {mod_res['acc'] - dense_res['acc']:+.3f}) "
              f"| k {mod_res['mean_k']:.1f}±{mod_res['std_k']:.1f} "
              f"quartile {[round(v, 1) for v in mod_res['quartile_k']]}", flush=True)


if __name__ == "__main__":
    main()
