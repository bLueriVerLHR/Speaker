"""Finetune training pipeline (split from train.py, P0 structure).

run_finetune(cfg): the training loop moved verbatim — op order, loss
composition (LM + aux + budget + KL + UL), logging cadence, plateau protocol
and final report are unchanged. cfg is a speaker.hparams.FinetuneConfig
(typed dataclass, built with explicit kwargs by finetune/cli.train).
"""
import os
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from speaker import SpeakerConfig, convert_to_speaker  # noqa: E402
from speaker.config import budget_pin_warnings  # noqa: E402
from speaker.metrics import distill_kl_loss, estimate_act_mb, per_token_nll  # noqa: E402
from speaker.evaluate import ema_update, eval_heldout, format_k_quartile  # noqa: E402
from speaker.ruler import AccRuler  # noqa: E402 (P0: one accuracy scale everywhere)
from speaker.dual import DualController, difficulty_mult  # noqa: E402 (P0.5 dual + ed8 difficulty shaping)
from speaker.ul import (  # noqa: E402 (P2: anti-repetition mechanism library, shared by both gate schemes)
    gt_repeat_rate,
    ngram_repeat_trigger,
    rollout_rep3_probe,
    rollout_unlikelihood,
    unlikelihood_loss,
)
from speaker.converge import StopOnPlateau  # noqa: E402 (ed3 unified convergence rule)
from speaker.log import add_jsonl, emit, event, logger, setup_logger  # noqa: E402
from speaker.checkpoint import clean_base_state_dict, load_gate, save_gate  # noqa: E402
from speaker.train_common import (  # noqa: E402
    build_model,
    build_optimizer,
    build_param_groups,
    build_tok,
    dtype_of,
    enable_checkpointing,
    parse_csv_list,
    resolve_device,
    seed_all,
    split_train_eval,
    wrap_lora,
)
from data.sft import make_collate  # noqa: E402

# Historical import path shim: `from finetune.train import ngram_repeat_trigger,
# unlikelihood_loss` (tests/test_ul.py) keeps working through finetune/train.py,
# which re-exports these names from here.
__all__ = ["run_finetune"]


def run_finetune(hp):
    # logger first: every line below goes through console + save_dir/run.log
    setup_logger(run_dir=hp.save_dir)
    add_jsonl(os.path.join(hp.save_dir, "metrics.jsonl"), "metrics")
    add_jsonl(os.path.join(hp.save_dir, "layers.jsonl"), "layers")
    add_jsonl(os.path.join(hp.save_dir, "converge.jsonl"), "converge")
    if hp.seed is not None:
        seed_all(hp.seed)
    # P2 accelerator backend (opt-in): "none" (default) = legacy manual device
    # handling below, bit-identical reruns. "accelerate"/"fabric" wrap the
    # model+optimizer via speaker/accelerate_backend.py when requested.
    backend = None
    if hp.accelerator != "none":
        from speaker.accelerate_backend import get_backend
        backend = get_backend(hp.accelerator)
    device = resolve_device(hp.device)

    tok = build_tok(hp.model_id)

    model = build_model(hp.model_id, device, dtype_of(hp.dtype),
                        device_map=(hp.device_map or None))

    if hp.use_lora:
        model = wrap_lora(model, hp.lora_rank, hp.lora_alpha, hp.lora_targets)
        tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"lora enabled: rank {hp.lora_rank} trainable {tr / 1e6:.1f}M")

    full, eval_texts = split_train_eval(
        hp.data_path, hp.max_samples, hp.eval_samples,
        tok=tok if hp.use_chat_template else None, use_chat=hp.use_chat_template,
        single_turn=hp.single_turn)
    if not eval_texts:
        logger.warning(f"eval slice is empty ({len(full.samples)} lines in file, max_samples {hp.max_samples}); "
                       f"final eval skipped! Check that the build-time filter matches the SFTDataset protocol")
    ds = full
    coll_fn = make_collate(tok, device, hp.max_length,
                           hp.use_chat_template, hp.mask_user_tokens)

    dense_res = None
    ruler = AccRuler.from_cli(hp.acc_target, hp.acc_margin)
    if eval_texts:
        dense_res = eval_heldout(model, eval_texts, coll_fn, valid_mode=ruler.mode)
        logger.info(f"heldout dense baseline: loss {dense_res['loss']:.3f} acc {dense_res['acc']:.3f} "
                    f"({len(eval_texts)} samples)")
    acc_floor = ruler.resolve(dense_res["acc"] if dense_res else None)
    logger.info(f"[ruler] {ruler.describe()}")

    teacher = None
    t_device = None
    if hp.kl_coef > 0 or hp.diff_mode == "teacher":
        # the frozen dense serves KL distillation and/or the difficulty oracle (ed8 tiers);
        # one forward covers both when both are on
        t_device = torch.device(hp.teacher_device)
        teacher = AutoModelForCausalLM.from_pretrained(
            hp.teacher_model_id or hp.model_id, dtype=dtype_of(hp.dtype), device_map=None,
            trust_remote_code=True, low_cpu_mem_usage=True)
        teacher.to(t_device)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        logger.info(f"teacher ready on {t_device} kl_coef {hp.kl_coef} temp {hp.kl_temp}")

    if hp.gradient_checkpointing:
        enable_checkpointing(model, hp.use_lora)

    if hp.resume_dir:
        cfg = SpeakerConfig.from_json(os.path.join(hp.resume_dir, "mod_config.json"))
        # Reset end-state values to the training initial state (annealing/price rescheduling);
        # structure (always-on layers/gating/kmax) follows the ckpt to keep weights aligned
        cfg.temp_affinity = hp.temp_affinity
        cfg.gumbel_scale = hp.gumbel_scale
        cfg.sparsity_price = hp.sparsity_price
        cfg.over_budget_coef = hp.over_budget_coef
        cfg.acc_target = acc_floor
        cfg.price_warmup_steps = hp.price_warmup
        cfg.cos_reg_coef = hp.cos_reg_coef
        logger.info(f"resumed config from {hp.resume_dir}")
    else:
        overrides = dict(
            gate_mode=hp.gate_mode,
            kmax=hp.kmax, select_mode=hp.select_mode, top_p=hp.top_p,
            top_k=hp.top_k, min_layers=hp.min_layers,
            weight_mode=hp.weight_mode,
            always_on_head=hp.always_head,
            always_on_tail=hp.always_tail, temp_affinity=hp.temp_affinity,
            gumbel_scale=hp.gumbel_scale, sparsity_price=hp.sparsity_price,
            budget_form=hp.budget_form, budget_target=hp.budget_target,
            over_budget_coef=hp.over_budget_coef,
            tail_coef=hp.tail_coef, tail_temp=hp.tail_temp,
            price_adapt=hp.price_adapt, acc_target=acc_floor,
            diff_easy_nll=hp.diff_easy_nll, diff_hard_nll=hp.diff_hard_nll,
            diff_easy_mult=hp.diff_easy_mult, diff_hard_mult=hp.diff_hard_mult,
            price_warmup_steps=hp.price_warmup, tau_init=hp.tau_init,
            cos_reg_coef=hp.cos_reg_coef)
        if hp.always_layers.strip():
            overrides["always_on_layers"] = parse_csv_list(hp.always_layers)
        cfg = SpeakerConfig.from_model_config(model.config, **overrides)
        if hp.always_layers.strip():
            dropped = [i for i in parse_csv_list(hp.always_layers)
                       if not 0 <= i < cfg.num_hidden_layers]
            if dropped:
                logger.warning(f"always_layers {dropped} out of range "
                               f"[0,{cfg.num_hidden_layers}) silently dropped")
    # Baked decode recipe (ed9/C): explicit CLI wins on fresh and on resume; None = legacy
    decode_updates = {}
    if hp.decode_rep_penalty is not None:
        decode_updates["repetition_penalty"] = hp.decode_rep_penalty
    if hp.decode_no_repeat_ngram is not None:
        decode_updates["no_repeat_ngram_size"] = hp.decode_no_repeat_ngram
    if decode_updates:
        cfg.decode = {**(cfg.decode or {}), **decode_updates}
        logger.info(f"baked decode recipe {cfg.decode} into mod_config.json")
    logger.info(cfg.summary())
    for _w in budget_pin_warnings(cfg):
        logger.warning(_w)
    mod_model = convert_to_speaker(model, cfg)
    if not hp.device_map:
        # whole-card placement; under device_map sharding the wrapper's per-layer device
        # moves (already cross-device) handle placement and .to() would collapse the shards
        mod_model = mod_model.to(device)
    if hp.resume_dir:
        missing, unexp = load_gate(mod_model, hp.resume_dir)
        logger.info(f"resumed gate.pt, missing {len(missing)} unexpected {len(unexp)}")

    if hp.tau_calib_batches > 0 and not hp.resume_dir and hp.gate_mode == "threshold":
        mod_model.set_skip_mode("soft")
        calib = []
        for b in DataLoader(ds, batch_size=hp.batch_size, shuffle=False,
                            collate_fn=coll_fn):
            calib.append(b)
            if len(calib) >= hp.tau_calib_batches:
                break
        new_taus = mod_model.calibrate_tau(calib, spread=hp.tau_spread)
        if new_taus:
            vals = list(new_taus.values())
            logger.info(f"tau calibrated: [{min(vals):+.2f},{max(vals):+.2f}] "
                        f"(init {hp.tau_init:+.2f} spread {hp.tau_spread})")
        mod_model.train()

    if hp.router_calib_batches > 0 and not hp.resume_dir and hp.gate_mode == "moe":
        calib = []
        for b in DataLoader(ds, batch_size=hp.batch_size, shuffle=False,
                            collate_fn=coll_fn):
            calib.append(b)
            if len(calib) >= hp.router_calib_batches:
                break
        info = mod_model.calibrate_router_temp(
            calib, target_k=(hp.router_start_k or None))
        if info:
            logger.info(f"router temp calibrated: {info['router_temp']:.3f} "
                        f"(k0 {info['k_before']:.1f} -> {info['k_after']:.1f}, "
                        f"target {info['target_k']})")
        mod_model.train()

    groups, base_params, gate_params = build_param_groups(
        mod_model, hp.lr, hp.router_lr)
    n_base_train = sum(1 for p in base_params if p.requires_grad)
    n_gate_train = sum(1 for p in gate_params if p.requires_grad)
    logger.info(f"base {len(base_params)}(train {n_base_train}) gate {len(gate_params)}(train {n_gate_train})")

    opt = build_optimizer(groups)
    if backend is not None:
        mod_model, opt = backend.prepare(mod_model, opt)
    dl = DataLoader(ds, batch_size=hp.batch_size, shuffle=True, collate_fn=coll_fn)

    os.makedirs(hp.save_dir, exist_ok=True)
    step = 0
    ema_lm = None
    dual = DualController(mod_model, cfg)  # accuracy dual: ema + warmup gate + λ adaptation
    window_t0 = time.time()
    window_tokens = 0
    stopper = StopOnPlateau(eval_every=hp.eval_every, patience=hp.patience,
                            max_steps=hp.max_steps)  # ed3 unified convergence rule (cadence/cap adjustable)
    stop_subset = eval_texts[:40] if eval_texts else []  # subset for plateau checks; the final eval still uses the full set
    mod_model.train()
    for epoch in range(1000):
        for b in dl:
            step += 1
            if stopper.capped(step):
                break
            mod_model.anneal(step, hp.anneal_steps, ta_end=hp.ta_end, g_end=0.0)
            out = mod_model(**b)
            lm_loss = out.loss
            aux = mod_model.get_aux_loss()
            kl = None
            if teacher is not None:
                with torch.no_grad():
                    tb = {"input_ids": b["input_ids"].to(t_device),
                          "attention_mask": b["attention_mask"].to(t_device)}
                    t_out = teacher(**tb)
                kl = hp.kl_coef * distill_kl_loss(
                    out.logits, t_out.logits.to(device), b["labels"], hp.kl_temp)
            with torch.no_grad():
                # ruler: one accuracy scale for the dual, the plateau eval and the final
                # report (labels mode == the historical inline block under any collate)
                acc_item = ruler.batch_acc(out.logits, b)
            # Pure Lagrangian: regularizer = lambda*mean(k), lower is better; accuracy is held
            # by the acc dual adjustment; no easy/hard split
            lmap = None
            if hp.diff_mode == "teacher" and teacher is not None:
                # ed8 difficulty shaping: frozen-teacher NLL tiers -> per-token λ map
                # (easy pressed harder, hard allowed depth; the dual still owns λ_base)
                with torch.no_grad():
                    tb = {"input_ids": b["input_ids"].to(t_device),
                          "attention_mask": b["attention_mask"].to(t_device)}
                    t_nll = per_token_nll(teacher(**tb).logits.float(),
                                          b["labels"].to(t_device))
                lmap = difficulty_mult(t_nll.to(device), hp.diff_easy_nll,
                                       hp.diff_hard_nll, hp.diff_easy_mult,
                                       hp.diff_hard_mult)
            task = mod_model.get_budget_loss(b["attention_mask"], lambda_map=lmap)
            if task is not None and hp.budget_ramp > 0 and step < hp.budget_ramp:
                task = task * (step / hp.budget_ramp)  # gradual pressure: protect accuracy and let the gating learn first, then go sparse
            # sharded backbones: aux/task scalars may land on a different card than
            # lm_loss (whose device = lm_head); align before adding (single-device no-op)
            ldev = lm_loss.device
            on = (lambda x: x.to(ldev) if torch.is_tensor(x) and x.device != ldev else x)
            loss = lm_loss + on(aux or 0) + on(task or 0) + on(kl or 0)
            ema_lm = ema_update(ema_lm, lm_loss.item())
            ema_acc = dual.observe(step, acc_item)
            # stats: count every layer that computes for a token
            # (gated selections + always-on) — this is the deployment number
            with torch.no_grad():
                counts = mod_model.get_total_counts()
                valid = b["attention_mask"].bool()
                kv = valid.to(counts.device) if counts is not None else valid
                if counts is not None and kv.any():
                    ks = counts[kv].float()
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
            if hp.ul_mode == "gt":
                # GT-side UL: positions where the target token continues an n-gram already seen
                # earlier in the context (repetition forming points); suppress their probability
                trg = ngram_repeat_trigger(b["input_ids"], hp.ul_n, b["labels"] != -100)
                ul = unlikelihood_loss(out.logits[:, :-1], b["input_ids"][:, 1:], trg[:, 1:])
            elif hp.ul_mode == "rollout" and step % hp.rollout_every == 0:
                ul = rollout_unlikelihood(mod_model, b, tok, device,
                                          hp.rollout_prompt, hp.rollout_tokens,
                                          hp.ul_n)
            if ul is not None:
                loss = loss + hp.ul_coef * ul
            opt.zero_grad()
            if backend is not None:
                backend.backward(loss)
            else:
                loss.backward()
            torch.nn.utils.clip_grad_norm_(mod_model.parameters(), 1.0)
            opt.step()
            if step % hp.log_interval == 0:
                rate = window_tokens / max(time.time() - window_t0, 1e-6)
                sp = mod_model.get_layer_sparsity()
                # skip rate over all layers (always-on layers never skip)
                spars = sum(sp.values()) / max(len(sp), 1)
                mem_gb = torch.cuda.memory_allocated(device) / 1024**3 \
                    if device.type == "cuda" else 0.0
                # repetition visibility (P2): GT-side n-gram recurrence density, near-zero
                # cost at log cadence; the deployment-side signal is the plateau probe below
                rep_gt = gt_repeat_rate(b["input_ids"], b["labels"] != -100, hp.ul_n)
                emit("metrics", step=step, lm=round(lm_loss.item(), 4), ema_lm=round(ema_lm, 4),
                            acc=round(acc_item, 4), ema_acc=round(ema_acc, 4),
                            k=round(mean_k, 3), k_std=round(std_k, 3), spars=round(spars, 4),
                            aux=round(float(aux.detach()), 5) if aux is not None else 0.0,
                            task=round(float(task.detach()), 4) if task is not None else 0.0,
                            dlambda=(round(float(lmap[b["labels"] != -100].mean()), 3)
                                     if lmap is not None else None),
                            tot=round(float(loss.detach()), 4),
                            kl=round(float(kl.detach()), 4) if kl is not None else None,
                            ul=round(float(ul.detach()), 4) if ul is not None else None,
                            price=round(cfg.sparsity_price, 5), ta=round(cfg.temp_affinity, 3),
                            gumbel=round(cfg.gumbel_scale, 3), tok_s=round(rate, 1),
                            mem_gb=round(mem_gb, 2), est_mb=round(est_mb, 2),
                            rep_gt=round(rep_gt, 4))
                logger.info(
                    f"step {step} lm {ema_lm:.3f} "
                    f"acc {ema_acc:.2f} k {mean_k:.1f}±{std_k:.1f} spars {spars:.0%} "
                    f"price {cfg.sparsity_price:.4f} {rate:.0f}tok/s mem {mem_gb:.1f}GB"
                    + (f" ul {ul.item():.2f}" if ul is not None else ""))
                window_t0 = time.time()
                window_tokens = 0
            if step % stopper.eval_every == 0 and stop_subset:
                # plateau check (eval_heldout restores train mode itself); per-layer load written to layers.jsonl at low frequency
                chk = eval_heldout(mod_model, stop_subset, coll_fn, valid_mode=ruler.mode)
                usage = mod_model.get_layer_usage()
                emit("layers", step=step,
                     exec={str(i): round(v[0], 4) for i, v in sorted(usage.items())},
                     always_on=sorted(cfg.always_on_layers))
                emit("converge", step=step, subset_loss=chk["loss"],
                     best=stopper.best, ema_lm=ema_lm, k=chk.get("mean_k"))
                # long-run insurance: overwrite a small ckpt at every plateau beat (recoverable on crash, no waiting for the final state)
                save_gate(mod_model, hp.save_dir, extra_marks=("lora_",))
                cfg.to_json(os.path.join(hp.save_dir, "mod_config.json"), extra={
                    "use_lora": bool(hp.use_lora), "lora_rank": hp.lora_rank,
                    "lora_alpha": hp.lora_alpha, "lora_targets": hp.lora_targets,
                })
                mk = chk.get("mean_k")
                event(f"step {step} subset_loss {chk['loss']:.4f} "
                      f"best {stopper.best} k {mk if mk is not None else '-'}, ckpt saved")
                if hp.rep_probe:
                    # repetition visibility (P2, r4 gen-probe resurrected): the plateau
                    # stopper only watches subset_loss — a model can "converge" straight
                    # into a repetition attractor without this signal ever moving it
                    r3 = rollout_rep3_probe(mod_model, stop_subset, tok, device)
                    event(f"step {step} rep3 probe {r3:.3f} "
                          f"(5 prompts x 24 tok greedy, trend-only)")
                if stopper.check(step, chk["loss"]):
                    event(f"converged at step {step}, "
                          f"best heldout-subset loss {stopper.best:.3f}")
                    break
            if stopper.capped(step):
                break
        if stopper.capped(step):
            break

    cfg.to_json(os.path.join(hp.save_dir, "mod_config.json"), extra={
        "use_lora": bool(hp.use_lora), "lora_rank": hp.lora_rank,
        "lora_alpha": hp.lora_alpha, "lora_targets": hp.lora_targets,
    })
    save_gate(mod_model, hp.save_dir, extra_marks=("lora_",))
    if hp.save_full:
        assert not hp.use_lora, "--save_full only supports non-LoRA (peft requires adapter save)"
        mod_model.hf_model.save_pretrained(hp.save_dir,
                                           state_dict=clean_base_state_dict(mod_model))
        tok.save_pretrained(hp.save_dir)
    logger.info(f"saved to {hp.save_dir}")
    # Final eval: same-distribution held-out; accuracy delta = Speaker - dense baseline, at a glance
    if eval_texts and dense_res is not None:
        mod_res = eval_heldout(mod_model, eval_texts, coll_fn, valid_mode=ruler.mode)
        logger.info(f"heldout | dense loss {dense_res['loss']:.3f} acc {dense_res['acc']:.3f} "
                    f"| mod loss {mod_res['loss']:.3f} acc {mod_res['acc']:.3f} "
                    f"(Δloss {mod_res['loss'] - dense_res['loss']:+.3f} Δacc {mod_res['acc'] - dense_res['acc']:+.3f}) "
                    f"| {format_k_quartile(mod_res)}")


