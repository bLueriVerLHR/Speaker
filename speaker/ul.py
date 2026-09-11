"""Anti-repetition mechanism library (P2 consolidation): n-gram unlikelihood training
terms + repetition-rate probes.

Why a library (r4/r7 evidence): repetition collapse is invisible to the LM loss and
the plateau stopper (r7 thr "converged" into a rep3 0.583 graveyard), the UL terms
lived only in finetune/train.py (off by default, forgotten by the r7 protocol), and
repetition metrics existed only post-hoc in tools/eval_gen.py. Centralizing here:

- the training terms (gt / rollout) are reusable by any route — both our gate
  schemes already share them via finetune/train.py; the import path
  `from finetune.train import ngram_repeat_trigger, unlikelihood_loss` keeps working
  (re-exported) for historical callers/tests;
- gt_repeat_rate gives the training loop a near-zero-cost visibility signal
  (logged as rep_gt at log cadence);
- rollout_rep3_probe resurrects the r4 gen-probe: a tiny hard-greedy rollout at
  plateau beats, reported as a RunLogger event (trend-only, 5 prompts is noisy);
  restores skip_mode/train state and draws no RNG (greedy, eval mode), so training
  numerics are untouched.
"""
from __future__ import annotations

import torch


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


def rollout_unlikelihood(mod_model, batch, tok, device, prompt_len, gen_tokens, n):
    """Scheme 2: hard greedy rollout with the current model -> trigger n-gram UL on the
    self-generated segment.
    Side effect: the UL forward pass with gradients gives the gating/LoRA gradients on
    drifted prefixes (DAgger style)."""
    prompt = batch["input_ids"][:1, :prompt_len]
    mod_model.eval()
    mod_model.set_skip_mode("hard")  # rollout takes the deployment path (hard layer skipping + sparse KV)
    with torch.no_grad():
        g = mod_model.generate(input_ids=prompt, attention_mask=torch.ones_like(prompt),
                               max_new_tokens=gen_tokens, do_sample=False,
                               pad_token_id=tok.pad_token_id, use_cache=True)
    mod_model.set_skip_mode("soft")
    mod_model.train()
    seq = g[:, :prompt.shape[1] + gen_tokens]  # [1, Lp+G] (shorter if eos truncates early)
    valid = torch.zeros_like(seq, dtype=torch.bool)
    valid[0, prompt.shape[1]:] = True  # trigger only on the self-generated segment; the prompt segment only enters the context
    trg = ngram_repeat_trigger(seq, n, valid)
    out = mod_model(input_ids=seq, attention_mask=torch.ones_like(seq))
    return unlikelihood_loss(out.logits[:, :-1], seq[:, 1:], trg[:, 1:])


def rep3_rate(s: str) -> float:
    """Fraction of recurring character 3-grams (same formula as tools/eval_gen.py,
    single source going forward)."""
    g = [s[i:i + 3] for i in range(max(len(s) - 2, 0))]
    return 0.0 if not g else 1.0 - len(set(g)) / len(g)


def gt_repeat_rate(ids: torch.Tensor, valid: torch.Tensor, n: int) -> float:
    """Training-time visibility: fraction of supervised positions where an n-gram is
    recurring (repetition forming), i.e. the GT-side trigger density."""
    trg = ngram_repeat_trigger(ids, n, valid)
    v = valid.float().sum().item()
    return trg.float().sum().item() / max(v, 1.0)


def rollout_rep3_probe(mod_model, texts, tok, device, n_prompts=5, prompt_len=32,
                       gen_tokens=24, n=3) -> float:
    """Cheap repetition probe at plateau beats (r4 gen-probe): hard greedy rollouts on
    the first n_prompts eval texts, mean 3-gram repeat rate of the generated segment.
    Trend-only (small n, high variance). Restores skip_mode/train state; draws no RNG
    (greedy decode, eval mode) so training numerics are untouched."""
    was_training = mod_model.training
    prev_mode = mod_model.mod_config.skip_mode
    mod_model.eval()
    mod_model.set_skip_mode("hard")
    rates = []
    with torch.no_grad():
        for text in texts[:n_prompts]:
            ids = tok(text, truncation=True, max_length=prompt_len,
                      return_tensors="pt")["input_ids"].to(device)
            g = mod_model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                                   max_new_tokens=gen_tokens, do_sample=False,
                                   pad_token_id=tok.pad_token_id, use_cache=True)
            cont = tok.decode(g[0, ids.shape[1]:], skip_special_tokens=True)
            if cont.strip():
                rates.append(rep3_rate(cont))
    mod_model.set_skip_mode(prev_mode)
    if was_training:
        mod_model.train()
    return sum(rates) / max(len(rates), 1)
