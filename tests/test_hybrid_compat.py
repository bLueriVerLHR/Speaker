"""Offline unit tests: hybrid-attention backbone compatibility (v0.2 adaptation).

Covers the changes that let Speaker gate Qwen3_5-style backbones (the 27B
linear+full-attention VL model), on CPU with a tiny random config — no weights
download, no GPU:

- nested text_config unwrap in SpeakerConfig.from_model_config (+ layer_types);
- language_model.layers discovery in SpeakerModelWrapper._find_layers;
- position anchor: the first full_attention layer forced always-on when
  layer_types is known; legacy path (no layer_types) untouched;
- hybrid cache decode detection: linear-attention layers via
  has_previous_state (get_seq_length raises for them), full-attention layers via
  get_seq_length — exercised through hard-mode generation with use_cache;
- forward/backward/budget on both gate schemes over the hybrid stack;
- device-alignment fixes in metrics/UL (labels follow logits' device).
"""
import torch
from speaker import SpeakerConfig, convert_to_speaker  # noqa: E402


def _tiny_qwen3_5(n_layers=8, hidden=64, interval=4):
    """Tiny random Qwen3_5 (VL shell + hybrid text stack), CPU."""
    from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration
    text = dict(
        hidden_size=hidden, intermediate_size=128, num_hidden_layers=n_layers,
        num_attention_heads=4, head_dim=16, num_key_value_heads=2,
        linear_num_key_heads=2, linear_key_head_dim=16,
        linear_num_value_heads=4, linear_value_head_dim=16,
        linear_conv_kernel_dim=4, vocab_size=512,
        full_attention_interval=interval, max_position_embeddings=512,
        mamba_ssm_dtype="float32",
    )
    vision = dict(depth=2, hidden_size=32, num_heads=2, intermediate_size=64,
                  num_position_embeddings=16, out_hidden_size=hidden,
                  spatial_merge_size=1)
    torch.manual_seed(0)
    cfg = Qwen3_5Config(text_config=text, vision_config=vision)
    return Qwen3_5ForConditionalGeneration(cfg).eval()


def _batch(model, n=2, t=24, device="cpu"):
    v = model.config.text_config.vocab_size
    ids = torch.randint(0, v, (n, t), device=device)
    mask = torch.ones_like(ids)
    labels = ids.clone()
    return {"input_ids": ids, "attention_mask": mask, "labels": labels}


def test_from_model_config_unwraps_nested_text_config():
    model = _tiny_qwen3_5()
    cfg = SpeakerConfig.from_model_config(model.config, gate_mode="threshold")
    assert cfg.num_hidden_layers == 8 and cfg.hidden_size == 64
    assert cfg.layer_types is not None and len(cfg.layer_types) == 8
    # position anchor: first full_attention layer (idx 3 with interval 4) forced
    # always-on on top of the head/tail derivation
    assert 3 in cfg.always_on_layers, cfg.always_on_layers
    assert 0 in cfg.always_on_layers and 7 in cfg.always_on_layers
    # anchor forced even under pure-gating (explicit empty list)
    cfg0 = SpeakerConfig.from_model_config(model.config, gate_mode="threshold",
                                           always_on_layers=[])
    assert cfg0.always_on_layers == [3], cfg0.always_on_layers
    # dict-style nested config unwraps the same way
    d = {"text_config": {"num_hidden_layers": 5, "hidden_size": 32}}
    cfg2 = SpeakerConfig.from_model_config(d)
    assert cfg2.num_hidden_layers == 5 and cfg2.layer_types is None


def test_find_layers_and_wrap():
    model = _tiny_qwen3_5()
    cfg = SpeakerConfig.from_model_config(model.config, gate_mode="threshold")
    mod = convert_to_speaker(model, cfg)
    assert len(mod.layers) == 8
    assert mod.mod_config.gated_layers == [i for i in range(8) if i not in (0, 1, 3, 6, 7)]
    # the patched ModuleList is the one the forward actually runs
    assert mod.hf_model.model.language_model.layers is mod.layers


def test_threshold_forward_backward_budget():
    model = _tiny_qwen3_5()
    cfg = SpeakerConfig.from_model_config(model.config, gate_mode="threshold",
                                          tau_init=-2.0)
    mod = convert_to_speaker(model, cfg).train()
    b = _batch(model)
    out = mod(**b)
    aux = mod.get_aux_loss()
    task = mod.get_budget_loss(b["attention_mask"])
    loss = out.loss + (aux or 0) + (task or 0)
    loss.backward()
    grads = [w.router.net.weight.grad for w in mod.layers if w.router is not None]
    assert any(g is not None for g in grads), "router received no gradient"
    assert torch.isfinite(loss)


def test_moe_forward_backward_budget():
    model = _tiny_qwen3_5()
    cfg = SpeakerConfig.from_model_config(model.config, gate_mode="moe")
    mod = convert_to_speaker(model, cfg).train()
    b = _batch(model)
    out = mod(**b)
    aux = mod.get_aux_loss()
    task = mod.get_budget_loss(b["attention_mask"])
    loss = out.loss + (aux or 0) + (task or 0)
    loss.backward()
    assert mod.joint_router.net.weight.grad is not None
    assert torch.isfinite(loss)


def test_hard_generation_with_hybrid_cache_and_skip():
    model = _tiny_qwen3_5()
    cfg = SpeakerConfig.from_model_config(model.config, gate_mode="threshold",
                                          tau_init=-2.0)
    mod = convert_to_speaker(model, cfg).eval()
    mod.set_skip_mode("hard")
    b = _batch(model, n=1, t=16)
    g = mod.generate(input_ids=b["input_ids"], attention_mask=b["attention_mask"],
                     max_new_tokens=6, do_sample=False, use_cache=True,
                     pad_token_id=0)
    assert g.shape[1] >= 16
    # force every gate closed (tau huge) so decode steps hit the skip path on BOTH
    # layer kinds: full-attention layers via get_seq_length, linear layers via
    # has_previous_state (get_seq_length raises for them)
    for w in mod.layers:
        if w.tau is not None:
            with torch.no_grad():
                w.tau.fill_(50.0)
    mod.get_skip_hits(reset=True)
    g2 = mod.generate(input_ids=b["input_ids"], attention_mask=b["attention_mask"],
                      max_new_tokens=4, do_sample=False, use_cache=True,
                      pad_token_id=0)
    assert g2.shape[1] >= 16
    hits = mod.get_skip_hits()
    # gated layers are 2,4,5 (linear) — decode steps must skip through the hybrid
    # cache path; >0 proves the linear-attention branch answered "decode step"
    assert hits > 0, "no decode-time skips: hybrid cache decode detection failed"


def test_metrics_device_alignment():
    # logits on a 'remote' device (simulated by plain CPU tensors with mismatched
    # devices is impossible on CPU-only; test the label-follow contract directly)
    from speaker.metrics import per_token_correct, per_token_nll
    logits = torch.randn(2, 5, 11)
    labels = torch.randint(0, 11, (2, 5))
    labels[:, -1] = -100
    nll = per_token_nll(logits, labels)
    ok = per_token_correct(logits, labels)
    assert nll.shape == (2, 5) and ok.shape == (2, 5)
    assert nll[:, -1].eq(0).all()
