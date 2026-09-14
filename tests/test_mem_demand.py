"""Unit tests for tools/mem_demand.py (offline, no weights)."""
import pytest

from tools.mem_demand import demand, kv_layer_bytes, kv_layer_bytes_from_config


def test_kv_gqa_shape():
    # Qwen2.5-7B: hidden 3584, 28 q heads, 4 kv heads, bf16
    assert kv_layer_bytes(3584, 28, 4, 2) == 2 * 4 * 128 * 2


def test_kv_mqa_fallback():
    assert kv_layer_bytes(1024, 8, 0, 2) == 2 * 8 * 128 * 2


def test_kv_from_config_dict_and_object():
    d = {"hidden_size": 3584, "num_attention_heads": 28, "num_key_value_heads": 4}
    assert kv_layer_bytes_from_config(d, 2) == 2048

    class C:
        hidden_size = 1024
        num_attention_heads = 8
        num_key_value_heads = 8
    assert kv_layer_bytes_from_config(C(), 2) == 2 * 8 * 128 * 2


def test_demand_scales_with_k():
    a = demand(12.0, 28, 12.0, 2048, ctx=1024)
    b = demand(24.0, 28, 12.0, 2048, ctx=1024)
    assert a["flop_ratio"] == pytest.approx(12.0 / 28)
    assert b["active_w_gb"] == pytest.approx(2 * a["active_w_gb"])
    assert b["kv_mb_ctx"] == pytest.approx(2 * a["kv_mb_ctx"])
    # 12 layers x 2048 B x 1024 ctx = ~25.2 MB
    assert a["kv_mb_ctx"] == pytest.approx(12 * 2048 * 1024 / 1e6, rel=1e-6)


def test_demand_dense_matches_history_scale():
    # history: dense 7B thr weight ~13GB/tok scale, KV ~57MB @1024ctx
    full = demand(28.0, 28, 12.0, 2048, ctx=1024)
    assert full["active_w_gb"] == pytest.approx(12.0)
    assert full["kv_mb_ctx"] == pytest.approx(28 * 2048 * 1024 / 1e6, rel=1e-6)
    assert full["flop_ratio"] == pytest.approx(1.0)
