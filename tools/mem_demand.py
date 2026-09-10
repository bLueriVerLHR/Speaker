#!/usr/bin/env python3
"""r7 per-token inference memory demand aggregator.

Reads r7 P2 probe_kdist JSONs; for each method reports the per-token layer-weight
demand (the user's metric: token t needs k layers of weights -> demand k), converted
to GB using exact Qwen2.5-7B config (layer-only params; embed/lm_head are always-on
and excluded from the per-layer demand), plus sparse-KV bytes per token (only executed
layers write K/V).
"""
import os
import glob
import json
import sys

MODEL_CFG = "/home/hdd/model/Qwen2.5-7B-Instruct/config.json"
N_LAYERS = 28
BYTES_PER_PARAM = 2  # bf16


def layer_bytes():
    c = json.load(open(MODEL_CFG))
    h, i = c["hidden_size"], c["intermediate_size"]
    nh, nkv = c["num_attention_heads"], c["num_key_value_groups"] if "num_key_value_groups" in c else c["num_key_value_heads"]
    hd = c.get("head_size") or c.get("head_dim") or c["hidden_size"] // nh
    attn = h * nh * hd + h * nkv * hd * 2 + nh * hd * h
    mlp = h * i * 3
    per_layer = attn + mlp  # Qwen gates: gate+up+down = 3 mats
    kv_per_token_layer = nkv * hd * 2 * BYTES_PER_PARAM  # K+V
    return per_layer * BYTES_PER_PARAM, kv_per_token_layer


LOG_DIR = os.environ.get("MEM_DEMAND_DIR", "./.logs")


def main():
    lb, kvb = layer_bytes()
    print(f"Qwen2.5-7B: layer weights {lb/1e9:.3f} GB/layer, KV {kvb} B/token/layer, "
          f"dense demand {N_LAYERS*lb/1e9:.2f} GB layer-weights + {N_LAYERS*kvb/1024:.0f} KB KV per token\n")
    files = sorted(glob.glob(os.path.join(LOG_DIR, "*_kdist_*.json")))
    if not files:
        sys.exit("no kdist files yet")
    rows = []
    for f in files:
        d = json.load(open(f))
        for name, v in d.items():
            if name == "meta" or not isinstance(v, dict):
                continue
            kt = v.get("k_total")
            if kt is None:  # dense baseline: k = all layers
                kt = {"mean": float(N_LAYERS), "std": 0.0,
                      "q01_10_25_50_75_90_100": [N_LAYERS] * 7, "min": N_LAYERS, "max": N_LAYERS}
            q = kt["q01_10_25_50_75_90_100"]
            rows.append({
                "method": name.split(":")[-1] if ":" in name else name,
                "k_mean": kt["mean"], "k_std": kt["std"],
                "q10": q[1], "q50": q[3], "q90": q[5], "kmin": kt["min"], "kmax": kt["max"],
                "w_gb_tok": kt["mean"] * lb / 1e9,
                "kv_kb_tok": kt["mean"] * kvb / 1e3,
                "w_save": 1 - kt["mean"] / N_LAYERS,
            })
    rows.sort(key=lambda r: -r["k_mean"])
    hdr = f"{'method':<12} {'k_mean':>6} {'±':>4} {'q10':>4} {'q50':>4} {'q90':>4} {'min':>3} {'max':>3} {'weightsGB/tok':>13} {'KV KB/tok':>9} {'weight save':>11}"
    print(hdr); print("-" * len(hdr))
    for r in rows:
        print(f"{r['method']:<12} {r['k_mean']:>6.1f} {r['k_std']:>4.1f} {r['q10']:>4.0f} {r['q50']:>4.0f} "
              f"{r['q90']:>4.0f} {r['kmin']:>3.0f} {r['kmax']:>3.0f} {r['w_gb_tok']:>13.2f} "
              f"{r['kv_kb_tok']:>9.1f} {r['w_save']:>10.0%}")
    json.dump(rows, open(os.path.join(LOG_DIR, "mem_demand.json"), "w"), indent=1)
    print(f"\n-> {os.path.join(LOG_DIR, 'mem_demand.json')}")


if __name__ == "__main__":
    main()
