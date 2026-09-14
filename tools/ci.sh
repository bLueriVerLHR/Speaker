#!/usr/bin/env bash
# Canonical local verification gate (offline, no GPU needed).
#
# Usage:
#   tools/ci.sh              # py_compile + ruff + vulture + all offline pytest suites
#   tools/ci.sh --fast       # core seams only (~seconds): speaker/ruler/UL/assemble/registry/tooling
#   tools/ci.sh --real       # additionally run the on-device CPU smoke (needs local
#                            # Qwen1.5-0.5B weights + the SFT data path from AGENTS.md)
#
# The GPU training smokes (finetune/pretrain 8-step runs) stay outside this gate —
# they belong to the neu-sbox submission flow (see AGENTS.md).
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PYTHON:-python3}
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

echo "== py_compile =="
"$PY" -m py_compile speaker/*.py finetune/*.py pretrain/*.py baselines/*.py \
    data/*.py tools/*.py tests/*.py
echo "OK"

echo "== ruff =="
if "$PY" -m ruff --version >/dev/null 2>&1; then
    "$PY" -m ruff check .
else
    echo "SKIP (ruff not installed; pip install ruff)"
fi

echo "== vulture (dead code, min-confidence 100) =="
if "$PY" -m vulture --version >/dev/null 2>&1; then
    "$PY" -m vulture speaker finetune pretrain data tools tests baselines/*.py \
        --min-confidence 100
else
    echo "SKIP (vulture not installed; pip install vulture)"
fi

echo "== offline unit tests =="
OFFLINE="tests/test_speaker.py tests/test_load_profile.py tests/test_baseline_modd.py \
tests/test_baseline_mdf.py tests/test_scheduler.py tests/test_ul.py \
tests/test_ruler_dual.py tests/test_assemble.py tests/test_registry.py \
tests/test_tooling.py tests/test_budget_form.py tests/test_hybrid_compat.py \
tests/test_logging.py tests/test_mem_demand.py tests/test_gen_metrics.py"
# --fast: core seams only (iteration inner loop; baselines/hybrid stay in the full gate)
FAST="tests/test_speaker.py tests/test_ruler_dual.py tests/test_ul.py \
tests/test_assemble.py tests/test_registry.py tests/test_tooling.py \
tests/test_budget_form.py tests/test_logging.py"
if [[ "${1:-}" == "--fast" ]]; then
    OFFLINE="$FAST"
fi
if "$PY" -m pytest --version >/dev/null 2>&1; then
    "$PY" -m pytest $OFFLINE
else
    echo "ERROR: pytest not installed; pip install pytest (the suites are pytest-only, no direct-run shims)" >&2
    exit 1
fi

if [[ "${1:-}" == "--real" ]]; then
    echo "== on-device CPU smoke (local weights) =="
    "$PY" tests/test_speaker_real.py
fi
echo "CI OK"
