"""Offline tests for the primary logger (speaker/log.py, loguru-based).

- human stream (console/run.log) and JSONL sinks are separated by extra["jsonl"];
- JSONL lines are valid JSON with the training data contracts unchanged
  (metrics/layers/converge keys match what plot_converge.py and the ad-hoc
  analysis read);
- levels filter as usual.
"""
import json
import os

from speaker.log import add_jsonl, emit, event, logger, setup_logger


def _teardown():
    setup_logger(run_dir=None, console=True)  # restore the session default


def _read_lines(path):
    with open(path, encoding="utf-8") as f:
        return [ln for ln in f.read().splitlines() if ln.strip()]


def test_human_and_jsonl_separation(tmp_path):
    run_dir = str(tmp_path)
    try:
        setup_logger(run_dir=run_dir, console=False)
        add_jsonl(os.path.join(run_dir, "metrics.jsonl"), "metrics")
        add_jsonl(os.path.join(run_dir, "converge.jsonl"), "converge")
        logger.info("step 8 lm 6.795")
        logger.warning("eval slice is empty")
        emit("metrics", step=8, lm=6.795, acc=0.109, k=10.4, dlambda=None)
        emit("converge", step=8, subset_loss=6.1, best=6.0, ema_lm=6.8, k=10.5)
        event("converged step 8, best heldout-subset loss 6.000")

        run_log = _read_lines(os.path.join(run_dir, "run.log"))
        assert any("step 8 lm 6.795" in ln for ln in run_log)
        assert any("eval slice is empty" in ln for ln in run_log)
        assert any("[event] converged step 8" in ln for ln in run_log)
        # jsonl records never leak into the human stream
        assert not any(ln.strip().startswith("{") for ln in run_log)

        mrows = [_ for _ in map(json.loads, _read_lines(
            os.path.join(run_dir, "metrics.jsonl")))]
        assert mrows == [{"step": 8, "lm": 6.795, "acc": 0.109,
                          "k": 10.4, "dlambda": None}]
        crows = [_ for _ in map(json.loads, _read_lines(
            os.path.join(run_dir, "converge.jsonl")))]
        assert crows == [{"step": 8, "subset_loss": 6.1, "best": 6.0,
                          "ema_lm": 6.8, "k": 10.5}]
    finally:
        _teardown()


def test_layers_contract(tmp_path):
    run_dir = str(tmp_path)
    try:
        setup_logger(run_dir=run_dir, console=False)
        add_jsonl(os.path.join(run_dir, "layers.jsonl"), "layers")
        emit("layers", step=200, exec={"0": 1.0, "12": 0.3456},
             always_on=[0, 1, 23])
        rows = [_ for _ in map(json.loads, _read_lines(
            os.path.join(run_dir, "layers.jsonl")))]
        assert rows == [{"step": 200, "exec": {"0": 1.0, "12": 0.3456},
                         "always_on": [0, 1, 23]}]
    finally:
        _teardown()


def test_level_routing(tmp_path):
    """DEBUG/INFO/WARNING/ERROR all land in run.log at DEBUG level; JSONL stays isolated."""
    run_dir = str(tmp_path)
    try:
        setup_logger(run_dir=run_dir, level="DEBUG", console=False)
        add_jsonl(os.path.join(run_dir, "metrics.jsonl"), "metrics")
        logger.debug("gate routing trace")
        logger.info("step 8 lm 6.795")
        logger.warning("decode key 'top_k' is not a GenerationConfig field — ignored")
        logger.error("placement budget too small for fixed layers")
        emit("metrics", step=8, lm=6.795)

        run_log = _read_lines(os.path.join(run_dir, "run.log"))
        assert any("gate routing trace" in ln for ln in run_log)
        assert any("step 8 lm 6.795" in ln for ln in run_log)
        assert any("decode key" in ln for ln in run_log)
        assert any("placement budget" in ln for ln in run_log)
        assert not any(ln.strip().startswith("{") for ln in run_log)

        mrows = [_ for _ in map(json.loads, _read_lines(
            os.path.join(run_dir, "metrics.jsonl")))]
        assert mrows == [{"step": 8, "lm": 6.795}]
    finally:
        _teardown()


def test_level_filter(tmp_path):
    run_dir = str(tmp_path)
    try:
        setup_logger(run_dir=run_dir, level="WARNING", console=False)
        logger.info("hidden at WARNING")
        logger.error("shown at WARNING")
        run_log = _read_lines(os.path.join(run_dir, "run.log"))
        assert not any("hidden" in ln for ln in run_log)
        assert any("shown" in ln for ln in run_log)
    finally:
        _teardown()
