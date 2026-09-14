"""Optional distributed/precision backend (P2, opt-in).

Thin wrapper over HF Accelerate / Lightning Fabric so the custom training
loop (LM + λ·k + KL + UL rollout, per-layer device moves, sparse KV, dual
controller) keeps its exact op order and numerics. Default ``none`` = legacy
manual device handling, bit-identical reruns.

- ``none``: no-op backend (``prepare`` returns inputs unchanged,
  ``backward`` calls ``loss.backward()``);
- ``accelerate``: HF Accelerate (installed: 1.13.0) — ``Accelerator.prepare``
  + ``accelerator.backward``; multi-GPU via ``accelerate launch`` config;
- ``fabric``: Lightning Fabric — used only when the ``lightning`` package is
  present, otherwise raises a clear error telling how to run without it.

Only ``finetune/pipeline.py`` consults this (via ``args.accelerator``);
baselines keep their single-card regime. No caller changes behavior unless
``--accelerator`` is explicitly passed.
"""
from __future__ import annotations


class NoneBackend:
    """Legacy path: manual device handling (default)."""

    name = "none"

    def prepare(self, model, opt):
        return model, opt

    def backward(self, loss):
        loss.backward()


class AccelerateBackend:
    """HF Accelerate wrapper (mixed precision + DDP/FSDP via accelerate config)."""

    name = "accelerate"

    def __init__(self, **kw):
        try:
            from accelerate import Accelerator
        except ImportError as e:
            raise ImportError(
                "accelerate backend requested (--accelerator accelerate) but the "
                "'accelerate' package is not installed; run with --accelerator none "
                "(default) or pip install accelerate") from e
        self.acc = Accelerator(**kw)

    def prepare(self, model, opt):
        return self.acc.prepare(model, opt)

    def backward(self, loss):
        self.acc.backward(loss)


class FabricBackend:
    """Lightning Fabric wrapper (explicit control over the training loop)."""

    name = "fabric"

    def __init__(self, **kw):
        try:
            from lightning.fabric import Fabric
        except ImportError as e:
            raise ImportError(
                "fabric backend requested (--accelerator fabric) but the "
                "'lightning' package is not installed; run with --accelerator none "
                "(default) or pip install lightning") from e
        self.fabric = Fabric(**kw)
        self._model = None
        self._opt = None

    def prepare(self, model, opt):
        self._model, self._opt = self.fabric.setup(model, opt)
        return self._model, self._opt

    def backward(self, loss):
        self.fabric.backward(loss)


_BACKENDS = {"none": NoneBackend, "accelerate": AccelerateBackend, "fabric": FabricBackend}


def get_backend(name: str = "none", **kw):
    """Registry lookup; raises ValueError on unknown names."""
    try:
        return _BACKENDS[name](**kw) if name != "none" else NoneBackend()
    except KeyError:
        raise ValueError(
            f"unknown accelerator backend {name!r} "
            f"(expected one of {sorted(_BACKENDS)})") from None


def list_backends() -> list:
    return sorted(_BACKENDS)


__all__ = [
    "NoneBackend", "AccelerateBackend", "FabricBackend",
    "get_backend", "list_backends",
]
