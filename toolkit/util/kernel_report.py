"""Shared fallback reporting for fused-kernel wrappers (repo policy).

A silent fallback is a regression-in-waiting: the run looks fine and the
optimized path is dead. Every wrapper that can fall back reports HERE,
once per distinct (what, detail), with enough context (shapes, dtype,
versions) to search for a kernel fix. Copy the CALL PATTERN into new
wrappers; do not re-copy this function.
"""

import torch

_warned: set = set()

try:  # checkpoint control-flow signal, never a kernel failure
    from torch.utils.checkpoint._checkpoint_error import _StopRecomputationError
except Exception:  # pragma: no cover
    try:
        from torch.utils.checkpoint import _StopRecomputationError  # type: ignore
    except Exception:
        _StopRecomputationError = None  # type: ignore


def reraise_if_stop_recompute(e: BaseException):
    if _StopRecomputationError is not None and isinstance(
        e, _StopRecomputationError
    ):
        raise e


def report_missing(what: str, detail: str, tag: str = "kernels"):
    """Print once per distinct failure; never raise (the caller falls back)."""
    key = (tag, what, detail)
    if key in _warned:
        return
    _warned.add(key)
    try:
        import triton as _t

        tv = getattr(_t, "__version__", "?")
    except Exception:
        tv = "absent"
    print(
        f"[{tag}] {what} UNAVAILABLE, eager fallback in use "
        f"(triton {tv}, torch {torch.__version__}, "
        f"cuda {torch.version.cuda}, gpu {torch.cuda.get_device_name(0)} "
        f"sm_{torch.cuda.get_device_capability(0)[0]}{torch.cuda.get_device_capability(0)[1]}): "
        f"{detail}"
    )
