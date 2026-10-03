"""Off-the-shelf fused-kernel dispatch for the H3 transformer.

Routes elementwise/norm ops to liger-kernel fused CUDA kernels when
importable; otherwise (and always on first failure) falls back to the
eager aten ops and prints the reason once, so a missing kernel can
never silently degrade the training path.

Set H3_LIGER_FUSIONS=1 to enable the fused kernels. Default off until
the model's own live A/B has verified them (krea2's measured gain is
not assumed to transfer).
"""
import os

import torch
import torch.nn.functional as F

_ENABLED = os.environ.get("H3_LIGER_FUSIONS", "0") != "0"
_warned = set()


def _fallback(name, why):
    if name not in _warned:
        _warned.add(name)
        print(f"[H3 kernels] {name}: using eager fallback ({why})")
    return False


try:
    if not _ENABLED:
        raise ImportError("disabled by default; set H3_LIGER_FUSIONS=1")
    from liger_kernel.ops.rms_norm import LigerRMSNormFunction
    from liger_kernel.ops.swiglu import LigerSiLUMulFunction
except Exception as e:  # noqa: BLE001 - any import/env failure disables routing
    LigerRMSNormFunction = None
    LigerSiLUMulFunction = None
    if _ENABLED:
        _fallback("import", str(e)[:120])


def fused_norm(x, module):
    """nn.RMSNorm-module call routed through the fused kernel."""
    return rms_norm(x, module.weight, module.eps)


def rms_norm(x, weight, eps):
    """nn.RMSNorm(x) equivalent. 'llama' casting = fp32 accumulate, fp32
    cast-back before the weight multiply, matching aten RMSNorm bf16
    behaviour to rounding order. fp32 tensors keep the aten path: the
    model's float32 islands (adaln projections, final-layer heads)
    must not pass through the kernel's reduced-precision casting."""
    if LigerRMSNormFunction is not None and x.dtype in (torch.bfloat16, torch.float16):
        try:
            return LigerRMSNormFunction.apply(x, weight, eps, 0.0, "llama", False)
        except Exception as e:  # noqa: BLE001
            _fallback("rms_norm", f"{type(e).__name__}: {str(e)[:120]}")
    return F.rms_norm(x, (x.shape[-1],), weight=weight, eps=eps)


def swiglu_mul(gate, up):
    """F.silu(gate) * up fused fwd+bwd (one launch, saved-activation
    recompute instead of two stored tensors). fp32 keeps the aten path
    (float32 islands, see rms_norm)."""
    if LigerSiLUMulFunction is not None and gate.dtype in (torch.bfloat16, torch.float16):
        try:
            return LigerSiLUMulFunction.apply(gate, up)
        except Exception as e:  # noqa: BLE001
            _fallback("swiglu", f"{type(e).__name__}: {str(e)[:120]}")
    return F.silu(gate) * up
