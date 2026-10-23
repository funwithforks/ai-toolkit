"""Fused Triton RMSNorm core for flux3 (QKNorm sites, src/transformer.py).

Replaces only the promotion chain inside RMSNorm.forward (transformer.py
:178-185): ``x.float()`` copy, ``x**2`` pow kernel, fp32 mean-reduce,
``+eps`` add, rsqrt, fp32 mul, bf16 downcast -- six eager kernels and an
fp32 full-size transient per call. The ``* self.scale`` multiplication is
LEFT IN TORCH on purpose: it is ATen's own bf16 mul (fp32 opmath, one
rounding), so the double-rounding structure ``(x*rrms).to(bf16) * scale``
and the exact grad-scale reduction are byte-for-byte the ops in use today.
The only intentional numeric difference is the mean's reduction order
(atomic-free tree in tl.sum vs ATen Reduce.cuh): bounded at <=1 ulp on
rrms, verified in tests/f3_norm_fused_parity.py with a ulp cap.

Layouts (this model): q/k into QKNorm are (B, H, L, 128) bf16 STRIDED
views of the fused qkv chunks (head stride 128, row stride 3*H*D); the
kernel takes x through its four strides and writes the normalized tensor
contiguous (matching what today's cast produced). Backward is the
analytic RMSNorm adjoint in fp32 (the eager graph is a composite whose
fp32 chain differs only by reduction order); dx is written as an
empty_strided gradient matching x's strides.

FLUX3_NORM_FUSE=0 restores the eager chain (kill switch for A/B). A
launch failure prints once with the real shapes and falls back to eager
-- it is never silent.
"""

import os

import torch

from toolkit.util.kernel_report import (
    report_missing,
    reraise_if_stop_recompute,
)

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except Exception as _e:  # pragma: no cover
    _HAS_TRITON = False
    print(
        f"[flux3.norm_fusion] triton import FAILED ({_e}); "
        "flux3 rmsnorm runs the eager path"
    )

_FUSE_OK = (
    _HAS_TRITON
    and torch.cuda.is_available()
    and os.environ.get("FLUX3_NORM_FUSE", "1") != "0"
)
_engaged = False


def _banner_once(x):
    global _engaged
    if not _engaged:
        _engaged = True
        print(
            f"[flux3.norm_fusion] fused rmsnorm ACTIVE: first launch "
            f"x={tuple(x.shape)} strides={x.stride()} {x.dtype} "
            f"(eager control: FLUX3_NORM_FUSE=0)"
        )


if _HAS_TRITON:

    @triton.jit
    def _rms_fwd2(x_ptr, y_ptr, r_ptr, L, H, eps,
                  sx0, sx1, sx2, D: tl.constexpr, BLOCK: tl.constexpr):
        bh = tl.program_id(0)
        l = tl.program_id(1)
        b = bh // H
        h = bh % H
        offs = tl.arange(0, BLOCK)
        base = x_ptr + b * sx0 + h * sx1 + l * sx2
        x = tl.load(base + offs).to(tl.float32)
        ss = tl.sum(x * x, axis=0)
        rrms = tl.rsqrt(ss / D + eps)
        y = (x * rrms).to(tl.bfloat16)
        tl.store(y_ptr + (bh * L + l) * D + offs, y)
        tl.store(r_ptr + bh * L + l, rrms)

    @triton.jit
    def _rms_bwd(dy_ptr, x_ptr, dx_ptr, r_ptr, L, H,
                 sx0, sx1, sx2,
                 sdx0, sdx1, sdx2,
                 D: tl.constexpr, BLOCK: tl.constexpr):
        bh = tl.program_id(0)
        l = tl.program_id(1)
        b = bh // H
        h = bh % H
        offs = tl.arange(0, BLOCK)
        rrms = tl.load(r_ptr + bh * L + l)
        g = tl.load(dy_ptr + (bh * L + l) * D + offs).to(tl.float32)
        xbase = x_ptr + b * sx0 + h * sx1 + l * sx2
        x = tl.load(xbase + offs).to(tl.float32)
        # analytic adjoint of y = x * rsqrt(mean(x^2)+eps), fp32
        c = tl.sum(g * x, axis=0) / D
        dx = rrms * g - (rrms * rrms * rrms) * c * x
        dxbase = dx_ptr + b * sdx0 + h * sdx1 + l * sdx2
        tl.store(dxbase + offs, dx.to(tl.bfloat16))


def _supported(x, scale):
    return (
        x.is_cuda
        and x.dtype in (torch.bfloat16, torch.float16)
        and x.dim() == 4
        and x.stride(3) == 1
        and x.shape[3] == scale.shape[0]
        and x.shape[3] <= 1024
        and scale.is_cuda
        and scale.dtype == x.dtype
    )


_launch_fail_reported = False


def _run(x, eps=1e-6):
    B, H, L, D = x.shape
    y = torch.empty((B * H, L, D), device=x.device, dtype=x.dtype)
    rrms = torch.empty((B * H, L), device=x.device, dtype=torch.float32)
    BLOCK = triton.next_power_of_2(D)
    _rms_fwd2[(B * H, L)](
        x, y, rrms, L, H, eps,
        x.stride(0), x.stride(1), x.stride(2),
        D=D, BLOCK=BLOCK, num_warps=4 if BLOCK <= 512 else 8,
    )
    return _FusedRMS.apply(x, rrms, y.view(B, H, L, D), eps)


class _FusedRMS(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, rrms, y, eps):
        ctx.save_for_backward(x, rrms)
        return y

    @staticmethod
    def backward(ctx, dy):
        x, rrms = ctx.saved_tensors
        B, H, L, D = x.shape
        if not dy.is_contiguous():
            dy = dy.contiguous()
        dx = torch.empty_strided(
            x.shape, x.stride(), device=x.device, dtype=x.dtype
        )
        _rms_bwd[(B * H, L)](
            dy, x, dx, rrms, L, H,
            x.stride(0), x.stride(1), x.stride(2),
            dx.stride(0), dx.stride(1), dx.stride(2),
            D=D, BLOCK=triton.next_power_of_2(D),
            num_warps=4 if triton.next_power_of_2(D) <= 512 else 8,
        )
        return dx, None, None, None


def try_rmsnorm(x, scale):
    """Fused RMSNorm core + torch ``* scale``; None => use eager."""
    global _launch_fail_reported
    if not _FUSE_OK or not _supported(x, scale):
        return None
    try:
        _banner_once(x)
        out = _run(x)
        return out * scale
    except Exception as e:
        if reraise_if_stop_recompute(e):
            raise
        if not _launch_fail_reported:
            _launch_fail_reported = True
            report_missing(
                "flux3 fused rmsnorm",
                detail=f"launch failed {type(e).__name__}: {e} at "
                       f"x={tuple(x.shape)} strides={x.stride()} {x.dtype}; "
                       "falling back to eager for all calls",
            )
        return None
