"""Fused Triton kernels for the krea2 SingleStreamDiT hot elementwise paths.

* rope_apply: eager (mmdit.ropeapply) upcasts (B,H,L,D) q/k to fp32,
  rotates each pair with the stored fp32 [[cos,-sin],[sin,cos]] columns
  and downcasts. Every eager fp32 op uses the same rounding sequence as
  this kernel (mul, mul, add in fp32) and the only bf16 rounding is the
  store in both paths, so the fused kernel is bitwise identical while
  deleting the up/downcast copies and the separate mul/add/neg launches
  on the step's biggest non-GEMM tensors.
"""

from __future__ import annotations

import os

import torch

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except Exception as _e:  # pragma: no cover
    _HAS_TRITON = False
    print(
        f"[krea2.kernels] triton import FAILED ({_e}); all krea2 fused "
        f"kernels fall back to eager"
    )


if _HAS_TRITON:

    @triton.jit
    def _rope_kernel(
        x_ptr, y_ptr, f_ptr,
        n_elems,          # elements of x (B*H*L*D)
        hl,               # H * L
        L,                # sequence length
        half_d,           # D // 2 (pairs per row; also freq pairs per row)
        BLOCK: tl.constexpr,   # contiguous elements per program (even)
        SIGN: tl.constexpr,    # +1 forward, -1 backward (inverse rotation)
    ):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n_elems
        x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        x0, x1 = tl.split(tl.reshape(x, (BLOCK // 2, 2)))

        e = pid * BLOCK + tl.arange(0, BLOCK // 2) * 2   # even element offsets
        emask = e < n_elems
        p = e // 2                       # global pair index
        r = p // half_d                  # (b, h, l) row index
        pp = p - r * half_d              # pair within the row
        b = r // hl
        l = r % L
        # freqs fp32 contiguous (B, L, half_d, 2, 2); cos = [..,0,0],
        # sin = [..,1,0] -> element offset ((b*L + l)*half_d + pp)*4
        f = ((b * L + l) * half_d + pp) * 4
        cos_v = tl.load(f_ptr + f, mask=emask, other=1.0)
        sin_v = tl.load(f_ptr + f + 2, mask=emask, other=0.0) * SIGN
        o0 = cos_v * x0 - sin_v * x1
        o1 = sin_v * x0 + cos_v * x1
        y = tl.reshape(tl.interleave(o0, o1), (BLOCK,))
        tl.store(y_ptr + offs, y.to(y_ptr.dtype.element_ty), mask=mask)

    def _rope_launch(x4: torch.Tensor, freq4: torch.Tensor, sign: int):
        b, h, l, d = x4.shape
        x = x4.contiguous().view(-1)
        freq4 = freq4.contiguous()  # kernel indexes the (B,L,D//2,2,2) flat layout
        y = torch.empty_like(x)
        n = x.numel()
        BLOCK = 512
        grid = (triton.cdiv(n, BLOCK),)
        _rope_kernel[grid](
            x, y, freq4, n, h * l, l, d // 2,
            BLOCK=BLOCK, SIGN=sign, num_warps=4, enable_fp_fusion=False,
        )
        return y.view(x4.shape)

    class _RopeApply(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, freq):
            assert not freq.requires_grad
            ctx.save_for_backward(freq)
            return _rope_launch(x, freq, 1)

        @staticmethod
        def backward(ctx, dy):
            (freq,) = ctx.saved_tensors
            return _rope_launch(dy.contiguous(), freq, -1), None

_TRITON_OK = _HAS_TRITON and torch.cuda.is_available()
_ROPE_FUSE = _TRITON_OK

# --- cuDNN frontend FROST attention (sm120 native fwd/bwd) -----------------
from toolkit.util.kernel_report import (
    report_missing as _report_missing_shared,
    reraise_if_stop_recompute as _reraise_if_stop_recompute,
)


def _report_missing(what: str, detail: str):
    _report_missing_shared(what, detail, tag="krea2.kernels")


def fused_ropeapply(xq, xk, freqs):
    """Bitwise-identical fused mmdit.ropeapply; None if unavailable.

    xq/xk: (B,Hq,L,D)/(B,Hk,L,D) bf16 (GQA: head counts may differ);
    freqs: (B,L,D//2,2,2) fp32 contiguous, broadcast over heads.
    """
    if not _ROPE_FUSE:
        return None
    try:
        return _RopeApply.apply(xq, freqs), _RopeApply.apply(xk, freqs)
    except Exception as e:  # kernel failed on this shape/arch
        _reraise_if_stop_recompute(e)
        _report_missing(
            "rope_fusion",
            f"launch failed for q{tuple(xq.shape)} k{tuple(xk.shape)} {xq.dtype}: "
            f"{type(e).__name__}: {e}",
        )
        return None


# fused_sdpa (cuDNN FROST sm120 SDPA) is model-agnostic and lives in
# toolkit.attention.frost; re-exported for existing import sites.
from toolkit.attention.frost import fused_sdpa  # noqa: E402,F401
