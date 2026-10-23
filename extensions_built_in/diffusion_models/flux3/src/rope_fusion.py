"""Fused Triton RoPE for flux3 (one kernel launch per q/k per pass).

Replaces the eager apply_rope chain (src/transformer.py:86-91): fp32 upcast
of q/k, ``freqs_cis[...,0]*x0 + freqs_cis[...,1]*x1`` per pair, fp32 add,
bf16 downcast. The eager ops are separate fp32 kernels -- each mul rounds
to fp32, the add rounds -- so the fused kernel mirrors the sequence
(mul, mul, add with enable_fp_fusion=False) and is BITWISE identical,
while deleting two fp32 full-size upcast copies, the neg materialization,
the two muls and the add per tensor per pass (at bs8 the fp32 pair copies
are [8,24,L,128]x4B ~= 400 MB transients per block pass).

Layouts (this model, file:line):
- x:    (B, H, L, D) bf16, q/k after QKNorm -- contiguous.
- freqs: EmbedND output (B, 1, L, D//2, 2, 2) fp32; squeeze(1) gives the
  (B, L, D//2, 2, 2) flat table the kernel indexes; row-major pairs are
  [[cos, -sin], [sin, cos]] (transformer.py:79-82 stack order), so
  cos = flat[...0], sin = flat[...2]; broadcast over heads by construction.

Backward is the inverse rotation, same kernel with SIGN=-1 -- the exact
adjoint of the eager two-mul-and-add graph (no cos/sin grads: freqs come
from position ids, never trainable here).

FLUX3_ROPE_FUSE=0 restores the eager path (kill switch for A/B). A launch
failure prints once with the real shapes and falls back to eager -- it is
never silent.
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
        f"[flux3.rope_fusion] triton import FAILED ({_e}); "
        "flux3 rope runs the eager path"
    )

_FUSE_OK = (
    _HAS_TRITON
    and torch.cuda.is_available()
    and os.environ.get("FLUX3_ROPE_FUSE", "1") != "0"
)
_engaged = False


def _banner_once(xq, freq4):
    # first-use notice (H3 varlen precedent): a feature whose non-engagement
    # looks costless is a regression-in-waiting
    global _engaged
    if _engaged or not torch.cuda.is_available():
        return
    _engaged = True
    print(
        f"[flux3.rope_fusion] fused rope ACTIVE: first launch "
        f"q={tuple(xq.shape)} freqs={tuple(freq4.shape)} {xq.dtype} "
        f"(eager control: FLUX3_ROPE_FUSE=0)"
    )

if _HAS_TRITON:

    @triton.jit
    def _rope_kernel(
        x_ptr, y_ptr, f_ptr,
        n_elems,          # B*H*L*D
        hl,               # H * L
        L,                # sequence length
        half_d,           # D // 2 (pairs per row; freq table entries per row / 4)
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
        # freq table fp32 (B, L, half_d, 2, 2): cos at +0, sin at +2
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
        freq4 = freq4.contiguous()  # kernel indexes the (B,L,D//2,2,2) flat
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


def fused_rope_apply(xq, xk, freqs_cis):
    """Bitwise-identical fused apply_rope; None (caller runs eager) if the
    fused path cannot take this call.

    xq/xk: (B,H,L,D) bf16/fp16 CUDA; freqs_cis: (B,1,L,D//2,2,2) fp32.
    """
    if not _FUSE_OK:
        return None
    freq4 = freqs_cis.squeeze(1) if freqs_cis.dim() == 6 else freqs_cis
    if (
        xq.dim() != 4
        or xq.dtype not in (torch.bfloat16, torch.float16)
        or not xq.is_cuda
        or xq.shape[-1] % 2 != 0
        or freq4.dtype != torch.float32
        or freq4.dim() != 5
    ):
        return None
    try:
        _banner_once(xq, freq4)
        return (
            _RopeApply.apply(xq, freq4),
            _RopeApply.apply(xk, freq4),
        )
    except Exception as e:
        reraise_if_stop_recompute(e)
        report_missing(
            "rope_fusion",
            f"launch failed for q{tuple(xq.shape)} k{tuple(xk.shape)} "
            f"freqs{tuple(freqs_cis.shape)} {xq.dtype}: "
            f"{type(e).__name__}: {e}",
            tag="flux3.rope_fusion",
        )
        return None
