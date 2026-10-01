"""Fused Triton kernels for the krea2 SingleStreamDiT hot elementwise paths.

* rope_apply: eager (mmdit.ropeapply) upcasts (B,H,L,D) q/k to fp32,
  rotates each pair with the stored fp32 [[cos,-sin],[sin,cos]] columns
  and downcasts. Every eager fp32 op uses the same rounding sequence as
  this kernel (mul, mul, add in fp32) and the only bf16 rounding is the
  store in both paths, so the fused kernel is bitwise identical while
  deleting the up/downcast copies and the separate mul/add/neg launches
  on the step's biggest non-GEMM tensors.

* gated_residual: ``x + m * y`` with m the block modulation gate
  broadcast (B,1,F). Eager computes bf16 mul then bf16 add (two round-
  ings, one temporary round-trip through DRAM); the fused kernel keeps
  fp32 and rounds once at the store. NOT bitwise identical (<=1 bf16
  ulp on the sum): default OFF behind KREA2_GATED_RESIDUAL_FUSION=1
  pending a ComfyUI sample check.
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

    @triton.jit
    def _gated_residual_kernel(
        x_ptr, y_ptr, m_ptr, o_ptr,
        n, LF, F,
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        y = tl.load(y_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        b = offs // LF
        fcol = offs % F
        m = tl.load(m_ptr + b * F + fcol, mask=mask, other=0.0).to(tl.float32)
        tl.store(o_ptr + offs, (x + m * y).to(o_ptr.dtype.element_ty), mask=mask)

    @triton.jit
    def _gated_residual_bwd_kernel(
        g_ptr, y_ptr, m_ptr, dx_ptr, dy_ptr, dm_ptr,
        n, LF, F, has_m,
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        g = tl.load(g_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        y = tl.load(y_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        b = offs // LF
        fcol = offs % F
        m = tl.load(m_ptr + b * F + fcol, mask=mask, other=0.0).to(tl.float32)
        tl.store(dx_ptr + offs, g.to(dx_ptr.dtype.element_ty), mask=mask)
        tl.store(dy_ptr + offs, (g * m).to(dy_ptr.dtype.element_ty), mask=mask)
        if has_m:
            tl.atomic_add(dm_ptr + b * F + fcol, g * y)

    class _GatedResidual(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, y, m):
            # x, y: (B, L, F) same-shape; m: (B, 1, F) broadcast gate
            out = torch.empty_like(x)
            bf, lf = m.shape[0] * m.shape[2], x.shape[-2] * x.shape[-1]
            n = x.numel()
            BLOCK = 512
            grid = (triton.cdiv(n, BLOCK),)
            _gated_residual_kernel[grid](
                x, y, m, out, n, lf, x.shape[-1], BLOCK=BLOCK, num_warps=4,
            )
            ctx.save_for_backward(y, m)
            ctx.x_requires = x.requires_grad
            ctx.y_requires = y.requires_grad
            ctx.m_requires = m.requires_grad
            ctx.lf = lf
            return out

        @staticmethod
        def backward(ctx, dout):
            y, m = ctx.saved_tensors
            dout = dout.contiguous()
            dx = torch.empty_like(dout)
            dy = torch.empty_like(dout) if ctx.y_requires else None
            dm = (
                torch.zeros(m.shape, dtype=torch.float32, device=dout.device)
                if ctx.m_requires else None
            )
            n = dout.numel()
            BLOCK = 512
            grid = (triton.cdiv(n, BLOCK),)
            _gated_residual_bwd_kernel[grid](
                dout, y, m, dx,
                dy if ctx.y_requires else dout,
                dm if ctx.m_requires else dout,
                n, ctx.lf, dout.shape[-1], ctx.m_requires,
                BLOCK=BLOCK, num_warps=4,
            )
            return (
                dx if ctx.x_requires else None,
                dy,
                dm.to(m.dtype) if ctx.m_requires else None,
            )


_TRITON_OK = _HAS_TRITON and torch.cuda.is_available()
_ROPE_FUSE = _TRITON_OK

# --- cuDNN frontend FROST attention (sm120 native fwd/bwd) -----------------
# torch's SDPA on consumer Blackwell runs the sm120 generated fprop but an
# sm80 WMMA bprop. cudnn-frontend >= 1.28 ships sm120 DSL kernels for both
# (FROST); torch cannot select them. Wrapped as one autograd.Function;
# numerics are the standard cudnn bwd class (non-deterministic two-kernel
# dQK route, bf16 grad rel noise <= ~1e-2 vs the sm80 path on real shapes).
# Default ON when available; KREA2_FROST_SDPA=0 restores plain torch SDPA.
os.environ.setdefault("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", "1")
from cudnn.sdpa.bwd import sdpa_bwd_wrapper_dsl_sm120 as _frost_bwd
from cudnn.sdpa.fwd import sdpa_fwd_wrapper_dsl_sm120 as _frost_fwd

_FROST_OK = os.environ.get("KREA2_FROST_SDPA", "1") != "0"
_GATED_FUSE = _TRITON_OK and os.environ.get("KREA2_GATED_RESIDUAL_FUSION", "0") == "1"

_warned: set = set()

try:  # checkpoint control-flow signal, never a kernel failure
    from torch.utils.checkpoint._checkpoint_error import _StopRecomputationError
except Exception:  # pragma: no cover
    try:
        from torch.utils.checkpoint import _StopRecomputationError  # type: ignore
    except Exception:
        _StopRecomputationError = None  # type: ignore


def _reraise_if_stop_recompute(e: BaseException):
    if _StopRecomputationError is not None and isinstance(
        e, _StopRecomputationError
    ):
        raise e


def _report_missing(what: str, detail: str):
    """Fallbacks must never be silent: print once per distinct failure so
    the missing kernel/error is visible in the training log with enough
    context (shape, dtype, error text) to look up a kernel for it."""
    key = (what, detail)
    if key in _warned:
        return
    _warned.add(key)
    try:
        import triton as _t

        tv = getattr(_t, "__version__", "?")
    except Exception:
        tv = "absent"
    print(
        f"[krea2.kernels] {what} UNAVAILABLE, eager fallback in use "
        f"(triton {tv}, torch {torch.__version__}, "
        f"cuda {torch.version.cuda}, gpu {torch.cuda.get_device_name(0)} "
        f"sm_{torch.cuda.get_device_capability(0)[0]}{torch.cuda.get_device_capability(0)[1]}): "
        f"{detail}"
    )


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


def fused_gated_residual(x, y, m):
    """Fused x + m*y ((B,1,F) gate); None if unavailable.

    NOT wired into the model: enabled under gradient checkpointing it
    OOMed at batch 4 (save_for_backward(y) retains each block's output
    activation through the no-grad pass; the eager chain lets the
    checkpoint machinery reclaim it). A checkpoint-aware rewrite (e.g.
    recomputing m*y from cheaper saved tensors) must land before the
    wiring is reinstated.
    """
    if not _GATED_FUSE or not x.is_cuda:
        return None
    if (
        x.shape != y.shape or x.dtype != y.dtype
        or not x.is_contiguous() or not y.is_contiguous()
        or m.shape[0] != x.shape[0] or m.shape[-1] != x.shape[-1]
        or x.shape[-1] % 8 != 0
    ):
        _report_missing(
            "gated_residual_fusion",
            f"unsupported layout: x {tuple(x.shape)} {x.dtype} contig={x.is_contiguous()}, "
            f"y {tuple(y.shape)} {y.dtype}, m {tuple(m.shape)} {m.dtype}",
        )
        return None
    try:
        return _GatedResidual.apply(x, y, m)
    except Exception as e:
        _reraise_if_stop_recompute(e)
        _report_missing(
            "gated_residual_fusion",
            f"launch failed for {tuple(x.shape)} {x.dtype} m{tuple(m.shape)}: "
            f"{type(e).__name__}: {e}",
        )
        return None


# --------------------------------------------------------------------------
# FROST sm120 SDPA (cudnn-frontend DSL kernels)
# --------------------------------------------------------------------------
_frost_notice = False


if _FROST_OK:

    class _FrostSdpa(torch.autograd.Function):
        @staticmethod
        def forward(ctx, q, k, v, scale):
            out = _frost_fwd(q, k, v, scale_softmax=scale)
            o = out["o_tensor"]
            lse = out["lse_tensor"]
            ctx.save_for_backward(q, k, v, o, lse)
            ctx.scale = scale
            return o

        @staticmethod
        def backward(ctx, do):
            q, k, v, o, lse = ctx.saved_tensors
            grads = _frost_bwd(
                q, k, v, o, do.contiguous(), lse.unsqueeze(-1),
                scale_softmax=ctx.scale,
            )
            return grads["dq_tensor"], grads["dk_tensor"], grads["dv_tensor"], None


def fused_sdpa(q, k, v, scale):
    """sm120-native non-causal SDPA (FROST); None if unavailable.

    q: (B,Hq,L,D), k/v: (B,Hk,L,D), Hq % Hk == 0 (engine GQA, verified
    against torch enable_gqa), bf16/fp16 contiguous, D in {64,128}.
    """
    global _frost_notice
    if not _FROST_OK:
        return None
    try:
        out = _FrostSdpa.apply(q, k, v, scale)
        if not _frost_notice:
            _frost_notice = True
            print(
                "[krea2.kernels] cuDNN FROST sm120 SDPA active "
                "(fwd+bwd DSL kernels; KREA2_FROST_SDPA=0 restores torch SDPA)"
            )
        return out
    except Exception as e:
        _reraise_if_stop_recompute(e)
        _report_missing(
            "frost_sdpa",
            f"engine rejected q{tuple(q.shape)} k{tuple(k.shape)} {q.dtype}: "
            f"{type(e).__name__}: {e}",
        )
        return None
