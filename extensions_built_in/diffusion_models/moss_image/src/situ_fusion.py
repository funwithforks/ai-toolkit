"""Fused SiTU activation for the MOSS-Image MLP (autograd wrapper).

situ(h, g) = (4 tanh(g/4) sigmoid(g)) * (25 tanh(h/25))

Eager runs this as an eight-kernel elementwise chain over the
(N, ffn=8192) bf16 streams, twice per block per pass (main + text), and
backward replays a backward kernel per chain op. Fused: one kernel
each direction; backward saves only (hidden, gate) -- the two chunk
views it already had -- instead of four intermediates per call.

gate/hidden arrive as the two chunk views of ffn_up_gate's output:
row stride 2*ffn, column stride 1; the kernel reads the row strides
directly, no contiguous copy.

Fused-or-raise: enabled means the kernel runs or the call raises; no
silent downgrade. DEFAULT OFF (MOSS_SITU_FUSE=1) -- the full-step win
must be re-confirmed by a live clock measurement before rollout.

Numerics: eager rounds every chain intermediate to bf16; the kernel
computes in fp32 and rounds once (and backward once). One rounding
class, not a new gradient path.
"""

import os

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

_ENABLE = os.environ.get("MOSS_SITU_FUSE", "0") != "0"
_notice = False


@triton.jit
def _situ_fwd_kernel(h_ptr, g_ptr, o_ptr, n_cols, h_stride, g_stride,
                     BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < n_cols
    h = tl.load(h_ptr + row * h_stride + cols, mask=mask).to(tl.float32)
    g = tl.load(g_ptr + row * g_stride + cols, mask=mask).to(tl.float32)
    t = 4.0 * libdevice.tanh(g / 4.0) * tl.sigmoid(g)
    u = 25.0 * libdevice.tanh(h / 25.0)
    tl.store(o_ptr + row * n_cols + cols, (t * u).to(o_ptr.dtype.element_ty),
             mask=mask)


@triton.jit
def _situ_bwd_kernel(h_ptr, g_ptr, do_ptr, dh_ptr, dg_ptr, n_cols,
                     h_stride, g_stride, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < n_cols
    h = tl.load(h_ptr + row * h_stride + cols, mask=mask).to(tl.float32)
    g = tl.load(g_ptr + row * g_stride + cols, mask=mask).to(tl.float32)
    do = tl.load(do_ptr + row * n_cols + cols, mask=mask).to(tl.float32)
    th = libdevice.tanh(h / 25.0)
    tg = libdevice.tanh(g / 4.0)
    sg = tl.sigmoid(g)
    u = 25.0 * th
    t = 4.0 * tg * sg
    # d(25 tanh(h/25))/dh = sech^2(h/25)
    dh = do * t * (1.0 - th * th)
    # dT/dg for T(g) = 4 tanh(g/4) sigmoid(g):
    # T' = sech^2(g/4) sigma(g) + 4 tanh(g/4) sigma'(g)
    dgate = sg * (1.0 - tg * tg) + 4.0 * tg * sg * (1.0 - sg)
    dg = do * u * dgate
    # grad outputs are contiguous (N, n_cols); only INPUTS carry strides
    tl.store(dh_ptr + row * n_cols + cols, dh.to(dh_ptr.dtype.element_ty),
             mask=mask)
    tl.store(dg_ptr + row * n_cols + cols, dg.to(dg_ptr.dtype.element_ty),
             mask=mask)


def _grid(n_rows, n_cols, block):
    return (n_rows, triton.cdiv(n_cols, block))


class _SituFused(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, gate):
        n_rows, n_cols = hidden.shape
        out = torch.empty(hidden.shape, device=hidden.device,
                          dtype=hidden.dtype)
        block = 1024
        _situ_fwd_kernel[_grid(n_rows, n_cols, block)](
            hidden, gate, out, n_cols, hidden.stride(0), gate.stride(0),
            BLOCK=block,
        )
        ctx.save_for_backward(hidden, gate)
        ctx.strides = (hidden.stride(0), gate.stride(0))
        return out

    @staticmethod
    def backward(ctx, grad_out):
        hidden, gate = ctx.saved_tensors
        n_rows, n_cols = hidden.shape
        dh = torch.empty(hidden.shape, device=hidden.device,
                         dtype=hidden.dtype)
        dg = torch.empty(gate.shape, device=gate.device,
                         dtype=gate.dtype)
        block = 1024
        _situ_bwd_kernel[_grid(n_rows, n_cols, block)](
            hidden, gate, grad_out, dh, dg, n_cols,
            ctx.strides[0], ctx.strides[1], BLOCK=block,
        )
        return dh, dg


def situ_fused(hidden, gate):
    global _notice
    if hidden.dim() != 2 or hidden.stride(1) != 1 or gate.stride(1) != 1:
        raise RuntimeError(
            "[moss-situ] fused path needs row-strided (N, C) views with "
            f"contiguous columns, got {tuple(hidden.shape)}/"
            f"{hidden.stride()} -- fix the caller, no silent fallback"
        )
    out = _SituFused.apply(hidden, gate)
    if not _notice:
        _notice = True
        print(
            "[moss-situ] fused SiTU active (MOSS_SITU_FUSE=1; eager "
            "eight-kernel chain returns at MOSS_SITU_FUSE=0)"
        )
    return out


def situ(hidden, gate):
    """The single SiTU entry point for both block flavors."""
    if _ENABLE:
        return situ_fused(hidden, gate)
    return (4.0 * torch.tanh(gate / 4.0) * torch.sigmoid(gate)) * (
        25.0 * torch.tanh(hidden / 25.0)
    )


def situ_fuse_ok() -> bool:
    return _ENABLE
