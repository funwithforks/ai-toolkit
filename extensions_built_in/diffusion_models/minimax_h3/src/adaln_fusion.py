"""Fused AdaLN modulation kernels for the H3 transformer blocks.

The eager modulation chain per block is:

    h = norm(x) * (1 + scale_tbl[idx].to(dt)) + shift_tbl[idx].to(dt)
    x = x + gate_tbl[idx].to(dt) * branch(h)

which materializes six (B, S, hidden) float32 table gathers plus six
bf16 casts per block per pass — ~2.5 GB of extra traffic for a table with
at most (unique_timesteps * modalities) <= 24 rows, and it is the source
of the MulBackward0/aten::add launch flood in the bs4 profile. These two
kernels fold the gather, the cast and the arithmetic into single passes:

    modulated_norm:    y = h * (1 + scale[idx]) + shift[idx]
    gated_residual:    out = x + gate[idx] * y

Tables stay float32 and are read row-wise through idx (int32); activations
are bf16. Arithmetic runs float32 and rounds once to bf16 on store, where
eager rounded the table rows to bf16 first; deviation is the one-ulp
elementwise class, not a new gradient path.

Backward note: activation grads (dh/dx/dy) are per-element and
deterministic. Table grads (d_scale/d_shift/d_gate) are row scatter-adds
accumulated with float32 atomics -> run-to-run ordering noise only, on
parameters that are themselves the sum of thousands of rows (same noise
class already accepted for the low-rank weight grads).

Set H3_ADALN_FUSE=0 to restore the eager path.
"""

import os

import torch
import triton
import triton.language as tl

_ENABLE = os.environ.get("H3_ADALN_FUSE", "1") != "0"
_notice = False


def _maybe_notice():
    global _notice
    if not _notice:
        _notice = True
        print(
            "[h3-adaln] fused AdaLN modulation active (gather+scale+shift "
            "and gate+residual in one kernel each; H3_ADALN_FUSE=0 restores "
            "the eager path)"
        )


@triton.jit
def _mod_norm_fwd(h_ptr, s_ptr, sh_ptr, idx_ptr, out_ptr, D, stride_t,
                  BLOCK: tl.constexpr):
    row = tl.program_id(0)
    t = tl.load(idx_ptr + row) * stride_t
    offs = tl.arange(0, BLOCK)
    for d0 in range(0, D, BLOCK):
        m = d0 + offs < D
        h = tl.load(h_ptr + row * D + d0 + offs, mask=m, other=0.0).to(tl.float32)
        s = tl.load(s_ptr + t + d0 + offs, mask=m, other=0.0)
        sh = tl.load(sh_ptr + t + d0 + offs, mask=m, other=0.0)
        y = h * (1.0 + s) + sh
        tl.store(out_ptr + row * D + d0 + offs, y.to(out_ptr.dtype.element_ty), mask=m)


@triton.jit
def _mod_norm_bwd(dy_ptr, h_ptr, s_ptr, idx_ptr, dh_ptr, ds_ptr, dsh_ptr, D,
                  stride_t, BLOCK: tl.constexpr):  # ds/dsh contiguous: stride D
    row = tl.program_id(0)
    t = tl.load(idx_ptr + row)
    offs = tl.arange(0, BLOCK)
    for d0 in range(0, D, BLOCK):
        m = d0 + offs < D
        dy = tl.load(dy_ptr + row * D + d0 + offs, mask=m, other=0.0).to(tl.float32)
        h = tl.load(h_ptr + row * D + d0 + offs, mask=m, other=0.0).to(tl.float32)
        s = tl.load(s_ptr + t * stride_t + d0 + offs, mask=m, other=0.0)
        tl.store(dh_ptr + row * D + d0 + offs,
                 (dy * (1.0 + s)).to(dh_ptr.dtype.element_ty), mask=m)
        tl.atomic_add(ds_ptr + t * D + d0 + offs, dy * h, mask=m)
        tl.atomic_add(dsh_ptr + t * D + d0 + offs, dy, mask=m)


class _ModulatedNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, h, scale_tbl, shift_tbl, idx):
        h2 = h.contiguous()
        idx32 = idx.to(torch.int32).contiguous()
        out = torch.empty_like(h2)
        D = h2.shape[-1]
        n_rows = h2.numel() // D
        _mod_norm_fwd[(n_rows,)](h2, scale_tbl, shift_tbl, idx32, out, D,
                                 scale_tbl.stride(0), BLOCK=2048, num_warps=8)
        ctx.save_for_backward(h2, scale_tbl, idx32)
        return out

    @staticmethod
    def backward(ctx, dy):
        h2, scale_tbl, idx32 = ctx.saved_tensors
        dy = dy.contiguous()
        D = h2.shape[-1]
        n_rows = h2.numel() // D
        dh = torch.empty_like(h2)
        ds = torch.zeros_like(scale_tbl, memory_format=torch.contiguous_format)
        dsh = torch.zeros_like(scale_tbl, memory_format=torch.contiguous_format)
        _mod_norm_bwd[(n_rows,)](dy, h2, scale_tbl, idx32, dh, ds, dsh, D,
                                 scale_tbl.stride(0), BLOCK=2048, num_warps=8)
        return dh, ds, dsh, None


@triton.jit
def _gate_res_fwd(x_ptr, g_ptr, y_ptr, idx_ptr, out_ptr, D, stride_t,
                  BLOCK: tl.constexpr):
    row = tl.program_id(0)
    t = tl.load(idx_ptr + row) * stride_t
    offs = tl.arange(0, BLOCK)
    for d0 in range(0, D, BLOCK):
        m = d0 + offs < D
        x = tl.load(x_ptr + row * D + d0 + offs, mask=m, other=0.0).to(tl.float32)
        y = tl.load(y_ptr + row * D + d0 + offs, mask=m, other=0.0).to(tl.float32)
        g = tl.load(g_ptr + t + d0 + offs, mask=m, other=0.0)
        tl.store(out_ptr + row * D + d0 + offs,
                 (x + g * y).to(out_ptr.dtype.element_ty), mask=m)


@triton.jit
def _gate_res_bwd(dy_ptr, y_ptr, g_ptr, idx_ptr, dx_ptr, dy_out_ptr, dg_ptr,
                  D, stride_t, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    t = tl.load(idx_ptr + row)
    offs = tl.arange(0, BLOCK)
    for d0 in range(0, D, BLOCK):
        m = d0 + offs < D
        dy = tl.load(dy_ptr + row * D + d0 + offs, mask=m, other=0.0).to(tl.float32)
        y = tl.load(y_ptr + row * D + d0 + offs, mask=m, other=0.0).to(tl.float32)
        g = tl.load(g_ptr + t * stride_t + d0 + offs, mask=m, other=0.0)
        tl.store(dx_ptr + row * D + d0 + offs, dy.to(dx_ptr.dtype.element_ty), mask=m)
        tl.store(dy_out_ptr + row * D + d0 + offs,
                 (g * dy).to(dy_out_ptr.dtype.element_ty), mask=m)
        tl.atomic_add(dg_ptr + t * D + d0 + offs, y * dy, mask=m)


class _GatedResidual(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, gate_tbl, y, idx):
        x2 = x.contiguous()
        y2 = y.contiguous()
        idx32 = idx.to(torch.int32).contiguous()
        out = torch.empty_like(x2)
        D = x2.shape[-1]
        n_rows = x2.numel() // D
        _gate_res_fwd[(n_rows,)](x2, gate_tbl, y2, idx32, out, D,
                                 gate_tbl.stride(0), BLOCK=2048, num_warps=8)
        ctx.save_for_backward(y2, gate_tbl, idx32)
        return out

    @staticmethod
    def backward(ctx, dy):
        y2, gate_tbl, idx32 = ctx.saved_tensors
        dy = dy.contiguous()
        D = y2.shape[-1]
        n_rows = y2.numel() // D
        dx = torch.empty_like(y2)
        dy_out = torch.empty_like(y2)
        dg = torch.zeros_like(gate_tbl, memory_format=torch.contiguous_format)
        _gate_res_bwd[(n_rows,)](dy, y2, gate_tbl, idx32, dx, dy_out, dg, D,
                                 gate_tbl.stride(0), BLOCK=2048, num_warps=8)
        return dx, dg, dy_out, None


def _fusable_or_die(h, scale_tbl, idx):
    """Enabled means fused: any condition that blocks the fused kernel is a
    hard error, never a silent downgrade to eager. Eager runs only via an
    explicit H3_ADALN_FUSE=0."""
    if not _ENABLE:
        return False
    why = None
    if not h.is_cuda:
        why = "activations not on cuda"
    elif h.dtype not in (torch.bfloat16, torch.float16):
        why = f"activation dtype {h.dtype}"
    elif scale_tbl.dtype != torch.float32:
        why = f"table dtype {scale_tbl.dtype}"
    elif scale_tbl.stride(1) != 1:
        why = "table column stride != 1"
    elif idx.dim() + 1 != h.dim():
        why = f"index rank {idx.dim()} vs activation rank {h.dim()}"
    if why is not None:
        raise RuntimeError(
            f"h3-adaln: fused modulation cannot run ({why}); this build "
            "never silently falls back. Set H3_ADALN_FUSE=0 to run the "
            "eager path explicitly."
        )
    return True


def modulated_norm(h, scale_tbl, shift_tbl, idx):
    """h * (1 + scale_tbl[idx]) + shift_tbl[idx], fused; eager fallback."""
    if _fusable_or_die(h, scale_tbl, idx):
        _maybe_notice()
        return _ModulatedNorm.apply(h, scale_tbl, shift_tbl, idx)
    dt = h.dtype
    return h * (1.0 + scale_tbl[idx].to(dt)) + shift_tbl[idx].to(dt)


def gated_residual(x, gate_tbl, y, idx):
    """x + gate_tbl[idx] * y, fused; eager fallback."""
    if _fusable_or_die(x, gate_tbl, idx):
        return _GatedResidual.apply(x, gate_tbl, y, idx)
    dt = x.dtype
    return x + gate_tbl[idx].to(dt) * y
