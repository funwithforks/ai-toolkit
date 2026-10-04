"""Fused Triton RoPE for minimax_h3 (one kernel per tensor per pass).

Mirrors apply_rotary_emb in src/transformer.py exactly for the H3 layout:
x (B, S, H, D) contiguous; cos/sin (B, S, rot) float32 with the duplicated
half layout (cos[..., :rot/2] == cos[..., rot/2:]), rotating the leading
``rot`` channels by the half-rotate rule and passing the tail through.

    y_j = cos_j * x_j + sign_j * sin_j * x_partner(j)
    sign_j = -1 for j < rot/2 (rotate out the partner), +1 for
    rot/2 <= j < rot, 0 past rot (pass-through, cos=1/sin=0 masked in).

The backward is the adjoint: dx_j = cos_j * dy_j - sign_j *
sin_partner(j) * dy_partner(j) -- the same kernel with the sign negated and
sin read at the partner index (the forward reads sin at j, which is exactly
what the torch path does; no duplicated-cos/sin layout is assumed).

Math is done in fp32 and rounded once to the input dtype (the torch path
rounds after every elementwise op, so results differ by <= a few bf16 ulps;
the analytic adjoint makes the backward exact for whatever forward was
actually computed). H3_ROPE_FUSE=0 restores the torch path.
"""

import os

import torch

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except Exception:
    _HAS_TRITON = False

_FUSE_OK = _HAS_TRITON and torch.cuda.is_available() and os.environ.get(
    "H3_ROPE_FUSE", "1"
) != "0"
_printed = False

if _HAS_TRITON:

    @triton.jit
    def _rope_kernel(
        x_ptr,
        y_ptr,
        c_ptr,
        s_ptr,
        n_rows,
        H: tl.constexpr,
        ROT: tl.constexpr,
        HALF: tl.constexpr,
        D: tl.constexpr,
        SIGN: tl.constexpr,
    ):
        row = tl.program_id(0)
        j = tl.arange(0, D)
        x = tl.load(x_ptr + row * D + j).to(tl.float32)
        in_rot = j < ROT
        partner = tl.where(j < HALF, j + HALF, j - HALF)
        xp = tl.load(x_ptr + row * D + partner, mask=in_rot, other=0.0).to(tl.float32)
        bh = row // H
        cj = tl.load(c_ptr + bh * ROT + j, mask=in_rot, other=1.0).to(tl.float32)
        s_off = partner if SIGN < 0 else j
        sj = tl.load(s_ptr + bh * ROT + s_off, mask=in_rot, other=0.0).to(tl.float32)
        sgn = tl.where(j < HALF, -1.0, 1.0) * SIGN
        y = tl.where(in_rot, x * cj + sgn * xp * sj, x)
        tl.store(y_ptr + row * D + j, y.to(y_ptr.dtype.element_ty))

    def _launch(x4, cos, sin, sign):
        b, s, h, d = x4.shape
        rot = cos.shape[-1]
        y = torch.empty_like(x4)
        n_rows = b * s * h
        _rope_kernel[(n_rows,)](
            x4,
            y,
            cos,
            sin,
            n_rows,
            H=h,
            ROT=rot,
            HALF=rot // 2,
            D=d,
            SIGN=sign,
            num_warps=4,
        )
        return y


class _RopeApply(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, cos, sin):
        assert not cos.requires_grad and not sin.requires_grad
        ctx.save_for_backward(cos, sin)
        return _launch(x.contiguous(), cos.contiguous(), sin.contiguous(), 1)

    @staticmethod
    def backward(ctx, dy):
        cos, sin = ctx.saved_tensors
        return _launch(dy.contiguous(), cos, sin, -1), None, None


def fused_rope(x, cos, sin):
    """Returns None when the fused path cannot serve this call."""
    global _FUSE_OK, _printed
    if not _FUSE_OK or x.dim() != 4 or not x.is_cuda or x.dtype not in (
        torch.bfloat16,
        torch.float16,
        torch.float32,
    ):
        return None
    rot = cos.shape[-1]
    if rot % 2 or rot > x.shape[-1] or cos.shape[:2] != x.shape[:2]:
        return None
    try:
        out = _RopeApply.apply(x, cos, sin)
    except Exception as e:
        from toolkit.util.kernel_report import report_missing

        report_missing("h3_rope_fuse", f"{type(e).__name__}: {e}", tag="minimax_h3")
        return None
    if not _printed:
        _printed = True
        print(
            "[h3-rope] fused Triton RoPE active (one kernel per q/k; "
            "H3_ROPE_FUSE=0 restores the torch path)"
        )
    return out
