"""Fused split-RoPE (NeoX half-split) for LTX-2.5, replacing the eager
apply_split_rotary_emb chain (diffusers transformer_ltx2.py:46-84).

Eager cost per application (video q/k at [1, 10080, 4096]): reshape + swapaxes
views, fp32 upcast copy, cos mul (fp32), two addcmul_ (fp32), reshape copy,
bf16 downcast -- 5+ launches forward, ~6 nodes back, on every rope'd tensor,
and gradient checkpointing replays the forward inside backward. Measured
census: rope is ~4.6k launches/step (6 tensors/block x 48 blocks x ~16).

This kernel:
  - reads (x1, x2, cos, sin) once per token-head, fp32 math, one bf16 store
    per half -> 1 launch forward, 1 launch backward (same kernel, sin
    negated: dx1 = dy1*cos + dy2*sin, dx2 = dy2*cos - dy1*sin)
  - handles LTX's per-head cos natively (strides over [B, H, T, r]); no
    per-head duplication or materialization of any kind
  - operates on the physical (b, t, h, d) storage of the flat training
    layout [B, T, H*d] via strides -- zero layout copies in either pass
  - cos/sin stay non-differentiable: they are leaf buffers (fp32 from the
    fp64-widened exponent grid); backward returns no gradient for them
  - numerics vs eager: BITWISE-equal forward and backward, verified at all
    7 live call signatures (fwd signature dump + per-signature live-tuple
    backward replay) and in the isolated parity harness: fwd mirrors eager's
    mul-then-addcmul CUDA fma via explicit tl.fma; bwd keeps products
    unfused (enable_fp_fusion=False) to mirror autograd's two separately
    rounded fp32 products + fp32 accumulate, one bf16 store. Parity test:
    docs/kernel_opt/tests/ltx2_rope_fused_parity.py (campaign-local).

Shape contract matches eager: x is [B, T, dim] with 4-D cos/sin
[B, H, T, r] (reshaped/swapaxes exactly as eager does), or already-4-D x.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _split_rope_kernel(
    x_ptr, o_ptr, cos_ptr, sin_ptr,
    sxb, sxh, sxt,               # x strides for batch/head/token (elems)
    soxb, soxh, soxt,            # out strides
    scb, sch, sct,               # cos/sin token strides (sin shares layout)
    T,
    R: tl.constexpr,             # half head-dim (real elements)
    R_POW2: tl.constexpr,        # arange width (power of 2 >= R)
    BLOCK_T: tl.constexpr,
    BACKWARD: tl.constexpr,
):
    pid_t = tl.program_id(0)
    h = tl.program_id(1)         # grid axis1 = heads
    b = tl.program_id(2)         # grid axis2 = batch
    offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    tmask = offs_t < T
    cols = tl.arange(0, R_POW2)
    cmask = cols < R
    mask = tmask[:, None]

    x_base = x_ptr + b * sxb + h * sxh
    o_base = o_ptr + b * soxb + h * soxh
    c_base = cos_ptr + b * scb + h * sch
    s_base = sin_ptr + b * scb + h * sch

    offs_x = x_base + offs_t[:, None] * sxt + cols[None, :]
    offs_x2 = offs_x + R
    offs_o = o_base + offs_t[:, None] * soxt + cols[None, :]
    offs_o2 = offs_o + R
    offs_c = c_base + offs_t[:, None] * sct + cols[None, :]
    offs_s = s_base + offs_t[:, None] * sct + cols[None, :]

    m2 = mask & cmask[None, :]
    x1 = tl.load(offs_x, mask=m2, other=0.0).to(tl.float32)
    x2 = tl.load(offs_x2, mask=m2, other=0.0).to(tl.float32)
    c = tl.load(offs_c, mask=mask, other=0.0)
    s = tl.load(offs_s, mask=mask, other=0.0)

    if not BACKWARD:
        # mirror eager's rounding pattern exactly: eager is fp32 mul then
        # addcmul_ (CUDA fma): fma(-sin, x2, x1*c) / fma(sin, x1, x2*c)
        y1 = tl.fma(-s, x2, x1 * c)
        y2 = tl.fma(s, x1, x2 * c)
    else:
        # autograd's backward accumulates two separately-rounded fp32
        # products (mul-backward + mul-backward into an fp32 grad buffer);
        # keep muls unfused (launch sets enable_fp_fusion=False) so the
        # rounding pattern matches, then round the sum
        y1 = x1 * c + x2 * s
        y2 = x2 * c - x1 * s

    tl.store(offs_o, y1, mask=m2)
    tl.store(offs_o2, y2, mask=m2)


class _FusedSplitRope(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, cos, sin):
        # mirror eager layout handling (transformer_ltx2.py:50-56)
        x_dtype = x.dtype
        needs_reshape = False
        if x.ndim != 4 and cos.ndim == 4:
            b, h, t, r = cos.shape
            x4 = x.reshape(b, t, h, -1).swapaxes(1, 2)   # -> [B,H,T,D] view
            needs_reshape = True
        else:
            b, h, t, r = cos.shape
            x4 = x
        d = x4.shape[-1]
        assert d == 2 * r, f"head dim {d} != 2 * rope dim {r}"

        # output with x4's stride pattern: writes land in the physical
        # (b, t, h, d) order for the flat path -- zero layout copies
        out4 = torch.empty_strided(x4.shape, x4.stride(),
                                   dtype=x_dtype, device=x.device)
        BLOCK_T = 32
        grid = (triton.cdiv(t, BLOCK_T), h, b)
        _split_rope_kernel[grid](
            x4, out4, cos, sin,
            x4.stride(0), x4.stride(1), x4.stride(2),
            out4.stride(0), out4.stride(1), out4.stride(2),
            cos.stride(0), cos.stride(1), cos.stride(2),
            t,
            R=r, R_POW2=triton.next_power_of_2(r),
            BLOCK_T=BLOCK_T, BACKWARD=False,
            num_warps=4,
        )

        ctx.save_for_backward(cos, sin)
        ctx.needs_reshape = needs_reshape
        ctx.x_shape = x.shape
        if needs_reshape:
            return out4.swapaxes(1, 2).reshape(b, t, d * h)
        return out4

    @staticmethod
    def backward(ctx, go):
        cos, sin = ctx.saved_tensors
        b, h, t, r = cos.shape
        if ctx.needs_reshape:
            g4 = go.reshape(b, t, h, -1).swapaxes(1, 2)
        else:
            g4 = go.contiguous() if go.stride(3) != 1 else go
        gx = torch.empty_strided(g4.shape, g4.stride(),
                                 dtype=go.dtype, device=go.device)
        BLOCK_T = 32
        grid = (triton.cdiv(t, BLOCK_T), h, b)
        _split_rope_kernel[grid](
            g4, gx, cos, sin,
            g4.stride(0), g4.stride(1), g4.stride(2),
            gx.stride(0), gx.stride(1), gx.stride(2),
            cos.stride(0), cos.stride(1), cos.stride(2),
            t,
            R=r, R_POW2=triton.next_power_of_2(r),
            BLOCK_T=BLOCK_T, BACKWARD=True,
            num_warps=4, enable_fp_fusion=False,
        )
        if ctx.needs_reshape:
            gx = gx.swapaxes(1, 2).reshape(*ctx.x_shape)
        # cos/sin are leaf buffers: never a gradient (gate-checked)
        return gx, None, None


def fused_apply_split_rotary_emb(x, freqs):
    """Drop-in replacement for diffusers apply_split_rotary_emb (split
    rope_type only); same signature (x, (cos, sin))."""
    cos, sin = freqs
    return _FusedSplitRope.apply(x, cos, sin)
