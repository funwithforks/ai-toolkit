"""Autograd wrapper around torch's flash-attention varlen kernels (MOSS).

Mirror of ``minimax_h3/src/flash_varlen.py`` (campaign playbook:
docs/kernel_opt/FLASH_VARLEN_METHOD.md, audit: MOSS_PACKED_AUDIT.md).

The varlen ops (aten::_flash_attention_forward/_backward with cum_seq_*)
compute exact self-attention over a packed stream where each sample
attends to all of its own live rows -- identical semantics to MOSS's
``live_valid[:, None, None, :]`` key-padding mask, but the mask form
forces torch's SDPA dispatcher onto the mem-efficient backend whose
backward is several times the cost (flash rejects any non-null
attn_mask; upstream measured ~4x on this model's masked backward in
``88360c6``). MOSS pads on ANY caption-length variance, so every padded
bs>1 step pays that today; bs1 and sampling are never padded and keep
their exact dense-SDPA dispatch (this module is not in that path).

q/k/v: (nnz, H, D) packed b-major; cu: (B+1,) int32 offsets; max_len:
python int. Kernels are the same flash class torch's dense SDPA picks on
its own; the wrapper only carries the philox state through backward.
"""

import os

import torch

_VARLEN_OK = os.environ.get("MOSS_VARLEN_ATTN", "1") != "0"
_notice = False


class _FlashVarlen(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, cu, max_len, scale):
        out, lse, rng_state, unused, _ = torch.ops.aten._flash_attention_forward(
            q,
            k,
            v,
            cu,
            cu,
            max_len,
            max_len,
            0.0,
            False,
            False,
            scale=scale,
        )
        ctx.save_for_backward(q, k, v, out, lse, rng_state, unused)
        ctx.cu = cu
        ctx.max_len = max_len
        ctx.scale = scale
        return out

    @staticmethod
    def backward(ctx, grad_out):
        q, k, v, out, lse, rng_state, unused = ctx.saved_tensors
        dq, dk, dv = torch.ops.aten._flash_attention_backward(
            grad_out.contiguous(),
            q,
            k,
            v,
            out,
            lse,
            ctx.cu,
            ctx.cu,
            ctx.max_len,
            ctx.max_len,
            0.0,
            False,
            rng_state,
            unused,
            scale=ctx.scale,
        )
        return dq, dk, dv, None, None, None


def varlen_ok() -> bool:
    return _VARLEN_OK


def flash_varlen_sdpa(q, k, v, cu, max_len, scale):
    """(nnz,H,D) packed inputs -> (nnz,H,D) out; exact padded-mask semantics."""
    global _notice
    out = _FlashVarlen.apply(q, k, v, cu, max_len, scale)
    if not _notice:
        _notice = True
        print(
            "[moss-varlen] packed flash attention active (live rows on the "
            "flash kernels; MOSS_VARLEN_ATTN=0 restores the masked "
            "mem-efficient path)"
        )
    return out
