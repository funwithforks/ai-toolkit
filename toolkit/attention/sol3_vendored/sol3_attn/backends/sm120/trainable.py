"""Autograd entry: sol3 walk forward + backward for training.

Route tensors are never saved: buffers() hands out one shared
shape-keyed set, so every backward recomputes the (deterministic)
route and the global covariance from the saved q/k/v. Prefix query
rows are NOT computed here — the caller runs a plain SDPA branch on
them and concatenates, so autograd accumulates the prefix gradients
from dense attention and the video-row gradients from this Function.
"""
import math
import torch

from . import bwd as bwd_mod
from . import fused_attention as compact
from ._providers import sol3_kv_covariance
from .routing import DEFAULT_COVARIANCE, make_route


class Sol3WalkFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, prefix_tokens, gamma, bk, warps):
        b, t, h, d = q.shape
        if b != 1 or d != 128 or q.dtype != torch.bfloat16 or not q.is_cuda:
            raise ValueError("sol3 trainable expects contiguous CUDA bf16 "
                             "[1, T, H, 128] q/k/v")
        if not (q.is_contiguous() and k.is_contiguous() and v.is_contiguous()):
            raise ValueError("sol3 trainable requires contiguous q/k/v")
        if not 0 <= prefix_tokens < t:
            raise ValueError("prefix_tokens must be in [0, T)")
        from .attention import _PROFILE, _report_footprint
        cfg_bk, cfg_warps = _PROFILE.joint_launch if t >= 16385 else (bk, warps)
        route = make_route(q, k, v, {}, prefix_tokens)
        covariance = (sol3_kv_covariance.global_covariance(k, v, prefix_tokens)
                      if gamma else None)
        _report_footprint(q, route, covariance, gamma)
        out = torch.empty_like(q)
        lse = torch.empty((b, t, h), device=q.device, dtype=torch.float32)
        n = (t + 63) // 64
        compact.packed_attention_walk[(n, b * h)](
            q, k, v, route['kc'], route['vm'], route['flags'], route['bits'],
            out, q, q, covariance if covariance is not None else q, lse,
            t, n, h, d, prefix_tokens, d ** -0.5 * math.log2(math.e),
            cfg_bk, False, False, GAMMA=float(gamma),
            QST=q.stride(1), KST=k.stride(1), VST=v.stride(1), OST=out.stride(1),
            WRITE_LSE=True, num_warps=cfg_warps, num_stages=1)
        ctx.save_for_backward(q, k, v, out, lse)
        ctx.prefix_tokens = prefix_tokens
        ctx.gamma = gamma
        ctx.bk, ctx.warps = bk, warps
        return out

    @staticmethod
    def backward(ctx, dO):
        q, k, v, out, lse = ctx.saved_tensors
        prefix = ctx.prefix_tokens
        gamma = ctx.gamma
        route = make_route(q, k, v, {}, prefix)
        covariance = (sol3_kv_covariance.global_covariance(k, v, prefix)
                      if gamma else None)
        dq, dk, dv = bwd_mod.backward(dO, q, k, v, out, route, lse, covariance,
                                      gamma=gamma, prefix=prefix,
                                      bk=ctx.bk, warps=ctx.warps)
        return dq, dk, dv, None, None, None, None


def sol3_attn_trainable(q, k, v, *, prefix_tokens=0, gamma=DEFAULT_COVARIANCE,
                        dense=None, bk=64, warps=4):
    """Training-capable sol3 attention for the H3 main blocks.

    Video rows go through the sparse walk Function; prefix rows through
    a differentiable dense SDPA (or the provided callback). Returns a
    tensor with q's shape; all of q/k/v receive gradients.
    """
    if gamma not in (0.0, 0.25, 0.5, 1.0, 1.5):
        raise ValueError("Unsupported covariance coefficient")
    if not 0 <= prefix_tokens <= q.shape[1]:
        raise ValueError("prefix_tokens must be in [0, T]")
    if prefix_tokens == q.shape[1]:
        cb = dense or (lambda qp, kk, vv: torch.nn.functional.scaled_dot_product_attention(
            qp.transpose(1, 2), kk.transpose(1, 2), vv.transpose(1, 2)).transpose(1, 2))
        return cb(q, k, v).contiguous()
    out_video = Sol3WalkFn.apply(q, k, v, prefix_tokens, gamma, bk, warps)
    if prefix_tokens == 0:
        return out_video
    if dense is not None:
        out_pre = dense(q[:, :prefix_tokens], k, v)
    else:
        out_pre = torch.nn.functional.scaled_dot_product_attention(
            q[:, :prefix_tokens].transpose(1, 2), k.transpose(1, 2),
            v.transpose(1, 2)).transpose(1, 2)
    return torch.cat((out_pre, out_video[:, prefix_tokens:]), dim=1)
