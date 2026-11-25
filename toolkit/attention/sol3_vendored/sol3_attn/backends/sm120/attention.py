"""Run the FastH3 SM120 joint exact/representative kernel.

Vendored from sol3-attn 0.1.0 (Apache-2.0; see LICENSE and
THIRD_PARTY_NOTICES.md in the vendored root). Local modification: the
per-head-chunk launch loop is replaced by a single launch over all heads
whenever the compaction index fits the memory budget; the chunked loop
remains as the budgeted fallback. Per-head results are unchanged -- each
kernel program handles one (batch, head) pair, so the chunk size never
entering the math is what makes both paths equivalent.
"""

import math
import os

import torch
import triton

from ...profiles import PROFILES
from ..workspace import GIB, Workspace, available_memory, choose_plan, count_launch
from . import fused_attention as compact
from ._providers import sol3_compact, sol3_kv_covariance
from .routing import DEFAULT_COVARIANCE, make_route


_PROFILE = PROFILES["fast"]
_WORKSPACE = Workspace("fast", __package__ + ".")

# Upper bound on the (batch*heads, blocks, tokens) int32 compaction index.
# Above it, heads are processed in chunks sized to fit the budget. The
# env override is the standard A/B escape hatch.
_DEFAULT_IDS_BUDGET_BYTES = 512 * 1024 * 1024


def _ids_budget_bytes():
    raw = os.environ.get("SOL3_IDS_BUDGET_MB", "")
    try:
        return int(raw) * 1024 * 1024 if raw else _DEFAULT_IDS_BUDGET_BYTES
    except ValueError:
        return _DEFAULT_IDS_BUDGET_BYTES


def _head_slice(route, start, end, heads, blocks, dimension):
    selected = dict(route, b=1, h=end - start)
    for key in ("flags", "bits"):
        selected[key] = route[key].view(heads, blocks, blocks)[start:end]
    for key in ("kc", "vm"):
        selected[key] = route[key].view(heads, blocks, dimension)[start:end]
    return selected


def _launch_joint(qs, ks, vs, part, out, ids, count, cov, gamma, prefix_tokens,
                  bk, warps):
    """One packed_attention launch over the heads in ``part``."""
    batch, tokens, heads, dimension = qs.shape
    blocks = triton.cdiv(tokens, 64)
    compact.packed_attention[(blocks, batch * heads)](
        qs, ks, vs, part["kc"], part["vm"], part["flags"], ids, count,
        out, qs, qs, cov if cov is not None else qs,
        tokens, blocks, heads, dimension, prefix_tokens,
        dimension ** -0.5 * math.log2(math.e), bk, False, False,
        GAMMA=gamma, QST=qs.stride(1), KST=ks.stride(1),
        VST=vs.stride(1), OST=out.stride(1), IDST=ids.stride(1),
        num_warps=warps, num_stages=1,
    )


def forward(q, k, v, *, counters=None, config=None, prefix_tokens=0, dense=None):
    q, k, v = (x.contiguous() for x in (q, k, v))
    batch, tokens, heads, dimension = q.shape
    if not 0 <= prefix_tokens < tokens or (prefix_tokens and dense is None):
        raise ValueError("A protected FastH3 prefix requires the native dense-query callback")
    cfg = dict(config or {})
    gamma = float(cfg.pop("cov_gamma", DEFAULT_COVARIANCE))
    if gamma not in (0.0, 0.25, 0.5, 1.0, 1.5):
        raise ValueError("Unsupported FastH3 covariance coefficient")

    with _WORKSPACE.call(q, counters) as state:
        plan = choose_plan(
            tokens, heads, dimension, available_memory(q),
            preferred_heads=_PROFILE.long_heads, use_score_cache=False, budget_limit=24 * GIB,
        )
        route = make_route(q, k, v, cfg, prefix_tokens)
        covariance = sol3_kv_covariance.global_covariance(k, v, prefix_tokens) if gamma else None
        blocks = triton.cdiv(tokens, 64)
        out = torch.empty_like(q)
        bk, warps = _PROFILE.joint_launch if state.optimized else (128, 8)

        # The compaction index is (batch*heads, blocks, tokens) int32 when
        # sized by T; prepare sizes it by the largest per-query-block
        # selection total instead (an order of magnitude less at video
        # token counts) and returns None when even that exceeds the
        # budget -- then the heads are processed in chunks, which is
        # purely a memory bound: each kernel program owns one
        # (batch, head) pair, so chunking never enters the math.
        budget = _ids_budget_bytes()
        prepared = compact.prepare(q, route, cap_bytes=budget)
        if prepared is not None:
            ids, count = prepared
            _launch_joint(q, k, v, route, out, ids, count, covariance, gamma,
                          prefix_tokens, bk, warps)
            count_launch("joint_single")
            if prefix_tokens:
                # one dense call for all heads; the callback owns its layout
                out[:, :prefix_tokens] = dense(q[:, :prefix_tokens], k, v)
            return out, route
        single = False
        chunk = max(1, budget // max(1, batch * blocks * tokens * 4))
        for start in range(0, heads, chunk):
            end = min(heads, start + chunk)
            part = _head_slice(route, start, end, heads, blocks, dimension)
            cov = covariance.view(heads, dimension, dimension)[start:end] if covariance is not None else None
            qs, ks, vs = (x[:, :, start:end] for x in (q, k, v))
            if not plan.minimum_fits:
                for name in ("flags", "bits", "kc", "vm"):
                    part[name] = part[name].contiguous()
                out[:, :, start:end] = sol3_compact.attention(
                    qs.contiguous(), ks.contiguous(), vs.contiguous(), part,
                    prefix_tokens=prefix_tokens, dense=dense, bk=128, warps=8, stages=1,
                    covariance=cov, cov_gamma=gamma,
                )
                count_launch("compact_fallback")
                continue
            ids, count = compact.prepare(qs, part)
            _launch_joint(qs, ks, vs, part, out[:, :, start:end], ids, count,
                          cov, gamma, prefix_tokens, bk, warps)
            count_launch("joint")
            count_launch(f"joint_bk{bk}_warps{warps}")
        if prefix_tokens and plan.minimum_fits:
            for start in range(0, heads, 4):
                end = min(heads, start + 4)
                out[:, :prefix_tokens, start:end] = dense(
                    q[:, :prefix_tokens, start:end], k[:, :, start:end], v[:, :, start:end],
                )
        return out, route
