"""Run the SM120 joint exact/representative kernel.

Vendored from sol3-attn 0.1.0 (Apache-2.0; see LICENSE and
THIRD_PARTY_NOTICES.md in the vendored root). Local rewrite: upstream's
per-head-chunk launch loop and its (batch*heads, blocks, tokens)
compaction index are replaced by the single-launch bitmap walk in
fused_attention.packed_attention_walk -- no ids buffer, no compaction
kernels, one launch over all heads. The protected prefix is one dense
call instead of a per-chunk loop. Per-head results are unchanged: each
kernel program owns one (batch, head) pair, so chunking was never a
math boundary (verified bitwise against the upstream ids path).

The launch tuple and route constants come from the package's only SM120
profile; they are inherited starting points pending this model's own
measured pass, never inherited verdicts.
"""

import torch

from ...profiles import PROFILES
from ..workspace import Workspace, count_launch
from . import fused_attention as compact
from ._providers import sol3_kv_covariance
from .routing import DEFAULT_COVARIANCE, make_route


_PROFILE = PROFILES["fast"]
_WORKSPACE = Workspace("fast", __package__ + ".")
_REPORTED = set()


def _report_footprint(q, route, covariance, gamma):
    # One-time route footprint per shape: the resident route tensors are
    # the sol3-vs-stock peak-memory delta, so they must be attributable
    # at a glance (768p verdict, docs/kernel_opt/sol3/SOL3.md).
    key = (tuple(q.shape), q.device)
    if key in _REPORTED:
        return
    _REPORTED.add(key)
    from ._providers import h3_mu_route
    nbytes = sum(x.numel() * x.element_size() for x in route.values()
                 if torch.is_tensor(x))
    for val in h3_mu_route.CACHE.values():
        tensors = val if isinstance(val, tuple) else (val,)
        nbytes += sum(x.numel() * x.element_size() for x in tensors
                      if torch.is_tensor(x))
    if covariance is not None:
        nbytes += covariance.numel() * covariance.element_size()
    print(f"  sol3 route footprint: {nbytes / 2**20:.1f} MiB resident "
          f"(shape {tuple(q.shape)}, cov_gamma={gamma})")


def forward(q, k, v, *, counters=None, config=None, prefix_tokens=0, dense=None):
    q, k, v = (x.contiguous() for x in (q, k, v))
    if not 0 <= prefix_tokens < q.shape[1] or (prefix_tokens and dense is None):
        raise ValueError("A protected prefix requires the native dense-query callback")
    cfg = dict(config or {})
    gamma = float(cfg.pop("cov_gamma", DEFAULT_COVARIANCE))
    if gamma not in (0.0, 0.25, 0.5, 1.0, 1.5):
        raise ValueError("Unsupported covariance coefficient")

    with _WORKSPACE.call(q, counters) as state:
        route = make_route(q, k, v, cfg, prefix_tokens)
        covariance = sol3_kv_covariance.global_covariance(k, v, prefix_tokens) if gamma else None
        _report_footprint(q, route, covariance, gamma)
        bk, warps = _PROFILE.joint_launch if state.optimized else (128, 8)
        out = compact.attention(
            q, k, v, route, prefix_tokens=prefix_tokens, dense=dense,
            bk=bk, warps=warps, covariance=covariance, cov_gamma=gamma,
        )
        count_launch("joint_walk")
        return out, route
