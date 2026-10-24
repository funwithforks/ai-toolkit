"""GPU chunk rule for the Qwen3.5 gated-delta (linear-attention) layers.

transformers' modeling_qwen3_5 ships no usable default: it needs
flash-linear-attention / causal-conv1d (not installed, not being added) and
its bundled loop-heavy python path costs ~60% of all CPU during moss_image
embedding caching (24 of 32 tower layers are linear-attention; measured with
a sampling profiler at 3.5-4 captions/s, 15% gpu utilization).

This module replaces that path at load. Math:

  * The chunk transition matrix inverse (I - M)^-1 is a finite geometric
    series -- M is strictly lower triangular, so it terminates at
    chunk_size terms -- evaluated as log2(chunk_size) batched bf16 doubling
    steps, instead of the 63-iteration per-layer python loop.
  * Intra-chunk attention is one batched matmul over ALL chunks up front.
  * The cross-chunk recurrence runs fully batched with every elementwise
    factor hoisted out of the loop.

Dtype policy: the I/O boundary is the model's own dtype (bf16 as shipped),
but the chunk matmuls and the recurrent carry run in fp32 INSIDE the rule
and are cast back on output. Measured reason (this repo, packed-caption
branch): the rule's bf16 batched matmuls are not batch-shape invariant --
row 0 of an identical tensor differs bitwise at bs 1 vs bs 5 (cuBLAS algo
selection) -- and the delta-rule carry amplifies that reduction-order
noise over the tower's 24 GDN layers to an end-to-end worst cosine of
0.957 when captions are encoded in bucketed rows (n_b > 1) instead of one
row per caption. fp32 rule-internal matmuls put the injection below the
bf16 quantisation floor (acceptance: packed-vs-serial worst cosine >= 0.98,
mean >= 0.995). The decay chain (``g``) still arrives fp32 from the model's
own layer. Side effect to carry: the rule's internal temporaries are 2x
their bf16 size, so per-model bytes/token budgets are measured WITH this
rule in place (see moss text_cache_bytes_per_token).

Installed at construction (see install_qwen35_fast_path), so every load path
gets it.
"""

import torch

# Optional activation accounting for the per-model batch-size tooling.
# None (default) = zero cost on every normal path. Set the module attribute
# to a list and each fast_chunk_gated_delta_rule call appends
# (site, nbytes) for the activation objects it allocates.
ACTIVITY_PROBE = None


def _probe(site: str, *tensors) -> None:
    if ACTIVITY_PROBE is not None:
        ACTIVITY_PROBE.append(
            (
                site,
                sum(t.numel() * t.element_size() for t in tensors if t is not None),
            )
        )


def _l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv_norm


def fast_chunk_gated_delta_rule(
    query,
    key,
    value,
    g,
    beta,
    chunk_size=64,
    initial_state=None,
    output_final_state=False,
    use_qk_l2norm_in_kernel=False,
):
    """(B, L, H, D) in, (B, L, H, D) out -- the tower's own dtype at the
    boundary, fp32 for every matmul and the carry inside (see module
    docstring for why)."""
    wd = query.dtype
    rd = torch.float32
    query, key, value = [x.to(rd) for x in (query, key, value)]
    if use_qk_l2norm_in_kernel:
        query = _l2norm(query, dim=-1, eps=1e-6)
        key = _l2norm(key, dim=-1, eps=1e-6)
    query, key, value = [x.transpose(1, 2).contiguous() for x in (query, key, value)]
    # beta and g arrive from the model's own layer (bf16 sigmoid / fp32 decay
    # chain); both continue in fp32 -- the boundary casts back to wd happen
    # once, on the output.
    beta = beta.transpose(1, 2).contiguous().to(rd)
    g = g.transpose(1, 2).contiguous().to(rd)

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query = torch.nn.functional.pad(query, (0, 0, 0, pad_size))
    key = torch.nn.functional.pad(key, (0, 0, 0, pad_size))
    value = torch.nn.functional.pad(value, (0, 0, 0, pad_size))
    beta = torch.nn.functional.pad(beta, (0, pad_size))
    g = torch.nn.functional.pad(g, (0, pad_size))
    total_sequence_length = sequence_length + pad_size
    query = query * (k_head_dim**-0.5)

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    _probe("gdn_cast_pad", query, key, value, beta, g, v_beta, k_beta)
    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1])
        for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)

    # chunk decay (per-chunk local cumsum)
    g = g.cumsum(dim=-1)
    decay_mask = (g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp()
    _probe("gdn_decay_mask", decay_mask)

    # UT transform: T = (I - M)^-1 for the STRICT-lower transition matrix M
    # (intra-chunk attention below keeps its diagonal -- different mask).
    causal = torch.tril(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device)
    )
    strict_lower = torch.tril(causal, diagonal=-1)
    m = ((k_beta @ key.transpose(-1, -2)) * -decay_mask).masked_fill(~strict_lower, 0)

    # UT inverse: M is strictly lower triangular, so the geometric series
    # I + M + M^2 + ... terminates at chunk_size terms and equals
    # (I - M)^-1 exactly -- the same forward-substitution result, evaluated
    # as log2(chunk_size) doubling steps instead of a loop.
    eye = torch.eye(chunk_size, dtype=m.dtype, device=m.device)
    t_matrix = eye + m  # 2 terms: I + M
    e = m @ m  # M^k, k = 2
    k = 2
    while k < chunk_size:
        t_matrix = t_matrix + t_matrix @ e  # -> 2k terms
        e = e @ e
        k *= 2
    del e
    value = t_matrix @ v_beta
    k_cumdecay = t_matrix @ (k_beta * g.exp().unsqueeze(-1))
    _probe("gdn_ut", m, m, t_matrix, value, k_cumdecay)

    # intra-chunk attention for ALL chunks in one batched matmul
    # (the q @ k^T temporary + the masked copy are both this size)
    intra = (query @ key.transpose(-1, -2) * decay_mask).masked_fill(~causal, 0)
    _probe("gdn_intra", intra, intra)

    # cross-chunk factors, hoisted
    g_exp = g.exp()
    g_last = g[..., -1:]
    carry = g_last.exp().unsqueeze(-1)
    k_decayed_scale = (g_last - g).exp().unsqueeze(-1)

    last_recurrent_state = (
        torch.zeros(
            batch_size, num_heads, k_head_dim, v_head_dim, device=value.device, dtype=rd
        )
        if initial_state is None
        else initial_state.to(rd)
    )
    core_attn_out = torch.zeros_like(value)
    _probe("gdn_out", core_attn_out)
    for i in range(total_sequence_length // chunk_size):
        state = last_recurrent_state
        v_new = value[:, :, i] - (k_cumdecay[:, :, i] @ state)
        core_attn_out[:, :, i] = (
            (query[:, :, i] * g_exp[:, :, i].unsqueeze(-1)) @ state
            + intra[:, :, i] @ v_new
        )
        last_recurrent_state = state * carry[:, :, i] + (
            key[:, :, i] * k_decayed_scale[:, :, i]
        ).transpose(-1, -2) @ v_new
        _probe("gdn_loop_iter", v_new, last_recurrent_state)

    if not output_final_state:
        last_recurrent_state = None
    elif last_recurrent_state is not None:
        last_recurrent_state = last_recurrent_state.to(wd)
    core_attn_out = core_attn_out.reshape(
        core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1]
    )
    core_attn_out = core_attn_out[:, :, :sequence_length]
    # boundary cast: fp32 stays inside the rule, the tower only ever sees
    # its own dtype back
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(wd)
    _probe("gdn_out", core_attn_out)
    return core_attn_out, last_recurrent_state


def install_qwen35_fast_path(model) -> int:
    """Bind the chunk rule on every Qwen3_5GatedDeltaNet module of a loaded
    tower. The module's __init__ leaves ``self.chunk_gated_delta_rule``
    unbound-then-default, so an instance-level swap at construction is the
    whole install -- weights and every other path are untouched. Returns the
    number of layers swapped."""
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet

    count = 0
    for module in model.modules():
        if isinstance(module, Qwen3_5GatedDeltaNet):
            module.chunk_gated_delta_rule = fast_chunk_gated_delta_rule
            count += 1
    return count
