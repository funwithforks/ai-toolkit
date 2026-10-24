"""GPU fast path for the Qwen3.5 gated-delta (linear-attention) layers.

transformers' modeling_qwen3_5 falls back to ``torch_chunk_gated_delta_rule``
whenever flash-linear-attention / causal-conv1d are not installed (they are
not, and neither is being added). The fallback is loop-heavy Python: a
63-iteration UT-transform loop PER LAYER to build the chunk transition matrix,
plus per-chunk element work inside the recurrence. On moss_image embedding
caching that is ~60% of all CPU samples (py-spy) -- 24 of the 32 decoder
layers are linear-attention, so one 150-token forward issued ~8k tiny kernel
launches (measured 3.5-4 caption/s at 15% GPU util).

The loop is the forward substitution for the unit lower-triangular inverse:
running it on the strict-lower transition matrix ``M`` and adding I equals
``(I - M)^-1`` exactly (verified numerically on the real input domain --
l2-normalized k, sigmoid beta, exp-of-negative-cumsum decay -- to 3e-8 fp32).
One batched ``torch.linalg.solve_triangular`` (cuSOLVER trsm, prebuilt) does
the same work in a single launch. The per-chunk recurrence is inherently
sequential across chunks but stays fully batched (4 batched matmuls per
chunk, no elementwise work in the loop); at the ~150-token captions it is 3
iterations.

Semantics (chunk_size, fp32 state, l2norm, scale, right-padding behavior)
mirror the fallback exactly; equivalence is gated on a real TE forward
before and after installation, not on reasoning.
"""

import torch

# Optional activation accounting for the per-model batch-size probe
# (testing/te_batch_probe.py). None (default) = zero cost on every normal
# path. Set the module attribute to a list and each fast_chunk_gated_delta_rule
# call appends (site, nbytes) for the real activation objects it allocates.
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
    """Drop-in for modeling_qwen3_5.torch_chunk_gated_delta_rule, vectorized.

    Same signature, same math: within-chunk UT transform as a triangular
    solve instead of a 63-step loop, intra-chunk attention computed for all
    chunks in one batched matmul, then the cross-chunk state recurrence with
    all elementwise factors precomputed."""
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = _l2norm(query, dim=-1, eps=1e-6)
        key = _l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32)
        for x in (query, key, value, beta, g)
    ]

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
    mask = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device),
        diagonal=0,
    )

    # chunk decay (per-chunk local cumsum, exactly as the fallback)
    g = g.cumsum(dim=-1)
    decay_mask = (((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril())
    _probe("gdn_decay_mask", decay_mask)

    # UT transform: the fallback builds T = (I - M)^-1 row by row over 63
    # iterations (M = -(k_beta @ k^T * decay), strict lower); solve_triangular
    # computes the same inverse in one batched kernel.
    m = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
    eye = torch.eye(chunk_size, dtype=m.dtype, device=m.device)
    # eye.expand_as(m).clone() and the (eye - m) temporary are each m-sized
    t_matrix = torch.linalg.solve_triangular(
        eye - m, eye.expand_as(m).clone(), upper=False, unitriangular=True
    )
    value = t_matrix @ v_beta
    k_cumdecay = t_matrix @ (k_beta * g.exp().unsqueeze(-1))
    _probe("gdn_ut", m, m, m, t_matrix, value, k_cumdecay)

    # intra-chunk attention for ALL chunks in one batched matmul (the fallback
    # recomputes it per loop iteration)
    intra = (query @ key.transpose(-1, -2) * decay_mask).masked_fill(
        torch.triu(
            torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device),
            diagonal=1,
        ),
        0,
    )
    # (q @ k^T temporary + the out-of-place masked_fill copy are both this size)
    _probe("gdn_intra", intra, intra)
    # cross-chunk factors, precomputed once
    g_exp = g.exp()  # (b, h, nc, c)
    g_last = g[..., -1:]  # (b, h, nc, 1)
    carry = g_last.exp().unsqueeze(-1)  # (b, h, nc, 1, 1) state decay
    k_decayed_scale = (g_last - g).exp().unsqueeze(-1)  # (b, h, nc, c, 1)

    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim, device=value.device)
        if initial_state is None
        else initial_state.to(value)
    )
    core_attn_out = torch.zeros_like(value)
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
    _probe("gdn_out", core_attn_out)
    core_attn_out = core_attn_out.reshape(
        core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1]
    )
    core_attn_out = core_attn_out[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    _probe("gdn_out", core_attn_out)
    return core_attn_out, last_recurrent_state


def install_qwen35_fast_path(model) -> int:
    """Swap the gated-delta chunk rule on every Qwen3_5GatedDeltaNet module of
    a loaded tower. The fallback is bound per-instance in the module's
    __init__ (``self.chunk_gated_delta_rule = ...``), so an instance-level
    swap after load is the whole install -- weights and every other path are
    untouched. Returns the number of layers swapped."""
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet

    count = 0
    for module in model.modules():
        if isinstance(module, Qwen3_5GatedDeltaNet):
            module.chunk_gated_delta_rule = fast_chunk_gated_delta_rule
            count += 1
    return count
