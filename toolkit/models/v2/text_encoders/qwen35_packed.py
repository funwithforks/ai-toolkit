"""Padding-free (packed) caption forwards for the Qwen3.5 text-only tower.

The padded cache pass feeds (B, L_max) rows: every caption is encoded to
the group's longest member (quadratic attention waste, a measured 4.3 GB
single-tensor spike at bs 207). Packing keeps ONE packed token stream
(1, total_tokens) with NO padding anywhere: the token-local layers (norms,
MLPs, all projections) run once over the whole stream -- they are per-token
math, so batch shape cannot change what a token sees -- and the two
token-MIXING layer types (8 full-attention + 24 gated-delta) run per caption
on its EXACT slice of the stream, one (1, L_i, C) row through each module's
own forward (attention with attention_mask=None for the per-row causal
fast path; the chunk rule pads internally exactly as the single-caption
encode does).

That per-caption row loop is the measured correctness floor, not a hedge.
Measured vs the serial encode (worst cosine over real datasets):
1.000000 (bitwise) over a 207-caption / 31.8k-token stream, but 0.9991
when short captions ride a small mixed stream -- the stream-level
token-local GEMMs pick batch-shape-dependent kernels, so some captions
get a 1-ulp input flip that the delta-rule carries can amplify to the
1e-3..1e-4 range. Treat bitwise as the observed best case, not a
guarantee, and never judge alternative mixers against anything looser
than the exact-row path in the SAME stream shape. Batched/bucketed
mixers were measured to diverge materially (worst 0.95-0.98): bf16
kernel selection inside the chunk rule and SDPA flips with the batch
dim. Vectorized boundary-safe mixers (torch's built-in varlen_attn for
attention; a seq_idx-masked chunk rule for GDN) are the speed path
later; the win over serial today is the fused token-local GEMMs and
zero pad waste.

Position ids restart per segment at the model entry point (the tower's mrope
expands 2-D position_ids itself), so the packed stream's cos/sin are
per-caption-correct by construction (verified bitwise).

Forward-only by design (TE embedding pass, no KV cache): packed mode with a
cache raises loudly instead of producing wrong states. Inert when no packed
plan is active: every module calls straight through to its original forward,
so sampling and any other Qwen3.5 use keep today's exact path.

Installed per loaded tower by install_qwen35_packed_path (instance-level
forward wrappers, bindings only, never weights) -- see qwen35.py.
"""

import contextlib
import itertools

import torch

# The active packed plan for the tower (single-threaded inference use:
# exactly one packed forward runs at a time, set by packed_mode()).
_CTX = None


class PackedPlan:
    """Segment boundaries for one packed forward: lengths + prefix sums."""

    def __init__(self, lengths, device=None):
        self.lengths = [max(int(L), 1) for L in lengths]
        self.cu = [0] + list(itertools.accumulate(self.lengths))


@contextlib.contextmanager
def packed_mode(lengths, device=None):
    """Activate a packed forward for lengths (per-caption token counts)."""
    global _CTX
    if _CTX is not None:
        raise RuntimeError("nested packed_mode is not supported")
    _CTX = PackedPlan(lengths, device=device)
    try:
        yield _CTX
    finally:
        _CTX = None


def _make_gdn_forward(module):
    original = module.forward

    def packed_gdn_forward(hidden_states, cache_params=None, attention_mask=None):
        if _CTX is None:
            return original(
                hidden_states, cache_params=cache_params, attention_mask=attention_mask
            )
        if cache_params is not None:
            raise RuntimeError(
                "packed Qwen3.5 forward is forward-only (no KV/recurrent cache)"
            )
        # per-caption exact rows; boundaries are segment loops, never
        # padding (see module docstring for the measured numbers)
        out = torch.empty_like(hidden_states)
        for seg, n in enumerate(_CTX.lengths):
            sl = slice(_CTX.cu[seg], _CTX.cu[seg + 1])
            o = original(hidden_states[0:1, sl], cache_params=None, attention_mask=None)
            out[0, sl] = o[0]
        return out

    return packed_gdn_forward


def _make_attn_forward(module):
    original = module.forward

    def packed_attn_forward(
        hidden_states,
        position_embeddings=None,
        attention_mask=None,
        past_key_values=None,
        **kwargs,
    ):
        if _CTX is None:
            return original(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                **kwargs,
            )
        if past_key_values is not None:
            raise RuntimeError("packed Qwen3.5 forward is forward-only (no KV cache)")
        # per-caption exact rows; cos/sin are plain slices of the packed
        # stream's rotary tables (segments are contiguous, positions
        # restart per segment)
        cos, sin = position_embeddings
        cos0, sin0 = cos[0], sin[0]
        out = torch.empty_like(hidden_states)
        for seg, n in enumerate(_CTX.lengths):
            sl = slice(_CTX.cu[seg], _CTX.cu[seg + 1])
            o, _ = original(
                hidden_states[0:1, sl],
                position_embeddings=(cos0[sl][None], sin0[sl][None]),
                attention_mask=None,
                past_key_values=None,
                **kwargs,
            )
            out[0, sl] = o[0]
        return out, None

    return packed_attn_forward


def install_qwen35_packed_path(model) -> int:
    """Wrap the token-mixing forwards of a loaded Qwen3.5 text tower for
    packed caption mode. Inert (straight through) unless packed_mode() is
    active on this thread. Returns the number of layers wrapped."""
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        Qwen3_5Attention,
        Qwen3_5GatedDeltaNet,
    )

    if getattr(model, "_qwen35_packed_ready", False):
        return 0
    count = 0
    for module in model.modules():
        if isinstance(module, Qwen3_5GatedDeltaNet):
            module.forward = _make_gdn_forward(module)
            count += 1
        elif isinstance(module, Qwen3_5Attention):
            module.forward = _make_attn_forward(module)
            count += 1
    model._qwen35_packed_ready = count > 0
    return count
