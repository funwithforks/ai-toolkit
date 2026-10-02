"""cuDNN FROST sm120 attention wrapper (model-agnostic).

torch's SDPA on consumer Blackwell (sm120) runs a generated sm120 fprop but
an sm80 WMMA backward. cudnn-frontend >= 1.28 ships sm120 DSL kernels for
BOTH directions (FROST) that torch will not select; this module wraps them
as one autograd.Function. Numerics are the standard cudnn backward class:
non-deterministic two-kernel dQK route, bf16 grad rel noise up to ~1e-2
vs the sm80 path on real shapes (verify quality end-to-end, not bitwise).

Knob: FROST_SDPA=0 (or the legacy KREA2_FROST_SDPA=0) restores torch SDPA.
A missing package disables this path loudly at import and is NOT fatal.

PACKED RAGGED CONTRACT (learned the hard way; do not "clean up"):
  * Ragged mode is expressed as per-sample int32 lengths (rows at/beyond
    the length are excluded from softmax). The LSE rows past the length
    carry the engine's -inf SKIP SENTINEL: zeroing them makes backward
    recompute exp(score - 0) for the skipped rows, which overflows to NaN
    at real post-modulation magnitudes (q/k reach +-600 legally). lse is
    allocated torch.empty -- keep the sentinel, never write lse.
  * Excluded OUTPUT rows (torch.empty garbage -> 0*NaN poisons dW) are
    zeroed deterministically: o beyond q_len in forward; dq/dk/dv beyond
    their lengths in backward. Small per-sample slice fills: one broadcast
    masked_fill over the whole tensor measured slower (full-tensor pass
    per call vs B tail writes).
  * lens_cpu (host list) must be copied ONCE per model forward by the
    caller. A .cpu() inside this Function drains the pipeline once per
    block call.
  * Callers that pass lengths DROP their mask on the promise exclusion
    happens: if fused_sdpa returns None with lens set, the caller must
    raise, never fall back to unmasked attention.
"""

import os

import torch

from ..util.kernel_report import report_missing, reraise_if_stop_recompute

FROST_OK = (
    os.environ.get("FROST_SDPA", "1") != "0"
    and os.environ.get("KREA2_FROST_SDPA", "1") != "0"
)
if FROST_OK:
    try:
        os.environ.setdefault("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", "1")
        from cudnn.sdpa.bwd import sdpa_bwd_wrapper_dsl_sm120 as _frost_bwd
        from cudnn.sdpa.fwd import sdpa_fwd_wrapper_dsl_sm120 as _frost_fwd
    except Exception as _frost_e:
        FROST_OK = False
        print(
            f"[frost] sm120 attention package unavailable ({_frost_e}); "
            f"callers fall back to torch SDPA"
        )

_notice = False


def _zero_rows_beyond(t, lens_cpu):
    n_rows = t.shape[2]
    for b, n in enumerate(lens_cpu):
        if n < n_rows:
            t[b, :, n:] = 0


class FrostSdpa(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, scale, q_lens=None, kv_lens=None, lens_cpu=None):
        if q_lens is None:
            out = _frost_fwd(q, k, v, scale_softmax=scale)
        else:
            # the ragged kernels read base pointers/strides directly;
            # callers hand us transpose views
            q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
            out = _frost_fwd(
                q,
                k,
                v,
                scale_softmax=scale,
                seq_q_lens=q_lens,
                seq_kv_lens=kv_lens,
            )
        o = out["o_tensor"]
        lse = out["lse_tensor"]
        if q_lens is not None:
            if lens_cpu is None:
                lens_cpu = q_lens.detach().cpu().tolist()
            ctx.lens = (tuple(lens_cpu), tuple(lens_cpu))
            ctx.q_lens = q_lens
            ctx.kv_lens = kv_lens
            _zero_rows_beyond(o, lens_cpu)
            # lse KEEPS its -inf sentinel past the length (module docstring)
        else:
            ctx.lens = None
        ctx.save_for_backward(q, k, v, o, lse)
        ctx.scale = scale
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, lse = ctx.saved_tensors
        if ctx.lens is None:
            grads = _frost_bwd(
                q, k, v, o, do.contiguous(), lse.unsqueeze(-1),
                scale_softmax=ctx.scale,
            )
        else:
            grads = _frost_bwd(
                q, k, v, o, do.contiguous(), lse.unsqueeze(-1),
                scale_softmax=ctx.scale,
                seq_q_lens=ctx.q_lens,
                seq_kv_lens=ctx.kv_lens,
            )
        dq = grads["dq_tensor"]
        dk = grads["dk_tensor"]
        dv = grads["dv_tensor"]
        if ctx.lens is not None:
            lens_cpu, kv_cpu = ctx.lens
            _zero_rows_beyond(dq, lens_cpu)
            _zero_rows_beyond(dk, kv_cpu)
            _zero_rows_beyond(dv, kv_cpu)
        return dq, dk, dv, None, None, None, None


def fused_sdpa(q, k, v, scale, q_lens=None, kv_lens=None, lens_cpu=None):
    """sm120-native non-causal SDPA (FROST); None if unavailable.

    q: (B,Hq,L,D), k/v: (B,Hk,L,D), Hq % Hk == 0 (engine GQA, verified
    against torch enable_gqa), bf16/fp16, D in {64,128}.
    q_lens/kv_lens: optional (B,) int32 per-sample lengths for the packed
    ragged layout (rows at/beyond the length are excluded from attention
    and returned zeroed); see the ragged contract in the module docstring.
    """
    global _notice
    if not FROST_OK:
        return None
    try:
        out = FrostSdpa.apply(q, k, v, scale, q_lens, kv_lens, lens_cpu)
        if not _notice:
            _notice = True
            print(
                "[frost] cuDNN FROST sm120 SDPA active "
                "(fwd+bwd DSL kernels; FROST_SDPA=0 restores torch SDPA)"
            )
        return out
    except Exception as e:
        reraise_if_stop_recompute(e)
        report_missing(
            "frost_sdpa",
            f"engine rejected q{tuple(q.shape)} k{tuple(k.shape)} {q.dtype}: "
            f"{type(e).__name__}: {e}",
            tag="frost",
        )
        return None
