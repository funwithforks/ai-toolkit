"""Public forward inference and workspace management for Sol³."""

from functools import partial
import sys

from .profiles import PROFILES


def _validate(q, k, v, *, contiguous):
    import torch

    if any(not isinstance(x, torch.Tensor) for x in (q, k, v)):
        raise TypeError("Sol3 expects PyTorch tensors for q, k and v")
    if (
        q.ndim != 4
        or q.shape[0] != 1
        or q.shape[-1] != 128
        or not q.shape[1]
        or not q.shape[2]
        or q.shape != k.shape
        or q.shape != v.shape
    ):
        raise ValueError("q, k and v must have the same [1, T, H, 128] shape with positive T and H")
    if not q.is_cuda or any(x.device != q.device or x.dtype != torch.bfloat16 for x in (q, k, v)):
        raise ValueError("q, k and v must be BF16 tensors on the same CUDA device")
    if contiguous and any(not x.is_contiguous() for x in (q, k, v)):
        raise ValueError("q, k and v must be contiguous in BTHD layout")
    return torch


def _dense_rows(callback, q, k, v):
    import torch

    if callback is None:
        out = torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
        ).transpose(1, 2)
    else:
        out = callback(q, k, v)
    if not isinstance(out, torch.Tensor) or (out.shape, out.dtype, out.device) != (q.shape, q.dtype, q.device):
        raise ValueError("dense(q, k, v) must return a tensor with q's shape, dtype and device")
    return out


def _forward(q, k, v, *, model=None, counters=None, **options):
    torch = _validate(q, k, v, contiguous=model is None)
    capability = torch.cuda.get_device_capability(q.device)
    if capability not in ((8, 0), (12, 0)) or (model is not None and PROFILES[model].capability != capability):
        raise ValueError(f"{model or 'Sol3'} does not support CUDA SM{capability[0]}{capability[1]}")
    with torch.cuda.device(q.device), torch.no_grad():
        if model is None:
            prefix_tokens = options.pop("prefix_tokens", 0)
            dense = options.get("dense")
            if type(prefix_tokens) is not int or not 0 <= prefix_tokens <= q.shape[1]:
                raise ValueError("prefix_tokens must be an integer between zero and the sequence length")
            if dense is not None and not callable(dense):
                raise TypeError("dense must be a callable or None")
            if prefix_tokens:
                options["dense"] = partial(_dense_rows, dense)
                if prefix_tokens == q.shape[1]:
                    return options["dense"](q, k, v).contiguous(), None
                if capability == (8, 0):
                    options.update(sink_start=0, sink_tokens=prefix_tokens)
                else:
                    options["prefix_tokens"] = prefix_tokens
        if capability == (8, 0):
            from .backends.sm80.attention import forward

            return forward(q, k, v, model=model or "wan", counters=counters, public_qkv=model is None, **options)
        from .backends.sm120.attention import forward

        return forward(q, k, v, counters=counters, **options)


def sol3_attn(q, k, v, *, prefix_tokens=0, dense=None):
    """Compute forward-only, bidirectional attention on SM80 or SM120.

    Inputs are contiguous CUDA BF16 tensors of shape [1, T, H, 128].
    Blocks overlapping the leading prefix are exact keys and values for video
    queries. Prefix queries use dense attention over the full sequence via
    PyTorch SDPA or dense(q_prefix, k, v). The callback may receive a slice of
    the heads. Return a tensor with the query's shape, dtype and device.
    """
    output, _route = _forward(q, k, v, prefix_tokens=prefix_tokens, dense=dense)
    return output


def sol3_attn_with_route(q, k, v, *, profile, config, **options):
    """Run a model kernel profile using its native configuration.

    Return (output, route). The repository integrations supply the native
    sparse-call options, including token layout and any dense callback.
    """
    if profile not in PROFILES:
        raise ValueError(f"Unknown Sol3 profile {profile!r}. Available profiles: {', '.join(PROFILES)}")
    if profile == "fast":
        return _forward(q, k, v, model=profile, config=config, **options)
    return _forward(q, k, v, model=profile, cfg=config, **options)


def release_workspace(profile=None):
    """Release cached scratch for one profile, or all initialized profiles. Return the number of entries freed."""
    if profile is not None and profile not in PROFILES:
        raise ValueError(f"Unknown Sol3 profile {profile!r}")
    released = 0
    names = PROFILES if profile is None else (profile,)
    for name in names:
        arch = "sm120" if PROFILES[name].capability == (12, 0) else "sm80"
        backend = sys.modules.get(f"{__package__}.backends.{arch}.attention")
        if backend is not None:
            released += backend.release() if arch == "sm120" else backend.release(name)
    return released
