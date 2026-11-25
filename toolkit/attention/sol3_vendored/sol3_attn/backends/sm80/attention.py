"""Orchestrate the complete SM80 paged Sol³ forward."""

from importlib import import_module

from ...profiles import LONG_SEQUENCE
from ..workspace import GIB, Workspace


_ENTRY = {
    "wan": "flashinfer_head_chunks",
    "hy": "sol3_flashinfer_firstorder",
    "sana": "flashinfer_head_chunks",
    "ltx": "flashinfer_head_chunks",
    "h3": "sol3_h3_mu_flashinfer",
}
_WORKSPACES = {model: Workspace(model, f"{__package__}._{model}.") for model in _ENTRY}

_WAN_CONFIG = {
    "third_selection": "global_mu_sigma",
    "global_budget": None,
    "beta1": 3.0,
    "beta2": 0.0,
    "beta3": 0.6875,
    "selector": "bf16",
    "summary_dtype": "bf16",
    "value_mode": "fixed",
    "fuse_pool": True,
    "organization": "fused",
    "route_qtile": 32,
    "attention_qtile": 64,
    "risk_keyvar_cap": None,
    "disable_risk_cap": True,
    "risk_keyvar_alpha": 1.0,
    "risk_value_contrast": False,
    "compact_mode": "fused",
    "route_pair": False,
    "mu_moment_impl": "cached_tiled",
    "mu_bq": 64,
    "mu_bk": 128,
    "mu_stages": 2,
    "compute_prefix": True,
    "head_chunk": 40,
    "rep_bk": 64,
    "rep_qtile": 32,
    "first_order_alpha": 1.0,
    "factorized_risk_mean": True,
    "tensorcore_classify": True,
    "fused_csr": True,
    "mu_score_chunk": 40,
    "mu_score_dtype": "fp32",
}
_OPTIONS = {
    "wan": {"parallel_csr": True},
    "hy": {"rep_qtile": 64, "parallel_csr": True, "parallel_csr_chunk": 32},
    "sana": {"csr_parallel_chunk": 32, "mu_store_early": False},
    "ltx": {"parallel_csr": True, "parallel_csr_chunk": 32, "parallel_csr_warps": 2},
    "h3": {"csr_chunk": 32},
}


# Normalize configuration from existing model controllers before planning or fallback.
def _model_config(model, cfg, capacity):
    config = dict(cfg, capacity=capacity)
    aliases = {
        "engineering_csr_fusion": "fused_csr",
    }
    if model in ("wan", "hy"):
        aliases["engineering_parallel_csr"] = "parallel_csr"
    if model == "hy":
        aliases["engineering_csr_chunk"] = "parallel_csr_chunk"
    if model == "sana":
        aliases["engineering_factorized_risk"] = "factorized_risk"
    for old, new in aliases.items():
        if old in config:
            config.setdefault(new, config.pop(old))
    return config


def _planned_config(q, provider, model, original, kwargs, scratch, state):
    caps = state.profile.long_heads if q.shape[1] >= LONG_SEQUENCE else None
    plan = scratch.plan_for(q, preferred_heads=caps or 4, reserve_bytes=(6 if model == "h3" else 2) * GIB)
    cfg = original.copy()
    if not plan.minimum_fits:
        if model == "h3":
            cfg["head_chunk"] = 1
        return cfg
    cfg["head_chunk"] = plan.head_chunk
    cfg.update(_OPTIONS[model])
    if model == "h3":
        state.stage_h3_scores = True
        state.h3_score_heads = plan.mu_score_chunk
        return cfg
    cfg["mu_score_chunk"] = plan.mu_score_chunk
    if model == "ltx":
        cfg["mu_chunk"] = plan.mu_score_chunk
        prefix = kwargs.get("sink_tokens", 0) if kwargs.get("sink_start") == 0 else 0
        workspace = provider.workspace(q, prefix // 64, cfg["head_chunk"])
        if not all(slot["static_plan"] for slot in workspace["slots"].values()):
            return original
    return cfg


def forward(q, k, v, *, model, counters=None, cfg=None, metadata=None, native=None, dense=None, public_qkv=False, **kwargs):
    try:
        provider = import_module(f"._{model}.{_ENTRY[model]}", __package__)
    except ModuleNotFoundError as exc:
        if exc.name in ("flashinfer", "flashinfer.prefill"):
            raise RuntimeError("The SM80 backend requires FlashInfer. Install sol3-attn[sm80].") from exc
        raise
    if cfg is None:
        if not public_qkv:
            raise TypeError("The model adapter must supply the native method configuration")
        cfg = _WAN_CONFIG
    q, k, v = (x.contiguous() for x in (q, k, v))
    scratch = _WORKSPACES[model]
    original = _model_config(model, cfg, int(q.shape[1]))
    with scratch.call(q, counters, native=native, dense=dense, public_qkv=public_qkv) as state:
        config = _planned_config(q, provider, model, original, kwargs, scratch, state)
        try:
            output = provider.fast(q, k, v, cfg=config, **kwargs)
        except RuntimeError as exc:
            if model != "wan" or str(exc) != "First-order requires static plan" or config == original:
                raise
            provider.CACHE.clear()
            output = provider.fast(q, k, v, cfg=original, **kwargs)
        if metadata is not None:
            metadata.clear()
            metadata.update(provider.LAST)
        return output, provider.LAST.get("route")


def release(model):
    return _WORKSPACES[model].release()
