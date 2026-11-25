"""FastH3 complete-current-call Mixed-token selection."""

from types import MappingProxyType

from ._providers.h3_mu_route import route_only


DEFAULT_COVARIANCE = 1.0
ROUTE_CONFIG = MappingProxyType({
    "global_budget": None,
    "third_selection": "global_mu_sigma",
    "beta1": 1.9,
    "beta2": 0.0,
    "beta3": 1.125,
    "route_qtile": 32,
    "summary_dtype": "bf16",
    "value_mode": "fixed",
    "disable_risk_cap": True,
    "risk_keyvar_alpha": 1.0,
    "risk_value_contrast": False,
    "fused_risk_classify": True,
    "mu_moment_impl": "parallel_recompute",
    "mu_bq": 32,
    "mu_bk": 256,
    "compact_representatives": True,
    "skip_prefix": True,
    "cov_gamma": 0.0,
})


def make_route(q, k, v, config, prefix_tokens):
    obsolete = {"keyvar_cap", "keyvar_mode", "mass_zero_mixed"}.intersection(config)
    if obsolete:
        raise ValueError(f"Unsupported representative variance settings: {sorted(obsolete)}")
    cfg = dict(ROUTE_CONFIG, **config)
    for key, value in ROUTE_CONFIG.items():
        if key not in ("beta1", "beta3") and cfg[key] != value:
            raise ValueError(f"The FastH3 route requires {key}={value!r}")
    return route_only(q, k, v, cfg=cfg, scale=q.shape[-1] ** -0.5, sink_tokens=prefix_tokens, sink_start=0)
