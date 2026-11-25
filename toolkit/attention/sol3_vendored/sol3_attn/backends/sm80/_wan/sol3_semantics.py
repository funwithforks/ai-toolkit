"""Sol³ uses representative weights without a variance mass correction."""
UNSUPPORTED_VARIANCE_OPTIONS = frozenset({"keyvar_cap", "keyvar_mode", "mass_zero_mixed"})

def require_cap0(cfg):
    removed = UNSUPPORTED_VARIANCE_OPTIONS.intersection(cfg)
    if removed:
        raise ValueError("Unsupported representative variance correction options: " + ", ".join(sorted(removed)))

SUPPORTED_FIRST_ORDER_COEFFICIENTS = (0., 0.5, 1., 1.5, 2.)

def require_first_order(cfg):
    alpha = cfg.get("first_order_alpha", 0.)
    if type(alpha) not in (int, float) or alpha not in SUPPORTED_FIRST_ORDER_COEFFICIENTS:
        raise ValueError("first_order_alpha must be one of 0, 0.5, 1, 1.5, 2")
    return float(alpha)
