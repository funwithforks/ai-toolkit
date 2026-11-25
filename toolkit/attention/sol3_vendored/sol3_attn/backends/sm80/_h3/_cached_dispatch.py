"""Select the H3 score kernel for the available workspace."""

from sol3_attn.backends.workspace import active_call

from . import h3_mu_cached, h3_mu_head_staged_pack as staged


staged.ORIGINAL_ROUTE = h3_mu_cached.route


def route(*args, **kwargs):
    call = active_call()
    if call is not None and call.stage_h3_scores:
        return staged.route(*args, **kwargs)
    return h3_mu_cached.route(*args, **kwargs)
