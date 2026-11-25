from .block_selector import *

def route_only(q, k, v, *, cfg=None, scale=None, tau=1.0, thresh_type='diag', kv_splits=1, sink_tokens=0, sink_start=None, query_threshold_bias=None, **kw):
    cfg = {} if cfg is None else cfg
    if tau <= -100:
        from sol3_attn.backends.workspace import native_attention as sol_attn
        return sol_attn(q, k, v, scale=scale, tau=tau, thresh_type=thresh_type, kv_splits=kv_splits, sink_tokens=sink_tokens, sink_start=sink_start)
    assert q.shape == k.shape == v.shape and q.is_contiguous() and k.is_contiguous() and v.is_contiguous() and (query_threshold_bias is None)
    with torch.cuda.device(q.device):
        b, t, h, d = q.shape
        n = triton.cdiv(t, 64)
        s = d ** (-0.5) if scale is None else scale
        b1 = cfg.get('beta1', tau + 1.0)
        b2 = cfg.get('beta2', 0.0)
        b3 = cfg.get('beta3', 1.0)
        global_budget = cfg.get('global_budget', None)
        capacity = cfg.get('capacity', min(t, triton.cdiv(math.ceil(t / (1 + b3 * b3)) + 64, 64) * 64) if global_budget is None else triton.cdiv(math.ceil(1.25 * t * global_budget), 64) * 64)
        bf = cfg.get('selector', 'bf16') == 'bf16'
        fixed = cfg.get('value_mode', 'fixed') == 'fixed'
        vm_bf16 = cfg.get('summary_dtype', 'bf16') == 'bf16'
        risk_cap = float('inf') if cfg.get('disable_risk_cap', True) else float(cfg.get('risk_keyvar_cap', 0.0))
        risk_alpha = float(cfg.get('risk_keyvar_alpha', 1.0))
        risk_contrast = bool(cfg.get('risk_value_contrast', False))
        full_risk = risk_cap > 0 or cfg.get('risk_statistics') == 'full'
        cap = float(0.0)
        st = (t - sink_tokens if sink_start is None else sink_start) if sink_tokens else 0
        s0 = st // 64
        s1 = triton.cdiv(st + sink_tokens, 64) if sink_tokens else 0
        qskip = sink_tokens if cfg.get('skip_prefix', True) and st == 0 else 0
        fused = cfg.get('organization', 'fused') == 'fused'
        attention_qtile = cfg.get('attention_qtile', 64)
        assert attention_qtile in [16, 32, 64]
        assert fused or attention_qtile == 64, 'Smaller compute tiles are only enabled in the fused kernel'
        kc, vm, c, mom, kv, qb, th, flags, bits, off, full, counts, index, us, ls, ms, th2 = buffers(q, capacity, vm_bf16, fused)
        if cfg.get('fuse_pool', False):
            pool_qkv[n, b * h](q, k, v, qb, kc, vm, c, kv, t, n, h, d, cap > 0 or risk_cap > 0, num_warps=4, num_stages=1)
        else:
            pool_fast[n, b * h](k, v, kc, vm, c, kv, t, n, h, d, cap > 0 or risk_cap > 0, num_warps=4, num_stages=1)
        moments_fast[b * h,](kc, c, mom, t, n, h, d, num_warps=4, num_stages=1)
        from .sol3 import query_pool
        if not cfg.get('fuse_pool', False):
            query_pool[n, b * h](q, qb, t, n, h, d, num_warps=4, num_stages=1)
        vc = c
        if risk_contrast:
            ckey = (tuple(q.shape), q.device)
            if ckey not in VALUE_CACHE:
                VALUE_CACHE[ckey] = (torch.empty((b * h, n), device=q.device, dtype=torch.float32), torch.empty((b * h, d), device=q.device, dtype=torch.float32))
            vc, vg = VALUE_CACHE[ckey]
            LAST_KERNELS['value_global_mean'] = value_global_mean[b * h,](vm, vg, t, n, d, num_warps=4, num_stages=1)
            LAST_KERNELS['value_contrast'] = value_contrast[n, b * h](vm, vg, vc, n, d, num_warps=4, num_stages=1)
        if full_risk:
            LAST_KERNELS['risk_keyvar_threshold'] = risk_keyvar_threshold[triton.cdiv(n, 16), b * h](qb, kc, c, kv, vc, th2, t, n, h, d, s, b2, risk_cap, risk_alpha, qskip, 16, 64, risk_contrast, num_warps=cfg.get('risk_warps', 4), num_stages=1)
        rt = cfg.get('route_qtile', 16)
        if global_budget is None and cfg.get('third_selection') != 'global_mu_sigma':
            LAST_KERNELS['route_bits'] = route_bits[triton.cdiv(n, rt), n, b * h](qb, k, kc, c, mom, kv, vc, th2, flags, bits, s, t, n, h, d, b1, b2, b3, s0, s1, full_risk, bf, qskip, rt, risk_cap, risk_alpha, risk_contrast, num_warps=cfg.get('route_warps', 4), num_stages=1)
        else:
            classify_blocks[triton.cdiv(n, rt), n, b * h](qb, k, kc, c, mom, kv, vc, th2, flags, bits, s, t, n, h, d, b1, b2, b3, s0, s1, full_risk, bf, qskip, rt, risk_cap, risk_alpha, risk_contrast, num_warps=cfg.get('route_warps', 4), num_stages=1)
            if cfg.get('third_selection') == 'global_mu_sigma':
                if cfg.get('third_moment_mode') == 'block_diag':
                    from .sol3_mu_diagonal import threshold
                    threshold(qb, kc, kv, flags, th, t=t, n=n, h=h, d=d, scale=s, qskip=qskip, beta=float(b3))
                elif cfg.get('mu_moment_impl') == 'cached_raw':
                    from .sol3_global_mu_cached import threshold_route
                    threshold_route(qb, k, flags, bits, th, t=t, n=n, h=h, d=d, scale=s, qskip=qskip, beta=float(b3), chunk=cfg.get('mu_score_chunk', 14), route_qtile=rt)
                elif cfg.get('mu_moment_impl') == 'cached_tiled':
                    if cfg.get('mu_compact_moments', False):
                        from .sol3_global_mu_compact import threshold_route
                    else:
                        from .mu64 import threshold_route
                    threshold_route(qb, k, flags, bits, th, t=t, n=n, h=h, d=d, scale=s, qskip=qskip, beta=float(b3), chunk=cfg.get('mu_score_chunk', h), bq=cfg.get('mu_bq', 32), bk=cfg.get('mu_bk', 256), stages=cfg.get('mu_stages', 2), route_qtile=rt)
                elif cfg.get('mu_moment_impl') == 'tiled':
                    from .sol3_global_mu_tiled import threshold
                    threshold(qb, k, flags, th, t=t, n=n, h=h, d=d, scale=s, qskip=qskip, beta=float(b3), bq=cfg.get('mu_bq', 16), bk=cfg.get('mu_bk', 128), warps=4, stages=cfg.get('mu_stages', 1))
                else:
                    from .sol3_global_mu import threshold
                    threshold(qb, k, flags, th, t=t, n=n, h=h, d=d, scale=s, qskip=qskip, beta=float(b3))
            else:
                samples = cfg.get('threshold_samples', 2048)
                key = (tuple(q.shape), q.device, samples)
                if key not in SAMPLE_CACHE:
                    SAMPLE_CACHE[key] = torch.empty((b * h, n, samples), device=q.device, dtype=torch.float32)
                scores = SAMPLE_CACHE[key]
                if key not in GOAL_CACHE:
                    GOAL_CACHE[key] = torch.empty((b * h, n), device=q.device, dtype=torch.int32)
                goals = GOAL_CACHE[key]
                LAST_KERNELS['budget_goals'] = budget_goals[n, b * h](flags, goals, t, n, triton.next_power_of_2(n), float(global_budget), num_warps=4)
                LAST_KERNELS['sample_scores'] = sample_scores[triton.cdiv(n, 16), triton.cdiv(samples, 128), b * h](qb, k, flags, goals, scores, t, n, h, d, samples, s, bf, num_warps=4, num_stages=1)
                if cfg.get('cut_method', 'sort') == 'radix':
                    LAST_KERNELS['sampled_cut'] = radix_cut[n, b * h](scores, goals, th, t, n, samples, num_warps=cfg.get('cut_warps', 4))
                elif cfg.get('cut_method', 'sort') == 'histogram':
                    LAST_KERNELS['sampled_cut'] = histogram_cut[n, b * h](scores, flags, th, t, n, samples, triton.next_power_of_2(n), float(global_budget), cfg.get('cut_bins', 64), num_warps=4)
                else:
                    LAST_KERNELS['sampled_cut'] = sampled_cut[n, b * h](scores, goals, th, t, n, samples, num_warps=cfg.get('cut_warps', 8))
            if cfg.get('mu_moment_impl') not in ('cached_raw', 'cached_tiled'):
                LAST_KERNELS['global_route'] = global_route[triton.cdiv(n, rt), n, b * h](qb, k, flags, bits, th, s, t, n, h, d, bf, qskip, rt, num_warps=cfg.get('route_warps', 4), num_stages=1)
        return dict(b=b, t=t, h=h, d=d, n=n, s=s, qskip=qskip, kc=kc, vm=vm, kv=kv, qb=qb, flags=flags, bits=bits, off=off, full=full, counts=counts)
