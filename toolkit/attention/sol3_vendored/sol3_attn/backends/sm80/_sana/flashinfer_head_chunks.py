from sol3_attn.backends.sm80.paged import launch_pages
from sol3_attn.backends.sm80.paged import run_paged
import math, pathlib, torch, triton, flashinfer
import triton.language as tl
from . import block_selector as selector
from .block_routing import route_only
from .pair_routing import route_only as route_pair
from .flashinfer_plan_vectorized import vectorize_plan
from .flashinfer_full_heads import calibrate, plan_info_scalars
from . import flashinfer_full_heads as fallback
from .head_chunk_merge import rep_merge
CACHE = {}
LAST = {}

@triton.jit
def csr_lengths(FLAGS, LENGTHS, T: tl.constexpr, N: tl.constexpr, ACTIVE: tl.constexpr, SKIP: tl.constexpr, NN: tl.constexpr):
    row = tl.program_id(0)
    bh = row // ACTIVE
    qi = row % ACTIVE + SKIP
    js = tl.arange(0, NN)
    flag = tl.load(FLAGS + (bh * N + qi) * N + js, js < N, -2)
    sizes = tl.minimum(64, tl.maximum(T - js * 64, 0))
    exact = tl.where(flag == -1, sizes, tl.maximum(flag, 0))
    tl.store(LENGTHS + row, tl.sum(tl.where(js < N, exact, 0), 0))

@triton.jit
def csr_lengths_qpack(Q, FLAGS, LENGTHS, QP, T: tl.constexpr, N: tl.constexpr, ACTIVE: tl.constexpr, SKIP: tl.constexpr, H: tl.constexpr, D: tl.constexpr, HEAD_BASE: tl.constexpr, NN: tl.constexpr):
    row = tl.program_id(0)
    lh = row // ACTIVE
    qi = row % ACTIVE + SKIP
    js = tl.arange(0, NN)
    flag = tl.load(FLAGS + (lh * N + qi) * N + js, js < N, -2)
    sizes = tl.minimum(64, tl.maximum(T - js * 64, 0))
    exact = tl.where(flag == -1, sizes, tl.maximum(flag, 0))
    tl.store(LENGTHS + row, tl.sum(tl.where(js < N, exact, 0), 0))
    qs = qi * 64 + tl.arange(0, 64)
    ds = tl.arange(0, D)
    values = tl.load(Q + (qs[:, None].to(tl.int64) * H + lh + HEAD_BASE) * D + ds[None, :], (qs < T)[:, None], 0)
    tl.store(QP + (lh * (T - SKIP * 64) + qs[:, None] - SKIP * 64).to(tl.int64) * D + ds[None, :], values, (qs < T)[:, None])

@triton.jit
def csr_scan(LENGTHS, PTR, G: tl.constexpr, NB: tl.constexpr):
    rs = tl.arange(0, NB)
    values = tl.load(LENGTHS + rs, rs < G, 0)
    prefix = tl.cumsum(values, 0)
    tl.store(PTR + rs + 1, prefix, rs < G)
    tl.store(PTR, 0)

@triton.jit
def csr_indices(FLAGS, BITS, PTR, INDEX, T: tl.constexpr, N: tl.constexpr, ACTIVE: tl.constexpr, SKIP: tl.constexpr, H: tl.constexpr, CHUNK: tl.constexpr, HEAD_BASE: tl.constexpr):
    row = tl.program_id(0)
    bh = row // ACTIVE
    qi = row % ACTIVE + SKIP
    startptr = tl.load(PTR + row)
    kk = tl.arange(0, CHUNK)
    ts = tl.arange(0, 64)
    carry = tl.full((), 0, tl.int32)
    for start in range(0, N, CHUNK):
        js = start + kk
        flag = tl.load(FLAGS + (bh * N + qi) * N + js, js < N, -2)
        word = tl.load(BITS + (bh * N + qi) * N + js, js < N, 0)
        keys = js[:, None] * 64 + ts[None, :]
        selected = (word[:, None] >> ts[None, :].to(tl.uint64) & 1 != 0) & (flag >= 0)[:, None]
        selected = (selected | (flag == -1)[:, None]) & (js < N)[:, None] & (keys < T)
        cnt = tl.sum(selected.to(tl.int32), 1)
        amount = tl.sum(cnt, 0)
        if amount > 0:
            rank = tl.cumsum(selected.to(tl.int32), 1) - 1
            off = tl.cumsum(cnt, 0) - cnt + carry
            dest = startptr + off[:, None] + rank
            tl.store(INDEX + dest, keys * H + bh % H + HEAD_BASE, selected)
        carry += amount

def workspace(q, skip, chunk):
    b, t, h, d = q.shape
    assert b == 1
    ta = t - skip * 64
    active = triton.cdiv(t, 64) - skip
    key = (tuple(q.shape), q.device, skip, chunk)
    if key not in CACHE:
        slots = {}
        for hc in sorted(set((min(chunk, h - start) for start in range(0, h, chunk)))):
            groups = hc * active
            capacity = groups * t
            assert capacity < 2 ** 31
            lengths = torch.empty(groups, device=q.device, dtype=torch.int32)
            ptr = torch.empty(groups + 1, device=q.device, dtype=torch.int32)
            pages = torch.empty(capacity, device=q.device, dtype=torch.int32)
            qp = torch.empty((hc, ta, d), device=q.device, dtype=q.dtype)
            os = torch.empty((hc * ta, 1, d), device=q.device, dtype=q.dtype)
            lse = torch.empty((hc * ta, 1), device=q.device)
            qo = (torch.arange(hc, dtype=torch.int32)[:, None] * ta + torch.arange(active, dtype=torch.int32)[None, :] * 64).flatten()
            qo = torch.cat([qo, torch.tensor([hc * ta], dtype=torch.int32)])
            fws = torch.empty(128 * 1024 * 1024, device=q.device, dtype=torch.uint8)
            wrapper = vectorize_plan(flashinfer.BatchPrefillWithPagedKVCacheWrapper(fws, kv_layout='NHD', backend='fa2'))
            base2 = calibrate(wrapper, q.device, d)
            last = torch.ones(groups, dtype=torch.int32)
            wrapper.plan(qo, torch.arange(groups + 1, dtype=torch.int32) * t, pages, last, 1, 1, d, 1, causal=False, q_data_type=q.dtype, kv_data_type=q.dtype, use_fp16_qk_reduction=False)
            info = plan_info_scalars(wrapper._plan_info)
            static = len(info) == 15 and info[14] == 0
            if static:
                wrapper._paged_kv_indptr_buf = ptr
                wrapper._paged_kv_indices_buf = pages
            slots[hc] = dict(lengths=lengths, ptr=ptr, pages=pages, qp=qp, os=os, lse=lse, qo=qo, last_cpu=last, float_workspace=fws, wrapper=wrapper, lse_base2=base2, plan_info=info, static_plan=static)
        CACHE[key] = dict(slots=slots, ta=ta, active=active)
    return CACHE[key]

def fast(q, k, v, *, cfg, **kw):
    with torch.cuda.device(q.device):
        return _fast(q, k, v, cfg=cfg, **kw)

def _fast(q, k, v, *, cfg, **kw):
    if kw.get('tau', 1.0) <= -100:
        from sol3_attn.backends.workspace import native_attention as sol_attn
        return sol_attn(q, k, v, **kw)
    assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous()
    prefix = kw.get('sink_tokens', 0) if kw.get('sink_start') == 0 else 0
    skip = prefix // 64
    chunk = min(int(cfg.get('head_chunk', 14)), q.shape[2])
    ws = workspace(q, skip, chunk)
    if not all((w['static_plan'] for w in ws['slots'].values())):
        if float(cfg.get('first_order_alpha', 0.0)):
            raise RuntimeError('First-order attention requires a static FlashInfer plan')
        out = fallback.fast(q, k, v, cfg=cfg, **kw)
        LAST.clear()
        LAST.update(fallback.LAST)
        LAST['fallback'] = True
        return out
    selected_route = route_pair if cfg.get('route_pair', False) else route_only
    if cfg.get('fused_risk_classify', False):
        from .fused_risk_route import route_only as selected_route
    route = selected_route(q, k, v, cfg=dict(cfg, capacity=1, organization='fused', compute_prefix=False), **kw)
    b, t, h, d, n, scale = [route[x] for x in ['b', 't', 'h', 'd', 'n', 's']]
    assert route['qskip'] == prefix
    fo_alpha = float(cfg.get('first_order_alpha', 0.0))
    if fo_alpha:
        from .global_first_order import global_first_order
        hm = global_first_order(k, v, route['kc'])
    active = ws['active']
    ta = ws['ta']
    out = torch.empty_like(q)
    for start in range(0, h, chunk):
        hc = min(chunk, h - start)
        w = ws['slots'][hc]
        flags = route['flags'][start:start + hc]
        bits = route['bits'][start:start + hc]
        groups = hc * active
        fused_csr = bool(cfg.get('fused_csr', False))
        if fused_csr:
            if cfg.get('csr_parallel_indices', False):
                if 'block_offsets' not in w:
                    w['block_offsets'] = torch.empty((groups, n), device=q.device, dtype=torch.int32)
                csr_lengths_qpack_offsets[groups,](q, flags, w['lengths'], w['qp'], w['block_offsets'], t, n, active, skip, h, d, start, triton.next_power_of_2(n), num_warps=4)
            else:
                csr_lengths_qpack[groups,](q, flags, w['lengths'], w['qp'], t, n, active, skip, h, d, start, triton.next_power_of_2(n), num_warps=4)
            csr_scan[1,](w['lengths'], w['ptr'], groups, triton.next_power_of_2(groups), num_warps=8)
        else:
            csr_lengths[groups,](flags, w['lengths'], t, n, active, skip, triton.next_power_of_2(n), num_warps=4)
            w['ptr'][0] = 0
            w['ptr'][1:] = torch.cumsum(w['lengths'], 0, dtype=torch.int32)
        if cfg.get('csr_parallel_indices', False):
            assert fused_csr
            ic = int(cfg.get('csr_parallel_chunk', 32))
            launch_pages(csr_indices_parallel, (triton.cdiv(n, ic), groups), flags, bits, w['ptr'], w['block_offsets'], w['pages'], t, n, active, skip, h, ic, start, num_warps=4)
        else:
            csr_indices[groups,](flags, bits, w['ptr'], w['pages'], t, n, active, skip, h, 16, start, num_warps=4)
        if not fused_csr:
            w['qp'].copy_(q[:, skip * 64:, start:start + hc].permute(0, 2, 1, 3).reshape(hc, ta, d))
        if fused_csr:
            w['wrapper']._kv_lens_buffer = w['lengths']
        else:
            w['wrapper']._kv_lens_buffer[:groups].copy_(w['lengths'])
        w['wrapper']._sm_scale = scale
        run_paged(w['wrapper'], w['qp'].view(-1, 1, d), (k.view(-1, 1, 1, d), v.view(-1, 1, 1, d)), out=w['os'], lse=w['lse'], return_lse=True)
        if fo_alpha:
            from .head_chunk_merge_first_order import rep_merge as first_merge
            first_merge[n, hc](q, route['kc'], route['vm'], route['flags'], hm, w['os'], w['lse'], out, scale, t, n, h, d, True, prefix, cfg.get('rep_bk', 64), True, True, skip * 64, w['lse_base2'], start, fo_alpha, num_warps=4, num_stages=1)
        else:
            rep_merge[n, hc](q, route['kc'], route['vm'], route['flags'], w['os'], w['lse'], out, scale, t, n, h, d, True, prefix, cfg.get('rep_bk', 64), True, True, skip * 64, w['lse_base2'], start, num_warps=4, num_stages=1)
    if prefix and cfg.get('compute_prefix', True):
        from sol3_attn.backends.workspace import dense_prefix as flash_attn_func
        out[:, :prefix] = flash_attn_func(q[:, :prefix], k, v, softmax_scale=scale)
    LAST.clear()
    LAST.update(route=route, skip_groups=skip, active_groups=active, workspace=ws, fallback=False)
    return out

@triton.jit
def csr_lengths_qpack_offsets(Q, FLAGS, LENGTHS, QP, OFF, T: tl.constexpr, N: tl.constexpr, ACTIVE: tl.constexpr, SKIP: tl.constexpr, H: tl.constexpr, D: tl.constexpr, HEAD_BASE: tl.constexpr, NN: tl.constexpr):
    row = tl.program_id(0)
    lh = row // ACTIVE
    qi = row % ACTIVE + SKIP
    js = tl.arange(0, NN)
    flag = tl.load(FLAGS + (lh * N + qi) * N + js, js < N, -2)
    sizes = tl.minimum(64, tl.maximum(T - js * 64, 0))
    exact = tl.where(js < N, tl.where(flag == -1, sizes, tl.maximum(flag, 0)), 0)
    prefix = tl.cumsum(exact, 0) - exact
    tl.store(OFF + row * N + js, prefix, js < N)
    tl.store(LENGTHS + row, tl.sum(exact, 0))
    qs = qi * 64 + tl.arange(0, 64)
    ds = tl.arange(0, D)
    values = tl.load(Q + (qs[:, None].to(tl.int64) * H + lh + HEAD_BASE) * D + ds[None, :], (qs < T)[:, None], 0)
    tl.store(QP + (lh * (T - SKIP * 64) + qs[:, None] - SKIP * 64).to(tl.int64) * D + ds[None, :], values, (qs < T)[:, None])

@triton.jit
def csr_indices_parallel(FLAGS, BITS, PTR, OFF, INDEX, T: tl.constexpr, N: tl.constexpr, ACTIVE: tl.constexpr, SKIP: tl.constexpr, H: tl.constexpr, CHUNK: tl.constexpr, HEAD_BASE: tl.constexpr):
    tile, row = (tl.program_id(0), tl.program_id(1))
    bh = row // ACTIVE
    qi = row % ACTIVE + SKIP
    js = tile * CHUNK + tl.arange(0, CHUNK)
    ts = tl.arange(0, 64)
    flag = tl.load(FLAGS + (bh * N + qi) * N + js, js < N, -2)
    word = tl.load(BITS + (bh * N + qi) * N + js, js < N, 0)
    keys = js[:, None] * 64 + ts[None, :]
    chosen = (word[:, None] >> ts[None, :].to(tl.uint64) & 1 != 0) & (flag >= 0)[:, None]
    chosen = (chosen | (flag == -1)[:, None]) & (js < N)[:, None] & (keys < T)
    if tl.sum(tl.sum(chosen.to(tl.int32), 1), 0) > 0:
        start = tl.load(PTR + row)
        block = tl.load(OFF + row * N + js, js < N, 0)
        rank = tl.cumsum(chosen.to(tl.int32), 1) - 1
        tl.store(INDEX + start + block[:, None] + rank, keys * H + bh % H + HEAD_BASE, chosen)
