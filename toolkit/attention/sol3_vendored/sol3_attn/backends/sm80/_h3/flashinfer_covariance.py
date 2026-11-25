from sol3_attn.backends.sm80.paged import run_paged
from sol3_attn.backends.sm80.representatives import launch_merge
import math, pathlib, torch, triton, flashinfer
import triton.language as tl
from . import block_selector as selector
from .block_routing import route_only
from .pair_routing import route_only as route_pair
from .flashinfer_plan_vectorized import vectorize_plan
from .flashinfer_full_heads import calibrate, plan_info_scalars
from . import flashinfer_full_heads as fallback
from .h3_cov_rep_merge import rep_merge
from .h3_global_covariance import global_covariance
from .h3_route_tiled import route_only as route_tiled
from .h3_route_factorized import route_only as route_factorized
from .h3_packed_kv import packed_kv
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
    if COMPACT_REP:
        keep = (js < N) & (flag != -1) & (sizes - tl.maximum(flag, 0) > 0)
        rank = tl.cumsum(keep.to(tl.int32), 0) - 1
        tl.store(RINDEX + row * N + rank, js, keep)
        tl.store(RCOUNTS + row, tl.sum(keep.to(tl.int32), 0))

@triton.jit
def csr_indices(FLAGS, BITS, PTR, INDEX, T: tl.constexpr, N: tl.constexpr, ACTIVE: tl.constexpr, SKIP: tl.constexpr, H: tl.constexpr, CHUNK: tl.constexpr, HEAD_BASE: tl.constexpr, PACKED_KV: tl.constexpr):
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
            mapped = (bh % H + HEAD_BASE) * T + keys if PACKED_KV else keys * H + bh % H + HEAD_BASE
            tl.store(INDEX + dest, mapped, selected)
        carry += amount

@triton.jit
def csr_lengths_qpack(Q, FLAGS, LENGTHS, QP, RINDEX, RCOUNTS, T: tl.constexpr, N: tl.constexpr, ACTIVE: tl.constexpr, SKIP: tl.constexpr, H: tl.constexpr, D: tl.constexpr, HEAD_BASE: tl.constexpr, NN: tl.constexpr, COMPACT_REP: tl.constexpr):
    row = tl.program_id(0)
    lh = row // ACTIVE
    qi = row % ACTIVE + SKIP
    js = tl.arange(0, NN)
    flag = tl.load(FLAGS + (lh * N + qi) * N + js, js < N, -2)
    sizes = tl.minimum(64, tl.maximum(T - js * 64, 0))
    exact = tl.where(flag == -1, sizes, tl.maximum(flag, 0))
    tl.store(LENGTHS + row, tl.sum(tl.where(js < N, exact, 0), 0))
    if COMPACT_REP:
        keep = (js < N) & (flag != -1) & (sizes - tl.maximum(flag, 0) > 0)
        rank = tl.cumsum(keep.to(tl.int32), 0) - 1
        tl.store(RINDEX + row * N + rank, js, keep)
        tl.store(RCOUNTS + row, tl.sum(keep.to(tl.int32), 0))
    ts = qi * 64 + tl.arange(0, 64)
    ds = tl.arange(0, D)
    h = lh + HEAD_BASE
    val = tl.load(Q + (ts[:, None].to(tl.int64) * H + h) * D + ds[None, :], (ts < T)[:, None], 0)
    dst = (lh * (T - SKIP * 64) + ts[:, None] - SKIP * 64).to(tl.int64) * D + ds[None, :]
    tl.store(QP + dst, val, (ts < T)[:, None])

@triton.jit
def csr_ptr(LENGTHS, PTR, ROWS: tl.constexpr, SIZE: tl.constexpr):
    r = tl.arange(0, SIZE)
    lens = tl.load(LENGTHS + r, r < ROWS, 0)
    end = tl.cumsum(lens, 0)
    tl.store(PTR + 1 + r, end, r < ROWS)
    tl.store(PTR, 0)

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
            rindex = torch.empty((groups, triton.cdiv(t, 64)), device=q.device, dtype=torch.int32)
            rcounts = torch.empty(groups, device=q.device, dtype=torch.int32)
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
            slots[hc] = dict(rindex=rindex, rcounts=rcounts, lengths=lengths, ptr=ptr, pages=pages, qp=qp, os=os, lse=lse, qo=qo, last_cpu=last, float_workspace=fws, wrapper=wrapper, lse_base2=base2, plan_info=info, static_plan=static)
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
        out = fallback.fast(q, k, v, cfg=cfg, **kw)
        LAST.clear()
        LAST.update(fallback.LAST)
        LAST['fallback'] = True
        return out
    route = (route_tiled if cfg.get('classify_tiled', False) else route_factorized if cfg.get('factorized_risk_mean', False) else route_pair if cfg.get('route_pair', False) else route_only)(q, k, v, cfg=dict(cfg, capacity=1, organization='fused', compute_prefix=False), **kw)
    b, t, h, d, n, scale = [route[x] for x in ['b', 't', 'h', 'd', 'n', 's']]
    assert route['qskip'] == prefix
    covariance = global_covariance(k, v, prefix, cfg.get('cov_blocks', 32)) if cfg.get('cov_gamma', 0.0) != 0 else route['kv']
    active = ws['active']
    ta = ws['ta']
    kk, vv = packed_kv(k, v, cfg.get('stage_block', 2048)) if cfg.get('packed_kv', False) else (k, v)
    out = torch.empty_like(q)
    for start in range(0, h, chunk):
        hc = min(chunk, h - start)
        w = ws['slots'][hc]
        flags = route['flags'][start:start + hc]
        bits = route['bits'][start:start + hc]
        groups = hc * active
        csr_lengths_qpack[groups,](q, flags, w['lengths'], w['qp'], w['rindex'], w['rcounts'], t, n, active, skip, h, d, start, triton.next_power_of_2(n), bool(cfg.get('compact_representatives', False)), num_warps=4)
        csr_ptr[1,](w['lengths'], w['ptr'], groups, triton.next_power_of_2(groups), num_warps=8)
        csr_indices[groups,](flags, bits, w['ptr'], w['pages'], t, n, active, skip, h, cfg.get('csr_chunk', 16), start, bool(cfg.get('packed_kv', False)), num_warps=4)
        w['wrapper']._kv_lens_buffer = w['lengths']
        w['wrapper']._sm_scale = scale
        run_paged(w['wrapper'], w['qp'].view(-1, 1, d), (kk.view(-1, 1, 1, d), vv.view(-1, 1, 1, d)), out=w['os'], lse=w['lse'], return_lse=True)
        launch_merge(rep_merge, (n, hc), q, route['kc'], route['vm'], route['flags'], covariance, w['rindex'], w['rcounts'], w['os'], w['lse'], out, scale, t, n, h, d, True, prefix, cfg.get('rep_bk', 64), True, True, skip * 64, w['lse_base2'], start, float(cfg.get('cov_gamma', 0.0)), bool(cfg.get('compact_representatives', False)), active, skip, num_warps=4, num_stages=1)
    if prefix and cfg.get('compute_prefix', True):
        from sol3_attn.backends.workspace import dense_prefix as flash_attn_func
        out[:, :prefix] = flash_attn_func(q[:, :prefix], k, v, softmax_scale=scale)
    LAST.clear()
    LAST.update(route=route, skip_groups=skip, active_groups=active, workspace=ws, fallback=False)
    return out
