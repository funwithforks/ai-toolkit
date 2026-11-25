"""Kernels for block pooling and sparse attention routing."""
import torch, triton
import triton.language as tl
from . import sol3 as base
from .risk_threshold_routing import risk_threshold
CACHE = {}

@triton.jit
def pool_keyvar(K, V, KC, VS, C, KV, T: tl.constexpr, N: tl.constexpr, H: tl.constexpr, D: tl.constexpr):
    j, bh = (tl.program_id(0), tl.program_id(1))
    b, h = (bh // H, bh % H)
    t = j * 64 + tl.arange(0, 64)
    d = tl.arange(0, D)
    ix = ((b * T + t[:, None]).to(tl.int64) * H + h) * D + d[None, :]
    valid = t < T
    n = tl.minimum(64, T - j * 64).to(tl.float32)
    k = tl.load(K + ix, valid[:, None], 0).to(tl.float32)
    v = tl.load(V + ix, valid[:, None], 0).to(tl.float32)
    km = tl.sum(k, 0) / n
    vs = tl.sum(v, 0)
    vm = vs / n
    var = tl.sum(tl.sum(tl.where(valid[:, None], (v - vm[None, :]) * (v - vm[None, :]), 0), 0), 0) / (n * D)
    rms = tl.sqrt(tl.maximum(var, 0))
    si = ((b * N + j) * H + h) * D + d
    tl.store(KC + si, km)
    tl.store(VS + si, vs)
    kvar = tl.sum(tl.where(valid[:, None], (k - km[None, :]) * (k - km[None, :]), 0), 0) / n
    tl.store(KV + si, kvar)
    tl.store(C + (b * N + j) * H + h, tl.log2(n * tl.maximum(rms, 1e-06)))

@triton.jit
def route_opt(QB, K, KC, C, M, TH2, FLAGS, MASK, SCALE: tl.constexpr, T: tl.constexpr, N: tl.constexpr, H: tl.constexpr, D: tl.constexpr, B1: tl.constexpr, B2: tl.constexpr, B3: tl.constexpr, SINK0: tl.constexpr, SINK1: tl.constexpr, USE_TH2: tl.constexpr, BF16: tl.constexpr, QSKIP: tl.constexpr):
    tile, j, bh = (tl.program_id(0), tl.program_id(1), tl.program_id(2))
    b, h = (bh // H, bh % H)
    qs = tile * 16 + tl.arange(0, 16)
    activeq = (qs < N) & ((qs + 1) * 64 > QSKIP)
    ds = tl.arange(0, D)
    ts = tl.arange(0, 64)
    qb = tl.load(QB + (bh * N + qs[:, None]) * D + ds[None, :], (qs < N)[:, None], 0)
    mk = tl.load(M + bh * (2 * D + 2) + ds)
    vk = tl.load(M + bh * (2 * D + 2) + D + ds)
    mc = tl.load(M + bh * (2 * D + 2) + 2 * D)
    vc = tl.load(M + bh * (2 * D + 2) + 2 * D + 1)
    s = SCALE * 1.4426950408889634
    mu = tl.sum(qb * mk[None, :], 1) * s
    var = tl.sum(qb * qb * vk[None, :], 1) * s * s
    th1 = mu + B1 * tl.sqrt(tl.maximum(var, 0) + 1e-06)
    th2 = mu + mc + B2 * tl.sqrt(tl.maximum(var + vc, 0) + 1e-06)
    kc = tl.load(KC + ((b * N + j) * H + h) * D + ds).to(tl.float32)
    c = tl.load(C + (b * N + j) * H + h)
    proxy = tl.sum(qb * kc[None, :], 1) * s
    if USE_TH2:
        th2 = tl.load(TH2 + bh * N + qs, qs < N, 0)
    ex = activeq & ((proxy > th1) | (tl.abs(qs - j) <= 1) | (j >= SINK0) & (j < SINK1))
    pure = activeq & ~ex & (proxy + c < th2)
    mix = activeq & ~ex & ~pure
    mask = tl.full((16,), 0, tl.uint64)
    count = tl.zeros((16,), tl.int32)
    if tl.sum(mix.to(tl.int32), 0) > 0:
        kt = j * 64 + ts
        valid = kt < T
        n = tl.minimum(64, T - j * 64).to(tl.float32)
        k = tl.load(K + ((b * T + kt[:, None]).to(tl.int64) * H + h) * D + ds[None, :], valid[:, None], 0).to(tl.float32)
        if BF16:
            a = tl.dot(qb.to(tl.bfloat16), tl.trans(k.to(tl.bfloat16))) * s
        else:
            a = tl.dot(qb, tl.trans(k), input_precision='tf32x3') * s
        am = tl.sum(tl.where(valid[None, :], a, 0.0), 1) / n
        av = tl.sum(tl.where(valid[None, :], (a - am[:, None]) * (a - am[:, None]), 0.0), 1) / n
        sel = mix[:, None] & valid[None, :] & (a > am[:, None] + B3 * tl.sqrt(tl.maximum(av, 0) + 1e-06)[:, None])
        count = tl.sum(sel.to(tl.int32), 1)
        bits = tl.full((64,), 1, tl.uint64) << ts.to(tl.uint64)
        mask = tl.sum(tl.where(sel, bits[None, :], 0), 1)
        rank = tl.cumsum(sel.to(tl.int32), 1) - 1
        ix = (bh * N + qs) * N + j
        tl.store(MASK + ix[:, None] * 64 + rank, tl.broadcast_to(ts[None, :], (16, 64)), sel)
    flags = tl.where(ex, -1, tl.where(pure, -2, count))
    ix = (bh * N + qs) * N + j
    tl.store(FLAGS + ix, flags, qs < N)

@triton.jit
def forward_opt(Q, K, V, KC, VS, C, M, O, FLAGS, MASK, SCALE: tl.constexpr, T: tl.constexpr, N: tl.constexpr, H: tl.constexpr, D: tl.constexpr, B1: tl.constexpr, B2: tl.constexpr, B3: tl.constexpr, SINK0: tl.constexpr, SINK1: tl.constexpr, BV: tl.constexpr, FIXED: tl.constexpr, QSKIP: tl.constexpr):
    vi, qi, bh = (tl.program_id(0), tl.program_id(1), tl.program_id(2))
    b, h = (bh // H, bh % H)
    ts = tl.arange(0, 64)
    ds = tl.arange(0, D)
    vd = vi * BV + tl.arange(0, BV)
    gs = tl.arange(0, 32)
    if (qi + 1) * 64 <= QSKIP:
        return
    qt = qi * 64 + ts
    validq = qt < T
    qo = ((b * T + qt[:, None]).to(tl.int64) * H + h) * D + ds[None, :]
    q = tl.load(Q + qo, validq[:, None], 0)
    qn = tl.minimum(64, T - qi * 64).to(tl.float32)
    qb = tl.sum(q.to(tl.float32), 0) / qn
    mk = tl.load(M + bh * (2 * D + 2) + ds)
    vk = tl.load(M + bh * (2 * D + 2) + D + ds)
    mc = tl.load(M + bh * (2 * D + 2) + 2 * D)
    vc = tl.load(M + bh * (2 * D + 2) + 2 * D + 1)
    s = SCALE * 1.4426950408889634
    mu = tl.sum(qb * mk, 0) * s
    var = tl.sum(qb * qb * vk, 0) * s * s
    th1 = mu + B1 * tl.sqrt(tl.maximum(var, 0) + 1e-06)
    th2 = mu + mc + B2 * tl.sqrt(tl.maximum(var + vc, 0) + 1e-06)
    acc = tl.zeros((64, BV), tl.float32)
    l = tl.zeros((64,), tl.float32)
    mx = tl.full((64,), -float('inf'), tl.float32)
    for start in range(0, N, 32):
        js = start + gs
        vj = js < N
        kc = tl.load(KC + ((b * N + js[:, None]) * H + h) * D + ds[None, :], vj[:, None], 0)
        scores = tl.dot(q, kc.T) * s
        proxy = tl.sum(scores, 0) / qn
        c = tl.load(C + (b * N + js) * H + h, vj, 0)
        flags = tl.load(FLAGS + (bh * N + qi) * N + js, vj, -2)
        exact = vj & (flags == -1)
        pure = vj & (flags != -1)
        mixed = vj & (flags >= 0)
        ns = tl.minimum(64, tl.maximum(0, T - js * 64)).to(tl.float32)
        rem = ns - tl.maximum(flags, 0).to(tl.float32)
        if FIXED:
            pure = pure & (rem > 0)
        has = tl.sum(pure.to(tl.int32), 0) > 0
        safe = tl.where(has, tl.where(pure[None, :], scores, -float('inf')), 0.0)
        nm = tl.where(has, tl.maximum(mx, tl.max(safe, 1)), mx)
        al = tl.exp2(tl.where(has, mx - nm, 0.0))
        p = tl.where(has & pure[None, :], tl.exp2(safe - tl.where(has, nm, 0.0)[:, None]), 0.0)
        vs = tl.load(VS + ((b * N + js[:, None]) * H + h) * D + vd[None, :], vj[:, None], 0)
        if FIXED:
            vs = vs * (rem / tl.maximum(ns, 1.0))[:, None]
        acc = acc * al[:, None] + tl.dot(p, vs, input_precision='tf32x3')
        ns = tl.minimum(64, tl.maximum(0, T - js * 64)).to(tl.float32)
        if FIXED:
            l = l * al + tl.sum(p * rem[None, :], 1)
        else:
            l = l * al + tl.sum(p * ns[None, :], 1)
        mx = nm
        eo = tl.where(exact, gs, 32)
        for z in range(tl.sum(exact.to(tl.int32), 0)):
            off = tl.min(eo, 0)
            j = start + off
            eo = tl.where(gs == off, 32, eo)
            kt = j * 64 + ts
            validk = kt < T
            ko = ((b * T + kt[:, None]).to(tl.int64) * H + h) * D + ds[None, :]
            k = tl.load(K + ko, validk[:, None], 0)
            exs = tl.dot(q, k.T) * s + tl.where(validk[None, :], 0.0, -float('inf'))
            nm = tl.maximum(mx, tl.max(exs, 1))
            al = tl.exp2(mx - nm)
            ep = tl.exp2(exs - nm[:, None])
            vo = ((b * T + kt[:, None]).to(tl.int64) * H + h) * D + vd[None, :]
            v = tl.load(V + vo, validk[:, None], 0)
            acc = acc * al[:, None] + tl.dot(ep.to(q.dtype), v)
            l = l * al + tl.sum(ep, 1)
            mx = nm
        counts = tl.maximum(flags, 0)
        ends = tl.cumsum(counts, 0)
        beginnings = ends - counts
        total = tl.sum(counts, 0)
        for chunk in range(tl.cdiv(total, 64)):
            cr = chunk * 64 + ts
            ok = cr < total
            bo = tl.minimum(tl.sum((cr[:, None] >= ends[None, :]).to(tl.int32), 1), 31)
            begin = tl.gather(beginnings, bo, 0)
            local = tl.minimum(tl.maximum(cr - begin, 0), 63)
            token = tl.load(MASK + ((bh * N + qi) * N + start + bo) * 64 + local, ok, 0).to(tl.int32)
            kt = (start + bo) * 64 + token
            ko = ((b * T + kt[:, None]).to(tl.int64) * H + h) * D + ds[None, :]
            sk = tl.load(K + ko, ok[:, None], 0)
            ms = tl.dot(q, sk.T) * s + tl.where(ok[None, :], 0.0, -float('inf'))
            means = tl.gather(scores, tl.broadcast_to(bo[None, :], (64, 64)), 1)
            nm = tl.maximum(mx, tl.max(ms, 1))
            al = tl.exp2(mx - nm)
            if FIXED:
                diff = tl.where(ok[None, :], tl.exp2(ms - nm[:, None]), 0.0)
            else:
                diff = tl.where(ok[None, :], tl.exp2(ms - nm[:, None]) - tl.exp2(means - nm[:, None]), 0.0)
            vo = ((b * T + kt[:, None]).to(tl.int64) * H + h) * D + vd[None, :]
            sv = tl.load(V + vo, ok[:, None], 0)
            acc = acc * al[:, None] + tl.dot(diff.to(q.dtype), sv)
            l = l * al + tl.sum(diff, 1)
            mx = nm
    oo = ((b * T + qt[:, None]).to(tl.int64) * H + h) * D + vd[None, :]
    tl.store(O + oo, acc / l[:, None], validq[:, None])

@triton.jit
def forward_branch(Q, K, V, KC, VS, C, M, UO, LO, MO, FLAGS, MASK, SCALE: tl.constexpr, T: tl.constexpr, N: tl.constexpr, H: tl.constexpr, D: tl.constexpr, B1: tl.constexpr, B2: tl.constexpr, B3: tl.constexpr, SINK0: tl.constexpr, SINK1: tl.constexpr, BV: tl.constexpr, FIXED: tl.constexpr, QSKIP: tl.constexpr, BRANCH: tl.constexpr):
    vi, qi, bh = (tl.program_id(0), tl.program_id(1), tl.program_id(2))
    b, h = (bh // H, bh % H)
    ts = tl.arange(0, 64)
    ds = tl.arange(0, D)
    vd = vi * BV + tl.arange(0, BV)
    gs = tl.arange(0, 32)
    if (qi + 1) * 64 <= QSKIP:
        return
    qt = qi * 64 + ts
    validq = qt < T
    qo = ((b * T + qt[:, None]).to(tl.int64) * H + h) * D + ds[None, :]
    q = tl.load(Q + qo, validq[:, None], 0)
    qn = tl.minimum(64, T - qi * 64).to(tl.float32)
    qb = tl.sum(q.to(tl.float32), 0) / qn
    mk = tl.load(M + bh * (2 * D + 2) + ds)
    vk = tl.load(M + bh * (2 * D + 2) + D + ds)
    mc = tl.load(M + bh * (2 * D + 2) + 2 * D)
    vc = tl.load(M + bh * (2 * D + 2) + 2 * D + 1)
    s = SCALE * 1.4426950408889634
    mu = tl.sum(qb * mk, 0) * s
    var = tl.sum(qb * qb * vk, 0) * s * s
    th1 = mu + B1 * tl.sqrt(tl.maximum(var, 0) + 1e-06)
    th2 = mu + mc + B2 * tl.sqrt(tl.maximum(var + vc, 0) + 1e-06)
    acc = tl.zeros((64, BV), tl.float32)
    l = tl.zeros((64,), tl.float32)
    mx = tl.full((64,), -float('inf'), tl.float32)
    for start in range(0, N, 32):
        js = start + gs
        vj = js < N
        kc = tl.load(KC + ((b * N + js[:, None]) * H + h) * D + ds[None, :], vj[:, None], 0)
        scores = tl.dot(q, kc.T) * s
        proxy = tl.sum(scores, 0) / qn
        c = tl.load(C + (b * N + js) * H + h, vj, 0)
        flags = tl.load(FLAGS + (bh * N + qi) * N + js, vj, -2)
        exact = vj & (flags == -1)
        pure = vj & (flags != -1)
        mixed = vj & (flags >= 0)
        if BRANCH == 0:
            ns = tl.minimum(64, tl.maximum(0, T - js * 64)).to(tl.float32)
            rem = ns - tl.maximum(flags, 0).to(tl.float32)
            if FIXED:
                pure = pure & (rem > 0)
            has = tl.sum(pure.to(tl.int32), 0) > 0
            safe = tl.where(has, tl.where(pure[None, :], scores, -float('inf')), 0.0)
            nm = tl.where(has, tl.maximum(mx, tl.max(safe, 1)), mx)
            al = tl.exp2(tl.where(has, mx - nm, 0.0))
            p = tl.where(has & pure[None, :], tl.exp2(safe - tl.where(has, nm, 0.0)[:, None]), 0.0)
            vs = tl.load(VS + ((b * N + js[:, None]) * H + h) * D + vd[None, :], vj[:, None], 0)
            if FIXED:
                vs = vs * (rem / tl.maximum(ns, 1.0))[:, None]
            acc = acc * al[:, None] + tl.dot(p, vs, input_precision='tf32x3')
            ns = tl.minimum(64, tl.maximum(0, T - js * 64)).to(tl.float32)
            if FIXED:
                l = l * al + tl.sum(p * rem[None, :], 1)
            else:
                l = l * al + tl.sum(p * ns[None, :], 1)
            mx = nm
        else:
            eo = tl.where(exact, gs, 32)
            for z in range(tl.sum(exact.to(tl.int32), 0)):
                off = tl.min(eo, 0)
                j = start + off
                eo = tl.where(gs == off, 32, eo)
                kt = j * 64 + ts
                validk = kt < T
                ko = ((b * T + kt[:, None]).to(tl.int64) * H + h) * D + ds[None, :]
                k = tl.load(K + ko, validk[:, None], 0)
                exs = tl.dot(q, k.T) * s + tl.where(validk[None, :], 0.0, -float('inf'))
                nm = tl.maximum(mx, tl.max(exs, 1))
                al = tl.exp2(mx - nm)
                ep = tl.exp2(exs - nm[:, None])
                vo = ((b * T + kt[:, None]).to(tl.int64) * H + h) * D + vd[None, :]
                v = tl.load(V + vo, validk[:, None], 0)
                acc = acc * al[:, None] + tl.dot(ep.to(q.dtype), v)
                l = l * al + tl.sum(ep, 1)
                mx = nm
            counts = tl.maximum(flags, 0)
            ends = tl.cumsum(counts, 0)
            beginnings = ends - counts
            total = tl.sum(counts, 0)
            for chunk in range(tl.cdiv(total, 64)):
                cr = chunk * 64 + ts
                ok = cr < total
                bo = tl.minimum(tl.sum((cr[:, None] >= ends[None, :]).to(tl.int32), 1), 31)
                begin = tl.gather(beginnings, bo, 0)
                local = tl.minimum(tl.maximum(cr - begin, 0), 63)
                token = tl.load(MASK + ((bh * N + qi) * N + start + bo) * 64 + local, ok, 0).to(tl.int32)
                kt = (start + bo) * 64 + token
                ko = ((b * T + kt[:, None]).to(tl.int64) * H + h) * D + ds[None, :]
                sk = tl.load(K + ko, ok[:, None], 0)
                ms = tl.dot(q, sk.T) * s + tl.where(ok[None, :], 0.0, -float('inf'))
                means = tl.gather(scores, tl.broadcast_to(bo[None, :], (64, 64)), 1)
                nm = tl.maximum(mx, tl.maximum(tl.max(ms, 1), tl.max(tl.where(ok[None, :], means, -float('inf')), 1)))
                al = tl.exp2(mx - nm)
                if FIXED:
                    diff = tl.where(ok[None, :], tl.exp2(ms - nm[:, None]), 0.0)
                else:
                    diff = tl.where(ok[None, :], tl.exp2(ms - nm[:, None]) - tl.exp2(means - nm[:, None]), 0.0)
                vo = ((b * T + kt[:, None]).to(tl.int64) * H + h) * D + vd[None, :]
                sv = tl.load(V + vo, ok[:, None], 0)
                acc = acc * al[:, None] + tl.dot(diff.to(q.dtype), sv)
                l = l * al + tl.sum(diff, 1)
                mx = nm
    oo = ((b * T + qt[:, None]).to(tl.int64) * H + h) * D + vd[None, :]
    tl.store(UO + oo, acc, validq[:, None])
    ro = (b * T + qt) * H + h
    tl.store(LO + ro, l, validq)
    tl.store(MO + ro, tl.where(l != 0, mx, 0.0), validq)

@triton.jit
def merge_partial(U0, L0, M0, U1, L1, M1, O, TOTAL: tl.constexpr, D: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    ds = tl.arange(0, D)
    valid = row < TOTAL
    l0 = tl.load(L0 + row, valid, 0)
    l1 = tl.load(L1 + row, valid, 0)
    m0 = tl.load(M0 + row, valid, 0)
    m1 = tl.load(M1 + row, valid, 0)
    mx = tl.maximum(m0, m1)
    a = tl.exp2(m0 - mx)
    b = tl.exp2(m1 - mx)
    u0 = tl.load(U0 + row[:, None] * D + ds[None, :], valid[:, None], 0)
    u1 = tl.load(U1 + row[:, None] * D + ds[None, :], valid[:, None], 0)
    o = (u0 * a[:, None] + u1 * b[:, None]) / (l0 * a + l1 * b)[:, None]
    tl.store(O + row[:, None] * D + ds[None, :], o, valid[:, None])
PARTIAL = {}

def get_buffers(q):
    b, t, h, d = q.shape
    n = triton.cdiv(t, 64)
    key = (tuple(q.shape), q.device)
    if key not in CACHE:
        kc = torch.empty((b, n, h, d), device=q.device, dtype=torch.bfloat16)
        vs = torch.empty_like(kc, dtype=torch.float32)
        c = torch.empty((b, n, h), device=q.device, dtype=torch.float32)
        m = torch.empty((b * h, 2 * d + 2), device=q.device, dtype=torch.float32)
        kv = torch.empty_like(kc, dtype=torch.float32)
        qb = torch.empty((b * h, n, d), device=q.device, dtype=torch.float32)
        th = torch.empty((b * h, n), device=q.device, dtype=torch.float32)
        flags = torch.empty((b * h, n, n), device=q.device, dtype=torch.int32)
        mask = torch.empty((b * h, n, n, 64), device=q.device, dtype=torch.uint8)
        CACHE[key] = (kc, vs, c, m, kv, qb, th, flags, mask)
    return CACHE[key]

def opt(q, k, v, *, cfg=None, scale=None, tau=1.0, thresh_type='diag', kv_splits=1, sink_tokens=0, sink_start=None, query_threshold_bias=None, **kw):
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
        cap = float(0.0)
        fixed = cfg.get('value_mode', 'residual') == 'fixed'
        full = cfg.get('moment', 'diag') == 'full'
        bf = cfg.get('selector', 'tf32x3') == 'bf16'
        st = (t - sink_tokens if sink_start is None else sink_start) if sink_tokens else 0
        s0 = st // 64
        s1 = triton.cdiv(st + sink_tokens, 64) if sink_tokens else 0
        qskip = sink_tokens if cfg.get('skip_prefix', True) and st == 0 else 0
        vv = cfg.get('bv', 128)
        ww = cfg.get('warps', 4)
        kc, vs, c, m, kv, qb, th, flags, mask = get_buffers(q)
        if cap > 0:
            pool_keyvar[n, b * h](k, v, kc, vs, c, kv, t, n, h, d, num_warps=4, num_stages=1)
        else:
            base.pool[n, b * h](k, v, kc, vs, c, t, n, h, d, num_warps=4, num_stages=1)
        base.moments[b * h,](kc, c, m, t, n, h, d, num_warps=4, num_stages=1)
        base.query_pool[n, b * h](q, qb, t, n, h, d, num_warps=4, num_stages=1)
        if full:
            risk_threshold[triton.cdiv(n, 16), b * h](qb, kc, c, m, th, float(b2), s, t, n, h, d, b1, s0, s1, False, False, num_warps=4, num_stages=1)
        route_opt[triton.cdiv(n, 16), n, b * h](qb, k, kc, c, m, th, flags, mask, s, t, n, h, d, b1, b2, b3, s0, s1, full, bf, qskip, num_warps=4, num_stages=1)
        o = torch.empty_like(v)
        if cfg.get('organization', 'fused') == 'split':
            key = (tuple(q.shape), q.device)
            if key not in PARTIAL:
                us = [torch.empty_like(q, dtype=torch.float32) for _ in range(2)]
                ls = [torch.empty((b, t, h), device=q.device, dtype=torch.float32) for _ in range(2)]
                ms = [torch.empty_like(ls[0]) for _ in range(2)]
                PARTIAL[key] = (us, ls, ms)
            us, ls, ms = PARTIAL[key]
            for z in range(2):
                forward_branch[d // vv, n, b * h](q, k, v, kc, vs, c, m, us[z], ls[z], ms[z], flags, mask, s, t, n, h, d, b1, b2, b3, s0, s1, vv, fixed, qskip, z, num_warps=ww, num_stages=1)
            merge_partial[triton.cdiv(b * t * h, 16),](us[0], ls[0], ms[0], us[1], ls[1], ms[1], o, b * t * h, d, 16, num_warps=4)
        else:
            forward_opt[d // vv, n, b * h](q, k, v, kc, vs, c, m, o, flags, mask, s, t, n, h, d, b1, b2, b3, s0, s1, vv, fixed, qskip, num_warps=ww, num_stages=1)
        if qskip and cfg.get('compute_prefix', True):
            from sol3_attn.backends.workspace import dense_prefix as flash_attn_func
            o[:, :qskip] = flash_attn_func(q[:, :qskip], k, v, softmax_scale=s)
        return o
