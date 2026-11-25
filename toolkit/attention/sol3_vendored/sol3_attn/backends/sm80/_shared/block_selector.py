"""Block selection with diagonal key variance for second-stage routing."""
import torch, triton, math
import triton.language as tl
from .sol3_opt import merge_partial
CACHE = {}
LAST_KERNELS = {}
VALUE_CACHE = {}

@triton.jit
def pool_fast(K, V, KC, VM, C, KV, T: tl.constexpr, N: tl.constexpr, H: tl.constexpr, D: tl.constexpr, KEYVAR: tl.constexpr):
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
    si = (bh * N + j) * D + d
    tl.store(KC + si, km)
    tl.store(VM + si, vm)
    if KEYVAR:
        kvar = tl.sum(tl.where(valid[:, None], (k - km[None, :]) * (k - km[None, :]), 0), 0) / n
        tl.store(KV + si, kvar)
    tl.store(C + bh * N + j, tl.log2(n * tl.maximum(rms, 1e-06)))

@triton.jit
def pool_qkv(Q, K, V, QB, KC, VM, C, KV, T: tl.constexpr, N: tl.constexpr, H: tl.constexpr, D: tl.constexpr, KEYVAR: tl.constexpr):
    j, bh = (tl.program_id(0), tl.program_id(1))
    b, h = (bh // H, bh % H)
    t = j * 64 + tl.arange(0, 64)
    d = tl.arange(0, D)
    ix = ((b * T + t[:, None]).to(tl.int64) * H + h) * D + d[None, :]
    valid = t < T
    n = tl.minimum(64, T - j * 64).to(tl.float32)
    qq = tl.load(Q + ix, valid[:, None], 0).to(tl.float32)
    tl.store(QB + (bh * N + j) * D + d, tl.sum(qq, 0) / n)
    k = tl.load(K + ix, valid[:, None], 0).to(tl.float32)
    v = tl.load(V + ix, valid[:, None], 0).to(tl.float32)
    km = tl.sum(k, 0) / n
    vs = tl.sum(v, 0)
    vm = vs / n
    var = tl.sum(tl.sum(tl.where(valid[:, None], (v - vm[None, :]) * (v - vm[None, :]), 0), 0), 0) / (n * D)
    rms = tl.sqrt(tl.maximum(var, 0))
    si = (bh * N + j) * D + d
    tl.store(KC + si, km)
    tl.store(VM + si, vm)
    if KEYVAR:
        kvar = tl.sum(tl.where(valid[:, None], (k - km[None, :]) * (k - km[None, :]), 0), 0) / n
        tl.store(KV + si, kvar)
    tl.store(C + bh * N + j, tl.log2(n * tl.maximum(rms, 1e-06)))

@triton.jit
def moments_fast(KC, C, M, T: tl.constexpr, N: tl.constexpr, H: tl.constexpr, D: tl.constexpr):
    bh = tl.program_id(0)
    b, h = (bh // H, bh % H)
    ds = tl.arange(0, D)
    js = tl.arange(0, 32)
    sm = tl.zeros((D,), tl.float32)
    ss = tl.zeros((D,), tl.float32)
    sc = tl.full((), 0.0, tl.float32)
    sc2 = tl.full((), 0.0, tl.float32)
    for start in range(0, N, 32):
        j = start + js
        valid = j < N
        k = tl.load(KC + (bh * N + j[:, None]) * D + ds[None, :], valid[:, None], 0).to(tl.float32)
        c = tl.load(C + bh * N + j, valid, 0)
        sm += tl.sum(k, 0)
        ss += tl.sum(k * k, 0)
        sc += tl.sum(c, 0)
        sc2 += tl.sum(c * c, 0)
    mean = sm / N
    var = tl.maximum(ss / N - mean * mean, 0)
    mc = sc / N
    vc = tl.maximum(sc2 / N - mc * mc, 0)
    tl.store(M + bh * (2 * D + 2) + ds, mean)
    tl.store(M + bh * (2 * D + 2) + D + ds, var)
    tl.store(M + bh * (2 * D + 2) + 2 * D, mc)
    tl.store(M + bh * (2 * D + 2) + 2 * D + 1, vc)

@triton.jit
def value_global_mean(VM, VG, T: tl.constexpr, N: tl.constexpr, D: tl.constexpr):
    bh = tl.program_id(0)
    ds = tl.arange(0, D)
    js0 = tl.arange(0, 32)
    total = tl.zeros((D,), tl.float32)
    for start in range(0, N, 32):
        js = start + js0
        valid = js < N
        ns = tl.minimum(64, tl.maximum(T - js * 64, 0)).to(tl.float32)
        vm = tl.load(VM + (bh * N + js[:, None]) * D + ds[None, :], valid[:, None], 0).to(tl.float32)
        total += tl.sum(vm * ns[:, None], 0)
    tl.store(VG + bh * D + ds, total / T)

@triton.jit
def value_contrast(VM, VG, VC, N: tl.constexpr, D: tl.constexpr):
    j, bh = (tl.program_id(0), tl.program_id(1))
    ds = tl.arange(0, D)
    vm = tl.load(VM + (bh * N + j) * D + ds).to(tl.float32)
    vg = tl.load(VG + bh * D + ds)
    rms = tl.sqrt(tl.sum((vm - vg) * (vm - vg), 0) / D)
    tl.store(VC + bh * N + j, rms)

@triton.jit
def risk_keyvar_threshold(QB, KC, C, KV, VC, TH2, T: tl.constexpr, N: tl.constexpr, H: tl.constexpr, D: tl.constexpr, SCALE: tl.constexpr, B2: tl.constexpr, RISK_CAP: tl.constexpr, RISK_ALPHA: tl.constexpr, QSKIP: tl.constexpr, BQ: tl.constexpr, BK: tl.constexpr, RISK_CONTRAST: tl.constexpr):
    tile, bh = (tl.program_id(0), tl.program_id(1))
    qs = tile * BQ + tl.arange(0, BQ)
    ds = tl.arange(0, D)
    js0 = tl.arange(0, BK)
    active = (qs < N) & ((qs + 1) * 64 > QSKIP)
    q = tl.load(QB + (bh * N + qs[:, None]) * D + ds[None, :], (qs < N)[:, None], 0)
    q2 = q * q
    ss = tl.zeros((BQ,), tl.float32)
    ss2 = tl.zeros((BQ,), tl.float32)
    for start in range(0, N, BK):
        js = start + js0
        valid = js < N
        kc = tl.load(KC + (bh * N + js[:, None]) * D + ds[None, :], valid[:, None], 0).to(tl.float32)
        c = tl.load(C + bh * N + js, valid, 0)
        score = tl.dot(q, tl.trans(kc), input_precision='tf32x3') * SCALE * 1.4426950408889634 + c[None, :]
        if RISK_CAP > 0:
            kv = tl.load(KV + (bh * N + js[:, None]) * D + ds[None, :], valid[:, None], 0)
            delta = 0.5 * SCALE * SCALE * tl.dot(q2, tl.trans(kv), input_precision='tf32x3')
            clipped = tl.minimum(tl.maximum(delta, 0.0), RISK_CAP)
            if RISK_CONTRAST:
                contrast = tl.load(VC + bh * N + js, valid, 0)
                ns = tl.minimum(64, tl.maximum(T - js * 64, 1)).to(tl.float32)
                rms = tl.exp2(c) / ns
                gamma = tl.exp2(clipped * 1.4426950408889634) - 1.0
                score += tl.log2(tl.maximum(rms[None, :] + RISK_ALPHA * gamma * contrast[None, :], 1e-06) / rms[None, :])
            else:
                score += RISK_ALPHA * clipped * 1.4426950408889634
        ss += tl.sum(tl.where(valid[None, :], score, 0.0), 1)
        ss2 += tl.sum(tl.where(valid[None, :], score * score, 0.0), 1)
    mu = ss / N
    var = tl.maximum(ss2 / N - mu * mu, 0.0)
    threshold = mu + B2 * tl.sqrt(var + 1e-06)
    tl.store(TH2 + bh * N + qs, threshold, qs < N)

@triton.jit
def route_bits(QB, K, KC, C, M, KV, VC, TH2, FLAGS, MASK, SCALE: tl.constexpr, T: tl.constexpr, N: tl.constexpr, H: tl.constexpr, D: tl.constexpr, B1: tl.constexpr, B2: tl.constexpr, B3: tl.constexpr, SINK0: tl.constexpr, SINK1: tl.constexpr, USE_TH2: tl.constexpr, BF16: tl.constexpr, QSKIP: tl.constexpr, QBATCH: tl.constexpr, RISK_CAP: tl.constexpr, RISK_ALPHA: tl.constexpr, RISK_CONTRAST: tl.constexpr):
    tile, j, bh = (tl.program_id(0), tl.program_id(1), tl.program_id(2))
    b, h = (bh // H, bh % H)
    qs = tile * QBATCH + tl.arange(0, QBATCH)
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
    kc = tl.load(KC + (bh * N + j) * D + ds).to(tl.float32)
    c = tl.load(C + bh * N + j)
    proxy = tl.sum(qb * kc[None, :], 1) * s
    if USE_TH2:
        th2 = tl.load(TH2 + bh * N + qs, qs < N, 0)
    ex = activeq & ((proxy > th1) | (tl.abs(qs - j) <= 1) | (j >= SINK0) & (j < SINK1))
    risk = proxy + c
    if RISK_CAP > 0:
        kv = tl.load(KV + (bh * N + j) * D + ds)
        delta = 0.5 * SCALE * SCALE * tl.sum(qb * qb * kv[None, :], 1)
        clipped = tl.minimum(delta, RISK_CAP)
        if RISK_CONTRAST:
            contrast = tl.load(VC + bh * N + j)
            ns = tl.minimum(64, T - j * 64).to(tl.float32)
            rms = tl.exp2(c) / ns
            gamma = tl.exp2(clipped * 1.4426950408889634) - 1.0
            risk += tl.log2(tl.maximum(rms + RISK_ALPHA * gamma * contrast, 1e-06) / rms)
        else:
            risk += RISK_ALPHA * clipped * 1.4426950408889634
    pure = activeq & ~ex & (risk < th2)
    mix = activeq & ~ex & ~pure
    mask = tl.full((QBATCH,), 0, tl.uint64)
    count = tl.zeros((QBATCH,), tl.int32)
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
    flags = tl.where(ex, -1, tl.where(pure, -2, count))
    ix = (bh * N + qs) * N + j
    tl.store(FLAGS + ix, flags, qs < N)
    tl.store(MASK + ix, mask, qs < N)

@triton.jit
def classify_blocks(QB, K, KC, C, M, KV, VC, TH2, FLAGS, MASK, SCALE: tl.constexpr, T: tl.constexpr, N: tl.constexpr, H: tl.constexpr, D: tl.constexpr, B1: tl.constexpr, B2: tl.constexpr, B3: tl.constexpr, SINK0: tl.constexpr, SINK1: tl.constexpr, USE_TH2: tl.constexpr, BF16: tl.constexpr, QSKIP: tl.constexpr, QBATCH: tl.constexpr, RISK_CAP: tl.constexpr, RISK_ALPHA: tl.constexpr, RISK_CONTRAST: tl.constexpr):
    tile, j, bh = (tl.program_id(0), tl.program_id(1), tl.program_id(2))
    b, h = (bh // H, bh % H)
    qs = tile * QBATCH + tl.arange(0, QBATCH)
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
    kc = tl.load(KC + (bh * N + j) * D + ds).to(tl.float32)
    c = tl.load(C + bh * N + j)
    proxy = tl.sum(qb * kc[None, :], 1) * s
    if USE_TH2:
        th2 = tl.load(TH2 + bh * N + qs, qs < N, 0)
    ex = activeq & ((proxy > th1) | (tl.abs(qs - j) <= 1) | (j >= SINK0) & (j < SINK1))
    risk = proxy + c
    if RISK_CAP > 0:
        kv = tl.load(KV + (bh * N + j) * D + ds)
        delta = 0.5 * SCALE * SCALE * tl.sum(qb * qb * kv[None, :], 1)
        clipped = tl.minimum(delta, RISK_CAP)
        if RISK_CONTRAST:
            contrast = tl.load(VC + bh * N + j)
            ns = tl.minimum(64, T - j * 64).to(tl.float32)
            rms = tl.exp2(c) / ns
            gamma = tl.exp2(clipped * 1.4426950408889634) - 1.0
            risk += tl.log2(tl.maximum(rms + RISK_ALPHA * gamma * contrast, 1e-06) / rms)
        else:
            risk += RISK_ALPHA * clipped * 1.4426950408889634
    pure = activeq & ~ex & (risk < th2)
    mix = activeq & ~ex & ~pure
    count = tl.zeros((QBATCH,), tl.int32)
    flags = tl.where(ex, -1, tl.where(pure, -2, count))
    ix = (bh * N + qs) * N + j
    tl.store(FLAGS + ix, flags, qs < N)

@triton.jit
def budget_goals(FLAGS, GOALS, T: tl.constexpr, N: tl.constexpr, NN: tl.constexpr, RHO: tl.constexpr):
    qi, bh = (tl.program_id(0), tl.program_id(1))
    js = tl.arange(0, NN)
    flag = tl.load(FLAGS + (bh * N + qi) * N + js, js < N, -2)
    ns = tl.minimum(64, tl.maximum(T - js * 64, 0))
    full = tl.sum(tl.where(flag == -1, ns, 0), 0)
    tl.store(GOALS + bh * N + qi, tl.maximum(int(T * RHO) - full, 0))

@triton.jit
def sample_scores(QB, K, FLAGS, GOALS, SCORES, T: tl.constexpr, N: tl.constexpr, H: tl.constexpr, D: tl.constexpr, SAMPLES: tl.constexpr, SCALE: tl.constexpr, BF16: tl.constexpr):
    tile, stile, bh = (tl.program_id(0), tl.program_id(1), tl.program_id(2))
    b, h = (bh // H, bh % H)
    qs = tile * 16 + tl.arange(0, 16)
    ds = tl.arange(0, D)
    ss = stile * 128 + tl.arange(0, 128)
    kt = tl.minimum(((ss.to(tl.float32) + 0.5) * T / SAMPLES).to(tl.int32), T - 1)
    goal = tl.load(GOALS + bh * N + qs, qs < N, 0)
    flag = tl.load(FLAGS + (bh * N + qs[:, None]) * N + kt[None, :] // 64, (qs < N)[:, None] & (ss < SAMPLES)[None, :], -2)
    useful = (flag >= 0) & (goal > 0)[:, None]
    a = tl.full((16, 128), -float('inf'), tl.float32)
    if tl.sum(tl.sum(useful.to(tl.int32), 1), 0) > 0:
        q = tl.load(QB + (bh * N + qs[:, None]) * D + ds[None, :], (qs < N)[:, None], 0)
        k = tl.load(K + ((b * T + kt[:, None]).to(tl.int64) * H + h) * D + ds[None, :], (ss < SAMPLES)[:, None], 0)
        if BF16:
            a = tl.dot(q.to(tl.bfloat16), tl.trans(k)) * SCALE * 1.4426950408889634
        else:
            a = tl.dot(q, tl.trans(k.to(tl.float32)), input_precision='tf32x3') * SCALE * 1.4426950408889634
        a = tl.where(useful, a, -float('inf'))
    tl.store(SCORES + (bh * N + qs[:, None]) * SAMPLES + ss[None, :], a, (qs < N)[:, None] & (ss < SAMPLES)[None, :])

@triton.jit
def sampled_cut(SCORES, GOALS, CUT, T: tl.constexpr, N: tl.constexpr, SAMPLES: tl.constexpr):
    qi, bh = (tl.program_id(0), tl.program_id(1))
    ss = tl.arange(0, SAMPLES)
    goal = tl.load(GOALS + bh * N + qi)
    cut = tl.full((), float('inf'), tl.float32)
    if goal > 0:
        a = tl.load(SCORES + (bh * N + qi) * SAMPLES + ss)
        sorted_a = tl.sort(a, descending=True)
        rank = tl.minimum(SAMPLES - 1, tl.maximum(tl.cdiv(goal * SAMPLES, T) - 1, 0))
        val = tl.gather(sorted_a, tl.full((1,), 0, tl.int32) + rank, 0)
        cut = tl.sum(val, 0)
    tl.store(CUT + bh * N + qi, cut)

@triton.jit
def radix_cut(SCORES, GOALS, CUT, T: tl.constexpr, N: tl.constexpr, SAMPLES: tl.constexpr):
    qi, bh = (tl.program_id(0), tl.program_id(1))
    ss = tl.arange(0, SAMPLES)
    bb = tl.arange(0, 16)
    goal = tl.load(GOALS + bh * N + qi)
    cut = tl.full((), float('inf'), tl.float32)
    if goal > 0:
        a = tl.load(SCORES + (bh * N + qi) * SAMPLES + ss)
        raw = a.to(tl.uint32, bitcast=True)
        ordered = tl.where(raw & 2147483648 != 0, ~raw, raw ^ 2147483648).to(tl.uint32)
        live = tl.full((SAMPLES,), True, tl.int1)
        rank = tl.minimum(SAMPLES - 1, tl.maximum(tl.cdiv(goal * SAMPLES, T) - 1, 0))
        for step in tl.static_range(8):
            digit = (ordered >> 28 - step * 4 & 15).to(tl.int32)
            hist = tl.histogram(digit, 16, mask=live)
            tail = tl.cumsum(hist, 0, reverse=True)
            chosen = tl.max(tl.where(tail > rank, bb, -1), 0)
            above = tl.sum(tl.where(bb == chosen, tail - hist, 0), 0)
            rank -= above
            live = live & (digit == chosen)
        pos = tl.min(tl.where(live, ss, SAMPLES), 0)
        one = tl.gather(a, tl.full((1,), 0, tl.int32) + pos, 0)
        cut = tl.sum(one, 0)
    tl.store(CUT + bh * N + qi, cut)

@triton.jit
def histogram_cut(SCORES, FLAGS, CUT, T: tl.constexpr, N: tl.constexpr, SAMPLES: tl.constexpr, NN: tl.constexpr, RHO: tl.constexpr, BINS: tl.constexpr):
    qi, bh = (tl.program_id(0), tl.program_id(1))
    js = tl.arange(0, NN)
    ss = tl.arange(0, SAMPLES)
    bb = tl.arange(0, BINS)
    flag = tl.load(FLAGS + (bh * N + qi) * N + js, js < N, -2)
    ns = tl.minimum(64, tl.maximum(T - js * 64, 0))
    full = tl.sum(tl.where(flag == -1, ns, 0), 0)
    goal = tl.maximum(int(T * RHO) - full, 0)
    desired = tl.cdiv(goal * SAMPLES, T)
    a = tl.load(SCORES + (bh * N + qi) * SAMPLES + ss)
    valid = a > -float('inf')
    count = tl.sum(valid.to(tl.int32), 0)
    lo = tl.min(tl.where(valid, a, float('inf')), 0)
    hi = tl.max(tl.where(valid, a, -float('inf')), 0)
    lo = tl.where(count > 0, lo, 0.0)
    hi = tl.where(count > 0, hi, 0.0)
    step = tl.maximum((hi - lo) / BINS, 1e-06)
    bucket = tl.minimum(BINS - 1, tl.maximum(0, ((tl.where(valid, a, lo) - lo) / step).to(tl.int32)))
    hist = tl.histogram(bucket, BINS, mask=valid)
    tail = tl.cumsum(hist, 0, reverse=True)
    cutbin = tl.max(tl.where(tail >= desired, bb, 0), 0)
    bin_count = tl.sum(tl.where(bb == cutbin, hist, 0), 0)
    above = tl.sum(tl.where(bb == cutbin, tail - hist, 0), 0)
    fraction = tl.minimum(1.0, tl.maximum(0.0, (desired - above).to(tl.float32) / tl.maximum(bin_count, 1)))
    cut = lo + (cutbin.to(tl.float32) + 1.0 - fraction) * step
    cut = tl.where(desired > count, -float('inf'), cut)
    cut = tl.where(goal > 0, cut, float('inf'))
    tl.store(CUT + bh * N + qi, cut)

@triton.jit
def global_route(QB, K, FLAGS, BITS, CUT, SCALE: tl.constexpr, T: tl.constexpr, N: tl.constexpr, H: tl.constexpr, D: tl.constexpr, BF16: tl.constexpr, QSKIP: tl.constexpr, QBATCH: tl.constexpr):
    tile, j, bh = (tl.program_id(0), tl.program_id(1), tl.program_id(2))
    b, h = (bh // H, bh % H)
    qs = tile * QBATCH + tl.arange(0, QBATCH)
    ds = tl.arange(0, D)
    ts = tl.arange(0, 64)
    kt = j * 64 + ts
    ix = (bh * N + qs) * N + j
    flag = tl.load(FLAGS + ix, qs < N, -2)
    cut = tl.load(CUT + bh * N + qs, qs < N, float('inf'))
    mix = (qs < N) & ((qs + 1) * 64 > QSKIP) & (flag >= 0) & (cut < float('inf'))
    count = tl.zeros((QBATCH,), tl.int32)
    bits = tl.zeros((QBATCH,), tl.uint64)
    if tl.sum(mix.to(tl.int32), 0) > 0:
        q = tl.load(QB + (bh * N + qs[:, None]) * D + ds[None, :], (qs < N)[:, None], 0)
        k = tl.load(K + ((b * T + kt[:, None]).to(tl.int64) * H + h) * D + ds[None, :], (kt < T)[:, None], 0)
        if BF16:
            a = tl.dot(q.to(tl.bfloat16), tl.trans(k)) * SCALE * 1.4426950408889634
        else:
            a = tl.dot(q, tl.trans(k.to(tl.float32)), input_precision='tf32x3') * SCALE * 1.4426950408889634
        selected = mix[:, None] & (kt < T)[None, :] & (a > cut[:, None])
        count = tl.sum(selected.to(tl.int32), 1)
        tokenbit = tl.full((64,), 1, tl.uint64) << ts.to(tl.uint64)
        bits = tl.sum(tl.where(selected, tokenbit[None, :], 0), 1)
    tl.store(FLAGS + ix, tl.where(flag >= 0, count, flag), qs < N)
    tl.store(BITS + ix, bits, qs < N)
SAMPLE_CACHE = {}
GOAL_CACHE = {}

@triton.jit
def prefix_compact(FLAGS, BITS, FULL, COUNTS, INDEX, T: tl.constexpr, N: tl.constexpr, CAPACITY: tl.constexpr, NN: tl.constexpr, CHUNK: tl.constexpr):
    qi, bh = (tl.program_id(0), tl.program_id(1))
    js = tl.arange(0, NN)
    flag = tl.load(FLAGS + (bh * N + qi) * N + js, js < N, -2)
    cnt = tl.where(js < N, tl.maximum(flag, 0), 0)
    raw = tl.sum(cnt, 0)
    fullmask = (js < N) & (flag == -1)
    fullrank = tl.cumsum(fullmask.to(tl.int32), 0) - 1
    fc = tl.sum(fullmask.to(tl.int32), 0)
    tl.store(FULL + (bh * N + qi) * N + fullrank, js, fullmask)
    tl.store(COUNTS + (bh * N + qi) * 3, tl.minimum(raw, CAPACITY))
    tl.store(COUNTS + (bh * N + qi) * 3 + 1, fc)
    tl.store(COUNTS + (bh * N + qi) * 3 + 2, tl.maximum(raw - CAPACITY, 0))
    if raw > 0:
        kk = tl.arange(0, CHUNK)
        ts = tl.arange(0, 64)
        carry = tl.full((), 0, tl.int32)
        for start in range(0, N, CHUNK):
            kb = start + kk
            fl = tl.load(FLAGS + (bh * N + qi) * N + kb, kb < N, -2)
            cc = tl.where(kb < N, tl.maximum(fl, 0), 0)
            count = tl.sum(cc, 0)
            if count > 0:
                word = tl.load(BITS + (bh * N + qi) * N + kb, kb < N, 0)
                selected = (word[:, None] >> ts[None, :].to(tl.uint64) & 1 != 0) & (cc > 0)[:, None]
                rank = tl.cumsum(selected.to(tl.int32), 1) - 1
                offsets = tl.cumsum(cc, 0) - cc + carry
                dest = offsets[:, None] + rank
                tl.store(INDEX + (bh * N + qi) * CAPACITY + dest, kb[:, None] * 64 + ts[None, :], selected & (dest < CAPACITY))
            carry += count

@triton.jit
def prefix_indices(FLAGS, OFF, FULL, COUNTS, T: tl.constexpr, N: tl.constexpr, H: tl.constexpr, CAPACITY: tl.constexpr, NN: tl.constexpr):
    qi, bh = (tl.program_id(0), tl.program_id(1))
    js = tl.arange(0, NN)
    ok = js < N
    flag = tl.load(FLAGS + (bh * N + qi) * N + js, ok, -2)
    cnt = tl.maximum(flag, 0)
    cnt = tl.where(ok, cnt, 0)
    end = tl.cumsum(cnt, 0)
    start = end - cnt
    tl.store(OFF + (bh * N + qi) * N + js, start, ok)
    isfull = (flag == -1) & ok
    frank = tl.cumsum(isfull.to(tl.int32), 0) - 1
    tl.store(FULL + (bh * N + qi) * N + frank, js, isfull)
    raw = tl.sum(cnt, 0)
    fc = tl.sum(isfull.to(tl.int32), 0)
    tl.store(COUNTS + (bh * N + qi) * 3, tl.minimum(raw, CAPACITY))
    tl.store(COUNTS + (bh * N + qi) * 3 + 1, fc)
    tl.store(COUNTS + (bh * N + qi) * 3 + 2, tl.maximum(raw - CAPACITY, 0))
    clipped = tl.minimum(cnt, tl.maximum(CAPACITY - start, 0))
    tl.store(FLAGS + (bh * N + qi) * N + js, tl.where(flag >= 0, clipped, flag), ok)

@triton.jit
def scatter_indices(BITS, OFF, FLAGS, INDEX, N: tl.constexpr, CAPACITY: tl.constexpr):
    j, qi, bh = (tl.program_id(0), tl.program_id(1), tl.program_id(2))
    ts = tl.arange(0, 64)
    flag = tl.load(FLAGS + (bh * N + qi) * N + j)
    if flag > 0:
        bits = tl.load(BITS + (bh * N + qi) * N + j)
        selected = bits >> ts.to(tl.uint64) & 1 != 0
        rank = tl.cumsum(selected.to(tl.int32), 0) - 1
        start = tl.load(OFF + (bh * N + qi) * N + j)
        tl.store(INDEX + (bh * N + qi) * CAPACITY + start + rank, j * 64 + ts, selected & (rank < flag) & (start + rank < CAPACITY))

@triton.jit
def scatter_batched(BITS, OFF, FLAGS, INDEX, N: tl.constexpr, CAPACITY: tl.constexpr, QBATCH: tl.constexpr):
    tile, j, bh = (tl.program_id(0), tl.program_id(1), tl.program_id(2))
    qs = tile * QBATCH + tl.arange(0, QBATCH)
    ts = tl.arange(0, 64)
    ix = (bh * N + qs) * N + j
    flag = tl.load(FLAGS + ix, qs < N, 0)
    if tl.sum((flag > 0).to(tl.int32), 0) > 0:
        bits = tl.load(BITS + ix, qs < N, 0)
        selected = bits[:, None] >> ts[None, :].to(tl.uint64) & 1 != 0
        rank = tl.cumsum(selected.to(tl.int32), 1) - 1
        start = tl.load(OFF + ix, qs < N, 0)
        tl.store(INDEX + (bh * N + qs[:, None]) * CAPACITY + start[:, None] + rank, j * 64 + ts[None, :], (qs < N)[:, None] & selected & (rank < flag[:, None]) & (start[:, None] + rank < CAPACITY))

@triton.jit
def scatter_keys_batched(BITS, OFF, FLAGS, INDEX, N: tl.constexpr, CAPACITY: tl.constexpr, JBATCH: tl.constexpr):
    tile, qi, bh = (tl.program_id(0), tl.program_id(1), tl.program_id(2))
    js = tile * JBATCH + tl.arange(0, JBATCH)
    ts = tl.arange(0, 64)
    ix = (bh * N + qi) * N + js
    flag = tl.load(FLAGS + ix, js < N, 0)
    if tl.sum((flag > 0).to(tl.int32), 0) > 0:
        bits = tl.load(BITS + ix, js < N, 0)
        selected = bits[:, None] >> ts[None, :].to(tl.uint64) & 1 != 0
        rank = tl.cumsum(selected.to(tl.int32), 1) - 1
        start = tl.load(OFF + ix, js < N, 0)
        tl.store(INDEX + (bh * N + qi) * CAPACITY + start[:, None] + rank, js[:, None] * 64 + ts[None, :], (js < N)[:, None] & selected & (rank < flag[:, None]) & (start[:, None] + rank < CAPACITY))

@triton.jit
def representatives(Q, KC, VM, FLAGS, U, L, M, SCALE: tl.constexpr, T: tl.constexpr, N: tl.constexpr, H: tl.constexpr, D: tl.constexpr, FIXED: tl.constexpr, QSKIP: tl.constexpr, BK: tl.constexpr, VM_BF16: tl.constexpr, USE_ACC: tl.constexpr):
    qi, bh = (tl.program_id(0), tl.program_id(1))
    b, h = (bh // H, bh % H)
    if (qi + 1) * 64 <= QSKIP:
        return
    qs = qi * 64 + tl.arange(0, 64)
    ds = tl.arange(0, D)
    js0 = tl.arange(0, BK)
    vq = qs < T
    q = tl.load(Q + ((b * T + qs[:, None]).to(tl.int64) * H + h) * D + ds[None, :], vq[:, None], 0)
    acc = tl.zeros((64, D), tl.float32)
    den = tl.zeros((64,), tl.float32)
    mx = tl.full((64,), -float('inf'), tl.float32)
    s = SCALE * 1.4426950408889634
    for start in range(0, N, BK):
        js = start + js0
        vj = js < N
        flag = tl.load(FLAGS + (bh * N + qi) * N + js, vj, -1)
        ns = tl.minimum(64, tl.maximum(T - js * 64, 0)).to(tl.float32)
        count = ns - tl.maximum(flag, 0).to(tl.float32) if FIXED else ns
        keep = vj & (flag != -1) & (count > 0)
        kc = tl.load(KC + (bh * N + js[:, None]) * D + ds[None, :], vj[:, None], 0)
        scores = tl.dot(q, tl.trans(kc)) * s
        scores += tl.log2(tl.maximum(count, 1.0))[None, :]
        scores = tl.where(keep[None, :], scores, -float('inf'))
        has = tl.sum(keep.to(tl.int32), 0) > 0
        nm = tl.where(has, tl.maximum(mx, tl.max(scores, 1)), mx)
        alpha = tl.exp2(tl.where(has, mx - nm, 0.0))
        p = tl.where(keep[None, :], tl.exp2(scores - tl.where(has, nm, 0.0)[:, None]), 0.0)
        vm = tl.load(VM + (bh * N + js[:, None]) * D + ds[None, :], vj[:, None], 0)
        if VM_BF16:
            if USE_ACC:
                acc = tl.dot(p.to(tl.bfloat16), vm, acc * alpha[:, None])
            else:
                acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), vm)
        elif USE_ACC:
            acc = tl.dot(p, vm, acc * alpha[:, None], input_precision='tf32x3')
        else:
            acc = acc * alpha[:, None] + tl.dot(p, vm, input_precision='tf32x3')
        den = den * alpha + tl.sum(p, 1)
        mx = nm
    ix = ((b * T + qs[:, None]).to(tl.int64) * H + h) * D + ds[None, :]
    ro = (b * T + qs) * H + h
    tl.store(U + ix, acc, vq[:, None])
    tl.store(L + ro, den, vq)
    tl.store(M + ro, tl.where(den != 0, mx, 0.0), vq)

@triton.jit
def selected_attention(Q, K, V, KC, INDEX, FULL, COUNTS, U, L, M, SCALE: tl.constexpr, T: tl.constexpr, N: tl.constexpr, H: tl.constexpr, D: tl.constexpr, CAPACITY: tl.constexpr, FIXED: tl.constexpr, QSKIP: tl.constexpr, BK: tl.constexpr, USE_ACC: tl.constexpr):
    qi, bh = (tl.program_id(0), tl.program_id(1))
    b, h = (bh // H, bh % H)
    if (qi + 1) * 64 <= QSKIP:
        return
    qs = qi * 64 + tl.arange(0, 64)
    ks = tl.arange(0, BK)
    ds = tl.arange(0, D)
    vq = qs < T
    q = tl.load(Q + ((b * T + qs[:, None]).to(tl.int64) * H + h) * D + ds[None, :], vq[:, None], 0)
    sc = tl.load(COUNTS + (bh * N + qi) * 3)
    fc = tl.load(COUNTS + (bh * N + qi) * 3 + 1)
    total = fc * 64 + sc
    acc = tl.zeros((64, D), tl.float32)
    den = tl.zeros((64,), tl.float32)
    mx = tl.full((64,), -float('inf'), tl.float32)
    s = SCALE * 1.4426950408889634
    for chunk in range(tl.cdiv(total, BK)):
        rank = chunk * BK + ks
        isfull = rank < fc * 64
        fj = tl.load(FULL + (bh * N + qi) * N + rank // 64, isfull, 0)
        si = rank - fc * 64
        sm = (si >= 0) & (si < sc)
        token = tl.load(INDEX + (bh * N + qi) * CAPACITY + si, sm, 0)
        kt = tl.where(isfull, fj * 64 + rank % 64, token)
        valid = (rank < total) & (kt < T)
        ix = ((b * T + kt[:, None]).to(tl.int64) * H + h) * D + ds[None, :]
        k = tl.load(K + ix, valid[:, None], 0)
        v = tl.load(V + ix, valid[:, None], 0)
        scores = tl.dot(q, tl.trans(k)) * s
        scores = tl.where(valid[None, :], scores, -float('inf'))
        if FIXED:
            nm = tl.maximum(mx, tl.max(scores, 1))
            p = tl.exp2(scores - nm[:, None])
        else:
            center = tl.load(KC + (bh * N + kt[:, None] // 64) * D + ds[None, :], (valid & ~isfull)[:, None], 0)
            approx = tl.dot(q, tl.trans(center)) * s
            approx = tl.where((valid & ~isfull)[None, :], approx, -float('inf'))
            nm = tl.maximum(mx, tl.maximum(tl.max(scores, 1), tl.max(approx, 1)))
            p = tl.exp2(scores - nm[:, None]) - tl.exp2(approx - nm[:, None])
        alpha = tl.exp2(mx - nm)
        if USE_ACC:
            acc = tl.dot(p.to(tl.bfloat16), v, acc * alpha[:, None])
        else:
            acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v)
        den = den * alpha + tl.sum(p, 1)
        mx = nm
    ix = ((b * T + qs[:, None]).to(tl.int64) * H + h) * D + ds[None, :]
    ro = (b * T + qs) * H + h
    tl.store(U + ix, acc, vq[:, None])
    tl.store(L + ro, den, vq)
    tl.store(M + ro, tl.where(den != 0, mx, 0.0), vq)

@triton.jit
def fused_attention(Q, K, V, KC, VM, FLAGS, INDEX, FULL, COUNTS, O, SCALE: tl.constexpr, T: tl.constexpr, N: tl.constexpr, H: tl.constexpr, D: tl.constexpr, CAPACITY: tl.constexpr, FIXED: tl.constexpr, QSKIP: tl.constexpr, BK: tl.constexpr, REP_BK: tl.constexpr, VM_BF16: tl.constexpr, USE_ACC: tl.constexpr, BQ: tl.constexpr):
    tile, bh = (tl.program_id(0), tl.program_id(1))
    qi = tile // (64 // BQ)
    b, h = (bh // H, bh % H)
    if (tile + 1) * BQ <= QSKIP:
        return
    qs = tile * BQ + tl.arange(0, BQ)
    ks = tl.arange(0, BK)
    ds = tl.arange(0, D)
    vq = qs < T
    q = tl.load(Q + ((b * T + qs[:, None]).to(tl.int64) * H + h) * D + ds[None, :], vq[:, None], 0)
    sc = tl.load(COUNTS + (bh * N + qi) * 3)
    fc = tl.load(COUNTS + (bh * N + qi) * 3 + 1)
    total = fc * 64 + sc
    acc = tl.zeros((BQ, D), tl.float32)
    den = tl.zeros((BQ,), tl.float32)
    mx = tl.full((BQ,), -float('inf'), tl.float32)
    s = SCALE * 1.4426950408889634
    js0 = tl.arange(0, REP_BK)
    for start in range(0, N, REP_BK):
        js = start + js0
        vj = js < N
        flag = tl.load(FLAGS + (bh * N + qi) * N + js, vj, -1)
        ns = tl.minimum(64, tl.maximum(T - js * 64, 0)).to(tl.float32)
        count = ns - tl.maximum(flag, 0).to(tl.float32) if FIXED else ns
        keep = vj & (flag != -1) & (count > 0)
        kc = tl.load(KC + (bh * N + js[:, None]) * D + ds[None, :], vj[:, None], 0)
        scores = tl.dot(q, tl.trans(kc)) * s
        scores += tl.log2(tl.maximum(count, 1.0))[None, :]
        scores = tl.where(keep[None, :], scores, -float('inf'))
        has = tl.sum(keep.to(tl.int32), 0) > 0
        nm = tl.where(has, tl.maximum(mx, tl.max(scores, 1)), mx)
        alpha = tl.exp2(tl.where(has, mx - nm, 0.0))
        p = tl.where(keep[None, :], tl.exp2(scores - tl.where(has, nm, 0.0)[:, None]), 0.0)
        vm = tl.load(VM + (bh * N + js[:, None]) * D + ds[None, :], vj[:, None], 0)
        if VM_BF16:
            if USE_ACC:
                acc = tl.dot(p.to(tl.bfloat16), vm, acc * alpha[:, None])
            else:
                acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), vm)
        elif USE_ACC:
            acc = tl.dot(p, vm, acc * alpha[:, None], input_precision='tf32x3')
        else:
            acc = acc * alpha[:, None] + tl.dot(p, vm, input_precision='tf32x3')
        den = den * alpha + tl.sum(p, 1)
        mx = nm
    for chunk in range(tl.cdiv(total, BK)):
        rank = chunk * BK + ks
        isfull = rank < fc * 64
        fj = tl.load(FULL + (bh * N + qi) * N + rank // 64, isfull, 0)
        si = rank - fc * 64
        sm = (si >= 0) & (si < sc)
        token = tl.load(INDEX + (bh * N + qi) * CAPACITY + si, sm, 0)
        kt = tl.where(isfull, fj * 64 + rank % 64, token)
        valid = (rank < total) & (kt < T)
        ix = ((b * T + kt[:, None]).to(tl.int64) * H + h) * D + ds[None, :]
        k = tl.load(K + ix, valid[:, None], 0)
        v = tl.load(V + ix, valid[:, None], 0)
        scores = tl.dot(q, tl.trans(k)) * s
        scores = tl.where(valid[None, :], scores, -float('inf'))
        if FIXED:
            nm = tl.maximum(mx, tl.max(scores, 1))
            p = tl.exp2(scores - nm[:, None])
        else:
            center = tl.load(KC + (bh * N + kt[:, None] // 64) * D + ds[None, :], (valid & ~isfull)[:, None], 0)
            approx = tl.dot(q, tl.trans(center)) * s
            approx = tl.where((valid & ~isfull)[None, :], approx, -float('inf'))
            nm = tl.maximum(mx, tl.maximum(tl.max(scores, 1), tl.max(approx, 1)))
            p = tl.exp2(scores - nm[:, None]) - tl.exp2(approx - nm[:, None])
        alpha = tl.exp2(mx - nm)
        if USE_ACC:
            acc = tl.dot(p.to(tl.bfloat16), v, acc * alpha[:, None])
        else:
            acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v)
        den = den * alpha + tl.sum(p, 1)
        mx = nm
    ix = ((b * T + qs[:, None]).to(tl.int64) * H + h) * D + ds[None, :]
    ro = (b * T + qs) * H + h
    tl.store(O + ix, acc / den[:, None], vq[:, None])

def buffers(q, capacity, vm_bf16, fused=False):
    b, t, h, d = q.shape
    n = triton.cdiv(t, 64)
    key = (tuple(q.shape), q.device, capacity, vm_bf16, fused)
    if key not in CACHE:
        CACHE.clear()
        SAMPLE_CACHE.clear()
        VALUE_CACHE.clear()
        kc = torch.empty((b * h, n, d), device=q.device, dtype=torch.bfloat16)
        vm = torch.empty_like(kc, dtype=torch.bfloat16 if vm_bf16 else torch.float32)
        kv = torch.empty_like(kc, dtype=torch.float32)
        c = torch.empty((b * h, n), device=q.device, dtype=torch.float32)
        mom = torch.empty((b * h, 2 * d + 2), device=q.device, dtype=torch.float32)
        qb = torch.empty_like(kc, dtype=torch.float32)
        th = torch.empty_like(c)
        th2 = torch.empty_like(c)
        flags = torch.empty((b * h, n, n), device=q.device, dtype=torch.int32)
        bits = torch.empty_like(flags, dtype=torch.uint64)
        off = torch.empty_like(flags)
        full = torch.empty_like(flags)
        counts = torch.empty((b * h, n, 3), device=q.device, dtype=torch.int32)
        index = torch.empty((b * h, n, capacity), device=q.device, dtype=torch.int32)
        if fused:
            us = ls = ms = []
        else:
            us = [torch.empty_like(q, dtype=torch.float32) for _ in range(2)]
            ls = [torch.empty((b, t, h), device=q.device, dtype=torch.float32) for _ in range(2)]
            ms = [torch.empty_like(ls[0]) for _ in range(2)]
        CACHE[key] = (kc, vm, c, mom, kv, qb, th, flags, bits, off, full, counts, index, us, ls, ms, th2)
    return CACHE[key]

def fast(q, k, v, *, cfg=None, scale=None, tau=1.0, thresh_type='diag', kv_splits=1, sink_tokens=0, sink_start=None, query_threshold_bias=None, **kw):
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
        if global_budget is None:
            LAST_KERNELS['route_bits'] = route_bits[triton.cdiv(n, rt), n, b * h](qb, k, kc, c, mom, kv, vc, th2, flags, bits, s, t, n, h, d, b1, b2, b3, s0, s1, full_risk, bf, qskip, rt, risk_cap, risk_alpha, risk_contrast, num_warps=cfg.get('route_warps', 4), num_stages=1)
        else:
            samples = cfg.get('threshold_samples', 2048)
            key = (tuple(q.shape), q.device, samples)
            if key not in SAMPLE_CACHE:
                SAMPLE_CACHE[key] = torch.empty((b * h, n, samples), device=q.device, dtype=torch.float32)
            scores = SAMPLE_CACHE[key]
            classify_blocks[triton.cdiv(n, rt), n, b * h](qb, k, kc, c, mom, kv, vc, th2, flags, bits, s, t, n, h, d, b1, b2, b3, s0, s1, full_risk, bf, qskip, rt, risk_cap, risk_alpha, risk_contrast, num_warps=cfg.get('route_warps', 4), num_stages=1)
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
            LAST_KERNELS['global_route'] = global_route[triton.cdiv(n, rt), n, b * h](qb, k, flags, bits, th, s, t, n, h, d, bf, qskip, rt, num_warps=cfg.get('route_warps', 4), num_stages=1)
        if cfg.get('compact_mode') == 'fused':
            LAST_KERNELS['prefix_compact'] = prefix_compact[n, b * h](flags, bits, full, counts, index, t, n, capacity, triton.next_power_of_2(n), cfg.get('compact_chunk', 16), num_warps=4)
        else:
            prefix_indices[n, b * h](flags, off, full, counts, t, n, h, capacity, triton.next_power_of_2(n), num_warps=4)
            batch = cfg.get('scatter_batch', 16)
            if cfg.get('scatter_layout', 'query') == 'key':
                scatter_keys_batched[triton.cdiv(n, batch), n, b * h](bits, off, flags, index, n, capacity, batch, num_warps=4)
            elif batch == 1:
                scatter_indices[n, n, b * h](bits, off, flags, index, n, capacity, num_warps=4)
            else:
                scatter_batched[triton.cdiv(n, batch), n, b * h](bits, off, flags, index, n, capacity, batch, num_warps=4)
        o = torch.empty_like(v)
        if fused:
            LAST_KERNELS['fused_attention'] = fused_attention[triton.cdiv(t, attention_qtile), b * h](q, k, v, kc, vm, flags, index, full, counts, o, s, t, n, h, d, capacity, fixed, qskip, cfg.get('selected_bk', 128), cfg.get('rep_bk', 64), vm_bf16, cfg.get('use_dot_accumulator', True), attention_qtile, num_warps=cfg.get('selected_warps', 4), num_stages=cfg.get('selected_stages', 1), **{'maxnreg': cfg['maxnreg']} if cfg.get('maxnreg') else {})
        else:
            LAST_KERNELS['representatives'] = representatives[n, b * h](q, kc, vm, flags, us[0], ls[0], ms[0], s, t, n, h, d, fixed, qskip, cfg.get('rep_bk', 64), vm_bf16, cfg.get('use_dot_accumulator', True), num_warps=cfg.get('rep_warps', 4), num_stages=1)
            LAST_KERNELS['selected_attention'] = selected_attention[n, b * h](q, k, v, kc, index, full, counts, us[1], ls[1], ms[1], s, t, n, h, d, capacity, fixed, qskip, cfg.get('selected_bk', 64), cfg.get('use_dot_accumulator', True), num_warps=cfg.get('selected_warps', 4), num_stages=cfg.get('selected_stages', 1))
            merge_partial[triton.cdiv(b * t * h, 16),](us[0], ls[0], ms[0], us[1], ls[1], ms[1], o, b * t * h, d, 16, num_warps=4)
        if qskip and cfg.get('compute_prefix', True):
            from sol3_attn.backends.workspace import dense_prefix as flash_attn_func
            o[:, :qskip] = flash_attn_func(q[:, :qskip], k, v, softmax_scale=s)
        return o
