import triton
import triton.language as tl

@triton.jit
def rep_merge(Q, KC, VM, FLAGS, HM, OS, LS, O, SCALE: tl.constexpr, T: tl.constexpr, N: tl.constexpr, H: tl.constexpr, D: tl.constexpr, FIXED: tl.constexpr, QSKIP: tl.constexpr, BK: tl.constexpr, VM_BF16: tl.constexpr, USE_ACC: tl.constexpr, SKIPTOK: tl.constexpr, LSE_BASE2: tl.constexpr, HEAD_BASE: tl.constexpr, FO_ALPHA: tl.constexpr, QT: tl.constexpr):
    tile = tl.program_id(0)
    qi = tile // (64 // QT)
    bh = tl.program_id(1) + HEAD_BASE
    b, h = (bh // H, bh % H)
    if (qi + 1) * 64 <= QSKIP:
        return
    tl.static_assert(QT == 64 or 2 == 2)
    qs = qi * 64 + tile % (64 // QT) * QT + tl.arange(0, QT)
    ds = tl.arange(0, D)
    js0 = tl.arange(0, BK)
    vq = qs < T
    q = tl.load(Q + ((b * T + qs[:, None]).to(tl.int64) * H + h) * D + ds[None, :], vq[:, None], 0)
    acc = tl.zeros((QT, D), tl.float32)
    den = tl.zeros((QT,), tl.float32)
    mx = tl.full((QT,), -float('inf'), tl.float32)
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
    live = vq & (qs >= QSKIP)
    packed = (bh - HEAD_BASE) * (T - SKIPTOK) + qs - SKIPTOK
    selected = tl.load(OS + packed[:, None].to(tl.int64) * D + ds[None, :], live[:, None], 0.0).to(tl.float32)
    lse = tl.load(LS + packed, live, -float('inf'))
    if not LSE_BASE2:
        lse *= 1.4426950408889634
    lr = tl.where(den > 0, mx + tl.log2(tl.maximum(den, 1e-30)), -float('inf'))
    common = tl.maximum(lr, lse)
    wr = tl.exp2(lr - common)
    ws = tl.exp2(lse - common)
    out = (wr[:, None] * (acc / tl.maximum(den, 1e-30)[:, None]) + ws[:, None] * selected) / (wr + ws)[:, None]
    hd = tl.arange(0, D)
    hm = tl.load(HM + (bh * D + ds[:, None]) * D + hd[None, :])
    correction = tl.dot(q, hm) * SCALE * FO_ALPHA
    out += (wr / tl.maximum(wr + ws, 1e-30))[:, None] * correction
    tl.store(O + ix, out, live[:, None])
