"""Backward kernels for the SM120 sol3 walk (exact + representative + covariance).

The forward is a softmax over a virtual token set: exact tokens (walk
bitmaps) plus, per non-full key block j, remaining_j virtual tokens at
score q.kc_j*SCALE + log2(remaining) with value vm_j; the covariance arm
adds gamma*D^-0.5*q@C after normalization. The route (flags/bits/kc/vm
and the selection statistics) is treated as a constant, per the package
contract; kc/vm are block means so their gradients scatter to member
tokens divided by the block's live count, matching pool_qkv.

Numerics mirror the forward where the softmax is involved: log2-domain
scores, SCALE with the log2(e) factor, dot operands rounded to bf16
like the forward, probabilities taken from the saved LSE (no running
max/sum is tracked). Masked loads of q/dO/lse/delta at prefix and
out-of-range rows return zeros, which makes every accumulation
self-zeroing there: the prefix query rows are handled by the dense SDPA
branch outside the autograd Function.

Grids mirror packed_attention_walk: (n, b*h).
"""
import math
import torch
import triton
import triton.language as tl

from .fused_attention import _walk_produce


@triton.jit
def _dq_flush(q, dO_t, lse, delta, K, V, ids, valid, dq, b, h, bh,
              T: tl.constexpr, D: tl.constexpr, SCALE: tl.constexpr,
              KST: tl.constexpr, VST: tl.constexpr):
    # dq contribution of one BK-wide exact chunk: mirrors the forward
    # _exact_flush chunk (same gather, same score form), replacing the
    # online-softmax update with the ds machinery.
    ds = tl.arange(0, D)
    kbase = (b * T + ids[:, None]).to(tl.int64) * KST + h * D
    vbase = (b * T + ids[:, None]).to(tl.int64) * VST + h * D
    k = tl.load(K + kbase + ds[None, :], valid[:, None], 0.)
    v = tl.load(V + vbase + ds[None, :], valid[:, None], 0.)
    score = tl.dot(q, tl.trans(k)) * SCALE
    score = tl.where(valid[None, :], score, -float('inf'))
    p = tl.exp2(score - lse[:, None])
    dp = tl.dot(dO_t, tl.trans(v))
    dS = p * (dp - delta[:, None])
    # softmax gradient in the exp2 domain: d/ds2 = ln2 * d/ds, and
    # ds2/dq = k * SCALE, so the constant is ln2 * SCALE = d^-0.5.
    dq += tl.dot(dS.to(tl.bfloat16), k) * (SCALE * 0.6931471805599453)
    return dq


@triton.jit
def dq_walk(Q, dO, O, K, V, KC, VM, F, BITS, LSE, DELTA, DQ, COV,
            T: tl.constexpr, N: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
            PREFIX: tl.constexpr, SCALE: tl.constexpr,
            BK: tl.constexpr = 64, BQ: tl.constexpr = 64,
            GAMMA: tl.constexpr = 0.,
            QST: tl.constexpr = 0, KST: tl.constexpr = 0, VST: tl.constexpr = 0,
            DST: tl.constexpr = 0):
    # dL/dq for the video rows; also writes DELTA = rowsum(dO*O) for the
    # dkdv pass (launched first on the same stream).
    qtile, bh = tl.program_id(0), tl.program_id(1)
    if (qtile + 1) * BQ <= PREFIX:
        return
    qi = qtile // (64 // BQ)
    b, h = bh // H, bh % H
    rows = qtile * BQ + tl.arange(0, BQ)
    in_rows = (rows >= PREFIX) & (rows < T)
    ds = tl.arange(0, D)
    q = tl.load(Q + (b * T + rows[:, None]).to(tl.int64) * QST + h * D + ds[None, :],
                in_rows[:, None], 0)
    dO_t = tl.load(dO + (b * T + rows[:, None]).to(tl.int64) * DST + h * D + ds[None, :],
                   in_rows[:, None], 0)
    o = tl.load(O + (b * T + rows[:, None]).to(tl.int64) * DST + h * D + ds[None, :],
                in_rows[:, None], 0).to(tl.float32)
    delta = tl.sum(dO_t.to(tl.float32) * o, 1)
    tl.store(DELTA + (b * T + rows).to(tl.int64) * H + h, delta, in_rows)
    lse = tl.load(LSE + (b * T + rows).to(tl.int64) * H + h, in_rows, 0.)

    dq = tl.zeros((BQ, D), tl.float32)
    # representative gradients: same score form as the forward rep pass
    for start in range(0, N, 64):
        js = start + tl.arange(0, 64)
        f = tl.load(F + (bh * N + qi) * N + js, js < N, -1)
        length = tl.minimum(64, tl.maximum(T - js * 64, 0))
        remaining = tl.where(f == -1, 0, length - tl.maximum(f, 0))
        kc = tl.load(KC + (bh * N + js[:, None]) * D + ds[None, :], (js < N)[:, None], 0)
        vm = tl.load(VM + (bh * N + js[:, None]) * D + ds[None, :], (js < N)[:, None], 0)
        score = tl.dot(q, tl.trans(kc)) * SCALE + tl.log2(tl.maximum(remaining, 1).to(tl.float32))[None, :]
        score = tl.where(((js < N) & (remaining > 0))[None, :], score, -float('inf'))
        p = tl.exp2(score - lse[:, None])
        dp = tl.dot(dO_t, tl.trans(vm))
        dS = p * (dp - delta[:, None])
        dq += tl.dot(dS.to(tl.bfloat16), kc) * (SCALE * 0.6931471805599453)
    # exact gradients over the selection bitmaps (same carry packing)
    ts = tl.arange(0, 64)
    idx = tl.arange(0, BK)
    carry = tl.zeros((BK,), tl.int32)
    ncarry = 0
    for j in range(0, N):
        cnt, sel, rank = _walk_produce(F, BITS, bh, qi, j, T, N)
        if cnt > 0:
            tok = j * 64 + ts
            cond = sel[:, None] & (rank[:, None] == ts[None, :])
            val = tl.where(cond, tok[:, None], 0)
            comp = tl.sum(val, 0)
            if ncarry + cnt < BK:
                gidx = tl.minimum(tl.maximum(idx - ncarry, 0), 63)
                comp_at = tl.gather(comp, gidx, 0)
                carry = tl.where((idx >= ncarry) & (idx < ncarry + cnt), comp_at, carry)
                ncarry = ncarry + cnt
            else:
                gidx = tl.minimum(tl.maximum(idx - ncarry, 0), 63)
                comp_at = tl.gather(comp, gidx, 0)
                carry = tl.where(idx >= ncarry, comp_at, carry)
                dq = _dq_flush(q, dO_t, lse, delta, K, V, carry, idx < BK, dq,
                               b, h, bh, T, D, SCALE, KST, VST)
                first = BK - ncarry
                gidx2 = tl.minimum(idx + first, 63)
                comp_at2 = tl.gather(comp, gidx2, 0)
                carry = tl.where(idx < cnt - first, comp_at2, 0)
                ncarry = ncarry + cnt - BK
    if ncarry > 0:
        dq = _dq_flush(q, dO_t, lse, delta, K, V, carry, idx < ncarry, dq,
                       b, h, bh, T, D, SCALE, KST, VST)
    if GAMMA != 0.:
        cd = tl.arange(0, D)
        cov = tl.load(COV + bh * D * D + ds[:, None] * D + cd[None, :])
        dq += tl.dot(dO_t, tl.trans(cov)) * (D ** -.5) * GAMMA
    tl.store(DQ + (b * T + rows[:, None]).to(tl.int64) * DST + h * D + ds[None, :],
             dq, (rows < T)[:, None])  # prefix rows accumulate as zeros


@triton.jit
def dkdv_keyblock(Q, dO, K, V, KC, VM, F, BITS, LSE, DELTA, DK, DV, DC,
                  T: tl.constexpr, N: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
                  PREFIX: tl.constexpr, SINK_BLOCKS: tl.constexpr, SCALE: tl.constexpr,
                  BQ: tl.constexpr = 64, GAMMA: tl.constexpr = 0.,
                  QST: tl.constexpr = 0, KST: tl.constexpr = 0, VST: tl.constexpr = 0,
                  DST: tl.constexpr = 0):
    # dL/dk, dL/dv for one key block: iterate query blocks; exact tile
    # contributions plus representative scalar contributions. The
    # representative gradients accumulate per key block and scatter to
    # member tokens divided by the live count (kc/vm are block means).
    j, bh = tl.program_id(0), tl.program_id(1)
    b, h = bh // H, bh % H
    toks = j * 64 + tl.arange(0, 64)
    live = toks < T
    ds = tl.arange(0, D)
    kbase = (b * T + toks[:, None]).to(tl.int64) * KST + h * D
    vbase = (b * T + toks[:, None]).to(tl.int64) * VST + h * D
    k_t = tl.load(K + kbase + ds[None, :], live[:, None], 0.)
    v_t = tl.load(V + vbase + ds[None, :], live[:, None], 0.)
    length = tl.minimum(64, tl.maximum(T - j * 64, 0))
    n_live = length.to(tl.float32)

    dk = tl.zeros((64, D), tl.float32)
    dv = tl.zeros((64, D), tl.float32)
    dkc = tl.zeros((D,), tl.float32)
    dvm = tl.zeros((D,), tl.float32)

    for qi in range(0, N):
        if (qi + 1) * 64 > PREFIX:
            rows = qi * 64 + tl.arange(0, 64)
            in_rows = (rows >= PREFIX) & (rows < T)
            q_t = tl.load(Q + (b * T + rows[:, None]).to(tl.int64) * QST + h * D + ds[None, :],
                          in_rows[:, None], 0)
            dO_t = tl.load(dO + (b * T + rows[:, None]).to(tl.int64) * DST + h * D + ds[None, :],
                           in_rows[:, None], 0)
            lse = tl.load(LSE + (b * T + rows).to(tl.int64) * H + h, in_rows, 0.)
            delta = tl.load(DELTA + (b * T + rows).to(tl.int64) * H + h, in_rows, 0.)
            f = tl.load(F + (bh * N + qi) * N + j)
            # exact contribution (-1 = whole block; >0 = bits-selected)
            if f != 0:
                if f == -1:
                    sel = live
                elif f > 0:
                    bits_word = tl.load(BITS + (bh * N + qi) * N + j)
                    ts = tl.arange(0, 64)
                    sel = (((bits_word >> ts.to(tl.uint64)) & 1) != 0) & live
                else:
                    sel = tl.zeros((64,), tl.int1)
                if tl.sum(sel.to(tl.int32), 0) > 0:
                    score = tl.dot(q_t, tl.trans(k_t)) * SCALE
                    score = tl.where(sel[None, :], score, -float('inf'))
                    p = tl.exp2(score - lse[:, None])
                    dp = tl.dot(dO_t, tl.trans(v_t))
                    dS = p * (dp - delta[:, None])
                    dk += tl.dot(tl.trans(dS.to(tl.bfloat16)), q_t) * (SCALE * 0.6931471805599453)
                    dv += tl.dot(tl.trans(p.to(tl.bfloat16)), dO_t)
            # representative contribution (never for full blocks)
            remaining = tl.where(f == -1, 0, length - tl.maximum(f, 0))
            if remaining > 0:
                kc_j = tl.load(KC + (bh * N + j) * D + ds)
                vm_j = tl.load(VM + (bh * N + j) * D + ds)
                s = tl.sum(q_t.to(tl.float32) * kc_j.to(tl.float32)[None, :], 1) * SCALE \
                    + tl.log2(remaining.to(tl.float32))
                p_r = tl.exp2(s - lse)
                dp_r = tl.sum(dO_t.to(tl.float32) * vm_j.to(tl.float32)[None, :], 1)
                dS_r = p_r * (dp_r - delta)
                dkc += tl.sum(dS_r[:, None] * q_t.to(tl.float32), 0) * (SCALE * 0.6931471805599453)
                dvm += tl.sum(p_r[:, None] * dO_t.to(tl.float32), 0)

    dk += dkc[None, :] / n_live
    dv += dvm[None, :] / n_live
    if GAMMA != 0.:
        # C = (1/T) sum_{j>=sink} (k_j - kbar_j)^T v_j (fp32 partials,
        # bf16 result); dL/dv_t = (k_t-kbar_j)@dC/T, dL/dk_t = (dC@v_t -
        # dC@vbar_j)/T. Blocks below the sink contribute nothing to C and
        # therefore receive no covariance gradient. The covariance kernel
        # centers over the live rows of each block, so recompute those
        # means from the loaded tiles.
        if j >= SINK_BLOCKS:
            cd = tl.arange(0, D)
            dC = tl.load(DC + bh * D * D + ds[:, None] * D + cd[None, :])
            km = tl.sum(k_t.to(tl.float32), 0) / n_live
            vm_live = tl.sum(v_t.to(tl.float32), 0) / n_live
            khat = (k_t.to(tl.float32) - km[None, :]).to(tl.bfloat16)
            # dC carries the gamma*D^-0.5 factor from cov_grad_partials and
            # arrives bf16 (a full fp32 (128,128) tile exceeds the 99 KB
            # SM120 shared-memory budget alongside the other tiles).
            dv += tl.dot(khat, dC).to(tl.float32) / T
            dk += (tl.dot(v_t, tl.trans(dC)).to(tl.float32)
                   - tl.sum(dC.to(tl.float32) * vm_live[None, :], 1)[None, :]) / T
    tl.store(DK + (b * T + toks[:, None]).to(tl.int64) * DST + h * D + ds[None, :],
             dk, live[:, None])
    tl.store(DV + (b * T + toks[:, None]).to(tl.int64) * DST + h * D + ds[None, :],
             dv, live[:, None])


@triton.jit
def cov_grad_partials(Q, dO, P, T: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
                      PREFIX: tl.constexpr, BLOCKS: tl.constexpr, SPLITS: tl.constexpr,
                      QST: tl.constexpr, DST: tl.constexpr, GAMMA: tl.constexpr):
    # dC_partial = gamma*D^-0.5 * sum_{q blocks in split} q^T dO (video rows)
    split, bh = tl.program_id(0), tl.program_id(1)
    b, h = bh // H, bh % H
    ds = tl.arange(0, D)
    acc = tl.zeros((D, D), tl.float32)
    for i0 in range(0, BLOCKS):
        qi = split * BLOCKS + i0
        if qi * 64 < T and (qi + 1) * 64 > PREFIX:
            rows = qi * 64 + tl.arange(0, 64)
            in_rows = (rows >= PREFIX) & (rows < T)
            q_t = tl.load(Q + (b * T + rows[:, None]).to(tl.int64) * QST + h * D + ds[None, :],
                          in_rows[:, None], 0)
            dO_t = tl.load(dO + (b * T + rows[:, None]).to(tl.int64) * DST + h * D + ds[None, :],
                           in_rows[:, None], 0)
            acc += tl.dot(tl.trans(q_t), dO_t)
    dest = ((bh * SPLITS + split) * D + ds[:, None]) * D + ds[None, :]
    tl.store(P + dest, acc * ((D ** -.5) * GAMMA))


@triton.jit
def cov_grad_reduce(P, OUT, D: tl.constexpr, SPLITS: tl.constexpr, SS: tl.constexpr,
                    BLOCK: tl.constexpr = 256):
    tile, bh = tl.program_id(0), tl.program_id(1)
    x = tile * BLOCK + tl.arange(0, BLOCK)
    sp = tl.arange(0, SS)
    val = tl.load(P + ((bh * SPLITS + sp[:, None]) * D * D + x[None, :]).to(tl.int64),
                  (sp < SPLITS)[:, None] & (x < D * D)[None, :], 0)
    tl.store(OUT + bh * D * D + x, tl.sum(val, 0), x < D * D)


_CACHE = {}


def _scratch(bh, splits, d, device):
    key = (bh, splits, d, device)
    if key not in _CACHE:
        _CACHE.clear()
        _CACHE[key] = (torch.empty((bh, splits, d, d), device=device, dtype=torch.float32),
                       torch.empty((bh, d, d), device=device, dtype=torch.float32),
                       torch.empty((bh, d, d), device=device, dtype=torch.bfloat16))
    return _CACHE[key]


def backward(dO, q, k, v, out, route, lse, covariance, *, gamma, prefix,
             bk=128, warps=8, dkdv_warps=8):
    """Return (dq, dk, dv) bf16 for the walk forward.

    route: dict from make_route (flags/bits/kc/vm); lse the forward's
    WRITE_LSE tensor (b,T,h) fp32; out the forward output (video rows);
    covariance the (b*h,D,D) tensor or None when gamma == 0.
    """
    b, t, h, d = q.shape
    n = triton.cdiv(t, 64)
    s = d ** -0.5 * math.log2(math.e)
    if not dO.is_contiguous():
        dO = dO.contiguous()
    delta = torch.empty((b, t, h), device=q.device, dtype=torch.float32)
    dq = torch.empty_like(q)
    if prefix:
        # dq_walk returns early on fully-prefix tiles, so those rows would
        # keep the empty_like initial contents; the prefix query gradients
        # are the dense SDPA branch's job and must read as zeros here.
        dq[:, :prefix].zero_()
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    dq_walk[(n, b * h)](q, dO, out, k, v, route['kc'], route['vm'],
                        route['flags'], route['bits'], lse, delta, dq,
                        covariance if covariance is not None else q,
                        t, n, h, d, prefix, s, BK=bk, GAMMA=float(gamma),
                        QST=q.stride(1), KST=k.stride(1), VST=v.stride(1),
                        DST=q.stride(1), num_warps=warps, num_stages=1)
    if gamma:
        splits = triton.cdiv(n, 16)
        parts, dc, dc_bf = _scratch(b * h, splits, d, q.device)
        cov_grad_partials[(splits, b * h)](q, dO, parts, t, h, d, prefix,
                                           triton.cdiv(n, splits), splits,
                                           q.stride(1), q.stride(1), float(gamma),
                                           num_warps=8, num_stages=1)
        cov_grad_reduce[(triton.cdiv(d * d, 256), b * h)](parts, dc, d, splits,
                                                          triton.next_power_of_2(splits),
                                                          num_warps=4)
        dc_bf.copy_(dc)
    else:
        dc_bf = q
    dkdv_keyblock[(n, b * h)](q, dO, k, v, route['kc'], route['vm'], route['flags'],
                              route['bits'], lse, delta, dk, dv, dc_bf,
                              t, n, h, d, prefix, (prefix + 63) // 64, s,
                              GAMMA=float(gamma) if gamma else 0.,
                              QST=q.stride(1), KST=k.stride(1), VST=v.stride(1),
                              DST=q.stride(1), num_warps=dkdv_warps, num_stages=1)
    return dq, dk, dv
