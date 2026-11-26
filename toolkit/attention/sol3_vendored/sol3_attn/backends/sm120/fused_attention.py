"""Exact-token bitmap walk and joint representative/exact online softmax.

Vendored from sol3-attn 0.1.0 (Apache-2.0; see LICENSE and
THIRD_PARTY_NOTICES.md in the vendored root). Local rewrite: the exact
token sequence is walked straight out of the route's flags/bits bitmaps
through a BK-wide carry register -- the (batch*heads, blocks, tokens)
compaction index and its compaction kernels are gone. The emitted
chunks are the same tokens in the same order and the same BK grouping
the ids path read (key-block-major, rank-ascending), so the
online-softmax chain is bit-reproduced.

Two Triton hazards found while building this (see
docs/kernel_opt/sol3/SOL3.md, walk-kernel status): a bare (64,) sel
tensor in the compaction where() broadcasts on the position axis, not
the lane axis -- unselected adjacent lanes then sum into the chunk
(the [192, 0, 391, 0, 399] corruption); and a 0-d-condition tl.where
feeding a (64,64) broadcast->where->reduce chain scrambles the value
tensor. The forms below avoid both: sel is indexed per-lane
(sel[:, None]), producers use a runtime scalar if, and the merge uses
tl.gather shifts instead of compare-select sums.
"""
import math
import torch
import triton
import triton.language as tl


@triton.jit
def _walk_produce(F, BITS, bh, qi, j, T, N: tl.constexpr):
    # Per-key-block exact-token scan, shared by the forward walk and the
    # dq kernel so the sparse pattern is single-sourced. Two Triton
    # hazards this form avoids (see module docstring): a bare (64,) sel
    # broadcasts on the position axis (lane-sum corruption), and a
    # 0-d-condition tl.where feeding a (64,64) where->reduce chain
    # scrambles values. Returns cnt (route scalar), the per-lane
    # selection, and the per-lane rank within the selection.
    ts = tl.arange(0, 64)
    f = tl.load(F + (bh * N + qi) * N + j)
    length = tl.minimum(64, tl.maximum(T - j * 64, 0))
    # cnt from the flag scalar, offsets_kernel's exact formula: no
    # reduction, and skipped/empty blocks select nothing even though
    # their bits words are uninitialized
    cnt = tl.where(f == -1, length, tl.maximum(f, 0))
    if cnt > 0:
        in_t = (j * 64 + ts) < T
        bits_word = tl.load(BITS + (bh * N + qi) * N + j)
        sel_bit = ((bits_word >> ts.to(tl.uint64)) & 1) != 0
        lower = (tl.full((64,), 1, tl.uint64) << ts.to(tl.uint64)) - 1
        rank_popc = tl.inline_asm_elementwise(
            "popc.b64 $0, $1;", "=r,l", [bits_word & lower], dtype=tl.int32, is_pure=True, pack=1)
        # scalar-if producer: a 0-d-condition tl.where feeding the
        # (64,64) broadcast->where->reduce chain scrambles the values
        if f == -1:
            sel = in_t
            rank = ts
        else:
            sel = sel_bit & in_t
            rank = rank_popc
    else:
        sel = tl.zeros((64,), tl.int1)
        rank = tl.zeros((64,), tl.int32)
    return cnt, sel, rank


@triton.jit
def _exact_flush(q, q8, K, V, qs, vs, ids, valid, m, l, acc, b, h, bh,
                 T: tl.constexpr, D: tl.constexpr, SCALE: tl.constexpr,
                 KST: tl.constexpr, VST: tl.constexpr,
                 HEAD_MAJOR: tl.constexpr, FP8: tl.constexpr):
    # verbatim body of the ids path's exact-pass chunk (fused_attention
    # upstream): one BK-wide dot pair over the given token indices.
    ds = tl.arange(0, D)
    if HEAD_MAJOR:
        kbase = (bh * T + ids[:, None]).to(tl.int64) * D
        vbase = (bh * T + ids[:, None]).to(tl.int64) * D
    else:
        kbase = (b * T + ids[:, None]).to(tl.int64) * KST + h * D
        vbase = (b * T + ids[:, None]).to(tl.int64) * VST + h * D
    k = tl.load(K + kbase + ds[None, :], valid[:, None], 0.)
    v = tl.load(V + vbase + ds[None, :], valid[:, None], 0.)
    if FP8:
        score = tl.dot(q8, tl.trans(k)) * qs[:, None] * SCALE
    else:
        score = tl.dot(q, tl.trans(k)) * SCALE
    score = tl.where(valid[None, :], score, -float('inf'))
    mm = tl.maximum(m, tl.max(score, 1))
    alpha = tl.exp2(m - mm)
    p = tl.exp2(score - mm[:, None])
    if FP8:
        delta = tl.dot((p * 448.).to(tl.float8e4nv), v) * (vs[None, :] / 448.)
    else:
        delta = tl.dot(p.to(tl.bfloat16), v)
    acc = acc * alpha[:, None] + delta
    l = l * alpha + tl.sum(p, 1)
    m = mm
    return m, l, acc


@triton.jit
def packed_attention_walk(Q, K, V, KC, VM, F, BITS, O, KS, VS, COV, LSE, OSM,
                          T: tl.constexpr, N: tl.constexpr, H: tl.constexpr,
                          D: tl.constexpr, PREFIX: tl.constexpr, SCALE: tl.constexpr,
                          BK: tl.constexpr = 64, HEAD_MAJOR: tl.constexpr = False,
                          FP8: tl.constexpr = False, BQ: tl.constexpr = 64,
                          GAMMA: tl.constexpr = 0., QST: tl.constexpr = 0,
                          KST: tl.constexpr = 0, VST: tl.constexpr = 0,
                          OST: tl.constexpr = 0, WRITE_LSE: tl.constexpr = False,
                          STORE_SM: tl.constexpr = False):
    # grid (n, b*h): one program per (query-block, batch*head). The exact
    # pass merges each key-block's selected tokens (rank-ascending) after
    # the pending carry; when the carry reaches BK tokens, one dot fires
    # with exactly the chunk the ids path would have read. A block adds
    # at most 64 tokens and the carry holds fewer than BK, so a block
    # spans at most one flush and the remainder stays under BK.
    qtile, bh = tl.program_id(0), tl.program_id(1)
    if (qtile + 1) * BQ <= PREFIX:
        return
    qi = qtile // (64 // BQ)
    b, h = bh // H, bh % H
    rows = qtile * BQ + tl.arange(0, BQ)
    ds = tl.arange(0, D)
    q = tl.load(Q + (b * T + rows[:, None]).to(tl.int64) * QST + h * D + ds[None, :], (rows < T)[:, None], 0)
    if FP8:
        ks_s = tl.load(KS + bh * D + ds)
        vs_s = tl.load(VS + bh * D + ds)
        scaled_q = q.to(tl.float32) * ks_s[None, :]
        qs_s = tl.maximum(tl.max(tl.abs(scaled_q), 1) / 448., 1.e-12)
        q8 = (scaled_q / qs_s[:, None]).to(tl.float8e4nv)
    else:
        qs_s = q
        vs_s = q
        q8 = q
    m = tl.full((BQ,), -float('inf'), tl.float32)
    l = tl.zeros((BQ,), tl.float32)
    acc = tl.zeros((BQ, D), tl.float32)
    # representative pass: fixed original-block value means; residual mass
    # is the remaining token count.
    for start in range(0, N, 64):
        js = start + tl.arange(0, 64)
        f = tl.load(F + (bh * N + qi) * N + js, js < N, -1)
        length = tl.minimum(64, tl.maximum(T - js * 64, 0))
        remaining = tl.where(f == -1, 0, length - tl.maximum(f, 0))
        kc = tl.load(KC + (bh * N + js[:, None]) * D + ds[None, :], (js < N)[:, None], 0)
        vm = tl.load(VM + (bh * N + js[:, None]) * D + ds[None, :], (js < N)[:, None], 0)
        score = tl.dot(q, tl.trans(kc)) * SCALE + tl.log2(tl.maximum(remaining, 1).to(tl.float32))[None, :]
        score = tl.where(((js < N) & (remaining > 0))[None, :], score, -float('inf'))
        mm = tl.maximum(m, tl.max(score, 1))
        safe = tl.where(mm == -float('inf'), 0., mm)
        alpha = tl.exp2(m - safe)
        p = tl.exp2(score - safe[:, None])
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), vm)
        l = l * alpha + tl.sum(p, 1)
        m = mm
    # exact pass over the selection bitmaps
    ts = tl.arange(0, 64)
    idx = tl.arange(0, BK)
    carry = tl.zeros((BK,), tl.int32)
    ncarry = 0
    for j in range(0, N):
        cnt, sel, rank = _walk_produce(F, BITS, bh, qi, j, T, N)
        if cnt > 0:
            tok = j * 64 + ts
            # named-axis compaction: sel gates LANES (axis 0), the rank
            # equality gates POSITIONS (axis 1). A bare (64,) sel would
            # broadcast onto positions and sum unselected neighbors in.
            cond = sel[:, None] & (rank[:, None] == ts[None, :])
            val = tl.where(cond, tok[:, None], 0)
            comp = tl.sum(val, 0)
            if ncarry + cnt < BK:
                # merge after the pending carry, no flush yet
                gidx = tl.minimum(tl.maximum(idx - ncarry, 0), 63)
                comp_at = tl.gather(comp, gidx, 0)
                carry = tl.where((idx >= ncarry) & (idx < ncarry + cnt), comp_at, carry)
                ncarry = ncarry + cnt
            else:
                # fill the carry to BK, flush the full chunk, keep the
                # remainder: the merged stream spans at most BK+63
                gidx = tl.minimum(tl.maximum(idx - ncarry, 0), 63)
                comp_at = tl.gather(comp, gidx, 0)
                carry = tl.where(idx >= ncarry, comp_at, carry)
                m, l, acc = _exact_flush(q, q8, K, V, qs_s, vs_s, carry, idx < BK,
                                         m, l, acc, b, h, bh, T, D, SCALE, KST, VST,
                                         HEAD_MAJOR, FP8)
                first = BK - ncarry
                gidx2 = tl.minimum(idx + first, 63)
                comp_at2 = tl.gather(comp, gidx2, 0)
                carry = tl.where(idx < cnt - first, comp_at2, 0)
                ncarry = ncarry + cnt - BK
    if ncarry > 0:
        # final partial chunk, masked exactly like the ids path's pos<total
        m, l, acc = _exact_flush(q, q8, K, V, qs_s, vs_s, carry, idx < ncarry,
                                 m, l, acc, b, h, bh, T, D, SCALE, KST, VST,
                                 HEAD_MAJOR, FP8)
    # covariance correction: additive on the finished softmax output
    # (out = acc/l + gamma * D^-0.5 * q@C). Upstream applied this between
    # the representative and exact passes, where the exact pass's rescaling
    # weighted it by the representative mass fraction (measured 0.53x at a
    # production route); the correction's defined semantics is additive.
    result_sm = acc / l[:, None]
    if STORE_SM:
        # softmax-only output for the training path: the backward's
        # delta = rowsum(dO * softmax_out) cancellation identity holds
        # only for the p-weighted mixture, so it must not see the arm.
        # Subtracting the arm from the bf16 result later would be pure
        # cancellation noise once the arm dominates.
        tl.store(OSM + (b * T + rows[:, None]).to(tl.int64) * OST + h * D + ds[None, :],
                 result_sm, ((rows < T) & (rows >= PREFIX))[:, None])
    if GAMMA != 0.:
        cd = tl.arange(0, D)
        cov = tl.load(COV + bh * D * D + ds[:, None] * D + cd[None, :])
        acc += l[:, None] * (tl.dot(q, cov) * (D ** -.5) * GAMMA)
        result = acc / l[:, None]
    else:
        result = result_sm
    tl.store(O + (b * T + rows[:, None]).to(tl.int64) * OST + h * D + ds[None, :], result,
             ((rows < T) & (rows >= PREFIX))[:, None])
    if WRITE_LSE:
        # log2-domain, matching the exp2 softmax above; the backward
        # kernels turn any score s into p = exp2(s - lse).
        tl.store(LSE + (b * T + rows).to(tl.int64) * H + h, m + tl.log2(l),
                 (rows < T) & (rows >= PREFIX))


def attention(q,k,v,route,*,prefix_tokens=0,dense=None,bk=64,head_major=False,warps=4,fp8=False,stages=1,covariance=None,cov_gamma=0.,lse=None):
    b,t,h,d=q.shape;n=triton.cdiv(t,64)
    out=torch.empty_like(q)
    original_k,original_v=k,v
    ks=vs=q
    if fp8:
        k,v,ks,vs=quantized_kv(k,v);head_major=True
    elif head_major:
        k=k.permute(0,2,1,3).contiguous();v=v.permute(0,2,1,3).contiguous()
    packed_attention_walk[(n,b*h)](q,k,v,route['kc'],route['vm'],route['flags'],route['bits'],out,ks,vs,covariance if covariance is not None else q,out if lse is None else lse,q,
                              t,n,h,d,prefix_tokens,d**-.5*math.log2(math.e),bk,head_major,fp8,GAMMA=float(cov_gamma),
                              QST=q.stride(1),KST=k.stride(1),VST=v.stride(1),OST=out.stride(1),
                              WRITE_LSE=lse is not None,
                              num_warps=warps,num_stages=stages)
    if prefix_tokens:
        if dense is None:raise RuntimeError('Selected common dense backend required')
        out[:,:prefix_tokens]=dense(q[:,:prefix_tokens],original_k,original_v)
    return out

@triton.jit
def channel_partial(K,V,PK,PV,T:tl.constexpr,H:tl.constexpr,D:tl.constexpr,NT:tl.constexpr,BT:tl.constexpr=256):
    tile,bh=tl.program_id(0),tl.program_id(1)
    ts=tile*BT+tl.arange(0,BT);ds=tl.arange(0,D)
    ix=((bh//H*T+ts[:,None]).to(tl.int64)*H+bh%H)*D+ds[None,:]
    k=tl.load(K+ix,(ts<T)[:,None],0).to(tl.float32)
    v=tl.load(V+ix,(ts<T)[:,None],0).to(tl.float32)
    tl.store(PK+(bh*NT+tile)*D+ds,tl.max(tl.abs(k),0))
    tl.store(PV+(bh*NT+tile)*D+ds,tl.max(tl.abs(v),0))
@triton.jit
def channel_scale(PK,PV,KS,VS,NT:tl.constexpr,ND:tl.constexpr,D:tl.constexpr,BD:tl.constexpr=32):
    bh,dt=tl.program_id(0),tl.program_id(1)
    ts=tl.arange(0,ND);ds=dt*BD+tl.arange(0,BD)
    ix=(bh*NT+ts[:,None])*D+ds[None,:]
    k=tl.load(PK+ix,(ts<NT)[:,None]&(ds<D)[None,:],0.)
    v=tl.load(PV+ix,(ts<NT)[:,None]&(ds<D)[None,:],0.)
    tl.store(KS+bh*D+ds,tl.maximum(tl.max(k,0)/448.,1.e-12),ds<D)
    tl.store(VS+bh*D+ds,tl.maximum(tl.max(v,0)/448.,1.e-12),ds<D)
@triton.jit
def quantize_channels(K,V,KS,VS,K8,V8,T:tl.constexpr,H:tl.constexpr,D:tl.constexpr,BT:tl.constexpr=128):
    tile,bh=tl.program_id(0),tl.program_id(1)
    ts=tile*BT+tl.arange(0,BT);ds=tl.arange(0,D)
    ix=((bh//H*T+ts[:,None]).to(tl.int64)*H+bh%H)*D+ds[None,:]
    k=tl.load(K+ix,(ts<T)[:,None],0).to(tl.float32);v=tl.load(V+ix,(ts<T)[:,None],0).to(tl.float32)
    ks=tl.load(KS+bh*D+ds);vs=tl.load(VS+bh*D+ds)
    dst=(bh*T+ts[:,None]).to(tl.int64)*D+ds[None,:]
    tl.store(K8+dst,(k/ks[None,:]).to(tl.float8e4nv),(ts<T)[:,None])
    tl.store(V8+dst,(v/vs[None,:]).to(tl.float8e4nv),(ts<T)[:,None])

def quantized_kv(k,v):
    b,t,h,d=k.shape;nt=triton.cdiv(t,256)
    pk=torch.empty((b*h,nt,d),device=k.device,dtype=torch.float32);pv=torch.empty_like(pk)
    ks=torch.empty((b*h,d),device=k.device,dtype=torch.float32);vs=torch.empty_like(pk)
    k8=torch.empty((b*h,t,d),device=k.device,dtype=torch.float8_e4m3fn);v8=torch.empty_like(k8)
    channel_partial[(nt,b*h)](k,v,pk,pv,t,h,d,nt,num_warps=8)
    channel_scale[(b*h,triton.cdiv(d,32))](pk,pv,ks,vs,nt,triton.next_power_of_2(nt),d,num_warps=4)
    quantize_channels[(triton.cdiv(t,128),b*h)](k,v,ks,vs,k8,v8,t,h,d,num_warps=4)
    return k8,v8,ks,vs
