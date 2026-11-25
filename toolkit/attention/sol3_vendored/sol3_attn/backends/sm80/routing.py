"""Consume the existing FP32 L3 scores and cutoffs, in contiguous key-block tiles."""
import triton
import triton.language as tl

from ..workspace import active_call, count_launch


@triton.jit
def _select(scores, flags, bits, cutoff, tokens: tl.constexpr, blocks: tl.constexpr,
            score_stride: tl.constexpr, head_start, prefix: tl.constexpr,
            rows: tl.constexpr, cols: tl.constexpr):
    keys = tl.program_id(0) * cols + tl.arange(0, cols)
    queries = tl.program_id(1) * rows + tl.arange(0, rows)
    head = tl.program_id(2)
    row = (head_start + head).to(tl.int64) * blocks + queries
    dest = row[:, None] * blocks + keys[None, :]
    valid = (queries < blocks)[:, None] & (keys < blocks)[None, :]
    flag = tl.load(flags + dest, valid, other=-2)
    cut = tl.load(cutoff + row, queries < blocks, other=float("inf"))
    mixed = valid & ((queries + 1) * 64 > prefix)[:, None] & (flag >= 0)
    token = tl.program_id(0) * (cols * 64) + tl.arange(0, cols * 64)
    read = tl.reshape(tl.broadcast_to(mixed[:, :, None], (rows, cols, 64)), (rows, cols * 64))
    raw_row = head.to(tl.int64) * blocks + queries
    score = tl.load(scores + raw_row[:, None] * score_stride + token[None, :],
                    read & (token < tokens)[None, :], other=-float("inf"))
    chosen = read & (token < tokens)[None, :] & (score > cut[:, None])
    chosen = tl.reshape(chosen, (rows, cols, 64))
    count = tl.sum(chosen.to(tl.int32), 2)
    lane = tl.arange(0, 64).to(tl.uint64)
    bit = tl.full((64,), 1, tl.uint64) << lane
    mask = tl.sum(tl.where(chosen, bit[None, None, :], 0), 2)
    tl.store(flags + dest, tl.where(flag >= 0, count, flag), valid)
    tl.store(bits + dest, mask, valid)


def select(scores, flags, bits, cutoff, tokens, blocks, head_start, prefix, amount,
           rows=4, cols=8, warps=4):
    """Scores are fresh for these heads. The last dimension can include padding."""
    grid = (triton.cdiv(blocks, cols), triton.cdiv(blocks, rows), amount)
    _select[grid](scores, flags, bits, cutoff, tokens, blocks, scores.stride(-2),
                  head_start, prefix, rows, cols, num_warps=warps, num_stages=1)


@triton.jit
def _moments(qb, k, flags, partial, scores, tokens: tl.constexpr, n: tl.constexpr,
             heads: tl.constexpr, d: tl.constexpr, scale: tl.constexpr, prefix: tl.constexpr,
             rq: tl.constexpr, kt: tl.constexpr, tiles: tl.constexpr, head0,
             pitch: tl.constexpr, key_first: tl.constexpr):
    if key_first:
        ktile, qtile = tl.program_id(0), tl.program_id(1)
    else:
        qtile, ktile = tl.program_id(0), tl.program_id(1)
    local = tl.program_id(2)
    bh = head0 + local
    qs = qtile * rq + tl.arange(0, rq)
    token = ktile * kt + tl.arange(0, kt)
    features = tl.arange(0, d)
    row = bh.to(tl.int64) * n + qs
    flag = tl.load(flags + row[:, None] * n + token[None, :] // 64,
                   (qs < n)[:, None] & (token < tokens)[None, :], -2)
    eligible = ((qs < n)[:, None] & ((qs + 1) * 64 > prefix)[:, None]
                & (token < tokens)[None, :] & (flag >= 0))
    count = tl.sum(eligible.to(tl.float32), 1)
    mean = tl.zeros((rq,), tl.float32)
    m2 = tl.zeros((rq,), tl.float32)
    if tl.sum(count, 0) > 0:
        q = tl.load(qb + row[:, None] * d + features[None, :], (qs < n)[:, None], 0).to(tl.bfloat16)
        krow = ((bh // heads * tokens + token).to(tl.int64) * heads + bh % heads) * d
        key = tl.load(k + krow[:, None] + features[None, :], (token < tokens)[:, None], 0)
        score = tl.dot(q, tl.trans(key)) * scale * 1.4426950408889634
        mean = tl.sum(tl.where(eligible, score, 0.), 1) / tl.maximum(count, 1.)
        m2 = tl.sum(tl.where(eligible, (score - mean[:, None]) * (score - mean[:, None]), 0.), 1)
        local_row = local.to(tl.int64) * n + qs
        tl.store(scores + local_row[:, None] * pitch + token[None, :], score, eligible)
    ix = ((local * n + qs) * tiles + ktile) * 3
    tl.store(partial + ix, count, qs < n)
    tl.store(partial + ix + 1, mean, qs < n)
    tl.store(partial + ix + 2, m2, qs < n)



def launch_selection(kernel, grid, *args, **options):
    call = active_call()
    if call is not None and args[5] > 256:
        scores, flags, bits, cutoff, tokens, blocks, head, prefix, _tile = args
        select(scores, flags, bits, cutoff, tokens, blocks, head, prefix, grid[2], rows=2, cols=16)
        count_launch("route_tiles")
        return
    return kernel[grid](*args, **options)


def launch_h3_scores(kernel, grid, *args, **options):
    call = active_call()
    if call is None or not call.stage_h3_scores or args[-1]:
        return kernel[grid](*args, **options)
    qb, k, flags, raw, partial, tokens, blocks, heads, dimension, scale, prefix, bq, bk, nt, pad, head, _ = args
    _moments[(grid[1], grid[0], grid[2])](
        qb, k, flags, partial, raw, tokens, blocks, heads, dimension, scale, prefix,
        bq, bk, nt, head, pad, False, **options,
    )
    count_launch("mixed_scores")
