"""Fused Triton update kernels for Automagic3FFT.

The eager hook body in ``automagic3fft._update_param`` fires ~13 full-size
PyTorch ops per parameter (fp32 cast, ``sq``, two axis reductions, broadcast
approx, RMS norm, clamps, wd add, stochastic round). Measured cost of that
eager chain inside backward: 1.61 s/step on the RTX PRO 6000 (bs16) and
1.00 s/step on the RTX 5090 (bs6) -- ~5x its own visible kernel region
(full-arm vs skip-arm backward timers). The damage is bandwidth
theft from wgrad GEMMs plus allocator churn from full-size fp32
temporaries allocated mid-backward.

This module replaces the chain with three full-size passes per 2-D param:

  A  grad (bf16) -> atomic row/col sums of squares          (read 2 B/elem)
  B  grad, r, c  -> atomic sum of the scaled update squared (read 2 B/elem)
  C  grad, r, c, p -> update, trust divide, clamp, wd, step,
     stochastic round -> bf16 write                         (read+write 6 B)

plus tiny R+C-sized vector work that stays in torch. ~10 B/elem vs the
eager ~40-60, and zero full-size temporaries. The math mirrors
``_update_param`` op-for-op (nan-sanitize; state update with eps added to
the mean BEFORE the (1-beta2) scale; the pre-rounding fp32 state feeds the
factors while the bf16-rounded value is written back; the trust scale is
computed by torch's own IEEE sqrt/div on the 1-element sum buffer --
tl.sqrt is the fast-approx flavor and must not be used here; clamp happens
BEFORE the wd term is added).

1-D params get a two-pass equivalent (reduce, then one kernel that
recomputes the moment in registers, updates state, and writes p).
Params this module cannot handle (dim > 2, fp32 masters, non-contiguous
layout, non-CUDA) raise UnsupportedParam -- fused mode never silently
runs half the model eager.

State layout is byte-compatible with the eager path, so optimizer.pt
checkpoints interchange between modes.
"""

import torch
import triton
import triton.language as tl


class UnsupportedParam(RuntimeError):
    """Fused stepping met a param the kernels do not implement.

    Loud on purpose (owner rule): fused mode has no silent eager fallback.
    A run that would half-fuse/half-eager dies at the first step naming
    the param instead of quietly paying eager cost for it.
    """


def _unsupported(p, why):
    raise UnsupportedParam(
        f"fused_update_param: {tuple(p.shape)} {p.dtype} "
        f"({why}) -- unset MOSS_OPT_FUSED or extend the kernels")


# --------------------------------------------------------------------- 2-D

@triton.jit
def _k_rowcol_sumsq(g_ptr, rows_ptr, cols_ptr, R, C,
                    BR: tl.constexpr, BC: tl.constexpr):
    """Kernel A: atomic row/col sums of g^2 over a sanitized bf16 grad."""
    pid_r = tl.program_id(0)
    pid_c = tl.program_id(1)
    rows = pid_r * BR + tl.arange(0, BR)
    cols = pid_c * BC + tl.arange(0, BC)
    offs = rows[:, None] * C + cols[None, :]
    m = (rows[:, None] < R) & (cols[None, :] < C)
    g = tl.load(g_ptr + offs, mask=m, other=0.0).to(tl.float32)
    g = tl.where(g != g, 0.0, g)
    g = tl.where(g == float("inf"), 0.0, g)
    g = tl.where(g == float("-inf"), 0.0, g)
    sq = g * g
    tl.atomic_add(rows_ptr + rows, tl.sum(sq, axis=1),
                  mask=rows < R, sem="relaxed")
    tl.atomic_add(cols_ptr + cols, tl.sum(sq, axis=0),
                  mask=cols < C, sem="relaxed")


@triton.jit
def _k_update_sumsq(g_ptr, r_ptr, c_ptr, out_ptr, R, C,
                    BR: tl.constexpr, BC: tl.constexpr):
    """Kernel B: atomic sum of update^2, update = g * r[row] * c[col]."""
    pid_r = tl.program_id(0)
    pid_c = tl.program_id(1)
    rows = pid_r * BR + tl.arange(0, BR)
    cols = pid_c * BC + tl.arange(0, BC)
    offs = rows[:, None] * C + cols[None, :]
    m = (rows[:, None] < R) & (cols[None, :] < C)
    g = tl.load(g_ptr + offs, mask=m, other=0.0).to(tl.float32)
    g = tl.where(g != g, 0.0, g)
    g = tl.where(g == float("inf"), 0.0, g)
    g = tl.where(g == float("-inf"), 0.0, g)
    r = tl.load(r_ptr + rows, mask=rows < R, other=0.0)
    c = tl.load(c_ptr + cols, mask=cols < C, other=0.0)
    u = (r[:, None] * c[None, :]) * g
    tl.atomic_add(out_ptr, tl.sum(u * u), sem="relaxed")


@triton.jit
def _k_apply_2d(g_ptr, p_ptr, r_ptr, c_ptr, m_ptr, R, C,
                lr, clip, wd, drop, seed, TRUNC: tl.constexpr,
                BR: tl.constexpr, BC: tl.constexpr):
    """Kernel C: full update + trust region + wd + SR write-back.

    Reads the r/c factor vectors and the torch-computed trust scale
    (sqrt/div done by torch on the 1-element sumsq buffer -- same IEEE
    ops eager uses, no .item() sync, no approximate in-kernel sqrt).
    """
    pid_r = tl.program_id(0)
    pid_c = tl.program_id(1)
    rows = pid_r * BR + tl.arange(0, BR)
    cols = pid_c * BC + tl.arange(0, BC)
    offs = rows[:, None] * C + cols[None, :]
    m = (rows[:, None] < R) & (cols[None, :] < C)

    m_scale = tl.load(m_ptr)

    g = tl.load(g_ptr + offs, mask=m, other=0.0).to(tl.float32)
    g = tl.where(g != g, 0.0, g)
    g = tl.where(g == float("inf"), 0.0, g)
    g = tl.where(g == float("-inf"), 0.0, g)
    r = tl.load(r_ptr + rows, mask=rows < R, other=0.0)
    c = tl.load(c_ptr + cols, mask=cols < C, other=0.0)

    u = (r[:, None] * c[None, :]) * g
    u = u / m_scale
    u = tl.minimum(tl.maximum(u, -clip), clip)

    p_old = tl.load(p_ptr + offs, mask=m, other=0.0).to(tl.float32)
    if wd != 0.0:
        u = u + wd * p_old
    newp = p_old - lr * u

    # stochastic truncate to the target mantissa (sign-magnitude int add,
    # wrapping, matching torch's int32 add_ semantics on the bit view)
    ip = newp.to(tl.int32, bitcast=True)
    if not TRUNC:
        rand = tl.randint(seed, offs) & ((1 << drop) - 1)
        ip = ip + rand
    ip = ip & (~((1 << drop) - 1))
    newp = ip.to(tl.float32, bitcast=True)
    tl.store(p_ptr + offs, newp.to(p_ptr.dtype.element_ty), mask=m)


# ------------------------------------------------------------------ 1-D

@triton.jit
def _k_vec_row(rs_ptr, sq_ptr, rf_ptr, acc_ptr, R, beta2, eps, inv_C,
               BS: tl.constexpr):
    """row_f32 = b2*row + (1-b2)*(sq/C + eps); write fp32 shadow + bf16
    state; accumulate sum(row_f32) for the global mean."""
    pid = tl.program_id(0)
    offs = pid * BS + tl.arange(0, BS)
    m = offs < R
    sq = tl.load(sq_ptr + offs, mask=m, other=0.0)
    v = tl.load(rs_ptr + offs, mask=m, other=0.0).to(tl.float32)
    t = sq * inv_C
    t = t + eps
    out = v * beta2 + (1.0 - beta2) * t
    tl.store(rf_ptr + offs, out, mask=m)
    tl.store(rs_ptr + offs, out.to(rs_ptr.dtype.element_ty), mask=m)
    tl.atomic_add(acc_ptr, tl.sum(out), mask=None, sem="relaxed")


@triton.jit
def _k_vec_col(cs_ptr, sq_ptr, cf_ptr, c_ptr, C, beta2, eps, inv_R,
               BS: tl.constexpr):
    """col state update + c = rsqrt(col_f32) in one pass (no global mean
    on the col side -- the factors take rsqrt directly)."""
    pid = tl.program_id(0)
    offs = pid * BS + tl.arange(0, BS)
    m = offs < C
    sq = tl.load(sq_ptr + offs, mask=m, other=0.0)
    v = tl.load(cs_ptr + offs, mask=m, other=0.0).to(tl.float32)
    t = sq * inv_R
    t = t + eps
    out = v * beta2 + (1.0 - beta2) * t
    tl.store(cf_ptr + offs, out, mask=m)
    tl.store(cs_ptr + offs, out.to(cs_ptr.dtype.element_ty), mask=m)
    tl.store(c_ptr + offs, tl.rsqrt(out), mask=m)


@triton.jit
def _k_row_factor(rf_ptr, acc_ptr, r_ptr, R, BS: tl.constexpr):
    """r = rsqrt(row_f32 / mean(row_f32)), mean from the acc accumulator."""
    pid = tl.program_id(0)
    offs = pid * BS + tl.arange(0, BS)
    m = offs < R
    mean = tl.load(acc_ptr) / R.to(tl.float32)
    v = tl.load(rf_ptr + offs, mask=m, other=0.0)
    tl.store(r_ptr + offs, tl.rsqrt(v / mean), mask=m)


@triton.jit
def _k_scale(sumsq_ptr, m_ptr, sqrtN, clip):
    """m = max(sqrt(sumsq)/sqrtN/clip, 1) -- sqrt_rn is the IEEE
    round-to-nearest flavor (tl.sqrt is the approximation that produced
    the 1.2e-3 scale error; anything feeding the global scale must not
    use it)."""
    s = tl.load(sumsq_ptr)
    rms = tl.math.sqrt_rn(s) / sqrtN
    m = rms / clip
    m = tl.maximum(m, 1.0)
    tl.store(m_ptr, m)


# --------------------------------------------------------------------- 1-D

@triton.jit
def _k_update_sumsq_1d(g_ptr, v_ptr, out_ptr, N, beta2, eps,
                       BS: tl.constexpr):
    """Kernel B (1-D): atomic sum of update^2, update = g * rsqrt(v_new+eps).

    v_new is recomputed in registers from the bf16 state; nothing is
    stored here -- kernel C redoes the moment update and writes state + p.
    """
    pid = tl.program_id(0)
    offs = pid * BS + tl.arange(0, BS)
    m = offs < N
    g = tl.load(g_ptr + offs, mask=m, other=0.0).to(tl.float32)
    g = tl.where(g != g, 0.0, g)
    g = tl.where(g == float("inf"), 0.0, g)
    g = tl.where(g == float("-inf"), 0.0, g)
    v = tl.load(v_ptr + offs, mask=m, other=0.0).to(tl.float32)
    v_new = v * beta2 + (1.0 - beta2) * (g * g)
    u = g * tl.rsqrt(v_new + eps)
    tl.atomic_add(out_ptr, tl.sum(u * u), sem="relaxed")


@triton.jit
def _k_apply_1d(g_ptr, p_ptr, v_ptr, m_ptr, N, beta2, eps,
                lr, clip, wd, drop, seed, TRUNC: tl.constexpr,
                BS: tl.constexpr):
    """Kernel C (1-D): moment update + state write + full step + SR."""
    pid = tl.program_id(0)
    offs = pid * BS + tl.arange(0, BS)
    m = offs < N

    m_scale = tl.load(m_ptr)

    g = tl.load(g_ptr + offs, mask=m, other=0.0).to(tl.float32)
    g = tl.where(g != g, 0.0, g)
    g = tl.where(g == float("inf"), 0.0, g)
    g = tl.where(g == float("-inf"), 0.0, g)
    v = tl.load(v_ptr + offs, mask=m, other=0.0).to(tl.float32)
    v_new = v * beta2 + (1.0 - beta2) * (g * g)
    tl.store(v_ptr + offs, v_new.to(v_ptr.dtype.element_ty), mask=m)

    u = g * tl.rsqrt(v_new + eps)
    u = u / m_scale
    u = tl.minimum(tl.maximum(u, -clip), clip)

    p_old = tl.load(p_ptr + offs, mask=m, other=0.0).to(tl.float32)
    if wd != 0.0:
        u = u + wd * p_old
    newp = p_old - lr * u

    ip = newp.to(tl.int32, bitcast=True)
    if not TRUNC:
        rand = tl.randint(seed, offs) & ((1 << drop) - 1)
        ip = ip + rand
    ip = ip & (~((1 << drop) - 1))
    newp = ip.to(tl.float32, bitcast=True)
    tl.store(p_ptr + offs, newp.to(p_ptr.dtype.element_ty), mask=m)


# ------------------------------------------------------------------ driver

import math

_WS = {}        # (R, C, device) -> [sq_row, sq_col, sumsq], reused forever
_WS1 = {}       # (N, device) -> [sumsq]
_seed_state = [0x1234ABCD]


def _next_seed():
    s = _seed_state[0]
    _seed_state[0] = (s * 0x85EBCA6B + 0x27D4EB2F) & 0x7FFFFFFF
    return s


def _ws2(R, C, device):
    ws = _WS.get((R, C, device))
    if ws is None:
        z = lambda n: torch.zeros(n, dtype=torch.float32, device=device)
        # order: sq_row, sq_col, sumsq, row_f32, col_f32, r, c, row_acc, m
        ws = [z(R), z(C), z(1), z(R), z(C), z(R), z(C), z(1), z(1)]
        _WS[(R, C, device)] = ws
    ws[0].zero_()
    ws[1].zero_()
    ws[2].zero_()
    ws[7].zero_()
    return ws


def _drop_bits(p):
    if p.dtype == torch.bfloat16:
        return 16
    if p.dtype == torch.float16:
        return 13
    return None


def _init_state(p, state):
    """Byte-identical to the eager first-step init (checkpoint-compatible)."""
    state["step"] = 0
    if p.dim() >= 2:
        state["exp_avg_sq_row"] = torch.zeros(
            p.shape[:-1], dtype=p.dtype, device=p.device)
        state["exp_avg_sq_col"] = torch.zeros(
            p.shape[:-2] + p.shape[-1:], dtype=p.dtype, device=p.device)
    else:
        state["exp_avg_sq"] = torch.zeros(
            p.shape, dtype=p.dtype, device=p.device)


def fused_update_param(p, group, state, truncate_only=False):
    """Fused-kernel mirror of Automagic3FFT._update_param.

    Runs to completion or raises UnsupportedParam -- there is no silent
    eager fallback. On success the grad is dropped (p.grad = None) exactly
    as the eager path does.
    """
    grad = p.grad
    if grad is None:
        return
    if p.dim() < 1 or p.dim() > 2:
        _unsupported(p, f"dim={p.dim()}, fused covers 1-D/2-D")
    if not p.is_cuda:
        _unsupported(p, "not on CUDA")
    if p.numel() > 2**31 - 1:
        _unsupported(p, "numel exceeds the int32 kernel indexing")
    if not p.is_contiguous() or not grad.is_contiguous():
        _unsupported(p, "non-contiguous p or grad")
    drop = _drop_bits(p)
    if drop is None:
        _unsupported(p, "master weights must be bf16/fp16")
    if grad.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        _unsupported(p, f"grad dtype {grad.dtype}")
    if len(state) == 0:
        _init_state(p, state)
    beta2 = group["beta2"]
    eps = group["eps"]
    clip = group["clip_threshold"]
    lr = group["lr"]
    wd = group["weight_decay"]
    seed = _next_seed()

    if p.dim() == 2:
        if (state["exp_avg_sq_row"].dtype != p.dtype
                or state["exp_avg_sq_col"].dtype != p.dtype):
            _unsupported(p, "state dtype != master dtype (fp32-state "
                            "checkpoints are an eager-path feature)")
        C = p.shape[-1]
        R = p.numel() // C
        rs = state["exp_avg_sq_row"]
        cs = state["exp_avg_sq_col"]
        sq_row, sq_col, sumsq, row_f, col_f, rb, cb, racc, mb = _ws2(
            R, C, p.device)
        BR, BC = 32, 128
        grid = (triton.cdiv(R, BR), triton.cdiv(C, BC))
        _k_rowcol_sumsq[grid](grad, sq_row, sq_col, R, C,
                              BR=BR, BC=BC, num_warps=4)
        # vector phase: state update + factors, all Triton on cached
        # buffers -- no torch ops, no per-param allocations.
        VB = 1024
        vgrid = (triton.cdiv(max(R, C), VB),)
        _k_vec_row[vgrid](rs, sq_row, row_f, racc, R, beta2, eps, 1.0 / C,
                          BS=VB)
        _k_vec_col[vgrid](cs, sq_col, col_f, cb, C, beta2, eps, 1.0 / R,
                          BS=VB)
        _k_row_factor[vgrid](row_f, racc, rb, R, BS=VB)
        _k_update_sumsq[grid](grad, rb, cb, sumsq, R, C,
                              BR=BR, BC=BC, num_warps=4)
        # Trust scale: sqrt_rn (IEEE) in a 1-CTA kernel; identity-tested
        # against the torch chain the eager path runs.
        sqrtN = math.sqrt(R * C)
        _k_scale[(1,)](sumsq, mb, sqrtN, clip)
        _k_apply_2d[grid](grad, p, rb, cb, mb, R, C,
                          lr, clip, wd, drop, seed,
                          TRUNC=truncate_only, BR=BR, BC=BC, num_warps=4)
    else:
        v = state["exp_avg_sq"]
        if v.dtype != p.dtype:
            _unsupported(p, "state dtype != master dtype")
        N = p.numel()
        key = (N, p.device)
        bufs = _WS1.get(key)
        if bufs is None:
            bufs = (torch.zeros(1, dtype=torch.float32, device=p.device),
                    torch.zeros(1, dtype=torch.float32, device=p.device))
            _WS1[key] = bufs
        buf, mb = bufs
        buf.zero_()
        BS = 1024
        grid = (triton.cdiv(N, BS),)
        _k_update_sumsq_1d[grid](grad, v, buf, N, beta2, eps, BS=BS)
        _k_scale[(1,)](buf, mb, math.sqrt(N), clip)
        _k_apply_1d[grid](grad, p, v, mb, N, beta2, eps,
                          lr, clip, wd, drop, seed, TRUNC=truncate_only,
                          BS=BS)

    state["step"] += 1
    p.grad = None
