"""Automagic3-FFT: Automagic3's update math with the polarity controller gutted.

Copied from ``toolkit/optimizers/automagic3.py`` (v3) and stripped for the
full-fine-tune regime. The adaptive-lr machinery is not configured weaker here,
it is ABSENT:

  * no sign-history ring            (v3: H/8 bytes per ELEMENT -- gigabytes at
    full-model scale, plus a packed uint8 plane write every step)
  * no pooled polarity vote         (bitwise-constant sign unpack/XOR/sums and
    the per-group up/down recomputation -- ~12% of the FFT step in the census)
  * no per-param ``state["lr"]`` tensor or group-lr nudge (lr is the fixed
    group lr; the only consumer of the vote in v3 was this nudge)
  * no non-fused / shape-family paths (the FFT regime runs fused: every param
    is stepped inside backward and its grad dropped immediately, so a full
    grad set never exists; gradient accumulation requires the unfused v3)

The math that REMAINS is v3's, carried verbatim: fp32 grad with non-finite
neutralisation, Adafactor factored row/col second moment (in p.dtype, fp32
math), RMS trust-region clip + per-element max-norm cap, and the stochastic
rounding on write-back that keeps sub-ULP bf16 updates alive. With v3 run at
min_lr == max_lr == lr (controller clamped off), this optimizer is bitwise
identical from a fresh start -- see docs/fullfinetune/moss/test_vote_identity.py.
"""

import os

import torch

# opt-in fused Triton update path (automagic3fft_kernels). Default off:
# every other model keeps the eager byte-for-byte. MOSS_OPT_FUSED=1 turns
# it on for Automagic3FFT after its identity test has passed.
_FUSED = os.environ.get("MOSS_OPT_FUSED") == "1"
_fused_import_error = None
try:
    from .automagic3fft_kernels import fused_update_param as _fused_update
except Exception as _e:
    _fused_update = None
    _fused_import_error = _e


class Automagic3FFT(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-6,
        beta2: float = 0.999,
        eps: float = 1e-30,
        clip_threshold: float = 1.0,
        weight_decay: float = 0.0,
    ):
        defaults = dict(
            lr=lr,
            beta2=beta2,
            eps=eps,
            clip_threshold=clip_threshold,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

        if _FUSED:
            if _fused_update is None:
                raise RuntimeError(
                    "MOSS_OPT_FUSED=1 but the fused Triton path is "
                    f"unavailable on this box: {_fused_import_error!r}. "
                    "Fix the environment or unset the flag -- this "
                    "optimizer never silently falls back to eager.")
            print("Automagic3FFT: fused Triton stepping ENABLED "
                  "(params outside the kernel set abort the run loudly)")

        self._hook_handles = []
        for group in self.param_groups:
            for p in group["params"]:
                if not p.requires_grad:
                    continue
                self._hook_handles.append(
                    p.register_post_accumulate_grad_hook(
                        self._make_backward_hook(group)
                    )
                )

        total = sum(p.numel() for g in self.param_groups for p in g["params"])
        print(
            f"Automagic3FFT: fused fixed-lr optimiser over {total:,} params "
            f"(no polarity ring: state is Adafactor row/col moments only)"
        )

    # ------------------------------------------------------------------ utils
    # (carried verbatim from automagic3.py — v3 reference semantics)

    @staticmethod
    def _rms(t: torch.Tensor) -> torch.Tensor:
        return t.norm(2) / (t.numel() ** 0.5)

    @staticmethod
    def _approx_sq_grad(row: torch.Tensor, col: torch.Tensor) -> torch.Tensor:
        r = (row / row.mean(dim=-1, keepdim=True)).rsqrt_().unsqueeze(-1)
        c = col.unsqueeze(-2).rsqrt()
        return torch.mul(r, c)

    @staticmethod
    def _sr_truncate(v_fp32: torch.Tensor, drop_bits: int) -> torch.Tensor:
        as_int = v_fp32.view(torch.int32)
        as_int.add_(torch.randint_like(as_int, 1 << drop_bits))
        as_int.bitwise_and_(-(1 << drop_bits))
        return v_fp32

    @staticmethod
    def _stochastic_round(v: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        finfo = torch.finfo(dtype)
        absv = v.abs().clamp_(min=finfo.tiny)
        ulp = torch.exp2(torch.floor(torch.log2(absv))).mul_(finfo.eps)
        noise = torch.rand_like(v).sub_(0.5).mul_(ulp)
        return v.add_(noise).to(dtype)

    @classmethod
    def _stochastic_copy_(cls, dst: torch.Tensor, src_fp32: torch.Tensor) -> None:
        if dst.dtype == torch.bfloat16:
            dst.copy_(cls._sr_truncate(src_fp32, 16))
        elif dst.dtype == torch.float16:
            dst.copy_(cls._sr_truncate(src_fp32, 13))
        else:
            dst.copy_(cls._stochastic_round(src_fp32, dst.dtype))

    # ------------------------------------------------------------------ hooks

    def _make_backward_hook(self, group):
        def _hook(p: torch.Tensor):
            if _FUSED:
                # No eager fallback by design: unsupported params raise.
                _fused_update(p, group, self.state[p])
            else:
                self._update_param(p, group)

        return _hook

    # ------------------------------------------------------------- per-param

    @torch.no_grad()
    def _update_param(self, p: torch.Tensor, group: dict) -> None:
        if p.grad is None:
            return
        state = self.state[p]
        if len(state) == 0:
            state["step"] = 0
            if p.dim() >= 2:
                state["exp_avg_sq_row"] = torch.zeros(
                    p.shape[:-1], dtype=p.dtype, device=p.device
                )
                state["exp_avg_sq_col"] = torch.zeros(
                    p.shape[:-2] + p.shape[-1:], dtype=p.dtype, device=p.device
                )
            else:
                state["exp_avg_sq"] = torch.zeros(
                    p.shape, dtype=p.dtype, device=p.device
                )

        grad = p.grad
        if grad.is_sparse:
            raise RuntimeError("Automagic3FFT does not support sparse gradients.")
        if grad.dtype != torch.float32:
            grad = grad.to(torch.float32)
        # Fused stepping runs inside backward, before the trainer's clip /
        # nan-skip; a single non-finite grad would poison the second-moment
        # EMA forever, so neutralise on our own fp32 copy (v3 rationale).
        grad.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)

        beta2 = group["beta2"]
        eps = group["eps"]
        sq = grad * grad

        # Second moment: Adafactor-factored for >=2D, full for 1D; state in
        # p.dtype with fp32 math and write-back (v3 verbatim).
        if p.dim() >= 2:
            row_state = state["exp_avg_sq_row"]
            col_state = state["exp_avg_sq_col"]
            if row_state.dtype == torch.float32:
                row, col = row_state, col_state
                row.mul_(beta2).add_(sq.mean(dim=-1).add_(eps), alpha=1.0 - beta2)
                col.mul_(beta2).add_(sq.mean(dim=-2).add_(eps), alpha=1.0 - beta2)
            else:
                row = row_state.to(torch.float32)
                col = col_state.to(torch.float32)
                row.mul_(beta2).add_(sq.mean(dim=-1).add_(eps), alpha=1.0 - beta2)
                col.mul_(beta2).add_(sq.mean(dim=-2).add_(eps), alpha=1.0 - beta2)
                row_state.copy_(row.to(row_state.dtype))
                col_state.copy_(col.to(col_state.dtype))
            update = self._approx_sq_grad(row, col).mul_(grad)
        else:
            v_state = state["exp_avg_sq"]
            if v_state.dtype == torch.float32:
                v = v_state
                v.mul_(beta2).add_(sq, alpha=1.0 - beta2)
            else:
                v = v_state.to(torch.float32)
                v.mul_(beta2).add_(sq, alpha=1.0 - beta2)
                v_state.copy_(v.to(v_state.dtype))
            update = v.add(eps).rsqrt().mul_(grad)

        # Trust region: RMS clip then per-element max-norm cap (v3 verbatim).
        update.div_((self._rms(update) / group["clip_threshold"]).clamp_(min=1.0))
        update.clamp_(-group["clip_threshold"], group["clip_threshold"])

        state["step"] += 1

        wd = group["weight_decay"]
        lr = group["lr"]

        if p.dtype == torch.float32:
            if wd != 0.0:
                update.add_(p, alpha=wd)
            p.add_(update, alpha=-lr)
        else:
            # fp32 update math + stochastic rounding back into the low
            # precision master weight — the reason a 1e-5 lr can move bf16
            # weights at all (v3 rationale).
            new_p_fp32 = p.to(torch.float32)
            if wd != 0.0:
                update.add_(new_p_fp32, alpha=wd)
            new_p_fp32.sub_(update, alpha=lr)
            self._stochastic_copy_(p, new_p_fp32)

        p.grad = None

    # --------------------------------------------------------- optimizer API

    @torch.no_grad()
    def step(self, closure=None):
        # Fused design: every param was already stepped in its backward hook.
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        return loss

    def get_learning_rates(self):
        return [float(group["lr"]) for group in self.param_groups]

    def get_avg_learning_rate(self):
        lrs = self.get_learning_rates()
        return sum(lrs) / len(lrs) if lrs else 0.0
