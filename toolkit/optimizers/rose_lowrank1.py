# Low-rank Rose. Stock rose-opt stays the rank > 16 optimizer.
# Derived from Rose v1.0.2 (Matthew Everet Kieren, Apache 2.0).
#
# Same update as Rose: one |max| - min range per output slice,
# theta -= lr * g / D. High lr still means a full-range step.
# No factored moments, no axis flip, no RMS denominator.
#
# Below width 32 the raw range is a short order statistic, biased low,
# so the same lr hammers harder and noisier than it does at rank 32.
# This corrects that and nothing else:
#   r_i   = |max(row)| - min(row)
#   r_ref = |max(G)| - min(G)
#   r_i  *= E[range_32] / E[range_n]          # bias correction
#   r_i   = clamp(r_i, 0.5 r_ref, r_ref)      # one spike cannot zero the row
#   D_i   = (1 - n/32) r_ref + (n/32) r_i     # width blend, not a CV collapse
# Centralize is off. On a 4-16 sample slice it deletes the offset Rose
# would otherwise hammer in.

import torch


# E[|max-min|] / sigma for n i.i.d. standard normals, n = 0..32.
_RANGE_FACTOR = (
    0.0, 0.0, 1.128, 1.693, 2.059, 2.326, 2.534, 2.704, 2.847,
    2.970, 3.078, 3.173, 3.258, 3.336, 3.406, 3.472, 3.532,
    3.588, 3.640, 3.689, 3.735, 3.778, 3.819, 3.858, 3.895,
    3.930, 3.964, 3.996, 4.027, 4.056, 4.085, 4.112, 4.138,
)


class RoseLowRank(torch.optim.Optimizer):
    """Rose for rank below 32.

    Purpose of this code change: stock Rose computes one
    ``|max| - min`` range per output slice as its denominator. At rank
    >= 32 that range is a reliable width-32 order statistic; at rank
    4-16 it is short-sampled -- biased low and noisy -- so the same lr
    produces a step that is both harder and less stable than at rank
    32. This optimizer's only job is to make Rose usable at rank < 32:
    it bias-corrects the short range by the expected-range ratio,
    clamps it against the whole-tensor range so one spike cannot zero
    a row, and blends toward the full range as width approaches 32.
    The update law is otherwise Rose v1 (no RMS denominator, no axis
    flip, centralize off by default).

    Status: limited testing is successful -- smoke runs at rank 4/8/16
    (60 steps, LoRA training) ran without instability, at speed on par
    with the rank-32 baseline, loss in the same band. Not yet validated
    at full training length; the quality gate is sample comparison.

    Served by the toolkit optimizer registry under the name ``rosev2``
    (v2/v4 slots were swapped 2026-10-08: v2 = low-rank, v4 = wide).
    """

    def __init__(
        self,
        params,
        lr: float,
        *,
        weight_decay: float | None = 1e-4,
        wd_schedule: bool | float = False,
        centralize: bool = False,
        stabilize: bool = True,
        bf16_sr: bool | torch.Generator = True,
        compute_dtype: torch.dtype | str | None = "fp32",
        batched: bool = True,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if weight_decay is not None and weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")

        if isinstance(bf16_sr, torch.Generator):
            self.bf16_sr_gen = bf16_sr
            bf16_sr = True
        else:
            self.bf16_sr_gen = None

        if isinstance(compute_dtype, str):
            dtype_lookup = {
                "float16": torch.float16, "fp16": torch.float16,
                "float32": torch.float32, "fp32": torch.float32,
                "float64": torch.float64, "fp64": torch.float64,
                "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
                "none": None, "null": None,
            }
            try:
                compute_dtype = dtype_lookup[compute_dtype.strip().lower()]
            except KeyError:
                raise ValueError(
                    f"Invalid compute_dtype string: {compute_dtype!r}. "
                    f"Valid options: {sorted(dtype_lookup)}"
                ) from None

        if bf16_sr and compute_dtype not in (torch.float32, torch.float64, None):
            raise ValueError(
                "bf16_sr=True has no useful effect unless compute_dtype is fp32, fp64, or None."
            )

        super().__init__(params, dict(
            lr=lr,
            centralize=centralize,
            stabilize=stabilize,
            weight_decay=weight_decay,
            wd_schedule=wd_schedule,
            bf16_sr=bf16_sr,
            compute_dtype=compute_dtype,
            batched=batched,
        ))

    _BATCH_MAX_ELEMS = 16_000_000

    def _wd_factor(self, group, lr):
        weight_decay = group["weight_decay"]
        wd_schedule = group["wd_schedule"]
        group.setdefault("initial_lr", lr)
        if weight_decay and wd_schedule:
            wd_lr = lr / (
                wd_schedule if isinstance(wd_schedule, float)
                else group.get("max_lr", group.get("initial_lr"))
            )
        else:
            wd_lr = lr
        return None if not weight_decay else max(0.0, 1.0 - wd_lr * weight_decay)

    def _range_denom(self, G, stabilize, batch_dims=0):
        """Per-output |max|-min, short-width corrected. Leading axis owns D."""
        lead = batch_dims
        trail = tuple(range(lead + 1, G.ndim))
        n = 1
        for ax in trail:
            n *= G.shape[ax]
        n = max(n, 2)

        r = G.amax(dim=trail, keepdim=True).abs_().sub_(G.amin(dim=trail, keepdim=True))
        ref_axes = tuple(range(lead, G.ndim))
        r_ref = G.amax(dim=ref_axes, keepdim=True).abs_().sub_(
            G.amin(dim=ref_axes, keepdim=True)
        )

        if stabilize and n < 32:
            scale = _RANGE_FACTOR[32] / _RANGE_FACTOR[min(n, 32)]
            r = r.mul(scale).clamp(min=r_ref * 0.5, max=r_ref)
            w = n / 32.0
            # out-of-place: r_ref is (..., 1, 1), r is (..., out, 1). add_ would
            # write the broadcast into the smaller tensor and throw.
            r = r_ref.mul(1.0 - w).add(r, alpha=w)

        r.masked_fill_(r == 0.0, 1.0)
        return r

    def _step_batched(self, group, wd_factor):
        compute_dtype = group["compute_dtype"]
        lr = group["lr"]
        by_key = {}
        for p in group["params"]:
            if p.grad is None or p.grad.is_sparse:
                continue
            key = (tuple(p.shape), p.dtype, p.device)
            by_key.setdefault(key, []).append(p)

        batched_ids = set()
        families = []
        for ps in by_key.values():
            if len(ps) < 2:
                continue
            families.append(ps)
            batched_ids.update(id(p) for p in ps)
        remaining = [
            p for p in group["params"]
            if p.grad is not None and id(p) not in batched_ids
        ]
        for ps in families:
            per = 1
            for s in ps[0].shape:
                per *= s
            chunk = max(1, self._BATCH_MAX_ELEMS // per)
            for i in range(0, len(ps), chunk):
                self._step_family(ps[i:i + chunk], group, wd_factor, compute_dtype, lr)
        return remaining

    @torch.no_grad()
    def _step_family(self, ps, group, wd_factor, compute_dtype, lr):
        n = len(ps)
        shape = tuple(ps[0].shape)
        fp32 = (group["bf16_sr"] and ps[0].dtype == torch.bfloat16
                and compute_dtype is None)
        work_dtype = torch.float32 if fp32 else (
            compute_dtype if compute_dtype is not None else ps[0].dtype
        )
        G = torch.empty((n, *shape), dtype=work_dtype, device=ps[0].device)
        P = torch.empty_like(G)
        torch._foreach_copy_(list(G.unbind()), [p.grad for p in ps])
        torch._foreach_copy_(list(P.unbind()), list(ps))

        if wd_factor is not None:
            P.mul_(wd_factor)

        if len(shape) == 0:
            P.add_(G.sign(), alpha=-lr)
        elif len(shape) == 1:
            g_min, g_max = G.aminmax(dim=1)
            denom = g_max.abs_().sub_(g_min).unsqueeze(1)
            denom.masked_fill_(denom == 0.0, 1.0)
            P.addcdiv_(G, denom, value=-lr)
        else:
            if group["centralize"] and int(torch.tensor(shape[1:]).prod()) >= 32:
                G.sub_(G.mean(dim=tuple(range(2, G.ndim)), keepdim=True))
            denom = self._range_denom(G, group["stabilize"], batch_dims=1)
            P.addcdiv_(G, denom, value=-lr)

        if group["bf16_sr"] and ps[0].dtype == torch.bfloat16:
            P32 = P.to(dtype=torch.float32)
            noise = torch.empty(P32.shape, dtype=torch.int32, device=P32.device)
            noise.random_(0, 0x10000, generator=self.bf16_sr_gen)
            out = P32.view(dtype=torch.int32).add_(noise).bitwise_and_(
                -0x10000).view(dtype=torch.float32)
            torch._foreach_copy_(list(ps), list(out.unbind()))
        else:
            torch._foreach_copy_(list(ps), list(P.unbind()))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            bf16_sr = group["bf16_sr"]
            compute_dtype = group["compute_dtype"]
            wd_factor = self._wd_factor(group, lr)

            params = group["params"]
            if group.get("batched", False):
                params = self._step_batched(group, wd_factor)

            for p in params:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError("RoseLowRank does not support sparse gradients")

                use_bf16_sr = bf16_sr and p.dtype == torch.bfloat16
                fp32 = use_bf16_sr and compute_dtype is None
                grad = p.grad.to(dtype=torch.float32 if fp32 else compute_dtype)
                param = p.to(dtype=torch.float32 if fp32 else compute_dtype)

                if wd_factor is not None:
                    param.mul_(wd_factor)

                if grad.ndim == 0:
                    param.add_(grad.sign(), alpha=-lr)
                elif grad.ndim == 1:
                    g_min, g_max = grad.aminmax()
                    denom = g_max.abs_().sub_(g_min)
                    denom.masked_fill_(denom == 0.0, 1.0)
                    param.addcdiv_(grad, denom, value=-lr)
                else:
                    n = 1
                    for s in grad.shape[1:]:
                        n *= s
                    if group["centralize"] and n >= 32:
                        grad = grad - grad.mean(dim=tuple(range(1, grad.ndim)), keepdim=True)
                    denom = self._range_denom(grad, group["stabilize"])
                    param.addcdiv_(grad, denom, value=-lr)

                if use_bf16_sr:
                    param = param.to(dtype=torch.float32)
                    p.copy_(
                        torch.empty_like(p, dtype=torch.int32)
                        .random_(0, 0x10000, generator=self.bf16_sr_gen)
                        .add_(param.view(dtype=torch.int32))
                        .bitwise_and_(-0x10000)
                        .view(dtype=torch.float32)
                    )
                elif param is not p:
                    p.copy_(param)

        return loss
