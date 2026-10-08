# Derived from "Rose: Range-Of-Slice Equilibration optimizer" v1.0.2
# by Matthew Everet Kieren, Apache License 2.0:
# http://www.apache.org/licenses/LICENSE-2.0
#
# Modifications (Rose v2):
# In the ndim >= 2 branch, the reduction axes are chosen by width rather
# than by position. When a 2D gradient's trailing product is shorter than
# `short` (e.g. a LoRA up-projection `[out, rank]` with small rank), the
# reduction is flipped to axis 0 so each slice's denominator is computed
# over a wide axis instead of a handful of values. The denominator
# formula is also width-gated: reductions of >= short elements use an
# RMS (outlier-robust), narrow reductions keep the `|max| - min` range.
# All defaults and every other branch are unchanged from v1.0.2.


import torch

class RoseV2(torch.optim.Optimizer):
    """Rose v2: Range-Of-Slice Equilibration optimizer, width-driven slices.

    Rose v2 rescales gradients using a per-slice denominator computed by
    reducing the widest available axis. Unlike Adam and other stateful
    optimizers, Rose maintains no per-parameter state between steps: no
    momentum buffers, variance estimates, or even step counters. Memory
    cost is parameters + gradients + working memory, nothing else.

    v2 changes the ndim >= 2 branch only:
      - Axis swap: for 2D gradients whose trailing axes hold fewer than
        32 elements in total, the reduction runs over axis 0 instead, so
        narrow trailing axes (small LoRA ranks) never produce a
        short-sampled denominator.
      - Width-gated denominator: reductions over >= 32 elements use the
        root-mean-square; narrower reductions keep the v1 `|max| - min`
        range. The axis swap and the mean used by `centralize` always
        share the same axes.
    Default behaviour at rank >= 32 is unchanged except that wide
    reductions now use RMS in place of the range.

    Args:
        params (iterable):
            Iterable of model parameters or parameter-group dictionaries.

        lr (float):
            --- Learning Rate ---

            The global step size. Because this optimizer uses range-based
            normalization rather than Adam's RMS-based normalization, the
            same `lr` value can correspond to very different effective
            update sizes. Tune `lr` independently rather than relying on
            Adam defaults.

        weight_decay (float or None, optional) [1e-4]:
            --- Decoupled Weight Decay ---

            A decoupled multiplicative weight-decay coefficient applied
            separately from the adaptive gradient step. It gently
            shrinks weights toward zero at each step and can help reduce
            overfitting. Set to `0` or `None` to disable it.

        wd_schedule (bool or float, optional) [False]:
            --- Schedule-Coupled Weight Decay ---

            Scales weight decay proportionally with the learning-rate
            schedule so that decay weakens as the learning rate drops,
            preventing it from overpowering small updates. The per-step
            factor becomes `1 - (lr / lr_ref) * weight_decay`.

            If `False`, standard decoupled weight decay is used.
            If `True`, `lr_ref` is the first available among
            `group["max_lr"]`, `group["initial_lr"]`, and the
            learning-rate passed at construction time.
            If a float is provided, it is used directly as `lr_ref`.

        centralize (bool, optional) [True]:
            --- Gradient Centralization ---

            Removes shared offsets from gradient slices before the
            denominator computation. This can improve generalization and
            training stability. Biases and other 1D parameters are not
            centralized. Centralization uses the same axes as the
            denominator, including when v2's axis swap is active.

        stabilize (bool, optional) [True]:
            --- Coefficient-of-Variation Trust Gating ---

            Computes a trust factor from the coefficient of variation of
            the per-slice denominator tensor, and then interpolates
            between the local denominator and a smoother global mean
            denominator. This can smooth noisy denominator estimates.
            Some models perform better with it enabled, others disabled;
            try both.

        bf16_sr (bool or torch.Generator, optional) [True]:
            --- Stochastic Rounding for BFloat16 ---

            Improves BF16 training by using stochastic rounding instead
            of plain truncation when writing parameters back. This has
            no effect on non-BF16 parameters.

            BF16 parameters are promoted to `compute_dtype` (or to FP32
            if `compute_dtype` is `None`) for the update and then
            cast to FP32 for write-back. The result is stochastically
            rounded by adding uniform noise to the lower 16 bits of the
            FP32 representation before truncation to BF16.

            If `False`, BF16 stochastic rounding is disabled.
            If `True`, uses the default random-number generator.
            If a `torch.Generator` is provided instead of a boolean,
            treats `bf16_sr` as enabled and forwards that generator
            to `random_`. This is useful when you want reproducible
            stochastic rounding noise.

        compute_dtype (torch.dtype, str, or None, optional) [fp64]:
            --- Internal Compute Precision ---

            Promotes parameters and gradients to this dtype for the
            update step, then casts them back on write-back. Setting
            this to `None` disables promotion and computes in each
            parameter's native dtype, except that BF16 parameters still
            use FP32 when `bf16_sr` is enabled.

            FP64 is recommended because the intermediate range and
            division arithmetic benefits from the extra precision.

            In addition to passing a `torch.dtype` or `None`, the
            following strings are also valid: `float16`, `fp16`,
            `bfloat16`, `bf16`, `float32`, `fp32`, `float64`, `fp64`,
            `none`, and `null`.
    """
    def __init__(
        self,
        params,
        lr: float,
        *,
        weight_decay: float | None = 1e-4,
        wd_schedule: bool | float = False,
        centralize: bool = True,
        stabilize: bool = True,
        bf16_sr: bool | torch.Generator = True,
        compute_dtype: torch.dtype | str | None = "fp64",
        batched: bool = True,
    ):
        if lr < 0.0:
            raise ValueError(f"\nInvalid learning rate: {lr}") from None
        if weight_decay is not None and weight_decay < 0.0:
            raise ValueError(f"\nInvalid weight_decay: {weight_decay}") from None

        if isinstance(bf16_sr, torch.Generator):
            self.bf16_sr_gen = bf16_sr
            bf16_sr = True
        else:
            self.bf16_sr_gen = None

        if isinstance(compute_dtype, str):
            dtype_lookup: dict[str, torch.dtype | None] = {
                "float16": torch.float16, "fp16": torch.float16,
                "float32": torch.float32, "fp32": torch.float32,
                "float64": torch.float64, "fp64": torch.float64,
                "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
                "none": None, "null": None
            }
            try:
                compute_dtype = dtype_lookup[compute_dtype.strip().lower()]
            except KeyError:
                raise ValueError(
                    f"\nInvalid compute_dtype string: {compute_dtype!r}.\n"
                    f"Valid options: {sorted(dtype_lookup)}"
                ) from None

        if bf16_sr and compute_dtype not in (torch.float32, torch.float64, None):
            raise ValueError(
                f"\nbf16_sr=True has no useful effect when compute_dtype is {compute_dtype}.\n"
                f"Use torch.float32, torch.float64, or None (same as fp32) instead."
            ) from None

        defaults = dict(
            lr=lr,
            centralize=centralize,
            stabilize=stabilize,
            weight_decay=weight_decay,
            wd_schedule=wd_schedule,
            bf16_sr=bf16_sr,
            compute_dtype=compute_dtype,
            batched=batched,
        )
        super().__init__(params, defaults)

    # Max stacked members per chunk: bounds the fp64 family buffers at
    # ~0.4 GB peak (G, P, square temp) regardless of family size. The
    # 32 GB fit tier cannot spare more (measured: 32M cap pushed run
    # peak to 32118 MiB vs 31540 control).
    _BATCH_MAX_ELEMS = 16_000_000

    def _wd_factor(self, group, lr):
        """Decoupled weight-decay factor, exactly as step() computes it."""
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

    def _step_batched(self, group, wd_factor):
        """Step same-shape families as stacked buffers; per-member math is
        identical to the loop in step(), with reductions over the member
        axis (slice reduction order/width unchanged). Returns params no
        family applies to (singletons), which the caller loops over."""
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
        # singletons (and anything family-batching declined, e.g. sparse)
        # stay on the per-parameter loop, whose sparse raise is preserved
        remaining = [
            p for p in group["params"]
            if p.grad is not None and id(p) not in batched_ids
        ]

        for ps in families:
            shape = tuple(ps[0].shape)
            per = 1
            for s in shape:
                per *= s
            chunk = max(1, self._BATCH_MAX_ELEMS // per)
            for i in range(0, len(ps), chunk):
                self._step_family(ps[i:i + chunk], group, wd_factor,
                                  compute_dtype, lr)
        return remaining

    @torch.no_grad()
    def _step_family(self, ps, group, wd_factor, compute_dtype, lr):
        n = len(ps)
        shape = tuple(ps[0].shape)
        fp32 = (group["bf16_sr"]
                and ps[0].dtype == torch.bfloat16
                and compute_dtype is None)
        # compute_dtype None means native-dtype compute in the loop path;
        # families are dtype-keyed so the native dtype is uniform
        work_dtype = torch.float32 if fp32 else (
            compute_dtype if compute_dtype is not None else ps[0].dtype
        )
        # stack + cast on copy (value-identical to per-param .to(dtype))
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
            short = 32
            trailing = 1
            for s in shape[1:]:
                trailing *= s
            if len(shape) == 2 and trailing < short:
                member_axes = (1,)          # member axis shifted from (0,)
            else:
                member_axes = tuple(range(2, 1 + len(shape)))
            reduced = 1
            for ax in member_axes:
                reduced *= G.shape[ax]

            if group["centralize"]:
                G.sub_(G.mean(dim=member_axes, keepdim=True))

            if reduced >= short:
                denom = G.square().mean(dim=member_axes, keepdim=True).sqrt()
            else:
                denom = (
                    G.amax(dim=member_axes, keepdim=True).abs_()
                    .sub_(G.amin(dim=member_axes, keepdim=True))
                )

            if group["stabilize"]:
                # per-member coefficient-of-variation over the member's
                # own denominator entries (dims >= 1): same set of values
                # the loop's std_mean(denom) sees for that param
                std, mean = torch.std_mean(
                    denom, dim=tuple(range(1, denom.ndim)),
                    keepdim=True, correction=0,
                )
                trust = mean.div(std.add_(mean).masked_fill_(mean == 0.0, 1.0))
                denom = mean.lerp(denom, trust)

            denom.masked_fill_(denom == 0.0, 1.0)  # SGD fallback
            P.addcdiv_(G, denom, value=-lr)

        if group["bf16_sr"] and ps[0].dtype == torch.bfloat16:
            P32 = P.to(dtype=torch.float32)
            noise = torch.empty(P32.shape, dtype=torch.int32,
                                device=P32.device)
            noise.random_(0, 0x10000, generator=self.bf16_sr_gen)
            out = P32.view(dtype=torch.int32).add_(noise).bitwise_and_(
                -0x10000).view(dtype=torch.float32)
            torch._foreach_copy_(list(ps), list(out.unbind()))
        else:
            torch._foreach_copy_(list(ps), list(P.unbind()))

    @torch.no_grad()
    def step(self, closure=None) -> torch.Tensor | None:
        """Perform a single Rose optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            use_stabilize = group["stabilize"]
            use_centralize = group["centralize"]
            bf16_sr = group["bf16_sr"]
            compute_dtype = group["compute_dtype"]

            # --- Decoupled Weight Decay Factor ---
            wd_factor = self._wd_factor(group, lr)

            params = group["params"]
            if group.get("batched", False):
                params = self._step_batched(group, wd_factor)

            for p in params:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError("Rose does not support sparse gradients")

                # --- Precision Handling ---
                use_bf16_sr = bf16_sr and p.dtype == torch.bfloat16
                fp32 = use_bf16_sr and compute_dtype is None
                grad = p.grad.to(dtype=torch.float32 if fp32 else compute_dtype)
                param = p.to(dtype=torch.float32 if fp32 else compute_dtype)

                # --- Decoupled Multiplicative Weight Decay ---
                if wd_factor is not None:
                    param.mul_(wd_factor)

                if grad.ndim == 0:
                    # --- 0D Scalar ---
                    # Plain signSGD update for single-value parameters.
                    param.add_(grad.sign(), alpha=-lr)

                elif grad.ndim == 1:
                    # --- Vectors / Degenerate Slices ---
                    g_min, g_max = grad.aminmax()
                    denom = g_max.abs_().sub_(g_min)
                    denom.masked_fill_(denom == 0.0, 1.0)
                    param.addcdiv_(grad, denom, value=-lr)

                else:
                    # --- Width-Driven Axis Selection (v2) ---
                    # Reduce the widest available axis. For 2D gradients
                    # whose trailing axes are shorter than `short`, the
                    # reduction is flipped to the leading axis so narrow
                    # trailing axes (small LoRA ranks) never produce a
                    # short-sampled denominator.
                    short = 32
                    trailing = 1
                    for s in grad.shape[1:]:
                        trailing *= s
                    if grad.ndim == 2 and trailing < short:
                        active_axes = (0,)
                    else:
                        active_axes = tuple(range(1, grad.ndim))
                    reduced = 1
                    for ax in active_axes:
                        reduced *= grad.shape[ax]

                    # --- Gradient Centralization ---
                    # Shares active_axes with the denominator below.
                    if use_centralize:
                        if grad is not p.grad:
                            grad.sub_(grad.mean(dim=active_axes, keepdim=True))
                        else:
                            grad = grad.sub(grad.mean(dim=active_axes, keepdim=True))

                    # --- Denominator, width-gated (v2) ---
                    # Wide reductions use RMS (outlier-robust);
                    # narrow reductions keep the v1 `|max| - min` range.
                    if reduced >= short:
                        denom = grad.square().mean(dim=active_axes, keepdim=True).sqrt()
                    else:
                        denom = (
                            grad.amax(dim=active_axes, keepdim=True).abs_()
                            .sub_(grad.amin(dim=active_axes, keepdim=True))
                        )

                    if use_stabilize:
                        # --- Coefficient-of-Variation Trust Gating ---
                        # Measures the self-consistency of per-slice denominators:
                        # Stable denominators preserve local detail.
                        # Noisy denominators use global mean for noise resistance.
                        std, mean = torch.std_mean(denom, correction=0)

                        # Trust factor:
                        # Higher when denominators are self-consistent.
                        # Lower when denominators are heterogeneous.
                        trust = mean.div(std.add_(mean).masked_fill_(mean == 0.0, 1.0))

                        # Blend each local denominator with the smoother mean estimate.
                        denom = mean.lerp(denom, trust)

                    # --- Update: theta -= lr * g / D(g) ---
                    denom.masked_fill_(denom == 0.0, 1.0)  # SGD fallback
                    param.addcdiv_(grad, denom, value=-lr)

                if use_bf16_sr:
                    # --- BF16 stochastic rounding ---
                    # Inspired by Nerogar's code snippet: https://github.com/pytorch/pytorch/issues/120376#issuecomment-1974828905

                    # P(round up) proportional to fractional distance, unbiased expectation
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
