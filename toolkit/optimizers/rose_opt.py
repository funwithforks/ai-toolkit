# Copyright 2026 Matthew Everet Kieren. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Vendored from the rose_opt v1.0.2 single-file package (previously
# installed into site-packages) so the repo owns the code. Modification:
# family-batched stepping (batched=True default) -- same-shape parameter
# families are stepped as stacked buffers with fused multi-tensor copies,
# bitwise-identical to the per-parameter loop (parity test:
# docs/kernel_opt/tests/rose_batched_parity.py; non-fp64 + stabilize
# families stay on the loop, see _step_batched).
# Dispatch for optimizer type "rose" points here.


import torch

class Rose(torch.optim.Optimizer):
    """Rose: Range-Of-Slice Equilibration optimizer.
    In loving memory of my mother, Rose Kieren.
    
    Copyright 2026 Matthew Everet Kieren. All Rights Reserved.
    Licensed under the Apache License, Version 2.0.
    
    Rose rescales gradients using a per-slice `|max| - min` range
    computed by reducing all dimensions beyond the leading axis.
    Unlike Adam and other stateful optimizers, Rose maintains no
    per-parameter state between steps: no momentum buffers,
    variance estimates, or even step counters. Memory cost is
    parameters + gradients + working memory, nothing else.
    
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
            
            Removes shared offsets from gradient slices before the range
            computation. This can improve generalization and training
            stability. Biases and other 1D parameters are not
            centralized.
        
        stabilize (bool, optional) [True]:
            --- Coefficient-of-Variation Trust Gating ---
            
            Computes a trust factor from the coefficient of variation of
            the per-slice range tensor, and then interpolates between the
            local range and a smoother global mean denominator. This can
            smooth noisy range estimates. Some models perform better with
            it enabled, others disabled; try both.
        
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
            update step, then casts them back on write-back. Setting this
            to `None` disables promotion and computes in each parameter's
            native dtype, except that BF16 parameters still use FP32 when
            `bf16_sr` is enabled.
            
            FP64 is recommended because the intermediate range and
            division arithmetic benefits from the extra precision.
            
            In addition to passing a `torch.dtype` or `None`, the
            following strings are also valid: `float16`, `fp16`,
            `bfloat16`, `bf16`, `float32`, `fp32`, `float64`, `fp64`,
            `none`, and `null`.

        batched (bool, optional) [True]:
            --- Family-Batched Stepping ---

            Steps same-shape parameter families as stacked buffers so the
            per-parameter math runs as a handful of whole-stack ops (plus
            fused multi-tensor copies) instead of one eager op per
            parameter per step. Numerically bitwise-identical to the
            per-parameter loop, including stochastic-rounding noise draws
            (all noise is drawn per parameter in group order). Set to
            `False` for the original per-parameter loop only.
    
    References:
        - Kingma, D. P. & Ba, J. (2014), Adam: A Method for
          Stochastic Optimization. arXiv:1412.6980
        - Loshchilov, I. & Hutter, F. (2017), Decoupled Weight Decay
          Regularization. arXiv:1711.05101
        - Hazan, E., Levy, K. Y., & Shalev-Shwartz, S. (2015),
          Beyond Convexity: Stochastic Quasi-Convex Optimization.
          arXiv:1507.02030
        - You, Y., Gitman, I., & Ginsburg, B. (2017), Large Batch
          Training of Convolutional Networks. arXiv:1708.03888
        - Yu, A. W., Huang, L., Lin, Q., Salakhutdinov, R., &
          Carbonell, J. (2017), Block-Normalized Gradient Method: An
          Empirical Study for Training Deep Neural Network.
          arXiv:1707.04822
        - Bernstein, J., Wang, Y. X., Azizzadenesheli, K., &
          Anandkumar, A. (2018), signSGD: Compressed Optimisation for
          Non-Convex Problems. arXiv:1802.04434
        - Yong, H., Huang, J., Hua, X. & Zhang, L. (2020), Gradient
          Centralization: A New Optimization Technique for Deep Neural
          Networks. arXiv:2004.01461
        - Zamirai, P., Zhang, J., Aberger, C. R. & De Sa, C. (2020),
          Revisiting BFloat16 Training. arXiv:2010.06192
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
        batched: bool = True
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
    # ~0.4 GB peak (G, P) regardless of family size.
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
        identical to the loop in step() (reductions keep each member's own
        width and order). Returns params no family applies to, which the
        caller loops over. Stochastic-rounding noise is drawn per parameter
        in group order -- the exact call sequence (shapes, count, order) of
        the loop path -- so a seeded run is bitwise-identical either way."""
        compute_dtype = group["compute_dtype"]
        lr = group["lr"]
        by_key = {}
        for p in group["params"]:
            if p.grad is None or p.grad.is_sparse:
                continue
            key = (tuple(p.shape), p.dtype, p.device)
            by_key.setdefault(key, []).append(p)

        families = []
        for ps in by_key.values():
            if len(ps) < 2:
                continue
            # Batched stabilize is bitwise only where the reduction widths
            # provably agree: std_mean over a stacked member row can pick a
            # different TensorIterator tree than the loop's per-tensor
            # reduce, a 1-ulp fp32/fp16 tie-flip (measured 1.2e-7 rel) --
            # in fp64 no flip was found. Non-fp64 + stabilize: those
            # families stay on the exact loop (they are not marked batched).
            cd = group["compute_dtype"]
            eff64 = cd == torch.float64 or (
                cd is None and ps[0].dtype == torch.float64)
            if group["stabilize"] and not eff64:
                continue
            families.append(ps)
        batched_ids = {id(p) for ps in families for p in ps}
        chunked = []
        for ps in families:
            per = max(1, ps[0].numel())
            chunk = max(1, self._BATCH_MAX_ELEMS // per)
            chunked.extend(ps[i:i + chunk] for i in range(0, len(ps), chunk))

        # SR noise pre-pass: one per-parameter random_ draw of p's shape,
        # in group order -- stream-identical to the loop's inline draws.
        self._sr_noise = {}
        if group["bf16_sr"]:
            for p in group["params"]:
                if p.grad is not None and p.dtype == torch.bfloat16:
                    self._sr_noise[id(p)] = (
                        torch.empty_like(p, dtype=torch.int32)
                        .random_(0, 0x10000, generator=self.bf16_sr_gen)
                    )

        for ps in chunked:
            self._step_family(ps, group, wd_factor, compute_dtype, lr)
        remaining = [
            p for p in group["params"]
            if p.grad is not None and id(p) not in batched_ids
        ]
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
            # v1 semantics: leading axis owns the denominator, every other
            # axis is reduced -- shifted by the member axis when stacked.
            active_axes = tuple(range(2, 1 + len(shape)))

            if group["centralize"]:
                G.sub_(G.mean(dim=active_axes, keepdim=True))

            raw_scale = (
                G.amax(dim=active_axes, keepdim=True).abs_()
                .sub_(G.amin(dim=active_axes, keepdim=True))
            )

            if group["stabilize"]:
                # per-member CV over the member's own scale entries: the
                # same set of values std_mean sees in the loop path
                std, mean = torch.std_mean(
                    raw_scale, dim=tuple(range(1, raw_scale.ndim)),
                    keepdim=True, correction=0,
                )
                trust = mean.div(std.add_(mean).masked_fill_(mean == 0.0, 1.0))
                denom = mean.lerp(raw_scale, trust)
            else:
                denom = raw_scale

            denom.masked_fill_(denom == 0.0, 1.0)  # SGD fallback
            P.addcdiv_(G, denom, value=-lr)

        if group["bf16_sr"] and ps[0].dtype == torch.bfloat16:
            P32 = P.to(dtype=torch.float32)
            NI = torch.empty((n, *shape), dtype=torch.int32,
                             device=ps[0].device)
            torch._foreach_copy_(list(NI.unbind()),
                                 [self._sr_noise.pop(id(p)) for p in ps])
            out = P32.view(dtype=torch.int32).add_(NI).bitwise_and_(
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
            self._sr_noise = {}
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
                    # --- Active Axes: all axes except the first ---
                    # Preserve the leading axis so that each
                    # slice receives its own scale.
                    active_axes = tuple(range(1, grad.ndim))
                    
                    # --- Gradient Centralization ---
                    if use_centralize:
                        if grad is not p.grad:
                            grad.sub_(grad.mean(dim=active_axes, keepdim=True))
                        else:
                            grad = grad.sub(grad.mean(dim=active_axes, keepdim=True))
                    
                    # --- Per-slice Range: `R = |max(g)| - min(g)` ---
                    # Reducing all non-leading dimensions at once
                    # yields one range value per leading-axis slice.
                    raw_scale = (
                        grad.amax(dim=active_axes, keepdim=True).abs_()
                        .sub_(grad.amin(dim=active_axes, keepdim=True))
                    )
                    
                    if use_stabilize:
                        # --- Coefficient-of-Variation Trust Gating ---
                        # Measures the self-consistency of per-slice ranges:
                        # Stable ranges preserve local detail.
                        # Noisy ranges use global mean for noise resistance.
                        std, mean = torch.std_mean(raw_scale, correction=0)
                        
                        # Trust factor:
                        # Higher when ranges are self-consistent.
                        # Lower when ranges are heterogeneous.
                        trust = mean.div(std.add_(mean).masked_fill_(mean == 0.0, 1.0))
                        
                        # Blend each local range with the smoother mean estimate.
                        denom = mean.lerp(raw_scale, trust)
                    else:
                        denom = raw_scale
                    
                    # --- Update: θ -= lr · g / D(g) ---
                    denom.masked_fill_(denom == 0.0, 1.0)  # SGD fallback
                    param.addcdiv_(grad, denom, value=-lr)
                
                if use_bf16_sr:
                    # --- BF16 stochastic rounding ---
                    # Inspired by Nerogar's code snippet: https://github.com/pytorch/pytorch/issues/120376#issuecomment-1974828905
                    
                    # P(round up) ∝ fractional distance → unbiased expectation
                    param = param.to(dtype=torch.float32)
                    noise = self._sr_noise.pop(id(p), None)
                    if noise is None:
                        noise = torch.empty_like(p, dtype=torch.int32).random_(
                            0, 0x10000, generator=self.bf16_sr_gen)
                    p.copy_(
                        noise
                        .add_(param.view(dtype=torch.int32))
                        .bitwise_and_(-0x10000)
                        .view(dtype=torch.float32)
                    )
                
                elif param is not p:
                    p.copy_(param)
        
        return loss
