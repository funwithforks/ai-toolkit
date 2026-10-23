# Low-rank native Rose. Use stock rose-opt at rank > 16.
# Derived from Rose v1.0.2 (Matthew Everet Kieren, Apache 2.0).
# The ndim >= 2 update is replaced. Weight decay, bf16 stochastic
# rounding, and the family batching scaffold match rose_v2.
#
# Denominator for a matrix G:
#   ms_trail = shrink(mean(G^2 over trailing axes), prior, n_trail)
#   ms_lead  = shrink(mean(G^2 over leading axis),  prior, n_lead)
#   D = sqrt(ms_trail * ms_lead / prior)
# shrink(ms, prior, n) = (n * ms + shrink * prior) / (n + shrink)
#
# Both LoRA orientations fall out of the same formula:
#   up   [out, rank]  absolute scale from `out`, rank only modulates
#   down [rank, in]   absolute scale from `in`,  rank only modulates
# No axis flip. No |max|-min. Centralize defaults off: on a short
# trailing axis it deletes the common mode, which is the update.

import torch


class RoseLowRank(torch.optim.Optimizer):
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
        shrink: int = 8,
        batched: bool = True,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if weight_decay is not None and weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        if shrink < 0:
            raise ValueError(f"Invalid shrink: {shrink}")

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
                f"bf16_sr=True has no useful effect when compute_dtype is {compute_dtype}. "
                "Use fp32, fp64, or None."
            )

        super().__init__(params, dict(
            lr=lr,
            centralize=centralize,
            stabilize=stabilize,
            weight_decay=weight_decay,
            wd_schedule=wd_schedule,
            bf16_sr=bf16_sr,
            compute_dtype=compute_dtype,
            shrink=shrink,
            batched=batched,
        ))

    # ~0.4 GB peak for the fp32 family buffers on a 32 GB card.
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

    def _equilibrate(self, G, centralize, stabilize, shrink, batch_dims=0):
        """Factored, shrunk RMS denominator. G is not mutated unless centralize."""
        if G.ndim < batch_dims + 2:
            raise RuntimeError("factored equilibration requires ndim >= 2")
        lead = batch_dims
        trail = tuple(range(lead + 1, G.ndim))
        n_lead = G.shape[lead]
        n_trail = 1
        for ax in trail:
            n_trail *= G.shape[ax]

        # Only centralize a trailing axis wide enough to have a mean
        # worth removing. Never centralize the leading axis.
        if centralize and n_trail >= 16:
            G = G - G.mean(dim=trail, keepdim=True)

        sq = G.square()
        prior = sq.mean(dim=tuple(range(lead, G.ndim)), keepdim=True)
        ms_trail = sq.mean(dim=trail, keepdim=True)
        ms_lead = sq.mean(dim=lead, keepdim=True)
        s = float(shrink)
        ms_trail = (n_trail * ms_trail + s * prior) / (n_trail + s)
        ms_lead = (n_lead * ms_lead + s * prior) / (n_lead + s)
        denom = (ms_trail * ms_lead).sqrt() / prior.sqrt().clamp_min(1e-16)

        if stabilize:
            dims = tuple(range(lead, denom.ndim))
            std, mean = torch.std_mean(denom, dim=dims, keepdim=True, correction=0)
            trust = mean.div(std.add(mean).masked_fill(mean == 0.0, 1.0))
            denom = mean.lerp(denom, trust)

        denom.masked_fill_(denom == 0.0, 1.0)
        return G, denom

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
            denom = G.square().mean(dim=1, keepdim=True).sqrt().clamp_min(1e-16)
            denom.masked_fill_(denom == 0.0, 1.0)
            P.addcdiv_(G, denom, value=-lr)
        else:
            G, denom = self._equilibrate(
                G, group["centralize"], group["stabilize"], group["shrink"],
                batch_dims=1,
            )
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
                    denom = grad.square().mean().sqrt().clamp_min(1e-16)
                    param.addcdiv_(grad, denom, value=-lr)
                else:
                    grad, denom = self._equilibrate(
                        grad, group["centralize"], group["stabilize"], group["shrink"],
                    )
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
