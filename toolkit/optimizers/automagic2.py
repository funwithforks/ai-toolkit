from typing import List
import torch


class Automagic2(torch.optim.Optimizer):
    """
    Automagic v2.

    A single scalar learning rate is kept per parameter (e.g. one lr for the
    full weight matrix of a Linear layer rather than one per element). The lr
    is nudged up when the per-element update direction stays consistent with
    the previous step and nudged down when it flips, clamped to [min_lr, max_lr].

    The optimizer step is fused into the backward pass via
    ``register_post_accumulate_grad_hook``: each parameter is updated and its
    grad freed as soon as autograd finishes accumulating into it. ``.step()``
    therefore does no real work and peak VRAM stays low.

    Second-moment EMA state is stored in ``p.dtype`` (math runs in fp32 when
    the state is lower precision). Stochastic rounding is applied only when
    writing back to a bf16 parameter.

    ``fused=False`` (opt-in) defers updates to ``step()`` and runs same-shape
    parameter families as ONE batched tensor (stack -> factored math ->
    foreach copy-back): ~30 launches per ~224-member family instead of
    224 x ~43, which is worth ~100 ms/step CPU on 5090 LoRA training.
    Deferral keeps ``p.grad`` alive until the step, so the trainer's
    ``max_grad_norm`` clip actually applies, and multi-micro-batch
    accumulation averages instead of applying N full updates (the hook mode
    does the latter by design). Row-wise math is unchanged; only stochastic
    rounding's RNG draw shape differs (same noise class).
    """

    def __init__(
        self,
        params,
        lr: float = 1e-6,
        min_lr: float = 1e-7,
        max_lr: float = 1e-3,
        lr_bump: float = 1e-6,
        beta2: float = 0.999,
        eps: float = 1e-30,
        clip_threshold: float = 1.0,
        weight_decay: float = 0.0,
        agreement_threshold: float = 0.5,
        fused: bool = True,
    ):
        if lr > 1e-3:
            print(f"Warning! Start lr {lr} is very high; forcing to 1e-6.")
            lr = 1e-6
        defaults = dict(
            lr=lr,
            min_lr=min_lr,
            max_lr=max_lr,
            lr_bump=lr_bump,
            beta2=beta2,
            eps=eps,
            clip_threshold=clip_threshold,
            weight_decay=weight_decay,
            agreement_threshold=agreement_threshold,
        )
        super().__init__(params, defaults)

        self.fused = bool(fused)
        self._hook_handles = []
        if self.fused:
            for group in self.param_groups:
                for p in group["params"]:
                    if p.requires_grad:
                        handle = p.register_post_accumulate_grad_hook(
                            self._make_backward_hook(group)
                        )
                        self._hook_handles.append(handle)
            print(
                "[Automagic2] updates run inside post-accumulate-grad hooks and "
                "free p.grad during backward: the trainer's max_grad_norm "
                "clip_grad_norm_ sees no grads and is a no-op. Update-magnitude "
                "control is per-param clip_threshold instead. "
                "(optimizer_params {fused: false} defers updates to step() and "
                "batches same-shape families.)"
            )
        else:
            print(
                "[Automagic2] deferred mode: grads stay on params until step(); "
                "max_grad_norm now applies; same-shape params update as one "
                "batched tensor."
            )

        total = sum(p.numel() for g in self.param_groups for p in g["params"])
        print(f"Total training paramiters: {total:,}")

    # ------------------------------------------------------------------ utils

    @staticmethod
    def _rms(t: torch.Tensor) -> torch.Tensor:
        return t.norm(2) / (t.numel() ** 0.5)

    @staticmethod
    def _approx_sq_grad(row: torch.Tensor, col: torch.Tensor) -> torch.Tensor:
        r = (row / row.mean(dim=-1, keepdim=True)).rsqrt_().unsqueeze(-1)
        c = col.unsqueeze(-2).rsqrt()
        return torch.mul(r, c)

    def _init_state(self, p: torch.Tensor, group: dict) -> None:
        state = self.state[p]
        state["step"] = 0
        state["lr"] = torch.full(
            (), float(group["lr"]), dtype=torch.float32, device=p.device
        )
        state["last_polarity"] = torch.zeros(p.shape, dtype=torch.bool, device=p.device)
        if p.dim() >= 2:
            state["exp_avg_sq_row"] = torch.zeros(
                p.shape[:-1], dtype=p.dtype, device=p.device
            )
            state["exp_avg_sq_col"] = torch.zeros(
                p.shape[:-2] + p.shape[-1:], dtype=p.dtype, device=p.device
            )
        else:
            state["exp_avg_sq"] = torch.zeros(p.shape, dtype=p.dtype, device=p.device)

    def _make_backward_hook(self, group):
        def _hook(p: torch.Tensor):
            self._update_param(p, group)

        return _hook

    # -------------------------------------------------------------- per-param

    @torch.no_grad()
    def _update_param(self, p: torch.Tensor, group: dict) -> None:
        if p.grad is None:
            return
        state = self.state[p]
        if len(state) == 0:
            self._init_state(p, group)

        grad = p.grad
        if grad.is_sparse:
            raise RuntimeError("Automagic2 does not support sparse gradients.")
        if grad.dtype != torch.float32:
            grad = grad.to(torch.float32)

        beta2 = group["beta2"]
        eps = group["eps"]
        sq = (grad * grad).add_(eps)

        if p.dim() >= 2:
            row_state = state["exp_avg_sq_row"]
            col_state = state["exp_avg_sq_col"]
            if row_state.dtype == torch.float32:
                row, col = row_state, col_state
                row.mul_(beta2).add_(sq.mean(dim=-1), alpha=1.0 - beta2)
                col.mul_(beta2).add_(sq.mean(dim=-2), alpha=1.0 - beta2)
            else:
                row = row_state.to(torch.float32)
                col = col_state.to(torch.float32)
                row.mul_(beta2).add_(sq.mean(dim=-1), alpha=1.0 - beta2)
                col.mul_(beta2).add_(sq.mean(dim=-2), alpha=1.0 - beta2)
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
            update = v.rsqrt().mul_(grad)

        update.div_((self._rms(update) / group["clip_threshold"]).clamp_(min=1.0))

        # Per-element sign agreement collapsed to a single bump decision.
        # Kept on-device as a 0-D tensor to avoid a CPU<->GPU sync in the hot path.
        cur_polarity = update > 0
        last_polarity = state["last_polarity"]
        agreement = (cur_polarity == last_polarity).to(torch.float32).mean()
        state["last_polarity"] = cur_polarity

        lr_t = state["lr"]
        if state["step"] > 0:
            direction = (agreement >= group["agreement_threshold"]).to(lr_t.dtype) * 2.0 - 1.0
            lr_t.add_(direction, alpha=group["lr_bump"]).clamp_(
                min=group["min_lr"], max=group["max_lr"]
            )
        state["step"] += 1

        update.mul_(lr_t)
        wd = group["weight_decay"]

        if p.dtype == torch.bfloat16:
            # Single bf16 -> fp32 conversion shared by weight decay and SR.
            new_p_fp32 = p.to(torch.float32)
            if wd != 0.0:
                update.addcmul_(new_p_fp32, lr_t, value=wd)
            new_p_fp32.sub_(update)
            # Stochastic rounding fp32 -> bf16: add random noise into the lower
            # 16 mantissa bits, then truncate. Done in place on new_p_fp32 so
            # we don't allocate a separate int32 work buffer.
            as_int = new_p_fp32.view(torch.int32)
            as_int.add_(torch.randint_like(as_int, 1 << 16)).bitwise_and_(-65536)
            p.copy_(new_p_fp32)
        else:
            if wd != 0.0:
                p_fp32 = p if p.dtype == torch.float32 else p.to(torch.float32)
                update.addcmul_(p_fp32, lr_t, value=wd)
            p.add_(update.to(p.dtype), alpha=-1.0)

        p.grad = None

    # ----------------------------------------------------------- optimizer API

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        if self.fused:
            return loss
        for group in self.param_groups:
            families: dict = {}
            leftovers = []
            for p in group["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                if (
                    p.dim() >= 2
                    and not p.grad.is_sparse
                    and len(st) > 0
                ):
                    families.setdefault((p.shape, p.dtype), []).append(p)
                else:
                    leftovers.append(p)
            for ps in families.values():
                if len(ps) >= 2:
                    self._update_family(ps, group)
                else:
                    self._update_param(ps[0], group)
            for p in leftovers:
                self._update_param(p, group)
        return loss

    @torch.no_grad()
    def _update_family(self, ps, group: dict) -> None:
        """Same-shape 2-D members updated as one (N, *shape) tensor.

        Mirrors _update_param exactly (factored second moments, update-RMS
        clip, sign-agreement lr nudge, stochastic bf16 round-back). The only
        intended numerical difference is the shape of the SR randint draw,
        which is the same noise class; row/col EMAs run the same per-row
        reductions.
        """
        n = len(ps)
        numel_each = ps[0].numel()
        G = torch.stack([p.grad for p in ps])
        if G.dtype != torch.float32:
            G = G.to(torch.float32)
        beta2 = group["beta2"]
        sq = (G * G).add_(group["eps"])

        rows = torch.stack([self.state[p]["exp_avg_sq_row"] for p in ps])
        cols = torch.stack([self.state[p]["exp_avg_sq_col"] for p in ps])
        rows = rows.to(torch.float32)
        cols = cols.to(torch.float32)
        rows.mul_(beta2).add_(sq.mean(dim=-1), alpha=1.0 - beta2)
        cols.mul_(beta2).add_(sq.mean(dim=-2), alpha=1.0 - beta2)
        r = (rows / rows.mean(dim=-1, keepdim=True)).rsqrt_().unsqueeze(-1)
        c = cols.unsqueeze(-2).rsqrt()
        update = torch.mul(r, c).mul_(G)

        u_rms = update.reshape(n, numel_each).norm(2, dim=1) / (numel_each**0.5)
        update.div_(
            (u_rms.view(n, *([1] * (update.dim() - 1))) / group["clip_threshold"]).clamp_(min=1.0)
        )

        cur_polarity = update > 0
        last = torch.stack([self.state[p]["last_polarity"] for p in ps])
        agreement = (cur_polarity == last).to(torch.float32).mean(
            dim=tuple(range(1, update.dim())), keepdim=True
        )

        L = torch.stack([self.state[p]["lr"] for p in ps]).float()
        first = self.state[ps[0]]["step"] == 0
        if not first:
            direction = (agreement >= group["agreement_threshold"]).to(L.dtype) * 2.0 - 1.0
            L.add_(direction.reshape(n), alpha=group["lr_bump"]).clamp_(
                min=group["min_lr"], max=group["max_lr"]
            )

        update.mul_(L.view(n, *([1] * (update.dim() - 1))))
        wd = group["weight_decay"]

        if ps[0].dtype == torch.bfloat16:
            new_p_fp32 = torch.stack([p for p in ps]).to(torch.float32)
            if wd != 0.0:
                update.addcmul_(new_p_fp32, L.view(n, *([1] * (update.dim() - 1))), value=wd)
            new_p_fp32.sub_(update)
            as_int = new_p_fp32.view(torch.int32)
            as_int.add_(torch.randint_like(as_int, 1 << 16)).bitwise_and_(-65536)
            torch._foreach_copy_(list(ps), list(new_p_fp32))
        else:
            P = torch.stack([p for p in ps])
            upd = update.to(P.dtype)
            if wd != 0.0:
                upd.addcmul_(P, L.view(n, *([1] * (update.dim() - 1))).to(P.dtype), value=wd)
            P.sub_(upd)
            torch._foreach_copy_(list(ps), list(P))

        # state write-back (per-member bf16 round, same as _update_param)
        torch._foreach_copy_(
            [self.state[p]["exp_avg_sq_row"] for p in ps], list(rows.to(ps[0].dtype))
        )
        torch._foreach_copy_(
            [self.state[p]["exp_avg_sq_col"] for p in ps], list(cols.to(ps[0].dtype))
        )
        torch._foreach_copy_(
            [self.state[p]["last_polarity"] for p in ps], list(cur_polarity)
        )
        torch._foreach_copy_([self.state[p]["lr"] for p in ps], list(L))
        for p in ps:
            self.state[p]["step"] += 1
            p.grad = None

    def get_learning_rates(self) -> List[float]:
        # per-param lr is a 0-D CUDA tensor; float() on each would be one
        # D2H round-trip per parameter (448/step from the tqdm postfix).
        # Stack first: a single transfer, same values in the same order.
        out = []
        for group in self.param_groups:
            lrs = [
                self.state[p]["lr"]
                for p in group["params"]
                if p in self.state and "lr" in self.state[p]
            ]
            if not lrs:
                out.append(float(group["lr"]))
                continue
            vals = torch.stack([t.reshape(()) for t in lrs]).float().cpu().tolist()
            out.append(sum(vals) / len(vals))
        return out

    def get_avg_learning_rate(self) -> float:
        lrs = self.get_learning_rates()
        return sum(lrs) / len(lrs) if lrs else float(self.defaults["lr"])

    def load_state_dict(self, state_dict):
        # Parent casts every fp state tensor to param.dtype; force lr back to fp32
        # so subsequent lr_bump (default 1e-6) isn't rounded away on bf16 weights.
        super().load_state_dict(state_dict)
        # Constructor args always win over whatever was saved in the checkpoint.
        for group in self.param_groups:
            for k, v in self.defaults.items():
                group[k] = v
            for p in group["params"]:
                st = self.state.get(p)
                if st is not None and isinstance(st.get("lr"), torch.Tensor):
                    st["lr"] = st["lr"].to(torch.float32)
