from __future__ import division
from __future__ import unicode_literals

from typing import Iterable, Optional
import threading
import weakref
import copy
import contextlib
from toolkit.optimizers.optimizer_utils import copy_stochastic, copy_stochastic_bf16

import torch


# Partially based on:
# https://github.com/tensorflow/tensorflow/blob/r1.13/tensorflow/python/training/moving_averages.py
class ExponentialMovingAverage:
    """
    Maintains (exponential) moving average of a set of parameters.

    Args:
        parameters: Iterable of `torch.nn.Parameter` (typically from
            `model.parameters()`).
            Note that EMA is computed on *all* provided parameters,
            regardless of whether or not they have `requires_grad = True`;
            this allows a single EMA object to be consistantly used even
            if which parameters are trainable changes step to step.

            If you want to some parameters in the EMA, do not pass them
            to the object in the first place. For example:

                ExponentialMovingAverage(
                    parameters=[p for p in model.parameters() if p.requires_grad],
                    decay=0.9
                )

            will ignore parameters that do not require grad.

        decay: The exponential decay.

        use_num_updates: Whether to use number of updates when computing
            averages.
    """

    def __init__(
            self,
            parameters: Iterable[torch.nn.Parameter] = None,
            decay: float = 0.995,
            use_num_updates: bool = False,
            # pulls the parameter toward the shadow each update
            use_feedback: bool = False,
            feedback_rate: float = 0.001,
            param_multiplier: float = 1.0,
            # apply the EMA once every `interval` update() calls (optimizer
            # steps). decay keeps its meaning as the per-STEP factor: an
            # applied update absorbs 1-(1-decay)**interval, so the averaging
            # horizon is unchanged by the interval. interval=1 (default) is
            # the historical every-step path, byte-for-byte. Note the legacy
            # use_num_updates ramp counts applied updates, so it stretches
            # with interval if that (unused by the toolkit) path is enabled;
            # use_feedback/param_multiplier stay per-applied-update.
            interval: int = 1,
            # park the shadow on the host (pinned) and run the blend on CPU.
            # update() then only issues an async D2H of the live params on a
            # side stream; a worker thread does the fp32 blend + stochastic
            # round while the GPU proceeds. Frees shadow-bytes of VRAM.
            # One flag, no fallback: if the host cannot back the pinned
            # copies, the allocation raises at setup and the run dies there.
            park_cpu: bool = False,
    ):
        if parameters is None:
            raise ValueError("parameters must be provided")
        if decay < 0.0 or decay > 1.0:
            raise ValueError('Decay must be between 0 and 1')
        interval = int(interval)
        if interval < 1:
            raise ValueError('interval must be a positive integer')
        self.decay = decay
        self.interval = interval
        self.step_counter = 0
        self.num_updates = 0 if use_num_updates else None
        self.use_feedback = use_feedback
        self.feedback_rate = feedback_rate
        self.param_multiplier = param_multiplier
        parameters = list(parameters)
        self.shadow_params = [
            p.clone().detach()
            for p in parameters
        ]
        self.collected_params = None
        self._is_train_mode = True
        # By maintaining only a weakref to each parameter,
        # we maintain the old GC behaviour of ExponentialMovingAverage:
        # if the model goes out of scope but the ExponentialMovingAverage
        # is kept, no references to the model or its parameters will be
        # maintained, and the model will be cleaned up.
        self._params_refs = [weakref.ref(p) for p in parameters]
        # CPU-parked state (see park_cpu in __init__ doc)
        self.park_cpu = False
        self._park_bytes = sum(p.numel() * p.element_size()
                               for p in self.shadow_params)
        self._staging = None
        self._copy_stream = None
        self._job_done = threading.Event()
        self._job_done.set()
        self._worker_error = None
        self._warned_busy = False
        if park_cpu:
            self._init_park()
            # daemon blend threads killed mid-torch-call during interpreter
            # finalization abort the process ("terminate called without an
            # active exception") when a save did not already join them --
            # run the join before threading tears daemons down. Print-only:
            # must never raise or stall a run that already succeeded.
            threading._register_atexit(self._join_at_exit)

    def _init_park(self):
        if not self.shadow_params:
            return
        dev = self.shadow_params[0].device
        if dev.type != "cuda":
            raise ValueError(
                "EMA park_cpu requires CUDA parameters (nothing to park "
                "when training runs on CPU)")
        if self.use_feedback or self.param_multiplier != 1.0:
            raise ValueError(
                "EMA park_cpu does not support use_feedback / "
                "param_multiplier (they rewrite the live params per apply); "
                "disable them or run GPU-resident")
        # one flag, no fallback: the shadow moves to pinned host RAM or the
        # allocation raises and the run dies at setup, not silently degraded
        self.shadow_params = [
            p.detach().to("cpu").pin_memory() for p in self.shadow_params
        ]
        self.park_cpu = True
        print(f"[ema] shadow parked on pinned host RAM: "
              f"{self._park_bytes / 2 ** 30:.2f} GiB freed from VRAM; "
              f"CPU blend runs async every {self.interval} step(s)")

    def _get_parameters(
            self,
            parameters: Optional[Iterable[torch.nn.Parameter]]
    ) -> Iterable[torch.nn.Parameter]:
        if parameters is None:
            parameters = [p() for p in self._params_refs]
            if any(p is None for p in parameters):
                raise ValueError(
                    "(One of) the parameters with which this "
                    "ExponentialMovingAverage "
                    "was initialized no longer exists (was garbage collected);"
                    " please either provide `parameters` explicitly or keep "
                    "the model to which they belong from being garbage "
                    "collected."
                )
            return parameters
        else:
            parameters = list(parameters)
            if len(parameters) != len(self.shadow_params):
                raise ValueError(
                    "Number of parameters passed as argument is different "
                    "from number of shadow parameters maintained by this "
                    "ExponentialMovingAverage"
                )
            return parameters

    def update(
            self,
            parameters: Optional[Iterable[torch.nn.Parameter]] = None
    ) -> None:
        """
        Update currently maintained parameters.

        Call this every time the parameters are updated, such as the result of
        the `optimizer.step()` call.

        Args:
            parameters: Iterable of `torch.nn.Parameter`; usually the same set of
                parameters used to initialize this object. If `None`, the
                parameters with which this `ExponentialMovingAverage` was
                initialized will be used.
        """
        parameters = self._get_parameters(parameters)
        # every call is an optimizer step; apply only on the interval phase
        self.step_counter += 1
        if self.interval > 1 and self.step_counter % self.interval != 0:
            return
        decay = self.decay
        if self.num_updates is not None:
            self.num_updates += 1
            decay = min(
                decay,
                (1 + self.num_updates) / (10 + self.num_updates)
            )
        one_minus_decay = 1.0 - decay
        if self.interval > 1:
            # horizon-preserving absorb for an update applied once per
            # `interval` steps: k steps of (1 - x) == one step of (1 - x)^k
            one_minus_decay = 1.0 - (1.0 - one_minus_decay) ** self.interval
        if self.park_cpu:
            self._parked_apply(one_minus_decay, parameters)
            return
        with torch.no_grad():
            for s_param, param in zip(self.shadow_params, parameters):
                s_param_float = s_param.float()
                if s_param.dtype != torch.float32:
                    s_param_float = s_param_float.to(torch.float32)
                param_float = param
                if param.dtype != torch.float32:
                    param_float = param_float.to(torch.float32)
                gap = (s_param_float - param_float)
                s_param_float.sub_(gap * one_minus_decay)
                
                update_param = False
                if self.use_feedback:
                    param_float.add_(gap * self.feedback_rate)
                    update_param = True
                
                if self.param_multiplier != 1.0:
                    param_float.mul_(self.param_multiplier)
                    update_param = True
                
                if s_param.dtype !=  torch.float32:
                    copy_stochastic(s_param, s_param_float)
                
                if update_param and param.dtype != torch.float32:
                    copy_stochastic(param, param_float)
                

    # ---- CPU-parked pipeline -------------------------------------------------

    def _parked_apply(self, one_minus_decay: float, parameters) -> None:
        """Issue async D2H of the live params on a side stream and hand the
        blend to a worker thread. Returns without touching the GPU default
        stream's critical path; the next apply (or any shadow reader) joins
        the worker. Assumes params are not rewritten until the copy stream
        drains, which the step cadence guarantees with seconds to spare."""
        self._sync_worker()
        first = parameters[0]
        if self._staging is None:
            self._staging = [
                torch.empty_like(s, pin_memory=True) for s in self.shadow_params
            ]
        if self._copy_stream is None:
            self._copy_stream = torch.cuda.Stream(device=first.device)
        src_ready = torch.cuda.Event()
        src_ready.record()   # after the optimizer step on the current stream
        events = []
        with torch.cuda.stream(self._copy_stream):
            self._copy_stream.wait_event(src_ready)
            for buf, param in zip(self._staging, parameters):
                if buf.shape != param.shape:
                    raise RuntimeError(
                        "parked EMA: param shape changed since init "
                        f"({tuple(buf.shape)} -> {tuple(param.shape)})")
                buf.copy_(param.detach(), non_blocking=True)
                ev = torch.cuda.Event()
                ev.record(self._copy_stream)
                events.append(ev)
        self._job_done.clear()
        threading.Thread(
            target=self._blend_job,
            args=(list(self._staging), events, one_minus_decay),
            daemon=True,
        ).start()

    def _blend_job(self, staging, events, one_minus_decay) -> None:
        """CPU mirror of the GPU update body: fp32 blend, stochastic-round
        writeback (copy_stochastic_bf16 is device-generic integer math)."""
        try:
            with torch.no_grad():
                for s_param, buf, ev in zip(self.shadow_params, staging, events):
                    ev.synchronize()   # this tensor's D2H landed
                    s_f = s_param.float()
                    gap = s_f - buf.float()
                    s_f.sub_(gap * one_minus_decay)
                    copy_stochastic_bf16(s_param, s_f)
                    del s_f, gap
        except BaseException as e:  # surfaced loudly at the next join point
            if self._worker_error is None:
                self._worker_error = e
        finally:
            self._job_done.set()

    def _join_at_exit(self) -> None:
        """atexit: let an in-flight blend finish so the daemon thread is
        not killed inside torch during finalization (-> SIGABRT). Bounded
        and print-only: a completed run must exit 0 regardless."""
        try:
            if not self._job_done.wait(timeout=60):
                print("[ema] WARNING: parked blend still running at exit; "
                      "shadow state may be one apply stale in any unsaved path")
            if self._worker_error is not None:
                print(f"[ema] WARNING: parked blend error at exit: "
                      f"{self._worker_error!r}")
        except Exception as e:  # interpreter is shutting down: never raise
            print(f"[ema] WARNING: parked EMA exit join failed: {e!r}")

    def _sync_worker(self) -> None:
        if not self.park_cpu:
            return
        if not self._job_done.is_set():
            if not self._warned_busy:
                print("[ema] parked EMA worker still blending; syncing "
                      "(CPU is falling behind the apply cadence)")
                self._warned_busy = True
            if not self._job_done.wait(timeout=600):
                raise RuntimeError("parked EMA blend did not finish in 600 s")
        if self._worker_error is not None:
            err, self._worker_error = self._worker_error, None
            raise RuntimeError(f"parked EMA blend failed: {err!r}") from err

    def copy_to(
            self,
            parameters: Optional[Iterable[torch.nn.Parameter]] = None
    ) -> None:
        """
        Copy current averaged parameters into given collection of parameters.

        Args:
            parameters: Iterable of `torch.nn.Parameter`; the parameters to be
                updated with the stored moving averages. If `None`, the
                parameters with which this `ExponentialMovingAverage` was
                initialized will be used.
        """
        parameters = self._get_parameters(parameters)
        self._sync_worker()   # parked: finish any in-flight blend first
        for s_param, param in zip(self.shadow_params, parameters):
            param.data.copy_(s_param.data)

    def store(
            self,
            parameters: Optional[Iterable[torch.nn.Parameter]] = None
    ) -> None:
        """
        Save the current parameters for restoring later.

        Args:
            parameters: Iterable of `torch.nn.Parameter`; the parameters to be
                temporarily stored. If `None`, the parameters of with which this
                `ExponentialMovingAverage` was initialized will be used.
        """
        parameters = self._get_parameters(parameters)
        self.collected_params = [
            param.clone()
            for param in parameters
        ]

    def restore(
            self,
            parameters: Optional[Iterable[torch.nn.Parameter]] = None
    ) -> None:
        """
        Restore the parameters stored with the `store` method.
        Useful to validate the model with EMA parameters without affecting the
        original optimization process. Store the parameters before the
        `copy_to` method. After validation (or model saving), use this to
        restore the former parameters.

        Args:
            parameters: Iterable of `torch.nn.Parameter`; the parameters to be
                updated with the stored parameters. If `None`, the
                parameters with which this `ExponentialMovingAverage` was
                initialized will be used.
        """
        if self.collected_params is None:
            raise RuntimeError(
                "This ExponentialMovingAverage has no `store()`ed weights "
                "to `restore()`"
            )
        parameters = self._get_parameters(parameters)
        for c_param, param in zip(self.collected_params, parameters):
            param.data.copy_(c_param.data)

    @contextlib.contextmanager
    def average_parameters(
            self,
            parameters: Optional[Iterable[torch.nn.Parameter]] = None
    ):
        r"""
        Context manager for validation/inference with averaged parameters.

        Equivalent to:

            ema.store()
            ema.copy_to()
            try:
                ...
            finally:
                ema.restore()

        Args:
            parameters: Iterable of `torch.nn.Parameter`; the parameters to be
                updated with the stored parameters. If `None`, the
                parameters with which this `ExponentialMovingAverage` was
                initialized will be used.
        """
        parameters = self._get_parameters(parameters)
        self.store(parameters)
        self.copy_to(parameters)
        try:
            yield
        finally:
            self.restore(parameters)

    def to(self, device=None, dtype=None) -> None:
        r"""Move internal buffers of the ExponentialMovingAverage to `device`.

        Args:
            device: like `device` argument to `torch.Tensor.to`
        """
        # .to() on the tensors handles None correctly
        # (parked shadows are host-resident by design; device moves must not
        # undo the park)
        if not self.park_cpu:
            self.shadow_params = [
                p.to(device=device, dtype=dtype)
                if p.is_floating_point()
                else p.to(device=device)
                for p in self.shadow_params
            ]
        if self.collected_params is not None:
            self.collected_params = [
                p.to(device=device, dtype=dtype)
                if p.is_floating_point()
                else p.to(device=device)
                for p in self.collected_params
            ]
        return

    def state_dict(self) -> dict:
        r"""Returns the state of the ExponentialMovingAverage as a dict."""
        # Following PyTorch conventions, references to tensors are returned:
        # "returns a reference to the state and not its copy!" -
        # https://pytorch.org/tutorials/beginner/saving_loading_models.html#what-is-a-state-dict
        self._sync_worker()   # never read a shadow mid-blend
        return {
            "decay": self.decay,
            "num_updates": self.num_updates,
            "shadow_params": self.shadow_params,
            "collected_params": self.collected_params,
            "step_counter": self.step_counter,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        r"""Loads the ExponentialMovingAverage state.

        Args:
            state_dict (dict): EMA state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        # deepcopy, to be consistent with module API
        state_dict = copy.deepcopy(state_dict)
        self.decay = state_dict["decay"]
        if self.decay < 0.0 or self.decay > 1.0:
            raise ValueError('Decay must be between 0 and 1')
        self.num_updates = state_dict["num_updates"]
        assert self.num_updates is None or isinstance(self.num_updates, int), \
            "Invalid num_updates"
        # older checkpoints have no phase record; restarting the interval
        # phase is harmless for a scale-free exponential EMA
        self.step_counter = int(state_dict.get("step_counter", 0))

        self.shadow_params = state_dict["shadow_params"]
        assert isinstance(self.shadow_params, list), \
            "shadow_params must be a list"
        assert all(
            isinstance(p, torch.Tensor) for p in self.shadow_params
        ), "shadow_params must all be Tensors"
        if self.park_cpu:
            # a loaded shadow must live where a parked shadow lives
            self.shadow_params = [
                p.detach().to("cpu").pin_memory() for p in self.shadow_params
            ]
            self._staging = None   # rebuild against loaded state

        self.collected_params = state_dict["collected_params"]
        if self.collected_params is not None:
            assert isinstance(self.collected_params, list), \
                "collected_params must be a list"
            assert all(
                isinstance(p, torch.Tensor) for p in self.collected_params
            ), "collected_params must all be Tensors"
            assert len(self.collected_params) == len(self.shadow_params), \
                "collected_params and shadow_params had different lengths"

        if len(self.shadow_params) == len(self._params_refs):
            # Consistant with torch.optim.Optimizer, cast things to consistant
            # device and dtype with the parameters
            params = [p() for p in self._params_refs]
            # If parameters have been garbage collected, just load the state
            # we were given without change.
            if not any(p is None for p in params):
                # ^ parameter references are still good
                for i, p in enumerate(params):
                    # parked shadows are host-resident by design: a loaded
                    # state must not yank them to GPU (the next parked
                    # update() would then rebuild staging with
                    # pin_memory on CUDA tensors and raise). dtype still
                    # follows the params.
                    self.shadow_params[i] = self.shadow_params[i].to(
                        device="cpu" if self.park_cpu else p.device,
                        dtype=p.dtype
                    )
                    if self.collected_params is not None:
                        self.collected_params[i] = self.collected_params[i].to(
                            device="cpu" if self.park_cpu else p.device,
                            dtype=p.dtype
                        )
        else:
            raise ValueError(
                "Tried to `load_state_dict()` with the wrong number of "
                "parameters in the saved state."
            )

    def eval(self):
        if self._is_train_mode:
            with torch.no_grad():
                self.store()
                self.copy_to()
                self._is_train_mode = False

    def train(self):
        if not self._is_train_mode:
            with torch.no_grad():
                self.restore()
                self._is_train_mode = True
