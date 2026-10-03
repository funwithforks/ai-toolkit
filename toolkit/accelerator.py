import torch
from accelerate import Accelerator
from diffusers.utils.torch_utils import is_compiled_module

global_accelerator = None


def get_accelerator() -> Accelerator:
    global global_accelerator
    if global_accelerator is None:
        # Blackwell training policy: bf16 autocast for the prepared
        # models (fp32 masters were a pre-mixed-precision holdover;
        # bf16/int8 bases gain no information from fp32 compute).
        # Programmatic Accelerator() ignores the machine's
        # accelerate default_config.yaml, so set it explicitly.
        import torch
        mp = 'no'
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            mp = 'bf16'
        global_accelerator = Accelerator(mixed_precision=mp)
    return global_accelerator

def unwrap_model(model):
    try:
        accelerator = get_accelerator()
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
    except Exception as e:
        pass
    return model


def drop_fp32_outcome_wrapper(model):
    # native-dtype training policy (established with krea2): accelerate's
    # prepare() rebinds model.forward to autocast(bf16) +
    # convert_outputs_to_fp32 when native_amp is on (bf16/fp16). That
    # fp32-outcome cast adds an elementwise pass at every model boundary
    # per step, and autocast's per-op cast dispatch is work the bf16/int8
    # bases already avoid by running in their own dtypes. fp32 LORA MASTER
    # PARAMS are a separate deliberate choice (update precision) and are
    # untouched here.
    # Safe to undo single-device: prepare_model's remaining work on a
    # non-multi-device model is .to(device) plus the _is_accelerate_prepared
    # marker (accelerator.py:1825-1836,1879,2074); the device move
    # survives a forward rebinding, and DDP/FSDP/dynamo paths never take
    # the autocast-binding branch we undo (we refuse anything but the
    # pristine wrapper). The wrapper is only ever set when native_amp,
    # so multi-device runs (which use that branch for real) are skipped
    # by the multi_device guard.
    if getattr(model, "_is_accelerate_prepared", False) and not get_accelerator().multi_device:
        original = getattr(model, "_original_forward", None)
        # the original must be the model's own bound method: anything else
        # (e.g. a MemoryManager-style plain-function forward, which python
        # leaves unbound on the instance) cannot be restored to a known
        # state, so we refuse and leave that model's wrapper in place.
        if original is None or getattr(original, "__self__", None) is not model:
            return False
        from types import MethodType, FunctionType
        import accelerate.utils.operations as _ops
        fwd = model.forward
        fn = fwd.__func__ if isinstance(fwd, MethodType) else fwd
        if isinstance(fn, _ops.ConvertOutputsToFp32):
            # prepare's other branch: convert_outputs_to_fp32(autocast_ctx(fwd))
            # bound straight onto the instance.
            pass
        elif (
            type(fn) is FunctionType
            and fn.__code__.co_name == "forward"
            and any(
                c.cell_contents.__class__ is _ops.ConvertOutputsToFp32
                for c in (fn.__closure__ or ())
            )
        ):
            # prepare's MethodType branch: forward(*args) closure around a
            # ConvertOutputsToFp32 (accelerator.py:1825-1836).
            pass
        else:
            return False
        del model.forward
        del model._original_forward
        return True
    return False
