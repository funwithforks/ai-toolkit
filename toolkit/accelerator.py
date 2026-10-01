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
