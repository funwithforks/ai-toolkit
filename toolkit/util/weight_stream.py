"""Stream-to-GPU loading primitives: assign tensors from a WeightSource into
a pre-created (meta) module one key at a time.

The shared home of the primitives the LTX-2.3 rail grew (ltx23_stream.py):
pure functions -- no model imports, no globals. Consumers build a plan
(module key -> (source, file key)) and fault tensors through
``WeightSource.get`` one key at a time, so no whole-checkpoint state dict is
ever materialized: peak host ram is the largest tensor in the file, never the
file's total size. Dtype is never touched unless the caller passes an
explicit ``cast_dtype``/``dtype`` -- as-shipped is the default, and the cast
then happens per tensor at stream time (peak: one tensor), never as a
whole-model post-load ``.to()``.

``stream_load_module`` is the single entry for standalone models that are not
in the comfy format (mask/control helpers, HF-layout checkpoints): identity
key plan, loud completeness gate, explicit device required -- no CPU model
construction, no CPU state dict, no silent placement.
"""

import os
from typing import Dict, Optional, Tuple

import torch


def remap_file_key(
    key: str, rename_dict: Dict[str, str], special_keys_remap: Dict[str, object]
) -> Optional[str]:
    """Apply the converters' key renames to one key. The special handlers are
    in-place dict key ops on the batched converters; emulated here on a
    one-key carrier (they never read the values). Returns None when a handler
    drops the key."""
    new_key = key
    for replace_key, rename_key in rename_dict.items():
        new_key = new_key.replace(replace_key, rename_key)
    for special_key, handler in special_keys_remap.items():
        if special_key not in new_key:
            continue
        carrier = {new_key: None}
        handler(new_key, carrier)
        if not carrier:
            return None  # key dropped by the handler
        new_key = next(iter(carrier))
    return new_key


def build_stream_plan(
    source,
    prefix: str,
    rename_dict: Dict[str, str],
    special_keys_remap: Dict[str, object],
    skip_prefixes: Tuple[str, ...] = (),
    keep_prefixes: Tuple[str, ...] = (),
) -> Dict[str, Tuple[object, str]]:
    """Map one WeightSource's keys onto meta-shell module keys.

    ``prefix`` filters the file (e.g. "model.diffusion_model." for the mono
    checkpoint's dit section, "" for a bare-key split file); keys under
    ``skip_prefixes`` (relative to the prefix) are left out, and when
    ``keep_prefixes`` is given only keys under one of them survive -- the
    connector/transformer split of one dit file classifies with the same
    tuple both ways."""
    plan: Dict[str, Tuple[object, str]] = {}
    for file_key in source.keys():
        if prefix and not file_key.startswith(prefix):
            continue
        key = file_key[len(prefix):] if prefix else file_key
        if any(key.startswith(sp) for sp in skip_prefixes):
            continue
        if keep_prefixes and not any(key.startswith(kp) for kp in keep_prefixes):
            continue
        mapped = remap_file_key(key, rename_dict, special_keys_remap)
        if mapped is not None:
            plan[mapped] = (source, file_key)
    return plan


def _probe_load_mem(tag: str, i: int, total: int) -> None:
    """Live rss/vram probe every 200 items (and on demand): the trajectory
    names the culprit if a load ooms mid-stream (per-tensor paths cap at
    largest-tensor ram; a whole-dict path ramps to model size). Gated behind
    AITK_LOAD_DEBUG=1 -- per-block prints flood a normal run's output."""
    if not os.environ.get("AITK_LOAD_DEBUG") == "1":
        return
    if i % 200:
        return
    for line in open("/proc/self/status"):
        if line.startswith("VmRSS"):
            rss = int(line.split()[1]) / 1024
            vram = (
                torch.cuda.memory_allocated() / 1e9
                if torch.cuda.is_available()
                else 0.0
            )
            print(
                f"[load-mem] {tag}: {i}/{total} rss {rss:.0f}MB vram {vram:.1f}GB",
                flush=True,
            )
            break


def _assign_to_owner(module: torch.nn.Module, mod_key: str, tensor) -> None:
    """Assign one tensor to its owning submodule, not the root.

    A root-level load_state_dict walks every child -- including modules a
    previous stream-quantize step already converted, whose custom
    _load_from_state_dict pops its keys unconditionally and raises on a
    partial dict. Deeper keys are delivered to their own parent (that
    subtree is quantization-free at assign time: blocks are fully assigned
    before they are quantized, and the extras live outside the block
    lists); top-level keys are assigned directly for the same reason.

    A delivery that matches nothing at its parent is a plan bug, not a
    skip: the parent-load result is checked and a wrong-key delivery
    raises here, on top of the closing stream_meta_check."""
    parent_path, _, _ = mod_key.rpartition(".")
    if parent_path:
        rel_key = mod_key[len(parent_path) + 1:]
        target = module.get_submodule(parent_path)
        result = target.load_state_dict({rel_key: tensor}, assign=True, strict=False)
        if result.unexpected_keys or rel_key in result.missing_keys:
            raise KeyError(
                f"{mod_key} matched no parameter or buffer of "
                f"{type(target).__name__} (unexpected={result.unexpected_keys})"
            )
        return
    if mod_key in module._parameters:
        old = module._parameters[mod_key]
        module._parameters[mod_key] = torch.nn.Parameter(
            tensor, requires_grad=old.requires_grad
        )
    elif mod_key in module._buffers:
        module._buffers[mod_key] = tensor
    else:
        raise KeyError(
            f"{mod_key} is neither a top-level parameter nor a buffer of "
            f"{type(module).__name__}"
        )


def stream_assign(
    module: torch.nn.Module,
    plan: Dict[str, Tuple[object, str]],
    device,
    name: str,
    cast_dtype: Optional[torch.dtype] = None,
    orig_dtype: Optional[torch.dtype] = None,
) -> int:
    """Assign every planned tensor into the (meta) module, one get() at a
    time, handling comfy-quant marker groups module-at-a-time. Nothing is
    copied or recast unless ``cast_dtype`` is given -- that is the quantize
    path's job, mirroring the batched rail's whole-model ``.to(dtype)`` before
    quantization (the fp32 modulation tables ride that cast there too).
    Returns the number of pre-quantized layers attached. No completeness
    check here: stream in pieces with ``stream_meta_check`` at the end."""
    from toolkit.util.comfy_quant_import import import_comfy_quantized_layers

    orig_dtype = orig_dtype if orig_dtype is not None else torch.bfloat16
    num_quantized = 0
    consumed = set()
    for marker_key in sorted(k for k in plan if k.endswith(".comfy_quant")):
        prefix = marker_key[: -len(".comfy_quant")]
        group = {}
        group_files = []
        for mod_key, (source, file_key) in plan.items():
            if mod_key == prefix or mod_key.startswith(prefix + "."):
                group[mod_key] = source.get(file_key, device)
                group_files.append(file_key)
        remaining, n = import_comfy_quantized_layers(
            module, group, orig_dtype=orig_dtype
        )
        for mod_key, tensor in remaining.items():
            _assign_to_owner(module, mod_key, tensor)
        consumed.update(group_files)
        num_quantized += n
    total = len(plan)
    for mod_key, (source, file_key) in plan.items():
        if file_key in consumed:
            continue
        tensor = source.get(file_key, device)
        if cast_dtype is not None and tensor.dtype != cast_dtype:
            tensor = tensor.to(cast_dtype)
        _assign_to_owner(module, mod_key, tensor)
        _probe_load_mem(name, len(plan), total)
    return num_quantized


def stream_meta_check(module: torch.nn.Module, name: str) -> None:
    """The strict=True stand-in: any param/buffer still on meta after a
    stream means the plan missed a tensor -- fail loudly, never ship a
    silently-half-loaded model.

    Deliberately NOT state_dict(): the save-dequantization patch (if applied)
    turns that into a full-model dequantization materialized in host ram.
    named_parameters/named_buffers walk the live tensors -- params and
    persistent buffers, the same set the batch path's strict load covers."""
    still_meta = [
        n for n, t in module.named_parameters() if t.is_meta
    ] + [n for n, t in module.named_buffers() if t.is_meta]
    if still_meta:
        raise ValueError(
            f"{name} stream load left {len(still_meta)} unfilled params/"
            f"buffers, first few: {still_meta[:8]}"
        )


def plain_stream_load(
    module: torch.nn.Module,
    plan: Dict[str, Tuple[object, str]],
    device,
    name: str,
    orig_dtype: Optional[torch.dtype] = None,
) -> int:
    """The as-shipped rail: assign everything at the precision it ships in
    (quantize: none/false means zero quantization code runs), then prove the
    plan complete. Returns the number of pre-quantized layers attached (the
    caller runs the mixed-file post load when it is non-zero)."""
    num_quantized = stream_assign(module, plan, device, name, orig_dtype=orig_dtype)
    stream_meta_check(module, name)
    return num_quantized


def stream_load_module(
    module: torch.nn.Module,
    file_paths,
    device,
    *,
    dtype: Optional[torch.dtype] = None,
    plan: Optional[Dict[str, Tuple[object, str]]] = None,
    name: str = "model",
) -> int:
    """Stream safetensors file(s) straight onto ``device`` into a pre-created
    (meta) module, one ``get()`` at a time, and prove the module complete.

    The load entry for standalone models that are not in the comfy format
    (mask/control helpers, HF-layout checkpoints): file keys land as-is
    (identity plan) unless the caller passes an explicit plan built with
    ``build_stream_plan``. ``dtype`` is an explicit per-model decision
    applied per tensor at stream time (peak: one tensor); the default is
    as-shipped -- nothing recast. ``device`` is required: there is no CPU
    load path, no CPU state dict, no whole-model post-load cast. Buffers the
    module computes at construction (not in the file) stay on CPU until the
    caller places them; they are small by definition.

    Returns the number of pre-quantized layers attached (aux files carry no
    markers; non-zero belongs to the comfy rails)."""
    if device is None:
        raise ValueError(
            f"{name}: stream load requires an explicit target device "
            "(there is no CPU load path)."
        )
    from toolkit.util.weight_source import WeightSource

    if isinstance(file_paths, (str, os.PathLike)):
        file_paths = [file_paths]
    sources = [WeightSource.open(str(p)) for p in file_paths]
    if plan is None:
        plan = {}
        for source in sources:
            for file_key in source.keys():
                if file_key in plan:
                    raise ValueError(
                        f"{name}: key {file_key} appears in more than one "
                        "stream file; pass an explicit plan."
                    )
                plan[file_key] = (source, file_key)
    num_quantized = stream_assign(module, plan, device, name, cast_dtype=dtype)
    stream_meta_check(module, name)
    return num_quantized
