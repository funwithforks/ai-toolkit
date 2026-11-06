"""LTX-2.3 streamed loading helpers.

Pure functions: no model imports, no globals. The 2.3 loaders (ltx2.py) and
the ltx2 vae_id entry (ltx23_vae_id.py) both build plans with these and fault
tensors through a WeightSource one key at a time, so no whole-checkpoint
state dict is ever materialized -- the 46 GB mono file loads at a peak of one
tensor (as-shipped) or one block (stream-quantized).

Semantics come from the upstream 2.3 load path
(extensions_built_in/diffusion_models/ltx2/convert_ltx2_to_diffusers.py): the
rename tables and special handlers there are pure key operations (none reads
a tensor value), so the same tables that drive the batched converters drive a
per-key plan. The closing meta scan plays the batch path's strict=True role.
"""

import os
from typing import Dict, Optional, Tuple

import torch

from toolkit.util.quantize import (
    _quantize_extras_pass,
    _quantize_stream_block,
    get_qtype,
    group_stream_plan_by_blocks,
)


def gemma3_stream_key(key: str) -> Optional[str]:
    """The comfy gemma3 -> transformers key remap as a pure key function --
    convert_comfy_gemma3_to_transformers minus the value-touching dequantize
    pass (that one belongs to the batched route, which refuses nothing).
    Returns None for dropped keys. Mirrors the converter rule for rule:
    vision tower and projector remap into the model, language layers shift
    under model.language_model, spiece_model drops, everything else passes
    through unchanged. Hub repo shards carry the older
    ``language_model.model.*`` layout -- remapped to the same target; the
    modern ``model.language_model.*`` form passes through untouched."""
    nk = key
    if nk.startswith("module."):
        nk = nk[len("module."):]
    if nk == "spiece_model":
        return None
    # pieces the conditioning stack never uses: the vision tower and mm
    # projector (dropped by the legacy rail after loading), the connector
    # projections (a separate component with its own file), and embedded
    # tokenizer assets
    if nk.startswith((
        "vision_model.",
        "vision_tower.",
        "multi_modal_projector.",
        "text_embedding_projection.",
        "hf_asset__",
        "tokenizer_json",
    )):
        return None
    if nk.startswith("language_model.model."):
        return "model.language_model." + nk[len("language_model.model."):]
    if nk == "model.embed_tokens.weight":
        return "model.language_model.embed_tokens.weight"
    if nk.startswith("model.layers."):
        return "model.language_model.layers." + nk[len("model.layers."):]
    if nk.startswith("model.norm."):
        return "model.language_model.norm." + nk[len("model.norm."):]
    return nk


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
    lists); top-level keys are assigned directly for the same reason."""
    parent_path, _, _ = mod_key.rpartition(".")
    if parent_path:
        rel_key = mod_key[len(parent_path) + 1:]
        target = module.get_submodule(parent_path)
        target.load_state_dict({rel_key: tensor}, assign=True, strict=False)
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


def quantize_streamed_plan(
    module: torch.nn.Module,
    plan: Dict[str, Tuple[object, str]],
    block_names,
    *,
    name: str,
    qtype: str,
    quantize_device,
    dtype: torch.dtype,
    exclude=None,
    quantize_kwargs: Optional[dict] = None,
    keep_on_device: bool,
    status_fn=print,
) -> None:
    """Stream + quantize-as-you-go: each repeated block's tensors are assigned
    (cast to ``dtype``, mirroring the batched rail's whole-model cast before
    quantization) and quantized with the IDENTICAL per-block body
    quantize_module uses, before the next block streams. Peak memory is one
    block, not the model. The leftovers then run the identical extras pass.

    The caller is responsible for marking the module
    (``module.aitk_stream_quantized = qtype``) so a later aitk_post_load
    skips the now-redundant block pass."""
    from toolkit.dequantize import patch_dequantization_on_save

    quantization_type = get_qtype(qtype)
    exclude = list(exclude or [])
    quantize_kwargs = quantize_kwargs or {}

    groups, extras = group_stream_plan_by_blocks(plan, block_names)
    if groups:
        status_fn(
            f" - stream-quantizing {len(groups)} blocks ({qtype}) as they load"
        )
    skipped = 0
    for block_path, sub_plan in groups:
        # assign straight onto the quantize device at shipped precision --
        # the shared block body's .to(device, dtype) performs the job-dtype
        # cast at move time, exactly where the batched rail's cast lives
        stream_assign(
            module, sub_plan, quantize_device, f"{name}:{block_path}",
            orig_dtype=dtype,
        )
        block = module.get_submodule(block_path)
        was_skipped, _seg = _quantize_stream_block(
            block,
            quantization_type,
            device=quantize_device,
            dtype=dtype,
            exclude=exclude,
            quantize_kwargs=quantize_kwargs,
            keep_on_device=keep_on_device,
        )
        if torch.cuda.is_available() and os.environ.get("AITK_LOAD_DEBUG") == "1":
            print(
                f"[load-mem] {name}:{block_path} quantized; "
                f"vram {torch.cuda.memory_allocated()/1e9:.1f}GB",
                flush=True,
            )
        if was_skipped:
            skipped += 1
    if skipped:
        status_fn(
            f" - {skipped} blocks already quantized with a matching qtype; left untouched"
        )
    # leftovers: assign at the job dtype (the batched rail's whole-model cast
    # covers these too), then the identical extras pass
    stream_assign(
        module, extras, "cpu" if not keep_on_device else quantize_device,
        f"{name}:extras", cast_dtype=dtype, orig_dtype=dtype,
    )
    status_fn(" - quantizing extras")
    _quantize_extras_pass(
        module,
        quantization_type,
        device=quantize_device,
        exclude=exclude,
        quantize_kwargs=quantize_kwargs,
        keep_on_device=keep_on_device,
    )
    # completeness gate FIRST: after the save-dequantization patch below, a
    # state_dict() call would dequantize the whole model into host ram (the
    # gate itself now avoids state_dict entirely -- this ordering keeps the
    # two concerns from ever meeting)
    stream_meta_check(module, name)
    patch_dequantization_on_save(module)


def pack_audio_latents(latents: torch.Tensor) -> torch.Tensor:
    """The pipeline's default audio packing: [B, C, L, M] -> [B, L, C * M].
    (The patch-size branch is never taken on the 2.3 path -- encode_audio
    calls _pack_audio_latents with no patch args.) Bitwise-identical pure
    reshape, replicated so the vae bundle does not need a pipeline."""
    return latents.transpose(1, 2).flatten(2, 3)
