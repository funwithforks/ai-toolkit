"""LTX-2.3 streamed loading: model-specific key logic on top of the shared
stream primitives.

The general primitives (``build_stream_plan`` / ``stream_assign`` /
``stream_meta_check`` / ``plain_stream_load`` and their internals) live in
``toolkit.util.weight_stream`` -- the shared home for every stream-to-GPU
consumer. What stays here is LTX-2.3's own key semantics: the 2.3 loaders
(ltx2.py) and the ltx2 vae_id entry (ltx23_vae_id.py) build plans with these
and fault tensors through a WeightSource one key at a time, so no
whole-checkpoint state dict is ever materialized -- the 46 GB mono file loads
at a peak of one tensor (as-shipped) or one block (stream-quantized).

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
from toolkit.util.weight_stream import stream_assign, stream_meta_check


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
