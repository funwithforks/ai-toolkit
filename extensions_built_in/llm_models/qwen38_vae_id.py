"""qwen38 vae_id: the vision tower behind one id.

For this arch the "vae" role is the frozen vision tower: its embeddings are
what the latent caches store (packed in a DTO with the processor's expanded
media span and the MRoPE grid), so the tower plays the vae end-to-end --
phase 1 loads only it, the iron rule frees it via ``sd.vae = None``, and
phase 3 builds the language stack without it. The wrapper below is the
``sd.vae`` object and the tower's SOLE owner: if the tower also lived inside
``self.model``, that drop would free nothing.

The load streams only the checkpoint's ``model.visual.*`` keys, per tensor,
straight onto the vae device (never the whole-file dict -- that path peaks
at ~2x the 17 GB checkpoint). Quantized tower linears attach through the
comfy-quant importer on the prefix-stripped subset; shipped dtypes are kept,
and by owner decision the tower stays int8 convrot by default
(``vision_bf16`` dequants explicitly at the holder's call site).

Cache identity is the ``vae_id`` itself: files hash the id, so the DTO
caches are keyed on ``qwen38`` regardless of which checkpoint file served
the tensors.
"""

import torch

from toolkit.models.vae_registry import VaeEntry, register_vae_entry


class Qwen38VisionTower(torch.nn.Module):
    """The vae object: the bare tower plus the embedding call. A plain module
    holding the tower as a submodule -- placement (.to), .device and the
    phase-boundary free all follow the real parameters, no mirror needed."""

    def __init__(self, visual: torch.nn.Module):
        super().__init__()
        self.visual = visual

    @property
    def device(self) -> torch.device:
        return self.visual.patch_embed.proj.weight.device

    @property
    def dtype(self) -> torch.dtype:
        return self.visual.patch_embed.proj.weight.dtype

    def get_image_features(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        """Merged vision embeddings [N_merged, hidden] for one or more images / videos.

        The compute dtype is the tower's own weight dtype; the caller's pixel
        values are moved to it and nothing else is cast. No grad policy here:
        the caller decides -- the cache pass runs under no_grad, and a future
        tower-LoRA train step backprops through this."""
        visual = self.visual
        out = visual(pixel_values.to(visual.patch_embed.proj.weight.dtype), grid_thw=grid_thw, return_dict=True)
        return out.pooler_output


def _load_qwen38_vision(model):
    """-> Qwen38VisionTower resident on model.vae_device_torch. Reads the same
    checkpoint the transformer phase resolves (one door, one file) and the
    config the holder stashed at ``model._config`` (loading it here too when
    the holder has not run yet -- the header read is idempotent)."""
    from transformers import AutoConfig

    from .src.qwen38_load import dequantize_vision_tower, load_vision_tower

    config = getattr(model, "_config", None)
    if config is None:
        config = AutoConfig.from_pretrained(model._config_source())
        model._config = config
    if not hasattr(config, "vision_config"):
        raise RuntimeError("qwen38 checkpoint config carries no vision_config")

    model.print_and_status_update("Loading qwen38 vision tower")
    tower = load_vision_tower(
        model._checkpoint_path(),
        config,
        model.vae_device_torch,
        orig_dtype=model.torch_dtype,
        status_fn=model.print_and_status_update,
    )
    if model.vision_bf16:
        n = dequantize_vision_tower(tower, model.torch_dtype)
        print_acc(f"Vision tower: {n} int8 linears dequantized to {model.torch_dtype} (model_kwargs.vision_bf16)")
    return Qwen38VisionTower(tower)


register_vae_entry(
    VaeEntry(
        vae_id="qwen38",
        components={"vision": Qwen38VisionTower},
        load=_load_qwen38_vision,
    )
)
