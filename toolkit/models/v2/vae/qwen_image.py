"""The Qwen-Image (wan-style video) VAE: one physical encoder, one id.

This module IS the ``qwen_image`` vae_id contract: the component class
(candidate metadata, key layout, loader), the encode/decode primitives, and
the registry registration. Models join it by declaring ``vae_id =
"qwen_image"`` (krea2 today); a future model on this encoder adds nothing
here. Alternate encode paths for THIS vae would live here as this vae's
implementation -- they never fork the id.

Loading is vendored here on purpose (the same call the minimax_h3 vae
loaders make): the module shell is built with meta parameters, the
checkpoint is streamed straight onto the target device one tensor at a time
(no process-side copy of the model), and the tensors are assigned as they
land. There is no ``dtype`` argument anywhere in this path -- a vae is
loaded at the precision it ships in, and a blanket cast to some other dtype
is the loss-of-quality pattern this fork does not keep. Upstream's
``convert_state_dict_on_load`` / ``aitk_cast_on_load`` machinery is bypassed
entirely rather than opted out of.

File format is not identity: the comfy single-file repack and the diffusers
folder hold the same weights tensor-for-tensor, so reading the comfy file is
a pure rename (verified: 194/194 keys, no shape or bitwise differences). The
repack uses the wan-native module naming (root ``conv1``/``conv2``,
``downsamples``/``upsamples``, ``head``, ``middle``) while this class's tree
is the original Qwen-Image one (``quant_conv``, flat ``encoder.down_blocks``
per stage, ``decoder.up_blocks`` holding resnet lists plus ``upsamplers``).
Convs stay 5-dim: unlike the Qwen-Image-2.1 vae -- a differently trained,
different-shaped vae with its own id and its own module -- this checkpoint's
dimensionality already matches the module tree.
"""

import time

import torch
from diffusers import AutoencoderKLQwenImage

from ...vae_registry import VaeEntry, register_vae_entry
from ....util.weight_source import WeightSource
from .._mixin import OstrisModelMixin
from .comfy_wan_layout import _COMFY_MID_PARTS, _comfy_resnet_suffix

# Architecture and latent normalization of this physical vae, vendored so a
# load needs nothing but the checkpoint file (no config folder, no hub
# round-trip). Values are Qwen/Qwen-Image ``vae/config.json``; the model
# cannot be built without them and they are what makes the latents the
# distribution this vae's models were trained against.
QWEN_IMAGE_VAE_CONFIG = {
    "base_dim": 96,
    "dim_mult": [1, 2, 4, 4],
    "num_res_blocks": 2,
    "attn_scales": [],
    "temperal_downsample": [False, True, True],
    "dropout": 0.0,
    "z_dim": 16,
    "latents_mean": [
        -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
        0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921,
    ],
    "latents_std": [
        2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
        3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.916,
    ],
}


class QwenImageVAE(AutoencoderKLQwenImage, OstrisModelMixin):
    """The Qwen-Image (wan-style video) VAE, shared by the qwen_image family,
    nucleus_image and krea2. Registered as the ``qwen_image`` vae_id entry at
    the bottom of this module.

    Use :meth:`load_to_device` (the entry's loader): it reads a comfy
    single-file checkpoint onto the target device at shipped precision.
    """

    aitk_subfolder = "vae"
    aitk_comfy_repo = "Comfy-Org/Qwen-Image"
    # one physical file behind every name this vae ships under
    aitk_comfy_weight_names = {
        "Comfy-Org/Qwen-Image": ["vae/qwen_image_vae.safetensors"],
        "Qwen/Qwen-Image": ["vae/qwen_image_vae.safetensors"],
        "Comfy-Org/Krea-2": ["vae/qwen_image_vae.safetensors"],
        "krea/Krea-2-Raw": ["vae/qwen_image_vae.safetensors"],
        "krea/Krea-2-Turbo": ["vae/qwen_image_vae.safetensors"],
    }

    @classmethod
    def load_to_device(cls, file_path: str, device) -> "QwenImageVAE":
        """Stream a comfy single-file checkpoint onto ``device``.

        Meta-parameter shell (``include_buffers=False`` keeps computed buffers
        real), tensors materialized on the device one at a time so peak ram is
        the largest tensor and never the model, then assigned in place.
        ``strict=True`` makes a key-layout mistake die at load instead of
        quietly leaving part of the network at its initialization. Nothing
        here takes a dtype: the vae keeps the precision it ships in.
        """
        from accelerate import init_empty_weights

        t0 = time.perf_counter()
        dev = torch.device(device)
        if dev.type == "cuda" and dev.index is None:
            dev = torch.device("cuda", torch.cuda.current_device())
        with init_empty_weights(include_buffers=False):
            model = cls(**QWEN_IMAGE_VAE_CONFIG)
        state = convert_comfy_qwen_image_state_dict(
            WeightSource.open(file_path).materialize(dev)
        )
        model.load_state_dict(state, strict=True, assign=True)
        del state
        # the meta shell leaves non-persistent buffers wherever they were
        # built; parameters are already resident so this only relocates those
        model.to(dev)
        model.eval().requires_grad_(False)
        print(
            f"[load-timing] QwenImageVAE: streamed to {dev} "
            f"{time.perf_counter() - t0:.1f}s"
        )
        return model

    @classmethod
    def aitk_config_from_state_dict(cls, state_dict):
        # the checkpoint carries weights only; the architecture and the
        # latent normalization are properties of this vae, declared above
        return dict(QWEN_IMAGE_VAE_CONFIG)

    @classmethod
    def convert_state_dict_on_load(cls, state_dict):
        # only reached by the generic loader (holders not on the registry);
        # the entry's loader maps the keys itself
        return convert_comfy_qwen_image_state_dict(state_dict)


# ----------------------------------------------------------------------
# comfy repack -> this class's tree (pure rename, verified tensor-for-tensor)
# ----------------------------------------------------------------------


def _decoder_stage_plan(state_dict) -> dict:
    """Destination path for each flat ``decoder.upsamples.<i>`` stage.

    The comfy repack lists decoder stages flat; this class groups them into
    blocks that hold a resnet list plus an optional upsampler. Walking the
    stages in order recovers the grouping: a stage carrying a
    ``resample``/``time_conv`` payload is the current block's upsampler and
    closes that block.
    """
    stages = {}
    for key in state_dict:
        parts = key.split(".")
        if len(parts) > 3 and parts[0] == "decoder" and parts[1] == "upsamples":
            stages.setdefault(int(parts[2]), set()).add(parts[3])
    plan, block, resnet = {}, 0, 0
    for index in sorted(stages):
        if stages[index] & {"resample", "time_conv"}:
            plan[index] = ["up_blocks", str(block), "upsamplers", "0"]
            block, resnet = block + 1, 0
        else:
            plan[index] = ["up_blocks", str(block), "resnets", str(resnet)]
            resnet += 1
    return plan


def _qwen_image_comfy_key(key: str, plan: dict) -> str:
    parts = key.split(".")
    # the quant convs sit at the checkpoint root
    if parts[0] == "conv1":
        return ".".join(["quant_conv"] + parts[1:])
    if parts[0] == "conv2":
        return ".".join(["post_quant_conv"] + parts[1:])
    side, rest = parts[0], parts[1:]
    if rest[0] == "conv1":
        return ".".join([side, "conv_in"] + rest[1:])
    if rest[0] == "head":
        # head.0 is the output norm, head.2 the output conv (head.1 is SiLU)
        tail = "norm_out" if rest[1] == "0" else "conv_out"
        return ".".join([side, tail] + rest[2:])
    if rest[0] == "middle":
        part, inner = _COMFY_MID_PARTS[rest[1]], rest[2:]
        if part.startswith("resnets"):
            inner = _comfy_resnet_suffix(inner)
        return ".".join([side, "mid_block", part] + inner)
    if rest[0] == "downsamples":
        # encoder: one resnet per stage, held flat on the block
        stage, inner = rest[1], rest[2:]
        if inner[0] in ("resample", "time_conv"):
            return ".".join([side, "down_blocks", stage] + inner)
        return ".".join([side, "down_blocks", stage] + _comfy_resnet_suffix(inner))
    if rest[0] == "upsamples":
        stage, inner = int(rest[1]), rest[2:]
        dest = plan[stage]
        if inner[0] in ("resample", "time_conv"):
            return ".".join([side] + dest + inner)
        return ".".join([side] + dest + _comfy_resnet_suffix(inner))
    raise ValueError(f"unmapped key in comfy Qwen-Image vae checkpoint: {key}")


def convert_comfy_qwen_image_state_dict(state_dict: dict) -> dict:
    """Comfy wan-layout Qwen-Image vae state dict -> this class's tree. A
    state dict already in this layout passes through untouched (no root
    ``conv1``). Shapes already agree, so nothing is reshaped."""
    if "conv1.weight" not in state_dict:
        return state_dict
    plan = _decoder_stage_plan(state_dict)
    return {
        _qwen_image_comfy_key(key, plan): value
        for key, value in state_dict.items()
    }


# ----------------------------------------------------------------------
# encode / decode primitives (owned by the vae, shared with holder models)
# ----------------------------------------------------------------------


def encode_images(model, image_list, device=None, dtype=None):
    """It is a video VAE: images ride in a single-frame dim, and latents
    normalize with the config's latents_mean/std."""
    if device is None:
        device = model.vae_device_torch
    if dtype is None:
        dtype = model.vae_torch_dtype
    vae = model.vae

    if vae.device == torch.device("cpu"):
        vae.to(device)
    vae.eval()
    vae.requires_grad_(False)

    image_list = [image.to(device, dtype=dtype) for image in image_list]
    images = torch.stack(image_list).to(device, dtype=dtype)
    images = images.unsqueeze(2)  # add the frame dim
    latents = vae.encode(images).latent_dist.sample()

    latents_mean = (
        torch.tensor(vae.config.latents_mean)
        .view(1, vae.config.z_dim, 1, 1, 1)
        .to(latents.device, latents.dtype)
    )
    latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(
        1, vae.config.z_dim, 1, 1, 1
    ).to(latents.device, latents.dtype)

    latents = (latents - latents_mean) * latents_std
    latents = latents.squeeze(2)  # drop the frame dim
    return latents.to(device, dtype=dtype)


def decode_latents(model, latents, device=None, dtype=None):
    if device is None:
        device = model.vae_device_torch
    if dtype is None:
        dtype = model.vae_torch_dtype
    vae = model.vae

    if vae.device == torch.device("cpu"):
        vae.to(device)

    latents = latents.to(device, dtype=dtype)
    latents = latents.unsqueeze(2)  # add the frame dim

    latents_mean = (
        torch.tensor(vae.config.latents_mean)
        .view(1, vae.config.z_dim, 1, 1, 1)
        .to(latents.device, latents.dtype)
    )
    latents_std = (
        torch.tensor(vae.config.latents_std)
        .view(1, vae.config.z_dim, 1, 1, 1)
        .to(latents.device, latents.dtype)
    )
    latents = latents * latents_std + latents_mean

    # full-resolution decode spikes VRAM; models opt in to tiling it
    tiled = getattr(model, "vae_decode_tiled_on_low_vram", False) and (
        model.model_config.low_vram
    )
    if tiled:
        vae.enable_tiling()
    try:
        images = vae.decode(latents).sample
    finally:
        if tiled:
            vae.disable_tiling()

    images = images.squeeze(2)  # drop the frame dim
    return images.to(device, dtype=dtype)


def _load_qwen_image_vae(model):
    """Resolve the comfy single file through the door (the generic
    ``vae_path`` model_kwargs override short-circuits it) and stream it onto
    the vae device at the precision it ships in."""
    path = model._resolve_comfy_file("vae", QwenImageVAE)
    model.print_and_status_update(f"Loading Qwen-Image VAE from {path}")
    return QwenImageVAE.load_to_device(path, model.vae_device_torch)


register_vae_entry(
    VaeEntry(
        vae_id="qwen_image",
        components={"vae": QwenImageVAE},
        load=_load_qwen_image_vae,
        encode_images=encode_images,
        decode_latents=decode_latents,
    )
)


class QwenImageVAEHolderMixin:
    """BaseModel-side encode_images/decode_latents for models whose self.vae
    is the Qwen-Image VAE but that are NOT yet on the vae_id registry
    (qwen_image, nucleus_image). The bodies live in this module's entry --
    one source of truth; migrating a model onto the id retires its use of
    this holder."""

    # tile the decode when low_vram (decode only; encode stays untiled)
    vae_decode_tiled_on_low_vram = False

    def encode_images(self, image_list, device=None, dtype=None):
        return encode_images(self, image_list, device=device, dtype=dtype)

    def decode_latents(self, latents, device=None, dtype=None):
        return decode_latents(self, latents, device=device, dtype=dtype)
