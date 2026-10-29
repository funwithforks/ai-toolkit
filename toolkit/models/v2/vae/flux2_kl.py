"""The BFL-style Flux2 KL autoencoder (32ch latents, 2x2 pixel-shuffle
packing to 128ch), shared by the flux2 family and ideogram4.

This module is also the ``flux2`` vae_id contract (registration at the
bottom): the comfy file resolution, the vendored no-dtype loader, and the
class's own ``encode_images`` / ``decode_latents``. Models join by declaring
``vae_id = "flux2"`` and carry no vae code of their own.

This vae loads the raw BFL ae.safetensors layout and uses encode/decode (with
the BatchNorm running-stats latent normalization); ideogram4 loads
diffusers-format checkpoints via convert_diffusers_state_dict and drives
encoder/decoder directly with its own patchify + latent-norm tables. The two
models pack the 128 latent channels in different orders — the
packing/normalization stays per-model.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import torch
import torch.utils.checkpoint as ckpt
from einops import rearrange
from torch import Tensor, nn

from ...vae_registry import VaeEntry, register_vae_entry
from .._mixin import OstrisModelMixin


@dataclass
class AutoEncoderParams:
    resolution: int = 256
    in_channels: int = 3
    ch: int = 128
    out_ch: int = 3
    ch_mult: list[int] = field(default_factory=lambda: [1, 2, 4, 4])
    num_res_blocks: int = 2
    z_channels: int = 32

@dataclass
class AutoEncoderSmallDecoderParams:
    resolution: int = 256
    in_channels: int = 3
    ch: int = 128
    ch_encoder: int = 96
    out_ch: int = 3
    ch_mult: list[int] = field(default_factory=lambda: [1, 2, 4, 4])
    num_res_blocks: int = 2
    z_channels: int = 32


def swish(x: Tensor) -> Tensor:
    return x * torch.sigmoid(x)


class AttnBlock(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels

        self.norm = nn.GroupNorm(
            num_groups=32, num_channels=in_channels, eps=1e-6, affine=True
        )

        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def attention(self, h_: Tensor) -> Tensor:
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        b, c, h, w = q.shape
        q = rearrange(q, "b c h w -> b 1 (h w) c").contiguous()
        k = rearrange(k, "b c h w -> b 1 (h w) c").contiguous()
        v = rearrange(v, "b c h w -> b 1 (h w) c").contiguous()
        h_ = nn.functional.scaled_dot_product_attention(q, k, v)

        return rearrange(h_, "b 1 (h w) c -> b c h w", h=h, w=w, c=c, b=b)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.proj_out(self.attention(x))


class ResnetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels

        self.norm1 = nn.GroupNorm(
            num_groups=32, num_channels=in_channels, eps=1e-6, affine=True
        )
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.norm2 = nn.GroupNorm(
            num_groups=32, num_channels=out_channels, eps=1e-6, affine=True
        )
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        if self.in_channels != self.out_channels:
            self.nin_shortcut = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=1, padding=0
            )

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = swish(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = swish(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)

        return x + h


class Downsample(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        # no asymmetric padding in torch conv, must do it ourselves
        self.conv = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=2, padding=0
        )

    def forward(self, x: Tensor):
        pad = (0, 1, 0, 1)
        x = nn.functional.pad(x, pad, mode="constant", value=0)
        x = self.conv(x)
        return x


class Upsample(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=1, padding=1
        )

    def forward(self, x: Tensor):
        x = nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        x = self.conv(x)
        return x


class Encoder(nn.Module):
    def __init__(
        self,
        resolution: int,
        in_channels: int,
        ch: int,
        ch_mult: list[int],
        num_res_blocks: int,
        z_channels: int,
    ):
        super().__init__()
        self.quant_conv = torch.nn.Conv2d(2 * z_channels, 2 * z_channels, 1)
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        # downsampling
        self.conv_in = nn.Conv2d(
            in_channels, self.ch, kernel_size=3, stride=1, padding=1
        )

        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        block_in = self.ch
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out))
                block_in = block_out
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in)

        # end
        self.norm_out = nn.GroupNorm(
            num_groups=32, num_channels=block_in, eps=1e-6, affine=True
        )
        self.conv_out = nn.Conv2d(
            block_in, 2 * z_channels, kernel_size=3, stride=1, padding=1
        )
        self.gradient_checkpointing = False

    def enable_gradient_checkpointing(self):
        self.gradient_checkpointing = True

    def forward(self, x: Tensor) -> Tensor:
        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                if torch.is_grad_enabled() and self.gradient_checkpointing:
                    h = ckpt.checkpoint(self.down[i_level].block[i_block], hs[-1])
                    if len(self.down[i_level].attn) > 0:
                        h = ckpt.checkpoint(self.down[i_level].attn[i_block], h)
                else:
                    h = self.down[i_level].block[i_block](hs[-1])
                    if len(self.down[i_level].attn) > 0:
                        h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                if torch.is_grad_enabled() and self.gradient_checkpointing:
                    hs.append(ckpt.checkpoint(self.down[i_level].downsample, hs[-1]))
                else:
                    hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            h = ckpt.checkpoint(self.mid.block_1, h)
            h = ckpt.checkpoint(self.mid.attn_1, h)
            h = ckpt.checkpoint(self.mid.block_2, h)
        else:
            h = self.mid.block_1(h)
            h = self.mid.attn_1(h)
            h = self.mid.block_2(h)
        # end
        h = self.norm_out(h)
        h = swish(h)
        h = self.conv_out(h)
        h = self.quant_conv(h)
        return h


class Decoder(nn.Module):
    def __init__(
        self,
        ch: int,
        out_ch: int,
        ch_mult: list[int],
        num_res_blocks: int,
        in_channels: int,
        resolution: int,
        z_channels: int,
    ):
        super().__init__()
        self.post_quant_conv = torch.nn.Conv2d(z_channels, z_channels, 1)
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.ffactor = 2 ** (self.num_resolutions - 1)

        # compute in_ch_mult, block_in and curr_res at lowest res
        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, z_channels, curr_res, curr_res)

        # z to block_in
        self.conv_in = nn.Conv2d(
            z_channels, block_in, kernel_size=3, stride=1, padding=1
        )

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks + 1):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out))
                block_in = block_out
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = nn.GroupNorm(
            num_groups=32, num_channels=block_in, eps=1e-6, affine=True
        )
        self.conv_out = nn.Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1)
        self.gradient_checkpointing = False

    def enable_gradient_checkpointing(self):
        self.gradient_checkpointing = True

    def forward(self, z: Tensor) -> Tensor:
        z = self.post_quant_conv(z)

        # get dtype for proper tracing
        upscale_dtype = next(self.up.parameters()).dtype

        # z to block_in
        h = self.conv_in(z)

        # middle
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            h = ckpt.checkpoint(self.mid.block_1, h)
            h = ckpt.checkpoint(self.mid.attn_1, h)
            h = ckpt.checkpoint(self.mid.block_2, h)
        else:
            h = self.mid.block_1(h)
            h = self.mid.attn_1(h)
            h = self.mid.block_2(h)

        # cast to proper dtype
        h = h.to(upscale_dtype)
        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                if torch.is_grad_enabled() and self.gradient_checkpointing:
                    h = ckpt.checkpoint(self.up[i_level].block[i_block], h)
                    if len(self.up[i_level].attn) > 0:
                        h = ckpt.checkpoint(self.up[i_level].attn[i_block], h)
                else:
                    h = self.up[i_level].block[i_block](h)
                    if len(self.up[i_level].attn) > 0:
                        h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                if torch.is_grad_enabled() and self.gradient_checkpointing:
                    h = ckpt.checkpoint(self.up[i_level].upsample, h)
                else:
                    h = self.up[i_level].upsample(h)

        # end
        h = self.norm_out(h)
        h = swish(h)
        h = self.conv_out(h)
        return h


class AutoEncoder(nn.Module, OstrisModelMixin):
    # ---- comfy weight sources (the flux2 vae_id's file resolution) ----
    # One physical file, one download source. The comfy repos of the models
    # that use this vae publish it byte-identical (sha256 868fe7b3...), so a
    # non-gated one is the default source and the job's own repo never has to
    # carry it. Chosen over the flux3 package repo's copy of the same name
    # because that one is not byte-identical to the file we train on (and is
    # gated); FLUX.2's own BFL/Comfy-Org repos are gated too.
    aitk_subfolder = "vae"
    aitk_comfy_repo = "Comfy-Org/MOSS-Image"
    # every standard name_or_path of a model on this vae maps to the same
    # properly-named file; a job naming any of them fetches one shared vae
    aitk_comfy_weight_names = {
        "black-forest-labs/flux-3-image-klein-comfy": ["vae/flux2-vae.safetensors"],
        "Comfy-Org/Fibo-2.0": ["vae/flux2-vae.safetensors"],
        "Comfy-Org/MOSS-Image": ["vae/flux2-vae.safetensors"],
        "OpenMOSS-Team/MOSS-Image-Base": ["vae/flux2-vae.safetensors"],
    }

    @classmethod
    def aitk_config_from_state_dict(cls, state_dict):
        # small decoder builds report 96ch at the first decoder block
        if state_dict["decoder.up.0.block.0.conv1.bias"].shape[0] == 96:
            return AutoEncoderSmallDecoderParams()
        return AutoEncoderParams()

    def __init__(self, params: AutoEncoderParams):
        super().__init__()
        self.params = params
        self.encoder = Encoder(
            resolution=params.resolution,
            in_channels=params.in_channels,
            ch=params.ch,
            ch_mult=params.ch_mult,
            num_res_blocks=params.num_res_blocks,
            z_channels=params.z_channels,
        )
        decoder_ch = params.ch
        if hasattr(params, "ch_encoder"):
            decoder_ch = params.ch_encoder
        self.decoder = Decoder(
            resolution=params.resolution,
            in_channels=params.in_channels,
            ch=decoder_ch,
            out_ch=params.out_ch,
            ch_mult=params.ch_mult,
            num_res_blocks=params.num_res_blocks,
            z_channels=params.z_channels,
        )

        self.bn_eps = 1e-4
        self.bn_momentum = 0.1
        self.ps = [2, 2]
        self.bn = torch.nn.BatchNorm2d(
            math.prod(self.ps) * params.z_channels,
            eps=self.bn_eps,
            momentum=self.bn_momentum,
            affine=False,
            track_running_stats=True,
        )
        self._gradient_checkpointing = False

    @property
    def gradient_checkpointing(self):
        return self._gradient_checkpointing

    @gradient_checkpointing.setter
    def gradient_checkpointing(self, value: bool):
        self._gradient_checkpointing = value
        self.encoder.gradient_checkpointing = value
        self.decoder.gradient_checkpointing = value

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def enable_gradient_checkpointing(self):
        self.gradient_checkpointing = True
        self.encoder.enable_gradient_checkpointing()
        self.decoder.enable_gradient_checkpointing()

    @classmethod
    def load_to_device(cls, file_path: str, device) -> "AutoEncoder":
        """Stream a comfy-layout checkpoint onto ``device`` at the precision
        it ships in (the file's own mix, fp32 for the shipped flux2 vae).

        Vendored on the minimax_h3 / qwen_image pattern: the shell is built
        with meta parameters, tensors materialize on the device one at a time
        (peak ram is the largest tensor, never the model), and the load is
        ``strict=True`` so a wrong file dies here instead of leaving part of
        the network at its initialization. There is no dtype argument: the
        upstream "cast the vae to the job's dtype" step is what rounds the
        latents, so this path cannot ask for it.
        """
        import time
        from accelerate import init_empty_weights
        from toolkit.util.weight_source import WeightSource

        t0 = time.perf_counter()
        dev = torch.device(device)
        if dev.type == "cuda" and dev.index is None:
            dev = torch.device("cuda", torch.cuda.current_device())
        with init_empty_weights(include_buffers=False):
            model = cls(cls._params_from_checkpoint(file_path))
        state = WeightSource.open(file_path).materialize(dev)
        model.load_state_dict(state, strict=True, assign=True)
        del state
        # the meta shell leaves non-persistent buffers at their init device;
        # parameters are already resident so this only relocates those
        model.to(dev)
        model.eval().requires_grad_(False)
        print(
            f"[load-timing] flux2 AutoEncoder: streamed to {dev} "
            f"{time.perf_counter() - t0:.1f}s"
        )
        return model

    @staticmethod
    def _params_from_checkpoint(file_path: str):
        """The file decides the architecture, from its header alone (no weight
        read): small-decoder builds report 96ch at the first decoder block."""
        from safetensors import safe_open

        with safe_open(file_path, framework="pt") as f:
            width = f.get_slice("decoder.up.0.block.0.conv1.bias").get_shape()[0]
        return AutoEncoderSmallDecoderParams() if width == 96 else AutoEncoderParams()

    def normalize(self, z):
        self.bn.eval()
        return self.bn(z)

    def inv_normalize(self, z):
        self.bn.eval()
        s = torch.sqrt(self.bn.running_var.view(1, -1, 1, 1) + self.bn_eps)
        m = self.bn.running_mean.view(1, -1, 1, 1)
        return z * s + m

    def encode(self, x: Tensor) -> Tensor:
        moments = self.encoder(x)
        mean = torch.chunk(moments, 2, dim=1)[0]

        z = rearrange(
            mean,
            "... c (i pi) (j pj)  -> ... (c pi pj) i j",
            pi=self.ps[0],
            pj=self.ps[1],
        )
        z = self.normalize(z)
        return z

    def decode(self, z: Tensor) -> Tensor:
        z = self.inv_normalize(z)
        z = rearrange(
            z,
            "... (c pi pj) i j -> ... c (i pi) (j pj)",
            pi=self.ps[0],
            pj=self.ps[1],
        )
        dec = self.decoder(z)
        return dec

    # -- the vae does the vae work ------------------------------------------------
    # These are what phase 1 caches and what sampling decodes, so they belong to
    # the vae rather than to any model: device and precision come from this
    # module's own parameters, which is what makes one cached latent valid for
    # every model that declares this id, and what stops a job config from
    # rounding the latents. device/dtype say where the CALLER wants the result;
    # they never change how it is computed. Residency is the holder's job.

    @torch.no_grad()
    def encode_images(self, image_list, device=None, dtype=None) -> Tensor:
        """Images (C, H, W) in [-1, 1], or a stacked batch -> normalized latents
        (B, 128, H/16, W/16): encode(), i.e. 2x2-packed + BatchNorm normalized,
        the exact form the DiT consumes."""
        batch = torch.stack([img.to(self.device, dtype=self.dtype) for img in image_list])
        z = self.encode(batch)
        return z if device is None and dtype is None else z.to(device, dtype=dtype)

    @torch.no_grad()
    def decode_latents(self, latents: Tensor, device=None, dtype=None) -> Tensor:
        """Normalized latents (B, 128, h, w) -> images in [-1, 1]."""
        images = self.decode(latents.to(self.device, dtype=self.dtype))
        return images if device is None and dtype is None else images.to(device, dtype=dtype)


_NUM_RESOLUTIONS = 4


def convert_diffusers_state_dict(src: dict[str, Tensor]) -> dict[str, Tensor]:
    out: dict[str, Tensor] = {}
    attn_substrings = (".mid.attn_1.",)
    for src_key, tensor in src.items():
        dst_key = _rewrite_diffusers_key(src_key)
        if dst_key is None:
            raise KeyError(f"Unrecognized diffusers VAE state-dict key: {src_key}")
        if (
            any(s in dst_key for s in attn_substrings)
            and dst_key.endswith(".weight")
            and tensor.ndim == 2
        ):
            tensor = tensor.unsqueeze(-1).unsqueeze(-1)
        out[dst_key] = tensor
    return out


def _rewrite_diffusers_key(key: str) -> str | None:
    if key.startswith("bn."):
        return key

    if key.startswith("quant_conv."):
        return key.replace("quant_conv.", "encoder.quant_conv.", 1)
    if key.startswith("post_quant_conv."):
        return key.replace("post_quant_conv.", "decoder.post_quant_conv.", 1)

    if key == "encoder.conv_norm_out.weight":
        return "encoder.norm_out.weight"
    if key == "encoder.conv_norm_out.bias":
        return "encoder.norm_out.bias"
    if key == "decoder.conv_norm_out.weight":
        return "decoder.norm_out.weight"
    if key == "decoder.conv_norm_out.bias":
        return "decoder.norm_out.bias"

    m = re.match(r"^(encoder|decoder)\.mid_block\.resnets\.(\d+)\.(.+)$", key)
    if m:
        side, idx, rest = m.group(1), int(m.group(2)), m.group(3)
        rest = rest.replace("conv_shortcut", "nin_shortcut")
        return f"{side}.mid.block_{idx + 1}.{rest}"
    m = re.match(r"^(encoder|decoder)\.mid_block\.attentions\.0\.(.+)$", key)
    if m:
        side, rest = m.group(1), m.group(2)
        rest = (
            rest.replace("group_norm.", "norm.")
            .replace("to_q.", "q.")
            .replace("to_k.", "k.")
            .replace("to_v.", "v.")
            .replace("to_out.0.", "proj_out.")
        )
        return f"{side}.mid.attn_1.{rest}"

    m = re.match(r"^encoder\.down_blocks\.(\d+)\.resnets\.(\d+)\.(.+)$", key)
    if m:
        level, res_idx, rest = m.group(1), m.group(2), m.group(3)
        rest = rest.replace("conv_shortcut", "nin_shortcut")
        return f"encoder.down.{level}.block.{res_idx}.{rest}"
    m = re.match(r"^encoder\.down_blocks\.(\d+)\.downsamplers\.0\.conv\.(.+)$", key)
    if m:
        return f"encoder.down.{m.group(1)}.downsample.conv.{m.group(2)}"

    m = re.match(r"^decoder\.up_blocks\.(\d+)\.resnets\.(\d+)\.(.+)$", key)
    if m:
        diffusers_idx = int(m.group(1))
        res_idx = m.group(2)
        rest = m.group(3).replace("conv_shortcut", "nin_shortcut")
        return (
            f"decoder.up.{_NUM_RESOLUTIONS - 1 - diffusers_idx}.block.{res_idx}.{rest}"
        )
    m = re.match(r"^decoder\.up_blocks\.(\d+)\.upsamplers\.0\.conv\.(.+)$", key)
    if m:
        diffusers_idx = int(m.group(1))
        return f"decoder.up.{_NUM_RESOLUTIONS - 1 - diffusers_idx}.upsample.conv.{m.group(2)}"

    if key.startswith(
        (
            "encoder.conv_in.",
            "encoder.conv_out.",
            "decoder.conv_in.",
            "decoder.conv_out.",
        )
    ):
        return key

    return None


# ----------------------------------------------------------------------
# the flux2 vae_id contract: file resolution + load
# ----------------------------------------------------------------------


def _load_flux2_vae(model):
    """Resolve the properly-named comfy file through the generic door (the
    ``vae_path`` model_kwargs override short-circuits it) and stream it onto
    the vae device at the precision it ships in. Candidate names and the
    download source are declared on the class, so no model carries vae file
    knowledge and no job's own repo has to ship the vae.

    One file, one build: the ``flux2-vae_small.safetensors`` sibling in the
    same repo is a decode-only build (identical encoder, narrowed decoder)
    and is deliberately not a candidate -- resolution is by exact filename,
    so it can never be picked up here."""
    path = model._resolve_comfy_file("vae", AutoEncoder)
    model.print_and_status_update(f"Loading image VAE from {path}")
    return AutoEncoder.load_to_device(path, model.vae_device_torch)


register_vae_entry(
    VaeEntry(
        vae_id="flux2",
        components={"vae": AutoEncoder},
        load=_load_flux2_vae,
    )
)
