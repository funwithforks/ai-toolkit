from diffusers import AutoencoderKL
from typing import Optional, Union
import torch
import torch.nn as nn
import numpy as np
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKLOutput
from diffusers.models.autoencoders.vae import DecoderOutput


class Config:
    in_channels = 3
    out_channels = 3
    down_block_types = ("1",)
    up_block_types = ("1",)
    block_out_channels = (1,)
    latent_channels = 3  # usually 4
    norm_num_groups = 1
    sample_size = 512
    scaling_factor = 1.0
    # scaling_factor = 1.8
    shift_factor = 0

    def __getitem__(cls, x):
        return getattr(cls, x)


class FakeVAE(nn.Module):
    def __init__(self, scaling_factor=1.0):
        super().__init__()
        self._dtype = torch.float32
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = Config()
        self.config.scaling_factor = scaling_factor

    @property
    def dtype(self):
        return self._dtype

    @dtype.setter
    def dtype(self, value):
        self._dtype = value

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, value):
        self._device = value

    # mimic to from torch
    def to(self, *args, **kwargs):
        # pull out dtype and device if they exist -- in keyword OR positional
        # form, so the soft _device/_dtype mirrors track every real placement
        # (the registry's placement step calls vae.to(device) positionally)
        if "dtype" in kwargs:
            self._dtype = kwargs["dtype"]
        if "device" in kwargs:
            self._device = kwargs["device"]
        for a in args:
            if isinstance(a, torch.dtype):
                self._dtype = a
            elif isinstance(a, torch.device):
                self._device = a
            elif isinstance(a, str):
                self._device = torch.device(a)
        return super().to(*args, **kwargs)

    def enable_xformers_memory_efficient_attention(self):
        pass

    # @apply_forward_hook
    def encode(
        self, x: torch.FloatTensor, return_dict: bool = True
    ) -> AutoencoderKLOutput:
        h = x

        # moments = self.quant_conv(h)
        # posterior = DiagonalGaussianDistribution(moments)

        if not return_dict:
            return (h,)

        class FakeDist:
            def __init__(self, x):
                self._sample = x

            def sample(self):
                return self._sample

        return AutoencoderKLOutput(latent_dist=FakeDist(h))

    def _decode(
        self, z: torch.FloatTensor, return_dict: bool = True
    ) -> Union[DecoderOutput, torch.FloatTensor]:
        dec = z

        if not return_dict:
            return (dec,)

        return DecoderOutput(sample=dec)

    # @apply_forward_hook
    def decode(
        self, z: torch.FloatTensor, return_dict: bool = True
    ) -> Union[DecoderOutput, torch.FloatTensor]:
        decoded = self._decode(z).sample

        if not return_dict:
            return (decoded,)

        return DecoderOutput(sample=decoded)

    def _set_gradient_checkpointing(self, module, value=False):
        pass

    def enable_tiling(self, use_tiling: bool = True):
        pass

    def disable_tiling(self):
        pass

    def enable_slicing(self):
        pass

    def disable_slicing(self):
        pass

    def set_use_memory_efficient_attention_xformers(self, value: bool = True):
        pass

    def forward(
        self,
        sample: torch.FloatTensor,
        sample_posterior: bool = False,
        return_dict: bool = True,
        generator: Optional[torch.Generator] = None,
    ) -> Union[DecoderOutput, torch.FloatTensor]:
        dec = sample

        if not return_dict:
            return (dec,)

        return DecoderOutput(sample=dec)

    # -- the vae does the vae work ------------------------------------------------
    # Registry-facing surface for pixel-space models: the latents ARE the
    # images in [-1, 1]. Device and precision come from the vae's own
    # _device/_dtype, so a job's dtype cannot round what the latent cache
    # stores. device/dtype say where the CALLER wants the result; they never
    # change how it is computed.

    @torch.no_grad()
    def encode_images(self, image_list, device=None, dtype=None) -> torch.FloatTensor:
        """Images (C, H, W) in [-1, 1], or a stacked batch -> the same
        tensor: identity encode."""
        if isinstance(image_list, (list, tuple)):
            batch = torch.stack([img.to(self.device, self.dtype) for img in image_list])
        else:
            batch = image_list.to(self.device, self.dtype)
        return batch if device is None and dtype is None else batch.to(device, dtype=dtype)

    def decode_latents(self, latents: torch.FloatTensor, device=None, dtype=None) -> torch.FloatTensor:
        """Identity decode: latents -> the same tensor. Left differentiable
        on purpose -- pixel-space losses may backprop through it, matching
        the legacy path's semantics. Range post-processing (the [-1, 1]
        export clamp) belongs to the caller, not the vae."""
        images = latents.to(self.device, self.dtype)
        return images if device is None and dtype is None else images.to(device, dtype=dtype)
