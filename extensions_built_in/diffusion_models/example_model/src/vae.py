"""The Example model's VAE: the vae does the vae.

Every new model's autoencoder gets this same shape (compare
``toolkit/models/v2/vae/qwen_image.py`` and ``flux2_kl.py`` for shared vaes,
``minimax_h3/src/vae.py`` for a bundle): the component class carries its own
comfy candidate metadata, its own loader, and its own ``encode_images`` /
``decode_latents``. The registration that turns it into a ``vae_id`` lives in
``../example_vae_id.py``.

Nothing here takes a dtype. A vae is loaded at the precision it ships in, and
the latents it hands back keep that precision -- a job's ``dtype`` rounding
them on the way into the latent cache is the loss-of-quality pattern this
fork does not keep. Upstream's ``AutoencoderKL.from_pretrained`` +
``aitk_cast_on_load`` machinery is not used at all: the module shell is built
with meta parameters, the checkpoint streams straight onto the target device
one tensor at a time (peak ram is the largest tensor, never the model), and
the load is ``strict=True`` so a wrong file dies here instead of leaving part
of the network at its initialization.

The architecture and the latent scaling are vendored below rather than read
from a ``vae/config.json``: a load needs the weights file and nothing else,
and they are what make the latents the distribution this vae's models train
against. Replace them with your own vae's numbers.
"""

import time
from typing import List, Union

import torch
from diffusers import AutoencoderKL

from toolkit.util.weight_source import WeightSource
from toolkit.models.v2._mixin import OstrisModelMixin

# This (fictional) vae's architecture + latent scaling, i.e. the contents of
# the config.json that would otherwise have to ship next to the weights.
# 4 latent channels, 8x spatial downsampling (matches the model's
# ``vae_scale_factor``), and the SD-style scaling the transformer was trained
# against.
EXAMPLE_VAE_CONFIG = {
    "in_channels": 3,
    "out_channels": 3,
    # the autoencoder's own block classes, spelled out. diffusers' legacy
    # "DownBlock2D"/"UpBlock2D" strings (still written in old public vae
    # config.json files) resolve to the UNET blocks on the diffusers this
    # repo installs, and those return multi-scale skip tuples -- the encoder
    # chain then dies feeding a tuple into its second resnet.
    "down_block_types": ("DownEncoderBlock2D",) * 4,
    "up_block_types": ("UpDecoderBlock2D",) * 4,
    "block_out_channels": (128, 256, 512, 512),
    "layers_per_block": 2,
    "latent_channels": 4,
    "norm_num_groups": 32,
    "act_fn": "silu",
    "scaling_factor": 0.18215,
    "shift_factor": 0.0,
}


class ExampleVAE(AutoencoderKL, OstrisModelMixin):
    """The Example model's frozen autoencoder.

    The file contract lives on the class, the same as every other component:
    the comfy single file is named after the vae it holds (no
    ``ae.safetensors``, no ``diffusion_pytorch_model.safetensors`` -- those
    names tell you nothing about which encoder they contain), it resolves
    through the comfy door, and it lands in ``MODELS_PATH/vae/``.

    Swap both values for your model's own repos when you adapt this template:
    ``aitk_comfy_weight_names`` is keyed by the standard ``name_or_path`` a job
    names, and ``aitk_comfy_repo`` is where a missing file is downloaded from.
    """

    aitk_subfolder = "vae"
    aitk_comfy_repo = "Comfy-Org/Example-Model"
    aitk_comfy_weight_names = {
        "Example-AI/Example-Model": ["vae/example-vae.safetensors"],
    }

    @classmethod
    def load_to_device(cls, file_path: str, device) -> "ExampleVAE":
        """Stream a comfy single-file checkpoint onto ``device``, at the
        precision it ships in. No dtype argument: see the module docstring."""
        from accelerate import init_empty_weights

        t0 = time.perf_counter()
        dev = torch.device(device)
        if dev.type == "cuda" and dev.index is None:
            dev = torch.device("cuda", torch.cuda.current_device())
        with init_empty_weights(include_buffers=False):
            model = cls(**EXAMPLE_VAE_CONFIG)
        state = WeightSource.open(file_path).materialize(dev)
        model.load_state_dict(state, strict=True, assign=True)
        del state
        # the meta shell leaves non-persistent buffers wherever they were
        # built; parameters are already resident so this only relocates those
        model.to(dev)
        model.eval().requires_grad_(False)
        print(
            f"[load-timing] ExampleVAE: streamed to {dev} "
            f"{time.perf_counter() - t0:.1f}s"
        )
        return model

    # -- the vae does the vae work ------------------------------------------------
    # These are what phase 1 caches and what sampling decodes, so they belong to
    # the vae rather than to the model: device and precision come from this
    # module's own parameters, which is what makes one cached latent valid for
    # every model declaring this id. device/dtype say where the CALLER wants the
    # result; they never change how it is computed. Residency (the cpu ->
    # vae-device move) is the holder's job, supplied by PhasedLoadMixin.

    @torch.no_grad()
    def encode_images(
        self, image_list: Union[List[torch.Tensor], torch.Tensor], device=None, dtype=None
    ) -> torch.Tensor:
        """Pixels -> latents. Used for latent caching and for control images.

        in:  image_list  list of (3, H, W) tensors -- or a (B, 3, H, W) batch --
                         with values in [-1, 1], already crop/bucket-sized
        out: (B, 4, H/8, W/8) latents, scaled the way the transformer expects
             ((z - shift_factor) * scaling_factor), in this vae's own precision
        """
        images = (
            image_list
            if isinstance(image_list, torch.Tensor)
            else torch.stack(list(image_list), dim=0)
        )
        images = images.to(self.device, dtype=self.dtype)
        z = self.encode(images).latent_dist.sample()
        z = (z - (self.config.shift_factor or 0.0)) * self.config.scaling_factor
        return z if device is None and dtype is None else z.to(device, dtype=dtype)

    @torch.no_grad()
    def decode_latents(self, latents: torch.Tensor, device=None, dtype=None) -> torch.Tensor:
        """Latents -> pixels. Used when rendering previews.

        in:  (B, 4, h, w) latents in the normalized space encode_images produces
        out: (B, 3, H, W) images in [-1, 1]
        """
        z = latents.to(self.device, dtype=self.dtype)
        z = z / self.config.scaling_factor + (self.config.shift_factor or 0.0)
        images = self.decode(z).sample
        return images if device is None and dtype is None else images.to(device, dtype=dtype)
