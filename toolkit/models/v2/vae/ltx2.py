from diffusers.models.autoencoders import (
    AutoencoderKLLTX2Audio as DiffusersAutoencoderKLLTX2Audio,
)
from diffusers.models.autoencoders import (
    AutoencoderKLLTX2Video as DiffusersAutoencoderKLLTX2Video,
)

from .._mixin import OstrisModelMixin
from ..diffusion_models.ltx2 import (
    LTX25_AUDIO_VAE_FILE,
    LTX25_REPO,
    LTX25_VIDEO_VAE_FILE,
)


class LTX2VideoVAE(DiffusersAutoencoderKLLTX2Video, OstrisModelMixin):
    aitk_subfolder = "vae"
    aitk_comfy_repo = LTX25_REPO
    # the "-conv-" file is the classic conv VAE; the default 2.5 vae file is a
    # new diffusion-decoder VAE that diffusers has no class for
    aitk_comfy_weight_names = {LTX25_REPO: [LTX25_VIDEO_VAE_FILE]}


class LTX2AudioVAE(DiffusersAutoencoderKLLTX2Audio, OstrisModelMixin):
    aitk_subfolder = "audio_vae"
    aitk_comfy_repo = LTX25_REPO
    # the file bundles the BWE vocoder under "vocoder." keys
    aitk_comfy_weight_names = {LTX25_REPO: [LTX25_AUDIO_VAE_FILE]}
