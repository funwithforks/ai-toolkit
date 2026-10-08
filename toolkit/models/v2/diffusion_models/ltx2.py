from diffusers.models.transformers import (
    LTX2VideoTransformer3DModel as DiffusersLTX2VideoTransformer3DModel,
)
from diffusers.pipelines.ltx2 import (
    LTX2TextConnectors as DiffusersLTX2TextConnectors,
)
from diffusers.pipelines.ltx2 import LTX2Vocoder as DiffusersLTX2Vocoder
from diffusers.pipelines.ltx2 import (
    LTX2VocoderWithBWE as DiffusersLTX2VocoderWithBWE,
)

from .._mixin import OstrisModelMixin


# ComfyUI-style split files for LTX-2.5 (no diffusers repo exists for 2.5).
# Only the shipped int8 ConvRot files are registered: the bf16 transformer/TE
# variants stay reachable exclusively via the explicit <component>_path
# model_kwargs overrides -- registering them as candidates would make a
# quantize-off run rank the clean file first and silently download a ~44GB
# file the box already has a trainable int8 twin of.
LTX25_REPO = "Lightricks/LTX-2.5"
LTX25_DIT_FILE = (
    "diffusion_models/ltx-2.5-22b-dev-transformer-comfy-int8-convrot.safetensors"
)
LTX25_VIDEO_VAE_FILE = "vae/ltx-2.5-video-vae-conv-bf16.safetensors"
# bundles the BWE vocoder alongside the audio VAE under "vocoder."/"audio_vae."
# prefixes; one file serves both classes
LTX25_AUDIO_VAE_FILE = "vae/ltx-2.5-audio-vae-bf16.safetensors"
LTX25_TE_FILE = (
    "text_encoders/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors"
)


class LTX2VideoTransformer3DModel(
    DiffusersLTX2VideoTransformer3DModel, OstrisModelMixin
):
    aitk_subfolder = "transformer"
    aitk_comfy_repo = LTX25_REPO
    aitk_comfy_weight_names = {LTX25_REPO: [LTX25_DIT_FILE]}

    @classmethod
    def get_transformer_block_names(cls):
        return ["transformer_blocks"]

    def get_offload_ignore_modules(self):
        # fp32 scale/shift tables (bare tensors) must stay resident
        mods = []
        for block in self.transformer_blocks:
            mods += [
                block.scale_shift_table,
                block.audio_scale_shift_table,
                block.video_a2v_cross_attn_scale_shift_table,
                block.audio_a2v_cross_attn_scale_shift_table,
            ]
        return mods + [self.scale_shift_table, self.audio_scale_shift_table]


class LTX2TextConnectors(DiffusersLTX2TextConnectors, OstrisModelMixin):
    aitk_subfolder = "connectors"
    aitk_comfy_repo = LTX25_REPO
    # connector weights ride inside the dit file ("model.diffusion_model.*"
    # keys split out by the holder); the per-modality projections ride in the
    # TE file and are lifted by the holder at build time
    aitk_comfy_weight_names = {LTX25_REPO: [LTX25_DIT_FILE]}


class LTX2Vocoder(DiffusersLTX2Vocoder, OstrisModelMixin):
    aitk_subfolder = "vocoder"


class LTX2VocoderWithBWE(DiffusersLTX2VocoderWithBWE, OstrisModelMixin):
    aitk_subfolder = "vocoder"
    aitk_comfy_repo = LTX25_REPO
    aitk_comfy_weight_names = {LTX25_REPO: [LTX25_AUDIO_VAE_FILE]}
