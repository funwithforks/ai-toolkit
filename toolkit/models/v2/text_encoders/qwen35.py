"""Qwen3.5 text towers as toolkit v2 loadables.

``Qwen35TextOnlyEncoder`` is the language tower alone (moss_image conditioning).
Text-only inputs, so the default arange positions match the multimodal model's
text path. A stock Qwen repo id resolves to the Comfy-Org single-file repack
under MODELS_PATH/text_encoders/ (config from the Qwen repo); its lm_head, mtp
head and vision tower are dropped on load.

EDIT TODO (for the moss_image edit pass): upstream also ships a
``Qwen35VLEncoder`` (language + vision tower, same comfy files, only lm_head
and mtp dropped, plus a ``patch_vision_patch_embed`` GEMM swap for the
kernel==stride Conv3d) for models that feed reference images through the
prompt. It lives in upstream's ``toolkit/models/v2/text_encoders/qwen35.py``
(ostris/ai-toolkit); port it when the vision tower gains a live consumer.
"""

from transformers import Qwen3_5TextModel

from .._mixin import OstrisTransformersMixin
from .qwen35_fast import install_qwen35_fast_path

QWEN35_COMFY_REPO = "Comfy-Org/Qwen3.5"


class Qwen35TextOnlyEncoder(Qwen3_5TextModel, OstrisTransformersMixin):
    """Qwen3.5 language tower only. A stock Qwen repo maps to the bf16 comfy
    repack here; models shipped inside a quantized model package (moss_image)
    subclass and re-register against their package repo's int8 convrot file,
    the Fibo2TextEncoder / LTX-2.5 convention."""

    aitk_subfolder = "text_encoder"
    aitk_tokenizer_subfolder = "tokenizer"
    # the int8 repacks keep A_log / gated-norm weights in fp32; one dtype
    # throughout, like the bf16 path
    aitk_cast_quantized_load = True

    aitk_comfy_repo = QWEN35_COMFY_REPO
    aitk_comfy_weight_names = {
        "Qwen/Qwen3.5-2B": ["text_encoders/qwen3.5_2b_bf16.safetensors"],
        "Qwen/Qwen3.5-4B": ["text_encoders/qwen3.5_4b_bf16.safetensors"],
        "Qwen/Qwen3.5-9B": ["text_encoders/qwen3.5_9b_bf16.safetensors"],
    }

    _PREFIXES = ("model.language_model.", "language_model.", "model.")
    _DROPPED = ("lm_head.", "mtp.", "model.visual.", "visual.")

    @classmethod
    def get_transformer_block_names(cls):
        return ["layers"]

    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        # transformers' gated-delta fallback is a 63-step python loop PER
        # LAYER per forward (the CPU hog behind moss_image embed caching);
        # swap the vectorized chunk rule in at construction -- the modules
        # exist by then on every load path (_load_single_file, from_pretrained
        # and the .load()/aitk_post_load chain all construct first), and the
        # swap touches bindings only, never weights. See qwen35_fast.
        install_qwen35_fast_path(self)

    @classmethod
    def aitk_load_config(cls, path, subfolder=None):
        config = super().aitk_load_config(path, subfolder=subfolder)
        # the Qwen repos carry the multimodal Qwen3_5Config
        return getattr(config, "text_config", config)

    @classmethod
    def convert_state_dict_on_load(cls, state_dict):
        out = {}
        for key, value in state_dict.items():
            if key.startswith(cls._DROPPED):
                continue
            for prefix in cls._PREFIXES:
                if key.startswith(prefix):
                    key = key[len(prefix) :]
                    break
            out[key] = value
        return out

    @classmethod
    def convert_state_dict_on_save(cls, state_dict):
        return {f"model.language_model.{k}": v for k, v in state_dict.items()}
