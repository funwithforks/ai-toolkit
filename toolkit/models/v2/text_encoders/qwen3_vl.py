import torch
import torch.nn.functional as F
from transformers import (
    Qwen3VLForConditionalGeneration,
    Qwen3VLModel,
    Qwen3VLTextModel,
)

from .._mixin import OstrisTransformersMixin


def patch_qwen_vl_patch_embed(model) -> int:
    """Qwen-VL's vision patch_embed is a Conv3d whose kernel == stride, i.e. a plain
    linear projection of each flattened patch. bf16 Conv3d has no fast cuDNN kernel and
    falls back to a slow, GPU-underutilizing path. Swap it for the equivalent F.linear
    (a GEMM). The weight is read lazily so this survives later .to(device)/dtype moves.
    Returns the number of patch_embed modules patched."""
    patched = 0
    for module in model.modules():
        proj = getattr(module, "proj", None)
        if isinstance(proj, torch.nn.Conv3d) and tuple(proj.kernel_size) == tuple(
            proj.stride
        ):

            def fast_forward(hidden_states, _proj=proj):
                w = _proj.weight.reshape(_proj.weight.shape[0], -1)
                x = hidden_states.view(-1, w.shape[1]).to(w.dtype)
                return F.linear(x, w, _proj.bias)

            module.forward = fast_forward
            patched += 1
    return patched


class Qwen3VLTextEncoder(Qwen3VLForConditionalGeneration, OstrisTransformersMixin):
    """Qwen3-VL conditioning stack (krea2, mageflow, nucleus_image, ideogram4,
    minimax_h3, ...). Loads from a checkpoint's text_encoder/ subfolder or a
    raw Qwen repo (pass subfolder=\"\" for the latter)."""

    aitk_subfolder = "text_encoder"
    aitk_processor_subfolder = "processor"

    # comfy repacks of this encoder, keyed by source repo: the krea2 file
    # is a text-tower repack (fp8-scaled), the H3 file is the 32B nvfp4/awq
    # one. A job only sees the candidates of the repo its name_or_path
    # points at (or its model's comfy_repo default).
    aitk_comfy_repo = "Comfy-Org/Krea-2"
    aitk_comfy_weight_names = {
        "Comfy-Org/Krea-2": [
            "text_encoders/qwen3vl_4b_fp8_scaled.safetensors",
        ],
        "Comfy-Org/MiniMax-H3": [
            "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
        ],
    }

    @classmethod
    def convert_state_dict_on_load(cls, state_dict):
        """Comfy repack of this encoder is a text-tower-only file: its keys
        sit at ``model.layers.*`` (Qwen3VLTextModel layout) while this class
        nests the decoder at ``model.language_model.layers.*``. Remap the
        prefix so the single-file path can attach its quantized linears and
        assign everything else. Hub-repo files already carry the nested
        layout and pass through untouched."""
        if not any(k.startswith("model.layers.") for k in state_dict):
            return state_dict
        converted = {}
        for key, value in state_dict.items():
            if key.startswith("model.visual."):
                # the visual tower rides along in this repack; it is dropped
                # after load (drop_vision_tower), its prefix already matches
                converted[key] = value
            elif key.startswith("model."):
                key = "model.language_model." + key[len("model."):]
            converted[key] = value
        # this repack carries no lm_head: it is the tied-embedding case, so
        # the head is the embedding table itself
        if "model.language_model.embed_tokens.weight" in converted and not any(
            k.startswith("lm_head.") for k in converted
        ):
            converted["lm_head.weight"] = converted["model.language_model.embed_tokens.weight"]
        return converted

    @classmethod
    def get_transformer_block_names(cls):
        return ["model.language_model.layers"]

    def drop_vision_tower(self):
        """Text-only conditioning: the vision tower is dead weight — drop it to
        free VRAM and skip loading its (bf16-slow) Conv3d patch_embed."""
        if getattr(self.model, "visual", None) is not None:
            self.model.visual = None
        return self

    def patch_vision_patch_embed(self) -> int:
        """Keep the vision tower (reference images ride into the embeddings)
        but swap its Conv3d patch_embed for an equivalent GEMM."""
        return patch_qwen_vl_patch_embed(self)


class Qwen3VLModelEncoder(Qwen3VLModel, OstrisTransformersMixin):
    """The inner Qwen3-VL base model (what AutoModel resolves): its
    last_hidden_state is the instruction feature boogu_image / ideogram4
    consume."""

    aitk_subfolder = "text_encoder"
    aitk_tokenizer_subfolder = "tokenizer"

    @classmethod
    def get_transformer_block_names(cls):
        return ["language_model.layers"]

    def drop_vision_tower(self):
        if getattr(self, "visual", None) is not None:
            self.visual = None
        return self

    def patch_vision_patch_embed(self) -> int:
        return patch_qwen_vl_patch_embed(self)


class Qwen3VLTextOnlyEncoder(Qwen3VLTextModel, OstrisTransformersMixin):
    """The Qwen3-VL text tower alone (prx_pixel)."""

    aitk_subfolder = "text_encoder"
    aitk_tokenizer_subfolder = "tokenizer"

    @classmethod
    def get_transformer_block_names(cls):
        return ["layers"]
