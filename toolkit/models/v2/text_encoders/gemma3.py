from transformers import Gemma3ForConditionalGeneration

from .._mixin import OstrisTransformersMixin


class Gemma3TextEncoder(Gemma3ForConditionalGeneration, OstrisTransformersMixin):
    """Gemma3 conditioning stack (ltx2 / ltx2.3)."""

    aitk_subfolder = "text_encoder"
    aitk_tokenizer_subfolder = "tokenizer"

    @classmethod
    def get_transformer_block_names(cls):
        # both layouts seen across transformers versions; missing paths skip
        return ["model.language_model.layers", "language_model.model.layers"]

    # embed_tokens is NOT an ignore module: the manager's bouncing embedding
    # keeps it cpu-resident with a cpu-side row gather (an ignore pin kept
    # 2GB on the gpu and stranded it when legacy .to("cpu") gestures moved
    # the resident set off-device)


try:
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextModel

    from ..diffusion_models.ltx2 import LTX25_REPO, LTX25_TE_FILE

    class Gemma4TextEncoder(Gemma4TextModel, OstrisTransformersMixin):
        """Gemma4 text decoder (ltx2.5's conditioning stack)."""

        aitk_subfolder = "text_encoder"
        aitk_tokenizer_subfolder = "tokenizer"
        # shipped int8 ConvRot file; also carries the connector projections
        # and the embedded tokenizer assets (holder-side, lifted separately)
        aitk_comfy_repo = LTX25_REPO
        aitk_comfy_weight_names = {LTX25_REPO: [LTX25_TE_FILE]}

        @classmethod
        def get_transformer_block_names(cls):
            return ["layers"]

        def get_offload_ignore_modules(self):
            # layer_scalar is a bare tensor buffer on each decoder layer; the
            # manager never enumerates it, so it must ride along explicitly.
            # (embed_tokens is handled by the bouncing embedding manager.)
            return [layer.layer_scalar for layer in self.layers]

except ImportError:
    Gemma4TextEncoder = None
