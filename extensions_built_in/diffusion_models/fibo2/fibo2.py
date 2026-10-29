"""Fibo 2.0 dev (briaai/Fibo-2.0-dev) for ai-toolkit -- text-to-image.

  - DiT: 44 Z-Image blocks, 3584 wide, a Perceiver turning the text into
    256-512 gist tokens and five cross-attention injection blocks
    (``src/transformer.py``, Comfy-Org/Fibo-2.0 key layout)
  - VAE: FLUX.2, 2x2 packed and BatchNorm normalized -- the shared toolkit BFL
    ``AutoEncoder`` (``toolkit/models/v2/vae/flux2_kl``), not a fibo2 copy.
  - text: Qwen3-VL-4B, six RMS-normalized bundles of three hidden layers;
    stored per item as (L, 46080) in an ``AdvancedPromptEmbeds``. These bundles
    are what the DiT consumes at every training step, so the caption cache
    stores the post-encoder bundle (the ``get_prompt_embeds`` return), not raw
    encoder output; ``embed_file_valid`` guards the format.

This is the phased-loading, text-to-image import. The Qwen3-VL vision tower is
dropped at load (t2i never runs it); the training process caches target latents
in phase 1 and bundles in phase 2, so the VAE and text encoder are never
resident during training (see docs/guide/phased-loading--component-phases.md).

EDIT IS NOT WIRED HERE. Fibo 2.0 makes an edit out of the same weights by
sending control images through the Qwen3-VL vision tower and as clean t=0 VAE
latents into the DiT. That needs the vision tower resident in phase 2 and the
reference latents cached in phase 1; until then every control/edit dataset is
refused loudly (get_prompt_embeds / get_noise_prediction). See EDIT TODO below.

``model.name_or_path`` is the package repo (default ``Comfy-Org/Fibo-2.0``) or a
local dir holding the component files. The DiT and text encoder resolve through
their classes' comfy-candidates registration -- the shipped int8 ConvRot repacks
are the trained identity (the LTX-2.5 / Krea-2 / flux3 convention: a
quantize: none run can never rank a higher-precision twin first). All fetches
land in the ComfyUI layout under MODELS_PATH, never the HF cache; the
``transformer_path`` / ``text_encoder_path`` / ``vae_path`` model_kwargs override
any component. Config, tokenizer and processor come from the public
Qwen/Qwen3-VL-4B-Instruct repo. Prompts are structured JSON captions (cleaned
and compacted as in training) or plain text.

Flow convention matches ai-toolkit exactly (t=1 noise -> t=0 clean,
target = noise - clean), so no time flips anywhere.

The conditioning cache boundary stops at the TE->DiT handoff ON PURPOSE. The
Perceiver/gist pass is deterministic and batch-invariant per caption, so its
output looks cacheable (train the LoRA without it) -- but caching it requires
freezing the Perceiver out of LoRA, and the vendor recipe (briaai/Fibo-2.0-dev
``fine_tuning/train_lora.py``, ``build_model``) adapts EVERY ``nn.Linear`` in
the transformer and asserts none were missed: the Perceiver is part of the
trainable identity. Measured 2026-10-23: the gist pass costs a flat ~6.3
ms/step (~1.7% of forward at 1024^2, ~5.2% at 512^2, bs=1, int8 ConvRot), so
the frozen-Perceiver trade was rejected and the cache stays pre-Perceiver. Do
not re-propose caching a LoRA-targeted module's output -- the same bar falls
on caching the injection blocks' ``to_kv`` projections.

EDIT TODO (for the edit-functionality pass -- cuts the research time):
  * keep the vision tower (drop the drop_vision_tower() call) and vendored
    ``src/references.py`` back from upstream (prepare_references/vlm_image/
    drop_repeats, ~129 lines) -- references.py was dropped here as dead t2i code.
  * set encode_control_in_text_embeddings / has_multiple_control_images /
    use_raw_control_images True (see the refuse points below for the flags).
  * ``get_text_embedding_space_version`` must fold the match_target_res rule into
    the cache key (a reference's vision tokens change with it) -- upstream had it.
  * the DiT's ``ref_latents`` path already works (src/transformer.forward handles
    a list of clean latents); it just needs them produced. Upstream re-encoded
    them from raw control images with the VAE resident EVERY step (the wart).
    The right shape is to cache reference latents in phase 1 like target latents
    (VAE-load, encode the per-item references, free), keyed on control image +
    bucket size + match_target_res -- then the VAE is never resident in training.
    Upstream's per-step encode + require_vae_during_training is the anti-pattern
    to avoid; nothing in the fork caches control latents today (control_cache_keys
    in dataloader_mixins.py:1272 is only krea2's in-RAM memo, not on disk).
"""

import os
from typing import TYPE_CHECKING, List, Optional

import torch

from toolkit.accelerator import unwrap_model
from toolkit.advanced_prompt_embeds import AdvancedPromptEmbeds
from toolkit.config_modules import GenerateImageConfig, ModelConfig
from toolkit.metadata import get_meta_for_safetensors
from toolkit.models.base_model import BaseModel
from toolkit.models.phased_load import PhasedLoadMixin
from toolkit.samplers.custom_flowmatch_sampler import (
    CustomFlowMatchEulerDiscreteScheduler,
)

from .src.pipeline import Fibo2Pipeline, scheduler_config, stack_text_embeds
from .src.text_encoder import QWEN3_VL_4B_REPO, MAX_TEXT_TOKENS, Fibo2TextEncoder, encode_prompts
from .src.transformer import Fibo2Transformer

if TYPE_CHECKING:
    from toolkit.data_transfer_object.data_loader import DataLoaderBatchDTO


HF_TOKEN = os.getenv("HF_TOKEN", None)

# The VAE is the shared flux2 vae_id entry (toolkit/models/v2/vae/flux2_kl.py):
# file resolution, load, encode and decode all live there. The DiT and TE files
# are registered as comfy candidates on their classes, not here.


class Fibo2Model(PhasedLoadMixin, BaseModel):
    arch = "fibo2"
    display_name = "Fibo 2.0"
    # the shared FLUX.2 packed AE; load/encode/decode are owned by that entry
    vae_id = "flux2"

    def __init__(
        self,
        device,
        model_config: ModelConfig,
        dtype="bf16",
        custom_pipeline=None,
        noise_scheduler=None,
        **kwargs,
    ):
        super().__init__(
            device, model_config, dtype, custom_pipeline, noise_scheduler, **kwargs
        )
        self.is_flow_matching = True
        self.is_transformer = True
        self.target_lora_modules = ["Fibo2Transformer"]
        self.supports_model_paths = True
        self.use_old_lokr_format = False
        # the base-model converter swaps transformer. <-> diffusion_model. on
        # save/load, matching the ComfyUI key layout
        self.lora_keys_use_comfy_prefix = True
        self.max_text_length = int(
            self.model_config.model_kwargs.get("max_text_length", MAX_TEXT_TOKENS)
        )
        # Qwen3-VL image processor, loaded with the text encoder. Text-only t2i
        # never feeds it pixels; it is kept because encode_prompts reads its
        # grid/merge config while building the (empty) image runs.
        self.image_processor = None

    @staticmethod
    def get_train_scheduler():
        return CustomFlowMatchEulerDiscreteScheduler(**scheduler_config)

    def get_bucket_divisibility(self):
        return 16  # f8 conv + 2x2 packing

    # ------------------------------------------------------------------
    # Loading (PhasedLoadMixin contract: one component per phase)
    # ------------------------------------------------------------------
    def _load_transformer(self) -> Fibo2Transformer:
        self.print_and_status_update("Loading transformer (Fibo2Transformer)")
        # comfy-candidates door: the class registers the shipped int8 convrot
        # dev repack; transformer_path overrides, a local package dir searches
        # as an extra root, downloads land in MODELS_PATH (comfy layout).
        dit_path = self._resolve_comfy_file("transformer", Fibo2Transformer)
        self.print_and_status_update(f" - transformer weights: {dit_path}")
        # config is sniffed from the file (Fibo2Transformer.
        # aitk_config_from_state_dict); the mixin loader reads the int8 file and
        # attaches the convrot groups as shipped (zero requantization). Placement
        # / quantization policy after this runs in aitk_post_load (the mixin).
        transformer = Fibo2Transformer.load_model(
            dit_path,
            dtype=self.torch_dtype,
            aitk_device=None if self.model_config.low_vram else self.device_torch,
        )
        return transformer

    def _load_text_encoder(self):
        te_path = self.model_config.te_name_or_path
        if te_path is None:
            # comfy-candidates door: the package's convrot repack of the 4B
            # encoder (text_encoder_path overrides through the door).
            te_path = self._resolve_comfy_file("text_encoder", Fibo2TextEncoder)
        self.print_and_status_update(f"Loading Qwen3-VL text encoder from {te_path}")
        text_encoder = Fibo2TextEncoder.load_model(
            te_path,
            dtype=self.torch_dtype,
            subfolder="",
            # the comfy file is weights-only; the public Qwen repo supplies the
            # model config (and the tokenizer / processor below)
            config_path=QWEN3_VL_4B_REPO,
            # as-shipped streaming onto the gpu (no cpu staging); low_vram parks
            # the encoder on cpu the old way
            aitk_device=None if self.model_config.low_vram else self.te_device_torch,
        )
        processor = Fibo2TextEncoder.load_processor()
        tokenizer = processor.tokenizer
        tokenizer.padding_side = "right"
        # text-only conditioning -- drop the vision tower (its conv3d patch_embed
        # is bf16-slow and unused here). See the EDIT TODO in the module header.
        text_encoder.drop_vision_tower()
        text_encoder.eval()
        text_encoder.requires_grad_(False)
        return tokenizer, text_encoder, {"image_processor": processor.image_processor}

    def _build_pipeline(self):
        # the phased mixin drives load_transformer/load_text_encoder/load_vae
        # (training phases, or all three at once for inference) and wires the
        # scheduler + pipeline + aitk_post_load placement per model_config.
        return Fibo2Pipeline(self)

    # ------------------------------------------------------------------
    # Embed cache format guard (the caption cache stores the post-encoder bundle)
    # ------------------------------------------------------------------
    def embed_file_valid(self, path: str) -> bool:
        # The caption cache stores the DiT's conditioning bundles (six RMS-normed
        # hidden layers concatenated), i.e. AdvancedPromptEmbeds -- what the
        # training step consumes, not raw encoder output. A plain PromptEmbeds
        # cache (or any other class) is a different space and must re-encode
        # through the text encoder phase. Only damaged/unreadable files count as
        # invalid; both callers check existence first, so a non-file error must
        # propagate (a swallowed bug here would silently re-encode every cache).
        from safetensors import _safetensors_rust, safe_open

        try:
            with safe_open(path, framework="pt") as f:
                if (f.metadata() or {}).get("class_name", "") != "AdvancedPromptEmbeds":
                    return False
                keys = set(f.keys())
        except (_safetensors_rust.SafetensorError, OSError):
            return False
        return "text_embeds" in keys

    # ------------------------------------------------------------------
    # Generation (training previews)
    # ------------------------------------------------------------------
    def get_generation_pipeline(self):
        return Fibo2Pipeline(self)

    def generate_single_image(
        self,
        pipeline: Fibo2Pipeline,
        gen_config: GenerateImageConfig,
        conditional_embeds: AdvancedPromptEmbeds,
        unconditional_embeds: AdvancedPromptEmbeds,
        generator: torch.Generator,
        extra: dict,
    ):
        if self.model.device == torch.device("cpu"):
            self.model.to(self.device_torch)

        sc = self.get_bucket_divisibility()
        gen_config.width = int(gen_config.width // sc * sc)
        gen_config.height = int(gen_config.height // sc * sc)

        # EDIT TODO: sample-time references (gen_config.ctrl_img*) condition both
        # passes as clean latents upstream; t2i refuses them for now.
        if any(
            p is not None
            for p in (
                gen_config.ctrl_img,
                gen_config.ctrl_img_1,
                gen_config.ctrl_img_2,
                gen_config.ctrl_img_3,
            )
        ):
            raise NotImplementedError(
                "Fibo 2.0 edit/reference sampling is not wired in this t2i build; "
                "remove ctrl_img from the sample prompt."
            )

        result = pipeline(
            conditional_embeds=conditional_embeds,
            unconditional_embeds=unconditional_embeds,
            height=gen_config.height,
            width=gen_config.width,
            num_inference_steps=gen_config.num_inference_steps,
            guidance_scale=gen_config.guidance_scale,
            latents=gen_config.latents,
            generator=generator,
        )
        return result[0]

    # ------------------------------------------------------------------
    # Training hooks
    # ------------------------------------------------------------------
    def get_noise_prediction(
        self,
        latent_model_input: torch.Tensor,  # (B, 128, h, w)
        timestep: torch.Tensor,  # 0..1000 scale
        text_embeddings: AdvancedPromptEmbeds,
        batch: "DataLoaderBatchDTO" = None,
        **kwargs,
    ):
        device = self.device_torch
        dtype = self.torch_dtype
        if self.model.device == torch.device("cpu"):
            self.model.to(device)

        batch_size = latent_model_input.shape[0]
        embeds_list = text_embeddings.text_embeds
        if len(embeds_list) == 1 and batch_size > 1:
            embeds_list = embeds_list * batch_size

        with torch.no_grad():
            # toolkit timestep (0..1000, 1000 = pure noise) -> the DiT's own t,
            # same convention, straight through (no /1000, no flip).
            t = timestep.to(device, dtype=torch.float32).reshape(-1)
            if t.shape[0] != batch_size:
                t = t.expand(batch_size)
            embeds, mask = stack_text_embeds(embeds_list, device, dtype)

            # Training is t2i-only. Edit conditioning needs a VAE encode per
            # batch (references as clean latents) and the vision tower in the
            # prompt, but neither is resident during training (see the phased
            # load and the module-header EDIT TODO). Refuse control datasets
            # loudly instead of silently ignoring their controls.
            if batch is not None and (
                batch.control_tensor_list is not None or batch.control_tensor is not None
            ):
                raise NotImplementedError(
                    "Fibo 2.0 training with control/edit images is not supported "
                    "in this build. Remove the control datasets."
                )

        return self.transformer(
            latent_model_input.to(device, dtype), t, embeds, mask, None
        )

    @torch.no_grad()
    def get_prompt_embeds(self, prompt, control_images=None) -> AdvancedPromptEmbeds:
        if control_images is not None:
            raise NotImplementedError(
                "Fibo 2.0 conditioning in this build is text-only; reference "
                "images are not supported (see the module-header EDIT TODO)"
            )
        if isinstance(prompt, str):
            prompt = [prompt]
        elif prompt is None or (isinstance(prompt, (list, tuple)) and len(prompt) == 0):
            # unset negatives arrive as None/empty; the null caption is the
            # trained unconditional
            prompt = [""]
        if self.text_encoder.device == torch.device("cpu"):
            self.text_encoder.to(self.device_torch)

        embeds = encode_prompts(
            self.text_encoder,
            self.tokenizer,
            self.image_processor,
            prompt,
            [[] for _ in prompt],
            self.device_torch,
            self.torch_dtype,
            max_length=self.max_text_length,
        )
        return AdvancedPromptEmbeds(text_embeds=embeds)

    def get_loss_target(self, *args, **kwargs):
        noise = kwargs.get("noise")
        batch = kwargs.get("batch")
        return (noise - batch.latents).detach()

    def get_model_has_grad(self):
        return False

    def get_te_has_grad(self):
        return False

    # ------------------------------------------------------------------
    # VAE encode / decode
    # ------------------------------------------------------------------
    # Both are the flux2 vae's own methods (toolkit/models/v2/vae/flux2_kl.py),
    # reached through PhasedLoadMixin -- there is no fibo2 encoding to keep in
    # step with them.

    # ------------------------------------------------------------------
    # Saving / bookkeeping
    # ------------------------------------------------------------------
    def save_model(self, output_path, meta, save_dtype):
        if not output_path.endswith(".safetensors"):
            output_path = output_path + ".safetensors"
        transformer: Fibo2Transformer = unwrap_model(self.model)
        transformer.save_model(
            output_path,
            dtype=save_dtype,
            metadata=get_meta_for_safetensors(meta, name=self.arch),
            quantized=False,
        )

    def get_base_model_version(self):
        return "fibo2"

    def get_transformer_block_names(self) -> Optional[List[str]]:
        return Fibo2Transformer.get_transformer_block_names()

    def get_quantization_exclude_modules(self):
        return Fibo2Transformer.get_quantization_exclude_modules()
