"""ExampleModel -- a fully documented template for adding a new model to ai-toolkit.

Read README.md in this folder first for the big picture (lifecycle, data flow,
registration, and how to adapt this template into an edit / video / i2v model).

Every override below documents:
  - WHEN ai-toolkit calls it
  - WHAT comes in (shapes, dtypes, scales)
  - WHAT must come out

The model itself is a made-up flow-matching DiT whose architecture lives in
./src/model.py and whose preview sampler lives in ./src/pipeline.py, simulating
the common case where diffusers does not ship your model and you vendor both.
"""

import os
from typing import List, Optional

import torch
import yaml
from safetensors.torch import load_file, save_file

from transformers import AutoTokenizer, AutoModel

from toolkit.accelerator import unwrap_model
from toolkit.advanced_prompt_embeds import AdvancedPromptEmbeds
from toolkit.basic import flush
from toolkit.config_modules import GenerateImageConfig, ModelConfig
from toolkit.models.base_model import BaseModel
from toolkit.models.phased_load import PhasedLoadMixin
from toolkit.samplers.custom_flowmatch_sampler import (
    CustomFlowMatchEulerDiscreteScheduler,
)

from .src.model import ExampleTransformer2DModel
from .src.pipeline import ExamplePipeline, pad_prompt_embeds

# registers the example vae_id (the vae class and its load live in src/vae.py)
from . import example_vae_id  # noqa: F401


# Config for the training/sampling noise scheduler. ai-toolkit's flow-matching
# models all use CustomFlowMatchEulerDiscreteScheduler; ``shift`` warps the
# timestep distribution toward the high-noise end (bigger = more high-noise
# steps, typical for high-resolution models).
scheduler_config = {
    "num_train_timesteps": 1000,
    "use_dynamic_shifting": False,
    "shift": 3.0,
}


class ExampleModel(PhasedLoadMixin, BaseModel):
    # ``arch`` is the unique id that ties everything together:
    #  - ``model.arch: "example"`` in the training config YAML selects this class
    #    (resolved by toolkit/util/get_model.py:get_model_class)
    #  - it is the default cache key for text-embedding / latent caches
    arch = "example"

    # short name used in the phased-load status prints
    display_name = "Example"

    # The vae is owned by its id, not by this class: PhasedLoadMixin.load_vae
    # resolves construction through the registered entry
    # (``example_vae_id.py``), and encode/decode are methods on the vae itself
    # (``src/vae.py``). A model that declares an id keeps no vae code, which is
    # what makes one cached latent valid for every model on the id.
    vae_id = "example"

    # ALL NEW MODELS should set this to False. ``BaseModel`` defaults it to True
    # only for backwards-compatibility with already-released LoKr checkpoints; the
    # newer LoKr weight format is the correct one for any new architecture.
    use_old_lokr_format = False

    def __init__(
        self,
        device,                  # "cuda:0" etc.
        model_config: ModelConfig,  # the parsed ``model:`` section of the YAML
        dtype="bf16",
        custom_pipeline=None,
        noise_scheduler=None,
        **kwargs,
    ):
        super().__init__(
            device, model_config, dtype, custom_pipeline, noise_scheduler, **kwargs
        )
        # --- flags the rest of the toolkit reads ---
        # flow matching (velocity prediction) vs ddpm-style epsilon prediction
        self.is_flow_matching = True
        # transformer (DiT) vs unet: affects LoRA naming ("transformer." prefix)
        self.is_transformer = True
        # Class names of modules whose Linear layers get LoRA'd. Matched against
        # type(module).__name__, so this must equal the class name in src/model.py.
        self.target_lora_modules = ["ExampleTransformer2DModel"]

        # --- values used by our own overrides below ---
        self.patch_size = 2       # transformer patch size (latent px per token)
        self.vae_scale_factor = 8  # pixels per latent px (8x downsampling VAE)
        # hard cap on prompt token length (truncation only -- embeds are stored
        # per-sample at natural length, see get_prompt_embeds)
        self.max_text_length = 512

        # Other flags you may need (all default False, set in BaseModel.__init__):
        #   self.encode_control_in_text_embeddings = True
        #       -> get_prompt_embeds receives control_images (vision-language TEs
        #          that look at the control image, e.g. qwen_image_edit)
        #   self.has_multiple_control_images = True
        #       -> control images arrive as a list (qwen_image_edit_plus)
        #   self.use_raw_control_images = True
        #       -> control images are not resized to match the target image
        #   self.is_multistage = True
        #       -> model has multiple experts trained on timestep ranges (wan22 14b)

    @staticmethod
    def get_train_scheduler():
        """Build the noise scheduler used for BOTH training and sampling.

        Called when loading the model, and again by the pipeline for every
        preview run (a fresh instance, because scheduler state is mutable).
        """
        return CustomFlowMatchEulerDiscreteScheduler(**scheduler_config)

    def get_bucket_divisibility(self):
        """Pixel multiple that dataset resolution buckets must snap to.

        The data loader crops every image so width/height are divisible by
        this. Latents are 1/8 the pixel size (VAE) and the transformer eats
        2x2 latent patches, so pixels must be divisible by 8 * 2 = 16.
        """
        return self.vae_scale_factor * self.patch_size

    # ------------------------------------------------------------------
    # Loading (phased -- see toolkit/models/phased_load.py)
    # ------------------------------------------------------------------
    # PhasedLoadMixin supplies load_transformer / load_text_encoder /
    # load_vae / load_model. The model supplies the _load_* builders below
    # (which return the raw components and assign nothing), plus
    # get_train_scheduler, _build_pipeline and display_name. The training
    # process calls the phases one at a time -- vae, text encoder,
    # transformer -- running dataset prep between phases, so no component
    # ever has to be shuffled to cpu to make room for prep work.
    #
    # MUST be set before the phases finish:
    #   self.model           the trainable denoiser (transformer/unet)
    #   self.vae             the (frozen) VAE
    #   self.text_encoder    one module or a list of modules (frozen unless
    #                        training the TE)
    #   self.tokenizer       one tokenizer or a list, parallel to text_encoder
    #   self.noise_scheduler from get_train_scheduler()
    #   self.pipeline        anything generate_single_image can use
    #
    # Where each component comes from:
    #   <name_or_path>/transformer/model.safetensors
    #   <name_or_path>/text_encoder/  + /tokenizer/  (transformers format)
    #   vae: NOT from this folder. It is the named comfy single file the vae
    #     class declares (src/vae.py), resolved through the comfy door and kept
    #     in MODELS_PATH/vae/ -- see the vae_id comment on the class above.

    def _load_transformer(self):
        """Build the transformer (the custom model from src/) and return it.

        Quantization/offload/placement is NOT done here: the mixin applies
        component_load_kwargs("transformer") via aitk_post_load right after
        this returns. Models whose checkpoint sourcing is standard can
        collapse the build into that step: ExampleTransformer2DModel.load(
        path, **kwargs)."""
        dtype = self.torch_dtype
        self.print_and_status_update("Loading transformer")
        model_path = self.model_config.name_or_path
        # Instantiate on the meta device (no RAM used), then materialize the
        # real tensors straight from the checkpoint with assign=True. This
        # avoids allocating the model twice. If your model has non-persistent
        # buffers, rebuild them after this (see ideogram4.py for an example).
        with torch.device("meta"):
            transformer = ExampleTransformer2DModel()
        state_dict = load_file(
            os.path.join(model_path, "transformer", "model.safetensors")
        )
        state_dict = {k: v.to(dtype) for k, v in state_dict.items()}
        transformer.load_state_dict(state_dict, assign=True)
        del state_dict
        flush()  # gc + empty cuda cache; call it after dropping anything big
        return transformer

    def _load_text_encoder(self):
        """Return (tokenizer, text_encoder, extras).

        extras is a dict of extra attributes the mixin assigns to self
        (vision processors etc.; empty here). aitk_post_load applies
        quantize_te / qtype_te and placement per model_config after this
        returns -- do not quantize by hand."""
        self.print_and_status_update("Loading text encoder")
        model_path = self.model_config.name_or_path
        tokenizer = AutoTokenizer.from_pretrained(model_path, subfolder="tokenizer")
        text_encoder = AutoModel.from_pretrained(
            model_path, subfolder="text_encoder", torch_dtype=self.torch_dtype
        )
        text_encoder.to(self.te_device_torch)
        # the TE is frozen here; only set requires_grad if you train it
        text_encoder.eval()
        text_encoder.requires_grad_(False)
        flush()
        return tokenizer, text_encoder, {}

    # load_vae: PhasedLoadMixin dispatches through the registered example
    # vae_id entry (example_vae_id.py) -- the vae class decides its own file
    # candidates and streams them to the vae device at shipped precision.

    def _build_pipeline(self):
        return ExamplePipeline(self)

    # ------------------------------------------------------------------
    # Sampling (training previews)
    # ------------------------------------------------------------------
    def get_generation_pipeline(self):
        """Return a fresh pipeline for a round of preview sampling.

        Called once per sampling round by BaseModel.generate_images. Our
        pipeline holds no state, so a new lightweight wrapper is enough.
        """
        return ExamplePipeline(self)

    def generate_single_image(
        self,
        pipeline: ExamplePipeline,
        gen_config: GenerateImageConfig,        # one sample_prompts entry: width,
                                                # height, seed, num_inference_steps,
                                                # guidance_scale, ctrl_img, num_frames...
        conditional_embeds: AdvancedPromptEmbeds,    # already-encoded prompt
        unconditional_embeds: AdvancedPromptEmbeds,  # already-encoded negative prompt
        generator: torch.Generator,             # seeded with gen_config.seed
        extra: dict,                            # adapter kwargs (controlnet etc.)
    ):
        """Render ONE preview image.

        The harness (BaseModel.generate_images) has already encoded the
        prompts with get_prompt_embeds -- the pipeline never sees text.

        Returns a PIL.Image (or for video models a list of PIL frames).
        """
        # low_vram: components may be parked on CPU between steps
        if self.model.device == torch.device("cpu"):
            self.model.to(self.device_torch)

        # snap requested size to the model's divisibility
        sc = self.get_bucket_divisibility()
        gen_config.width = int(gen_config.width // sc * sc)
        gen_config.height = int(gen_config.height // sc * sc)

        img = pipeline(
            conditional_embeds=conditional_embeds,
            unconditional_embeds=unconditional_embeds,
            height=gen_config.height,
            width=gen_config.width,
            num_inference_steps=gen_config.num_inference_steps,
            guidance_scale=gen_config.guidance_scale,
            latents=gen_config.latents,  # usually None; pre-made noise if set
            generator=generator,
        )[0]
        return img

    # ------------------------------------------------------------------
    # Training hooks
    # ------------------------------------------------------------------
    def get_noise_prediction(
        self,
        latent_model_input: torch.Tensor,
        timestep: torch.Tensor,
        text_embeddings: AdvancedPromptEmbeds,
        **kwargs,
    ):
        """The actual forward pass of the denoiser. Called every train step
        (with grads) via BaseModel.predict_noise, and also by some adapters.

        in:
          latent_model_input  (B, C, h, w) noisy latents: the output of
                              add_noise(clean_latents, noise, timestep), after
                              condition_noisy_latents (channel-concat models
                              would see extra channels here).
                              For video models this is (B, C, frames, h, w).
          timestep            (B,) float on the 0..1000 scale, 1000 = pure noise
          text_embeddings     AdvancedPromptEmbeds for the batch; every key you
                              stored in get_prompt_embeds holds a list of B
                              tensors (cached per-sample embeds are expanded /
                              concatenated for you)
          **kwargs            may include ``batch`` (DataLoaderBatchDTO),
                              guidance_embedding_scale, adapter residuals, ...
                              only passed if your signature declares them

        out:
          (B, C, h, w) the model prediction. For flow matching that is the
          velocity in the same convention as get_loss_target (here:
          noise - clean). Shape must match the TARGET latents -- if you
          concatenated control channels/tokens in, slice them off before
          returning (see ../flux_kontext/flux_kontext.py).
        """
        if self.model.device == torch.device("cpu"):
            self.model.to(self.device_torch)

        # toolkit timestep (0..1000) -> our model's flow time in [0, 1].
        # WATCH OUT: every model has its own time convention. If the original
        # repo uses t=1 for clean images, flip it here (see
        # ../ideogram4/src/pipeline.py predict_velocity for an example).
        t01 = timestep.to(self.device_torch, dtype=torch.float32) / 1000.0

        # per-sample embed lists -> padded batch tensor + attention mask
        llm_features, text_mask = pad_prompt_embeds(
            text_embeddings.text_embeds, self.device_torch, self.torch_dtype
        )

        noise_pred = self.model(
            hidden_states=latent_model_input.to(self.device_torch, self.torch_dtype),
            timestep=t01,
            encoder_hidden_states=llm_features,
            attention_mask=text_mask,
        )
        return noise_pred

    def get_prompt_embeds(self, prompt) -> AdvancedPromptEmbeds:
        """Encode prompt text into whatever conditioning the model eats.

        Called for dataset captions (optionally cached to disk per caption),
        for sample prompts, and for the empty string (unconditional).

        in:  prompt  a str or list[str]
        out: AdvancedPromptEmbeds. Each key holds a LIST of tensors, one per
             prompt, each at its natural (unpadded) length. Padding to the
             batch max is deferred to get_noise_prediction / the pipeline,
             which keeps caches small and lets any prompts share a batch.

             Each per-prompt tensor MUST be 2D ``(L, D)`` -- BaseModel infers the
             text batch size from the list and only treats it as one-per-prompt
             when the tensors are 2D; a 3D per-prompt tensor is misread as an
             already-batched ``(B, L, D)`` and training fails with a latents-vs-
             text batch-size mismatch. If your conditioning has an extra axis
             (e.g. N stacked encoder layers -> ``(L, N, D)``), flatten it here
             (``(L, N*D)``) and restore it (``reshape(B, Lt, N, D)``) at the
             model call.

             You can store any number of keys (pooled embeds, image features,
             ...). If a key must keep its dtype when everything else is cast
             (masks, token ids), list it in ``embeds.frozen_dtype_keys``.

        NOTE: if you change how embeddings are computed after release, bump
        ``text_embedding_space_version`` (a property on BaseModel) to
        invalidate users' on-disk caches.
        """
        if isinstance(prompt, str):
            prompt = [prompt]

        # low_vram support: TE might be parked on CPU
        if self.text_encoder.device == torch.device("cpu"):
            self.text_encoder.to(self.device_torch)

        embeds_list = []
        for p in prompt:
            tokens = self.tokenizer(
                p,
                truncation=True,
                max_length=self.max_text_length,
                return_tensors="pt",
            ).to(self.text_encoder.device)
            # no padding: encode each prompt at its own length
            with torch.no_grad():
                output = self.text_encoder(**tokens, output_hidden_states=True)
            # (L, D) -- drop the batch dim, one tensor per prompt
            embeds_list.append(output.last_hidden_state[0].to(self.torch_dtype))

        return AdvancedPromptEmbeds(text_embeds=embeds_list)

    def get_loss_target(self, *args, **kwargs):
        """The ground-truth tensor the prediction is MSE'd against.

        kwargs: noise (B, C, h, w), batch (DataLoaderBatchDTO with .latents =
        the clean latents), timesteps. For flow matching the velocity target
        is noise - clean. Must be detached.
        """
        noise = kwargs.get("noise")
        batch = kwargs.get("batch")
        return (noise - batch.latents).detach()

    def condition_noisy_latents(
        self, latents: torch.Tensor, batch
    ) -> torch.Tensor:
        """Optional hook: modify noisy latents before the model sees them.

        Called every train step right after noise is added. This is THE hook
        for editing / inpainting / i2v models that feed reference latents in
        alongside the noisy target (the reference is concatenated here, then
        consumed -- and sliced off the prediction -- in get_noise_prediction).

        in:  latents (B, C, h, w) noisy latents
             batch    DataLoaderBatchDTO -- batch.control_tensor holds the
                      control image(s) as (B, 3, H, W) in [0, 1] when the
                      dataset config has a control_path
        out: latents, conditioned (return .detach()'d -- no grads here)

        This base text-to-image model needs nothing, so it passes through.
        Real examples: ../flux_kontext/flux_kontext.py (concat control latents
        as extra tokens), ../qwen_image/qwen_image_edit.py.
        """
        return latents

    # ------------------------------------------------------------------
    # VAE encode / decode
    # ------------------------------------------------------------------
    # Both are the example vae's own methods (src/vae.py), reached through
    # PhasedLoadMixin.encode_images / decode_latents, which supply residency
    # only. There is no example-side encode to keep in step with them: a vae
    # with unusual latents (normalization, patchified, video frames) writes
    # them on its own class, where every model declaring the id shares one
    # implementation and one cached latent format.

    # ------------------------------------------------------------------
    # Saving / bookkeeping
    # ------------------------------------------------------------------
    def get_model_has_grad(self):
        """True only if the base denoiser weights themselves require grad
        (full fine-tune). LoRA training: False. Used to save/restore device
        and grad state around sampling."""
        return False

    def get_te_has_grad(self):
        """Same as above for the text encoder."""
        return False

    def save_model(self, output_path, meta, save_dtype):
        """Save the FULL model (fine-tune checkpoints; LoRA saving is handled
        elsewhere and only consults convert_lora_weights_before_save).

        ``output_path`` is a directory (no extension). Save in whatever layout
        load_model can read back; include aitk_meta.yaml for provenance.
        """
        transformer: ExampleTransformer2DModel = unwrap_model(self.model)
        os.makedirs(os.path.join(output_path, "transformer"), exist_ok=True)
        state_dict = {
            k: v.clone().to("cpu", dtype=save_dtype)
            for k, v in transformer.state_dict().items()
        }
        save_file(
            state_dict, os.path.join(output_path, "transformer", "model.safetensors")
        )
        with open(os.path.join(output_path, "aitk_meta.yaml"), "w") as f:
            yaml.dump(meta, f)

    def get_base_model_version(self):
        """Free-form version string written into LoRA metadata so other tools
        can identify the base model family."""
        return "example.1"

    def get_transformer_block_names(self) -> Optional[List[str]]:
        """Attribute name(s) on self.model that hold the repeated transformer
        blocks (a ModuleList). Used for LoRA block targeting; must match the
        attribute in src/model.py."""
        return ["blocks"]

    # LoRA keys save with the ecosystem-standard ``diffusion_model.`` prefix
    # (ComfyUI convention) and load back to the internal ``transformer.``
    # prefix; see BaseModel.convert_lora_weights_before_save/load
    lora_keys_use_comfy_prefix = True

