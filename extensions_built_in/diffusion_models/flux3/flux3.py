"""FLUX.3 (small, ~5.5B) for ai-toolkit.

A joint flow-matching DiT (``src/transformer.py``) that trains t2i and
samples t2i + i2i (reference images as clean conditioning tokens). Training
with control/edit datasets is not supported: training images are VAE-encoded
in phase 1 and cached, and the vae is never resident during training (see
docs/guide/phased-loading--component-phases.md).

  - image VAE: FLUX.2 packed AE, f16, 128 latent channels
  - text: Qwen3-VL-4B, 8 hidden layers stacked along the channel dim ->
    ``(B, L, 20480)``; the DiT's ``timesteps_ctx`` are zeros and there is no
    pooled vector. Embeds are stored per-sample as 2D ``(L, 20480)`` tensors
    in an ``AdvancedPromptEmbeds`` (padded to a multiple of 80 tokens -- the
    DiT has no text mask, pad tokens are part of the trained context).

``model.name_or_path`` is the package repo (default
``black-forest-labs/flux-3-image-klein-comfy``) or a local dir holding the
component files. The DiT and the text encoder resolve through their classes'
comfy-candidates registration -- the shipped int8 ConvRot repacks are the
trained identity (the Krea-2 / LTX-2.5 convention: a quantize: none run can
never rank a higher-precision twin first). The image VAE is the FLUX.2 packed
AE and loads through the shared ``toolkit/models/v2/vae/flux2_kl`` class.
All fetches land in the ComfyUI layout under MODELS_PATH, never the HF cache;
the ``transformer_path`` / ``text_encoder_path`` / ``vae_path`` model_kwargs
override any component. The step-distilled (``*_sd_*``) variants are not
registered and cannot be reached by a default run. Config, tokenizer and
processor for the encoder come from the public Qwen/Qwen3-VL-4B-Instruct repo.

Flow convention matches ai-toolkit exactly (t=1 noise -> t=0 clean,
target = noise - clean), so no time flips anywhere.
"""

import math
import os
from typing import TYPE_CHECKING, List, Optional

import torch
import torch.nn.functional as F
from PIL import Image
from safetensors.torch import save_file
from torchvision.transforms.functional import to_tensor

from toolkit.accelerator import unwrap_model
from toolkit.advanced_prompt_embeds import AdvancedPromptEmbeds
from toolkit.basic import flush
from toolkit.config_modules import GenerateImageConfig, ModelConfig
from toolkit.metadata import get_meta_for_safetensors
from toolkit.models.base_model import BaseModel
from toolkit.models.phased_load import PhasedLoadMixin
from toolkit.models.v2.resolver import resolve_comfy_file
from toolkit.models.v2.vae.flux2_kl import AutoEncoder
from toolkit.unloader import FakeTextEncoder
from toolkit.samplers.custom_flowmatch_sampler import (
    CustomFlowMatchEulerDiscreteScheduler,
)

from .src.pipeline import (
    DEFAULT_ALPHA,
    Flux3Pipeline,
    build_text_conditioning,
    pack_image_latents,
)
from .src.text_encoder import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_PAD_MULTIPLE,
    QWEN3_VL_4B_REPO,
    Flux3TextEncoder,
    encode_flux3_prompt,
)
from .src.transformer import Flux3Transformer, Flux3TransformerParams

if TYPE_CHECKING:
    from toolkit.data_transfer_object.data_loader import DataLoaderBatchDTO


HF_TOKEN = os.getenv("HF_TOKEN", None)
DEFAULT_REPO = "black-forest-labs/flux-3-image-klein-comfy"
# The package's VAE file in the comfy layout: the same BFL packed AE the flux2
# family ships, loaded by the shared v2 class. The DiT and TE files are
# registered as comfy candidates on their classes, not here.
VAE_FILE = "vae/flux2-vae.safetensors"

# Static flow-matching timeshift used for the training timestep/noise
# schedule (the package's sampler alpha).
DEFAULT_TIMESTEP_SHIFT = DEFAULT_ALPHA

scheduler_config = {
    "num_train_timesteps": 1000,
    "use_dynamic_shifting": False,
}


class Flux3Model(PhasedLoadMixin, BaseModel):
    arch = "flux3"
    display_name = "FLUX.3 Image"

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
        self.target_lora_modules = ["Flux3Transformer"]
        self.supports_model_paths = True
        self.use_old_lokr_format = False

        self.processor = None  # Qwen3-VL AutoProcessor
        self.max_text_length = int(
            self.model_config.model_kwargs.get("max_text_length", DEFAULT_MAX_TOKENS)
        )
        self.text_pad_multiple = int(
            self.model_config.model_kwargs.get("text_pad_multiple", DEFAULT_PAD_MULTIPLE)
        )
        # references are area-matched to the target by default; off = keep their
        # own size, only scaling down when larger than the target area
        self.match_target_res = bool(
            self.model_config.model_kwargs.get("match_target_res", True)
        )
        self.latent_space_version = "flux3_v1"

    @staticmethod
    def get_train_scheduler():
        return CustomFlowMatchEulerDiscreteScheduler(
            shift=DEFAULT_TIMESTEP_SHIFT, **scheduler_config
        )

    def get_bucket_divisibility(self):
        return 16  # f8 conv + 2x2 packing

    # ------------------------------------------------------------------
    # Package files
    # ------------------------------------------------------------------
    # Package files
    #
    # (none -- components resolve through the comfy-candidates door and the
    # generic resolver; see the loading section)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def _load_transformer(self) -> Flux3Transformer:
        self.print_and_status_update("Loading transformer (Flux3Transformer)")
        # comfy-candidates door: the class registers the shipped int8 convrot
        # repack; transformer_path overrides, a local package dir searches as
        # an extra root, downloads land in MODELS_PATH (comfy layout).
        dit_path = self._resolve_comfy_file("transformer", Flux3Transformer)
        # the mixin loader: WeightSource read, meta shell, convrot groups
        # attached as shipped (the registry carries no qtype: the int8 file
        # is the trained identity; placement/quantization policy after this
        # runs in aitk_post_load from PhasedLoadMixin). As-shipped mode
        # streams onto the gpu; low_vram keeps the legacy cpu parking.
        transformer = Flux3Transformer.load_model(
            dit_path,
            dtype=self.torch_dtype,
            config=Flux3TransformerParams(),
            aitk_device=None if self.model_config.low_vram else self.device_torch,
        )
        attn_backend = self.model_config.model_kwargs.get("attention_backend", None)
        if attn_backend is not None:
            transformer.set_attention_backend(attn_backend)
        flush()
        return transformer

    def _load_text_encoder(self):
        te_path = self.model_config.te_name_or_path
        if te_path is None:
            # comfy-candidates door: the package's convrot repack of the 4B
            # encoder (text_encoder_path overrides through the door).
            te_path = self._resolve_comfy_file("text_encoder", Flux3TextEncoder)
        self.print_and_status_update(f"Loading Qwen3-VL text encoder from {te_path}")
        text_encoder = Flux3TextEncoder.load_model(
            te_path,
            dtype=self.torch_dtype,
            subfolder="",
            # the comfy file is weights-only; the public repo supplies the
            # model config (and the processor/tokenizer below)
            config_path=QWEN3_VL_4B_REPO,
            # as-shipped streaming onto the gpu (no cpu staging); low_vram
            # parks the encoder on cpu the old way
            aitk_device=None if self.model_config.low_vram else self.te_device_torch,
        )
        processor_source = te_path if os.path.isdir(te_path) else None
        processor = Flux3TextEncoder.load_processor(processor_source)
        processor.tokenizer.padding_side = "right"
        # text-only conditioning -- drop the vision tower
        text_encoder.drop_vision_tower()
        text_encoder.eval()
        text_encoder.requires_grad_(False)
        flush()
        return processor.tokenizer, text_encoder, {"processor": processor}

    def _load_vae(self) -> AutoEncoder:
        # The FLUX.2 packed AE: the shared v2 class (toolkit/models/v2/vae/
        # flux2_kl), not a flux3 copy. Generic resolver door: vae_path
        # override, a local package dir as an extra root, a miss downloads
        # into MODELS_PATH/vae/ (comfy layout), never the HF cache.
        # Phase-1 component: it caches latents, is unloaded at the phase
        # boundary, and sampling rounds reload it through this hook.
        self.print_and_status_update("Loading image VAE")
        base = self.model_config.name_or_path or DEFAULT_REPO
        vae_path = resolve_comfy_file(
            VAE_FILE,
            base,
            override_path=self.model_config.model_kwargs.get("vae_path", None),
            extra_roots=[base] if os.path.isdir(base) else None,
            hf_token=HF_TOKEN,
            status_fn=self.print_and_status_update,
        )
        vae = AutoEncoder.load_model(
            vae_path,
            dtype=self.vae_torch_dtype,
            aitk_device=None if self.model_config.low_vram else self.vae_device_torch,
        )
        vae.eval()
        vae.requires_grad_(False)
        return vae

    def _build_pipeline(self):
        # the phased mixin drives load_transformer/load_text_encoder/load_vae
        # (training phases, or all three at once for inference) and wires the
        # scheduler + pipeline + aitk_post_load placement per model_config.
        return Flux3Pipeline(self)

    # ------------------------------------------------------------------
    # Generation (training previews)
    # ------------------------------------------------------------------
    def get_generation_pipeline(self):
        return Flux3Pipeline(self)

    def generate_single_image(
        self,
        pipeline: Flux3Pipeline,
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

        # cond and uncond were encoded separately, so their pad-to-80 windows
        # can differ. The DiT attends to real pad-token states, not zeros, so
        # re-encode the shorter prompt at the longer window when the TE is up
        # (cached embeddings: the TE is an unloaded stub; the shorter side is
        # zero-padded in build_text_conditioning instead)
        if (
            unconditional_embeds is not None
            and self.text_encoder is not None
            and not isinstance(self.text_encoder, FakeTextEncoder)
        ):
            cond_len = conditional_embeds.text_embeds[0].shape[0]
            uncond_len = unconditional_embeds.text_embeds[0].shape[0]
            if cond_len != uncond_len:
                target = max(cond_len, uncond_len)
                if cond_len < target:
                    conditional_embeds = self._encode_at_length(gen_config.prompt, target)
                else:
                    unconditional_embeds = self._encode_at_length(
                        gen_config.negative_prompt or "", target
                    )

        # references: ctrl_img_1 mirrors ctrl_img, so dedupe by path order
        ctrl_paths = []
        for path in (
            gen_config.ctrl_img,
            gen_config.ctrl_img_1,
            gen_config.ctrl_img_2,
            gen_config.ctrl_img_3,
        ):
            if path is not None and path not in ctrl_paths:
                ctrl_paths.append(path)
        target_pixels = gen_config.width * gen_config.height
        ctrl_imgs = [
            self._fit_reference(
                to_tensor(Image.open(path).convert("RGB")).unsqueeze(0), target_pixels
            )
            for path in ctrl_paths
        ]

        alpha = self.model_config.model_kwargs.get("sample_alpha", None)
        alpha = float(alpha) if alpha is not None else None

        result = pipeline(
            conditional_embeds=conditional_embeds,
            unconditional_embeds=unconditional_embeds,
            height=gen_config.height,
            width=gen_config.width,
            num_inference_steps=gen_config.num_inference_steps,
            guidance_scale=gen_config.guidance_scale,
            latents=gen_config.latents,
            generator=generator,
            ctrl_imgs=ctrl_imgs,
            alpha=alpha,
        )
        return result[0]

    def _fit_reference(self, img: torch.Tensor, target_pixels: int) -> torch.Tensor:
        """Size a (1, C, H, W) reference for the DiT. References keep their own
        aspect ratio on their own position grid; match_target_res scales the
        area to the target's, otherwise the image only scales down when it is
        larger than the target area. Snapped to the 16-px grid."""
        sc = self.get_bucket_divisibility()
        h, w = img.shape[-2], img.shape[-1]
        if self.match_target_res or h * w > target_pixels:
            ratio = h / w
            new_h = math.sqrt(target_pixels * ratio)
            new_w = new_h / ratio
        else:
            new_h, new_w = float(h), float(w)
        new_h = max(sc, int(round(new_h / sc)) * sc)
        new_w = max(sc, int(round(new_w / sc)) * sc)
        if (new_h, new_w) != (h, w):
            img = F.interpolate(img.float(), size=(new_h, new_w), mode="bilinear")
        return img

    def _encode_at_length(self, prompt: str, length: int) -> AdvancedPromptEmbeds:
        if self.text_encoder.device == torch.device("cpu"):
            self.text_encoder.to(self.device_torch)
        ctx = encode_flux3_prompt(
            self.text_encoder,
            self.processor,
            [prompt],
            max_length=length,
            dtype=self.torch_dtype,
            pad_multiple=None,
        )
        return AdvancedPromptEmbeds(text_embeds=[ctx[0]])

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

        with torch.no_grad():
            # toolkit timestep (0..1000, 1000 = pure noise) -> flow time in [0, 1]
            t = timestep.to(device, dtype=torch.float32) / 1000.0
            if t.dim() == 0:
                t = t.unsqueeze(0)
            if t.shape[0] != batch_size:
                t = t.expand(batch_size)

            ctx, ctx_ids, ts_ctx = build_text_conditioning(
                text_embeddings.text_embeds, device, dtype
            )

            latent_model_input = latent_model_input.to(device, dtype)
            _, _, h_lat, w_lat = latent_model_input.shape
            tokens, ids = pack_image_latents(latent_model_input)
            model_kwargs = {
                "x_image": tokens.to(dtype),
                "x_image_ids": ids,
                "x_image_timesteps": t.to(dtype)[:, None]
                * torch.ones(tokens.shape[:2], device=device, dtype=dtype),
            }

            # Training is t2i-only. Reference conditioning needs a VAE encode
            # per batch, but the vae is never resident during training (the
            # phase-1 caches cover every training image; see
            # BaseSDTrainProcess's phased load). Refuse edit datasets loudly
            # instead of silently ignoring their controls.
            if batch is not None and (
                batch.control_tensor_list is not None
                or batch.control_tensor is not None
            ):
                raise NotImplementedError(
                    "FLUX.3 training with control/edit images is not supported. "
                    "Reference conditioning is sampling-only; remove the control "
                    "datasets (refs may still ride on sample prompts)."
                )

        pred = self.transformer(
            ctx=ctx,
            ctx_ids=ctx_ids,
            timesteps_ctx=ts_ctx,
            **model_kwargs,
        )
        out = pred["x_image"]
        return out.reshape(batch_size, h_lat, w_lat, -1).permute(0, 3, 1, 2)

    def get_prompt_embeds(self, prompt, control_images=None) -> AdvancedPromptEmbeds:
        if control_images is not None:
            raise NotImplementedError(
                "FLUX.3 conditioning is text-only; reference images are VAE "
                "tokens, not text embeddings"
            )
        if isinstance(prompt, str):
            prompt = [prompt]
        elif prompt is None or (isinstance(prompt, (list, tuple)) and len(prompt) == 0):
            # unset negatives arrive as None/empty; the empty string is the
            # trained unconditional context
            prompt = [""]
        if self.text_encoder.device == torch.device("cpu"):
            self.text_encoder.to(self.device_torch)

        ctx = encode_flux3_prompt(
            self.text_encoder,
            self.processor,
            prompt,
            max_length=self.max_text_length,
            dtype=self.torch_dtype,
            pad_multiple=self.text_pad_multiple,
        )
        # one 2D (L, 20480) tensor per batch item
        return AdvancedPromptEmbeds(text_embeds=[ctx[i] for i in range(ctx.shape[0])])

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
    @torch.no_grad()
    def encode_images(self, image_list, device=None, dtype=None):
        """Images (C, H, W) in [-1, 1] -> latents (B, 128, H/16, W/16).

        Accepts a list of per-item tensors or a stacked batch tensor.
        """
        if device is None:
            device = self.vae_device_torch
        if dtype is None:
            dtype = self.vae_torch_dtype

        if self.vae.device == torch.device("cpu"):
            self.vae.to(device)

        vae_dtype = self.vae.dtype
        batch = torch.stack([img.to(device, dtype=vae_dtype) for img in image_list])
        latents = self.vae.encode(batch)
        return latents.to(device, dtype=dtype)

    @torch.no_grad()
    def decode_latents(self, latents: torch.Tensor, device=None, dtype=None):
        if device is None:
            device = self.vae_device_torch
        if dtype is None:
            dtype = self.vae_torch_dtype
        if self.vae.device == torch.device("cpu"):
            self.vae.to(device)
        images = self.vae.decode(latents.to(device, self.vae.dtype))
        return images.to(device, dtype)

    # ------------------------------------------------------------------
    # Saving / bookkeeping
    # ------------------------------------------------------------------
    def save_model(self, output_path, meta, save_dtype):
        from toolkit.util.quantize import dequantize_if_quantized

        if not output_path.endswith(".safetensors"):
            output_path = output_path + ".safetensors"
        transformer: Flux3Transformer = unwrap_model(self.model)
        state_dict = transformer.state_dict()
        save_dict = {}
        for k, v in state_dict.items():
            save_dict[k] = (
                dequantize_if_quantized(v).clone().to("cpu", dtype=save_dtype)
            )
        meta = get_meta_for_safetensors(meta, name=self.arch)
        save_file(save_dict, output_path, metadata=meta)

    def get_base_model_version(self):
        return "flux3"

    def get_transformer_block_names(self) -> Optional[List[str]]:
        return Flux3Transformer.get_transformer_block_names()

    def get_quantization_exclude_modules(self):
        # sensitive small modules kept in full precision (fnmatch patterns):
        #   emb_in*/txt_in       - token input projections
        #   time_in*             - timestep embedder; feeds every modulation
        #   *modulations*        - per-stream adaLN modulations
        #   final_layer*         - modulated output projections
        return [
            "emb_in*",
            "txt_in",
            "time_in*",
            "*modulations*",
            "final_layer*",
        ]

    # same as krea2 / flux2: the base-model converter swaps
    # transformer. <-> diffusion_model. on save/load
    lora_keys_use_comfy_prefix = True
