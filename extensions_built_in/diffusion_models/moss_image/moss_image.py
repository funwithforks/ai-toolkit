"""MOSS-Image (OpenMOSS-Team/MOSS-Image-Base) for ai-toolkit -- text-to-image.

  - DiT: 42-block single-stream transformer, 3072 wide, 128 latent channels
    (``src/transformer.py``, Comfy-Org/MOSS-Image key layout)
  - VAE: FLUX.2, 2x2 packed and BatchNorm normalized -- the shared toolkit BFL
    ``AutoEncoder`` (``toolkit/models/v2/vae/flux2_kl``), not a moss copy. The
    shared comfy flux2-vae file (full decoder) replaces the vendor's
    small-decoder copy by default.
  - text: Qwen3.5-9B final-normed last hidden state of the chat-templated
    prompt (thinking on, generation prompt appended), one unpadded forward per
    prompt; stored per item as (L, 4096) in an ``AdvancedPromptEmbeds``.

This is the phased-loading, text-to-image import. The training process caches
target latents in phase 1 and prompt embeds in phase 2, so the VAE and text
encoder are never resident during training (see
docs/guide/phased-loading--component-phases.md). ``vae.encode`` already returns
the packed, BatchNorm-normalized latents the DiT consumes -- the latent itself
is the phase-1 cache artifact; every further transform of it lives behind
LoRA-targetable transformer weights, so there is no output-cache opportunity
here. Likewise the cache stays pre-refiner: the DiT's 2-block text refiner is
LoRA-targetable conditioning, and caching a trainable module's output is the
fibo2-rejected pattern (see fibo2's module header, same bar).

``model.name_or_path`` defaults to the vendor Base repo; the Comfy repack, a
local diffusers dir or a ``.safetensors`` path (local or ``org/repo/file``)
also load. Components resolve through their classes' comfy-candidates
registration -- the shipped int8 ConvRot files are the trained identity (the
LTX-2.5 / Krea-2 / fibo2 convention), the sft/bf16 twins are unregistered and
reachable only through ``transformer_path`` / ``text_encoder_path`` /
``vae_path`` overrides. All fetches land in the ComfyUI layout under
MODELS_PATH, never the HF cache; the tokenizer comes from the vendor repo's
tokenizer/ folder (the thinking chat template is the vendor's).

Flow convention matches ai-toolkit exactly (t=1 noise -> t=0 clean,
target = noise - clean), so no time flips anywhere.

EDIT IS NOT WIRED HERE. Upstream trains this identity as an edit model by
sending references through the Qwen3.5 vision tower (prompt-side) and as clean
t=0 VAE latents after the text (the DiT path, ported intact in
``src/transformer.py``). That needs ``Qwen35VLEncoder`` + an image processor
resident in phase 2 and per-item reference latents cached in phase 1 keyed on
control image + bucket size + match_target_res -- the fibo2 EDIT TODO shape
with upstream's per-step VAE re-encode replaced by phase-1 caching. Until
then every control/edit dataset is refused loudly. Upstream's edit machinery:
``moss_image.py`` (``is_edit``, ``_encode_with_images``, ``_encode_ref_latents``,
``_prep_vlm_images``, ``_batch_ref_latents``) and upstream's
``Qwen35VLEncoder`` + ``patch_qwen_vl_patch_embed``.
"""

import os
from typing import TYPE_CHECKING, List, Optional

import torch
from safetensors import safe_open
from safetensors.torch import load_file
from transformers import Qwen2Tokenizer

from toolkit.accelerator import unwrap_model
from toolkit.advanced_prompt_embeds import AdvancedPromptEmbeds
from toolkit.config_modules import GenerateImageConfig, ModelConfig
from toolkit.metadata import get_meta_for_safetensors
from toolkit.models.base_model import BaseModel
from toolkit.network_mixins import set_multiplier_rows
from toolkit.models.phased_load import PhasedLoadMixin
from toolkit.models.v2.resolver import resolve_comfy_file
from toolkit.models.v2.vae.flux2_kl import AutoEncoder, convert_diffusers_state_dict
from toolkit.samplers.custom_flowmatch_sampler import (
    CustomFlowMatchEulerDiscreteScheduler,
)

from .src.pipeline import (
    DEFAULT_SHIFT,
    DEFAULT_UNCOND_SKIP_LAYERS,
    MossImagePipeline,
    stack_text_embeds,
)
from .src.text_encoder import (
    QWEN35_9B_REPO,
    MAX_TEXT_TOKENS,
    MossQwen35Encoder,
    encode_text_batch,
    encode_text_batch_packed,
)
from .src.transformer import COMFY_REPO, VENDOR_BASE_REPO, MossImageTransformer
from .src.flash_varlen import varlen_ok

if TYPE_CHECKING:
    from toolkit.data_transfer_object.data_loader import DataLoaderBatchDTO


HF_TOKEN = os.getenv("HF_TOKEN", None)

DEFAULT_REPO = VENDOR_BASE_REPO
# the shared FLUX.2 packed AE in the comfy layout (same file the flux2 family
# and fibo2 resolve), loaded by the shared v2 class
VAE_FILE = "vae/flux2-vae.safetensors"
# a local vendor checkpoint dir carries its own (small-decoder, diffusers
# layout) copy under this name
VENDOR_VAE_FILE = "vae/diffusion_pytorch_model.safetensors"

scheduler_config = {
    "num_train_timesteps": 1000,
    "use_dynamic_shifting": False,
}


def _is_hub_repo(path: str) -> bool:
    return (
        not os.path.exists(path)
        and not path.endswith(".safetensors")
        and path.count("/") == 1
    )


class MossImageModel(PhasedLoadMixin, BaseModel):
    arch = "moss_image"
    display_name = "MOSS-Image"

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
        self.target_lora_modules = ["MossImageTransformer"]
        self.supports_model_paths = True
        # phase-2 batched caption caching (declare token counting + bytes/token)
        self.supports_batched_text_cache = True
        # the packed (padding-free) encode path is installed by the TE
        # itself; captions pack by REAL tokens instead of padded ones
        self.supports_packed_text_cache = True
        self.use_old_lokr_format = False
        # the base-model converter swaps transformer. <-> diffusion_model. on
        # save/load, matching the ComfyUI key layout
        self.lora_keys_use_comfy_prefix = True
        self.latent_space_version = "moss_image"

        kw = self.model_config.model_kwargs
        self.max_text_length = int(kw.get("max_text_length", MAX_TEXT_TOKENS))
        self.sample_shift = float(kw.get("sample_shift", DEFAULT_SHIFT))
        self.uncond_skip_layers = tuple(
            kw.get("uncond_skip_layers", DEFAULT_UNCOND_SKIP_LAYERS) or ()
        )

    @staticmethod
    def get_train_scheduler():
        return CustomFlowMatchEulerDiscreteScheduler(
            shift=DEFAULT_SHIFT, **scheduler_config
        )

    def get_bucket_divisibility(self):
        return 16  # f8 conv + 2x2 packing

    # ------------------------------------------------------------------
    # Loading (PhasedLoadMixin contract: one component per phase)
    # ------------------------------------------------------------------
    def _load_transformer(self) -> MossImageTransformer:
        self.print_and_status_update("Loading transformer (MossImageTransformer)")
        # comfy-candidates door: the class registers the shipped int8 convrot
        # Base repack; transformer_path overrides, a local checkpoint dir
        # searches as an extra root, downloads land in MODELS_PATH (comfy
        # layout).
        dit_path = self._resolve_comfy_file("transformer", MossImageTransformer)
        self.print_and_status_update(f" - transformer weights: {dit_path}")
        # config is sniffed from the file (aitk_config_from_state_dict); the
        # mixin loader reads the int8 file and attaches the convrot groups as
        # shipped (zero requantization). Split-qkv community files are fused
        # onto this layout by convert_state_dict_on_load. Placement /
        # quantization policy after this runs in aitk_post_load (the mixin).
        transformer = MossImageTransformer.load_model(
            dit_path,
            dtype=self.torch_dtype,
            aitk_device=None if self.model_config.low_vram else self.device_torch,
        )
        return transformer

    def _load_text_encoder(self):
        te_path = self.model_config.te_name_or_path
        if te_path is None:
            # comfy-candidates door: the package's int8 convrot repack of the
            # 9B encoder (text_encoder_path overrides through the door).
            te_path = self._resolve_comfy_file("text_encoder", MossQwen35Encoder)
        self.print_and_status_update(f"Loading Qwen3.5 text encoder from {te_path}")
        # text-only conditioning (t2i build): the lm_head, mtp head and vision
        # tower are dropped on load. See the module-header EDIT TODO.
        text_encoder = MossQwen35Encoder.load_model(
            te_path,
            dtype=self.torch_dtype,
            subfolder="",
            # the comfy file is weights-only; the public Qwen repo supplies the
            # model config
            config_path=QWEN35_9B_REPO,
            # as-shipped streaming onto the gpu (no cpu staging); low_vram parks
            # the encoder on cpu the old way
            aitk_device=None if self.model_config.low_vram else self.te_device_torch,
        )
        text_encoder.eval()
        text_encoder.requires_grad_(False)
        tokenizer = self._load_tokenizer()
        return tokenizer, text_encoder, {}

    def _load_tokenizer(self):
        # from the vendor repo's tokenizer/ folder: the thinking chat template
        # (and vocab) is MOSS's own, not the stock Qwen3.5 repo's
        base = self.model_config.name_or_path or DEFAULT_REPO
        source = DEFAULT_REPO
        if os.path.isdir(os.path.join(base, "tokenizer")) or (
            _is_hub_repo(base) and base != COMFY_REPO
        ):
            source = base
        tokenizer = MossQwen35Encoder.load_tokenizer(
            source,
            subfolder="tokenizer",
            # the declared class: AutoTokenizer also resolves a config.json the
            # folder lacks, which fails offline
            tokenizer_cls=Qwen2Tokenizer,
            require_complete=True,
        )
        if tokenizer.chat_template is None or len(tokenizer) < 1000:
            raise ValueError(f"incomplete MOSS-Image tokenizer from {source}")
        return tokenizer

    def _load_vae(self) -> AutoEncoder:
        # The FLUX.2 packed AE: the shared v2 class (toolkit/models/v2/vae/
        # flux2_kl), not a moss copy. Phase-1 component: it caches latents
        # (encode() output is already packed + BN-normalized, the exact form
        # the DiT consumes), is unloaded at the phase boundary, and sampling
        # rounds reload it through this hook.
        self.print_and_status_update("Loading image VAE")
        base = self.model_config.name_or_path or DEFAULT_REPO
        vae_path = self.model_config.model_kwargs.get("vae_path", None)
        if vae_path is None and os.path.isfile(os.path.join(base, VENDOR_VAE_FILE)):
            # a local vendor checkpoint dir carries its own copy (diffusers
            # layout, small decoder); it wins over the shared comfy file
            vae_path = os.path.join(base, VENDOR_VAE_FILE)
        if vae_path is None:
            # generic resolver door: downloads come from the package repo
            # (which ships flux2-vae.safetensors), local comfy-layout files and
            # a local checkpoint dir win first
            vae_path = resolve_comfy_file(
                VAE_FILE,
                COMFY_REPO,
                override_path=self.model_config.model_kwargs.get("vae_path", None),
                extra_roots=[base] if os.path.isdir(base) else None,
                hf_token=HF_TOKEN,
                status_fn=self.print_and_status_update,
            )
        with safe_open(vae_path, framework="pt") as f:
            diffusers_layout = any(k.startswith("encoder.down_blocks.") for k in f.keys())
        if diffusers_layout:
            # the vendor copy (the BFL weights rounded to bf16): convert first
            state_dict = convert_diffusers_state_dict(load_file(vae_path))
            vae = AutoEncoder.load_from_state_dict(state_dict, self.vae_torch_dtype)
        else:
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
        return MossImagePipeline(self)

    def load_transformer(self):
        # phased-load contract: quantize/attach (cr8_qdata) runs inside
        # super(); per-module quantization flags must be applied after it.
        super().load_transformer()
        # model_kwargs int8_bwd (default off): route one weight-shape class
        # of quantized linears ("fc2" = in>out, "fc1" = in<out; true picks
        # fc2) onto the int8 STE input-gradient backward, fused CUTLASS NT
        # arm, forcing past the shared op's size gate (sized for small-m
        # models; MOSS batches are large enough to amortize). Cost and
        # quality are per-model decisions: enable only after a live
        # measurement on this model.
        _bwd = self.model_config.model_kwargs.get("int8_bwd", False)
        if _bwd is True:
            _bwd = "fc2"
        if _bwd in ("fc1", "fc2"):
            n = 0
            for m in self.model.modules():
                if getattr(m, "cr8_qdata", None) is None:
                    continue
                if (m.in_features > m.out_features) == (_bwd == "fc2"):
                    m.cr8_bwd_mode = "int8_force"
                    # fused NT arm; the value only needs != "off" -- layer
                    # selection is done right here
                    m.cr8_cutlass_bwd = _bwd
                    n += 1
            if n:
                self.print_and_status_update(
                    f"  - int8 STE dx ({_bwd}-class, gate-forced, FUSED "
                    f"CUTLASS NT arm) on {n} quantized linears "
                    f"(model_kwargs int8_bwd=false restores the eager backward)"
                )

        # int8 FORWARD arms (independent of the dx switch above). Both are
        # shipped in the shared convrot op, class-default off; each is a
        # per-model decision. Only reach modules with a lora folded into the
        # dequant epilogue (has_lora), which is MOSS's shipped path.
        #   epi_dot      -- tensor-core (tl.dot) lora-fold epilogue instead
        #                   of the fp32 broadcast-and-sum (same values, <=1
        #                   bf16 ulp in the lora term).
        #   cutlass_gemm -- fused CUTLASS GEMM+dequant forward engine; note
        #                   it moves the lora fold OUT of the epilogue to a
        #                   rank addmm, so it is tested separately.
        if self.model_config.model_kwargs.get("int8_epi_dot", False):
            n = 0
            for m in self.model.modules():
                if getattr(m, "cr8_qdata", None) is not None:
                    m.cr8_epi_dot = True
                    n += 1
            if n:
                self.print_and_status_update(
                    f"  - int8 lora-fold epilogue via the tl.dot kernel on "
                    f"{n} quantized linears (model_kwargs int8_epi_dot=false "
                    f"for the elementwise kernel)"
                )
        if self.model_config.model_kwargs.get("int8_cutlass_gemm", False):
            from toolkit.util.convrot_quant import _get_cutlass_int8

            if _get_cutlass_int8() is not None:
                n = 0
                for m in self.model.modules():
                    if getattr(m, "cr8_qdata", None) is not None:
                        m.cr8_cutlass_gemm = True
                        n += 1
                if n:
                    self.print_and_status_update(
                        f"  - int8 forward GEMM via fused CUTLASS on {n} "
                        f"quantized linears (model_kwargs "
                        f"int8_cutlass_gemm=false for torch._int_mm)"
                    )

    # ------------------------------------------------------------------
    # Embed cache format guard (the caption cache stores encoder output)
    # ------------------------------------------------------------------
    def embed_file_valid(self, path: str) -> bool:
        # The caption cache stores (L, 4096) final-normed states wrapped as
        # AdvancedPromptEmbeds -- what the training step consumes. A plain
        # PromptEmbeds cache (or any other class) is a different space and
        # must re-encode through the text encoder phase. Only damaged or
        # unreadable files count as invalid; both callers check existence
        # first, so a non-file error must propagate (a swallowed bug here
        # would silently re-encode every cache).
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
    # Batched caption caching (phase 2): the shared cache pass batches the
    # plain-text targets of models declaring supports_batched_text_cache.
    # Toolkit side: toolkit/text_cache_batching.py.
    # ------------------------------------------------------------------
    def count_text_tokens(self, captions):
        # token count of the EXACT presentation the tower sees (chat
        # template applied), capped at the encoder's truncation length
        enc = self.tokenizer(
            [self._chat(c) for c in captions], add_special_tokens=False
        )
        return [min(len(ids), self.max_text_length) for ids in enc["input_ids"]]

    def text_cache_bytes_per_token(self):
        # marginal live activation bytes per PADDED token, measured on this
        # tower from the slope of peak allocations between batched cache
        # groups on real captions: ~140,000 B/token (two independent
        # slopes: 136k, 140k). Round up to 150k for measurement precision.
        # (The old 1,520,000 came from the activation probe's single-slope
        # fit dominated by its L=4096 cells -- a long-sequence regime real
        # captions never run in; it over-reserved ~10x and capped batches
        # for no reason.)
        # Per-job override: model_kwargs.text_cache_bytes_per_token
        # (or 0 to disable the pass).
        override = self.model_config.model_kwargs.get("text_cache_bytes_per_token")
        if override is not None:
            return int(override)
        return 150_000

    # ------------------------------------------------------------------
    # Generation (training previews)
    # ------------------------------------------------------------------
    def get_generation_pipeline(self):
        return MossImagePipeline(self)

    def generate_single_image(
        self,
        pipeline: MossImagePipeline,
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

        # EDIT TODO: sample-time references (gen_config.ctrl_img*) ride both
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
                "MOSS-Image edit/reference sampling is not wired in this t2i build; "
                "remove ctrl_img from the sample prompt."
            )

        # sampling never packs (the pipeline is dense bs1/bs2-cond): start
        # the step with no packed map, so a stale training map can never
        # hijack the LoRA multiplier during generation
        set_multiplier_rows(())
        result = pipeline(
            conditional_embeds=conditional_embeds,
            unconditional_embeds=unconditional_embeds,
            height=gen_config.height,
            width=gen_config.width,
            num_inference_steps=gen_config.num_inference_steps,
            guidance_scale=gen_config.guidance_scale,
            latents=gen_config.latents,
            generator=generator,
            shift=self.sample_shift,
            uncond_skip_layers=self.uncond_skip_layers,
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

        # Every forward decides its own LoRA multiplier rows: clear first,
        # publish below iff this is a packed (ragged) call. The publish
        # OUTLIVES the call on purpose -- gradient checkpointing replays
        # block forwards during loss.backward(), which happens outside this
        # function (toolkit/guidance.py) -- the next entry clears or
        # overwrites, so no reader ever sees another stream's map.
        set_multiplier_rows(())

        with torch.no_grad():
            # toolkit timestep (0..1000, 1000 = pure noise) -> the DiT's own t,
            # same convention, straight through (no /1000, no flip).
            t = timestep.to(device, dtype=torch.float32).reshape(-1)
            if t.shape[0] != batch_size:
                t = t.expand(batch_size)

            # Training is t2i-only. Edit conditioning needs a VAE encode per
            # batch (references as clean latents) and the vision tower in the
            # prompt, but neither is resident during training (see the phased
            # load and the module-header EDIT TODO). Refuse control datasets
            # loudly instead of silently ignoring their controls.
            if batch is not None and (
                batch.control_tensor_list is not None or batch.control_tensor is not None
            ):
                raise NotImplementedError(
                    "MOSS-Image training with control/edit images is not supported "
                    "in this build. Remove the control datasets."
                )

        if batch_size > 1 and varlen_ok():
            # packed flash-varlen is the standard for padded training
            # batches: never pad. The ragged per-item states ride
            # straight into the DiT (transformer._forward_packed): live
            # rows only, no mask. bs1 never pads -> untouched dense
            # dispatch; MOSS_VARLEN_ATTN=0 keeps the stacked-mask path for
            # parity A/B.
            # The per-sample LoRA multiplier is (B,)-indexed and the packed
            # streams are ragged (interleave-by-B assumes uniform rows):
            # publish the row->sample maps for both streams; LoRA modules
            # expand per row, picking the map matching their own input rows
            # (network_mixins.set_multiplier_rows -- a registry, because the
            # modules' network weakrefs may not match sd.network's instance).
            num_image = int(latent_model_input.shape[-2]) * int(
                latent_model_input.shape[-1]
            )
            counts = torch.tensor([len(e) for e in embeds_list], device=device)
            bidx = torch.arange(batch_size, device=device)
            set_multiplier_rows((
                bidx.repeat_interleave(counts + num_image),  # main stream
                bidx.repeat_interleave(counts),              # text stream
            ))
            return self.transformer(
                latent_model_input.to(device, dtype), t, embeds_list
            )
        embeds, mask = stack_text_embeds(embeds_list, device, dtype)
        return self.transformer(
            latent_model_input.to(device, dtype), t, embeds, mask
        )

    def _chat(self, content: str) -> str:
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )

    @torch.no_grad()
    def get_prompt_embeds(self, prompt, control_images=None) -> AdvancedPromptEmbeds:
        if control_images is not None:
            raise NotImplementedError(
                "MOSS-Image conditioning in this build is text-only; reference "
                "images are not supported (see the module-header EDIT TODO)"
            )
        if isinstance(prompt, str):
            prompt = [prompt]
        elif prompt is None or (isinstance(prompt, (list, tuple)) and len(prompt) == 0):
            # unset negatives arrive as None/empty; the empty chat is the
            # unconditional used for classifier-free guidance
            prompt = [""]
        if self.text_encoder.device == torch.device("cpu"):
            self.text_encoder.to(self.device_torch)

        states = (
            encode_text_batch_packed(
                self.text_encoder,
                self.tokenizer,
                [self._chat(p) for p in prompt],
                self.max_text_length,
                self.device_torch,
            )
            if getattr(self.text_encoder, "_qwen35_packed_ready", False)
            else encode_text_batch(
                self.text_encoder,
                self.tokenizer,
                [self._chat(p) for p in prompt],
                self.max_text_length,
                self.device_torch,
            )
        )
        return AdvancedPromptEmbeds(text_embeds=[s.to(self.torch_dtype) for s in states])

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
        """Images (C, H, W) in [-1, 1] -> normalized latents (B, 128, H/16, W/16)."""
        if device is None:
            device = self.vae_device_torch
        if dtype is None:
            dtype = self.vae_torch_dtype
        if self.vae.device == torch.device("cpu"):
            self.vae.to(device)
        batch = torch.stack(
            [img.to(device, dtype=self.vae.dtype) for img in image_list]
        )
        return self.vae.encode(batch).to(device, dtype=dtype)

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
        if not output_path.endswith(".safetensors"):
            output_path = output_path + ".safetensors"
        transformer: MossImageTransformer = unwrap_model(self.model)
        transformer.save_model(
            output_path,
            dtype=save_dtype,
            metadata=get_meta_for_safetensors(meta, name=self.arch),
            quantized=False,
        )

    # LoRA keys: the module fuses q/k/v (like the Comfy files), but ComfyUI's
    # MOSS LoRA convention addresses the separate projections (mapped onto
    # fused files by row slices, and the only names unfused files have).
    # Saved as q/k/v, fused back on load.
    _QKV_SPLIT_ROWS = ("lora_B.weight", "lora_up.weight")
    _QKV_SHARED = ("lora_A.weight", "lora_down.weight", "alpha")

    def convert_lora_weights_before_save(self, state_dict):
        state_dict = super().convert_lora_weights_before_save(state_dict)
        out = {}
        for key, value in state_dict.items():
            base, sep, suffix = key.partition(".proj.qkv.")
            if not sep:
                out[key] = value
            elif suffix in self._QKV_SPLIT_ROWS:
                for name, part in zip(("q", "k", "v"), value.chunk(3, dim=0)):
                    out[f"{base}.proj.{name}.{suffix}"] = part.contiguous()
            elif suffix in self._QKV_SHARED:
                for name in ("q", "k", "v"):
                    out[f"{base}.proj.{name}.{suffix}"] = value.clone()
            else:
                out[key] = value  # e.g. LoKr: kept fused (maps on fused files)
        return out

    def convert_lora_weights_before_load(self, state_dict):
        state_dict = super().convert_lora_weights_before_load(state_dict)
        out = dict(state_dict)
        down_keys = [
            k for k in state_dict
            if k.endswith((".proj.q.lora_A.weight", ".proj.q.lora_down.weight"))
        ]
        for q_down in down_keys:
            base, _, down_suffix = q_down.partition(".proj.q.")
            up_suffix = "lora_B.weight" if down_suffix == "lora_A.weight" else "lora_up.weight"
            names = [f"{base}.proj.{p}." for p in ("q", "k", "v")]
            if not all(n + down_suffix in state_dict and n + up_suffix in state_dict for n in names):
                continue
            downs = [out.pop(n + down_suffix) for n in names]
            ups = [out.pop(n + up_suffix) for n in names]
            alphas = [out.pop(n + "alpha", None) for n in names]
            rank = downs[0].shape[0]
            if all(torch.equal(downs[0], d) for d in downs[1:]):
                # shared down projection (our saves): the fused rank-r LoRA
                down, up = downs[0], torch.cat(ups, dim=0)
            else:
                # independent q/k/v LoRAs: exact rank-3r block form
                down = torch.cat(downs, dim=0)
                up = torch.block_diag(*ups)
            out[f"{base}.proj.qkv.{down_suffix}"] = down
            out[f"{base}.proj.qkv.{up_suffix}"] = up
            if alphas[0] is not None:
                # keep alpha / rank, the per-projection scale
                out[f"{base}.proj.qkv.alpha"] = alphas[0] * (down.shape[0] / rank)
        return out

    def get_base_model_version(self):
        return "moss_image"

    def get_transformer_block_names(self) -> Optional[List[str]]:
        return MossImageTransformer.get_transformer_block_names()

    def get_quantization_exclude_modules(self):
        return MossImageTransformer.get_quantization_exclude_modules()
