"""Iris3BModel -- Sperid Labs' Iris-3B pixel-space text-to-image DiT
(https://huggingface.co/speridlabs/iris-3b) on the phased loading contract.

  - **Pixel space, no VAE.** The transformer denoises raw RGB (``in_channels=3``,
    ``patch_size=16``) through a 24-block hybrid trunk (8 dual-stream MM-DiT +
    16 single-stream) and a 4-block per-pixel PiT head. ``_load_vae`` returns a
    ``FakeVAE`` (identity): the toolkit's "latents" are the image in [-1, 1] and
    the latent caches store pixels through the vae phase unchanged. Buckets snap
    to 16 px. Architecture vendored in ``src/dit.py``.
  - **v-prediction rectified flow, shift 4.** Velocity = noise - clean on
    ``x_t = (1 - t) * clean + t * noise`` -- ai-toolkit's own convention, so no
    flip/negation (the process default flow-matching target is already the
    reference's). The model is conditioned on *shifted* model time
    ``1000 * sigma'``; ``train.timestep_type: shift`` on the arch-default
    scheduler (shift 4, the release schedule) reproduces the reference training
    distribution and the toolkit ``timestep`` feeds the model as-is.
  - **Qwen3-VL-4B-Instruct text encoder, 12 stacked hidden layers.** Prompts go
    through the Iris chat template and are padded to a fixed 300 tokens
    (``src/text_encoder.py``); the DiT's layerwise-attention adapter pools the
    layer stack. Encoder weights resolve through the shared
    ``Qwen3VLTextEncoder`` comfy candidates (same door Krea 2 uses; the
    ``text_encoder_path`` model_kwarg overrides). Embeds are stored per prompt
    as rank-2 ``(300, 12 * 2560)`` tensors plus their ``(300,)`` attention mask
    in an ``AdvancedPromptEmbeds`` -- the same flattened-layer convention Krea 2
    uses, so the caption/fixed caches and batching work untouched.
  - **Previews** use the reference FlowDPM-Solver++ (order 2) sampler
    (``src/pipeline.py``), non-staged: there is no real vae to page, "decode" is
    the identity FakeVAE followed by a clamp.

``name_or_path`` is the hub repo (``speridlabs/iris-3b``), a local export dir
(``model.safetensors`` + ``config.yaml``), or a ``.safetensors`` file. Only the
root text-to-image checkpoint is supported; the repo's ``depth/`` and
``upscaler/`` variants are out of scope. Weight resolution is MODELS_PATH-first
(comfy-layout ``diffusion_models/``); nothing is fetched into the HF cache. A
``config.yaml`` is read when it sits next to a local checkpoint; a hub-only run
uses the shipped defaults (which are the release config), refined by
``model_kwargs``: ``flow_shift`` (sampling schedule), ``sample_order``,
``cfg_interval``, ``checkpoint_filename``, ``model_config`` (arch overrides).
"""

import os
from typing import List, Optional

import torch
import yaml
from safetensors.torch import save_file
from transformers import AutoTokenizer

from toolkit.accelerator import unwrap_model
from toolkit.advanced_prompt_embeds import AdvancedPromptEmbeds
from toolkit.basic import flush
from toolkit.config_modules import GenerateImageConfig, ModelConfig
from toolkit.metadata import get_meta_for_safetensors
from toolkit.models.base_model import BaseModel
from toolkit.models.FakeVAE import FakeVAE
from toolkit.models.phased_load import PhasedLoadMixin
from toolkit.models.v2.resolver import resolve_component_file
from toolkit.models.v2.text_encoders.qwen3_vl import Qwen3VLTextEncoder
from toolkit.samplers.custom_flowmatch_sampler import (
    CustomFlowMatchEulerDiscreteScheduler,
)

from .src.dit import IrisConfig, IrisDiT
from .src.pipeline import Iris3BPipeline
from .src.text_encoder import (
    DEFAULT_HIDDEN_LAYERS,
    DEFAULT_TEXT_LEN,
    IrisPromptEncoder,
)

HF_TOKEN = os.getenv("HF_TOKEN", None)

# Reference flow settings (the release ``config.yaml`` flow section).
DEFAULT_FLOW_SHIFT = 4.0
DEFAULT_NUM_TRAIN_TIMESTEPS = 1000
DEFAULT_X_PRED_SIGMA_MIN = 0.05

# Qwen3-VL-4B-Instruct: config + tokenizer (tiny files) from the vendor repo;
# weights come from the shared comfy candidates (same file Krea 2 trains with).
QWEN3_VL_REPO = "Qwen/Qwen3-VL-4B-Instruct"
# repo whose comfy candidates Qwen3VLTextEncoder registers the 4B encoder file
# under (aitk_comfy_weight_names key). Iris trains that exact shared file, so
# it -- not iris's own model repo -- is the TE source repo for the door.
TE_COMFY_REPO = "Comfy-Org/Krea-2"


def _resolve_checkpoint(
    name_or_path: str,
    checkpoint_filename: Optional[str] = None,
    hf_token: Optional[str] = None,
    status_fn=None,
):
    """``(weights_path, config_yaml_path_or_None)`` for a local file, a local
    export dir, a hub repo id, or ``org/repo/sub/file.safetensors``.

    Hub references download into MODELS_PATH (comfy layout), never the HF
    cache. ``config.yaml`` is only used when it sits next to the weights on
    disk: a hub run trains the shipped defaults (= the release config), so no
    side-car file is fetched for it."""
    if os.path.isfile(name_or_path):
        cfg = os.path.join(os.path.dirname(name_or_path), "config.yaml")
        return name_or_path, (cfg if os.path.isfile(cfg) else None)
    if os.path.isdir(name_or_path):
        weights = os.path.join(name_or_path, checkpoint_filename or "model.safetensors")
        if not os.path.isfile(weights):
            candidates = [f for f in os.listdir(name_or_path) if f.endswith(".safetensors")]
            if len(candidates) != 1:
                raise FileNotFoundError(
                    f"Could not pick an Iris checkpoint in {name_or_path}: found {candidates}. "
                    "Set model.model_kwargs.checkpoint_filename."
                )
            weights = os.path.join(name_or_path, candidates[0])
        cfg = os.path.join(name_or_path, "config.yaml")
        return weights, (cfg if os.path.isfile(cfg) else None)

    ref = name_or_path
    if len(ref.split("/")) == 2 and not ref.endswith(".safetensors"):
        # plain hub repo id -> its root checkpoint
        ref = f"{ref}/{checkpoint_filename or 'model.safetensors'}"
    weights = resolve_component_file(
        ref,
        "diffusion_models",
        component="Iris-3B transformer",
        hf_token=hf_token,
        status_fn=status_fn,
    )
    cfg = os.path.join(os.path.dirname(weights), "config.yaml")
    return weights, (cfg if os.path.isfile(cfg) else None)


class Iris3BModel(PhasedLoadMixin, BaseModel):
    arch = "iris3b"
    display_name = "Iris 3B"

    def __init__(
        self,
        device,
        model_config: ModelConfig,
        dtype="bf16",
        custom_pipeline=None,
        noise_scheduler=None,
        **kwargs,
    ):
        super().__init__(device, model_config, dtype, custom_pipeline, noise_scheduler, **kwargs)
        self.is_flow_matching = True
        self.is_transformer = True
        # LoRA targets, matched by class name: the trunk blocks only. The
        # transformer_only filter is a substring match on "blocks", which also
        # catches pixel_blocks (per-pixel PiT head) and y_embedder.*.blocks (text
        # adapter); scoping the target classes keeps those frozen.
        self.target_lora_modules = ["MMDiTBlock", "SingleStreamBlock"]

        self.patch_size = 16
        self.vae_scale_factor = 1  # pixel space
        # flow / text settings; refreshed from a checkpoint config.yaml when a
        # local one is read in the transformer phase. The text-encoder phase
        # runs first and only ever targets the root release, whose text config
        # equals these defaults.
        self.flow_shift = float(self.model_config.model_kwargs.get("flow_shift", DEFAULT_FLOW_SHIFT))
        self.num_train_timesteps = DEFAULT_NUM_TRAIN_TIMESTEPS
        self.prediction = "v"
        self.x_pred_sigma_min = DEFAULT_X_PRED_SIGMA_MIN
        self.text_len = DEFAULT_TEXT_LEN
        self.text_hidden_layers = DEFAULT_HIDDEN_LAYERS
        self.te_config_repo = QWEN3_VL_REPO
        self.iris_raw_config: dict = {}
        self.prompt_encoder: Optional[IrisPromptEncoder] = None

    @staticmethod
    def get_train_scheduler(
        shift: float = DEFAULT_FLOW_SHIFT,
        num_train_timesteps: int = DEFAULT_NUM_TRAIN_TIMESTEPS,
    ):
        # class-level call site (BaseSDTrainProcess) precedes any checkpoint
        # read, so like every phased arch this returns the arch-default
        # schedule -- which is the release schedule. flow_shift overrides the
        # *sampling* grid (the preview pipeline) and the reference shift of a
        # non-default checkpoint.
        return CustomFlowMatchEulerDiscreteScheduler(
            num_train_timesteps=num_train_timesteps,
            shift=shift,
            use_dynamic_shifting=False,
        )

    def get_bucket_divisibility(self):
        return self.vae_scale_factor * self.patch_size

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def _read_checkpoint_config(self, cfg_path: Optional[str]) -> IrisConfig:
        raw = {}
        if cfg_path is not None:
            with open(cfg_path, "r") as f:
                raw = yaml.safe_load(f) or {}
        self.iris_raw_config = raw

        flow = raw.get("flow", {}) or {}
        if "flow_shift" not in self.model_config.model_kwargs:
            self.flow_shift = float(flow.get("shift", DEFAULT_FLOW_SHIFT))
        self.num_train_timesteps = int(flow.get("num_train_timesteps", DEFAULT_NUM_TRAIN_TIMESTEPS))
        self.prediction = str(flow.get("prediction", "v"))
        if self.prediction not in ("v", "x"):
            raise ValueError(f"unknown flow.prediction '{self.prediction}' (v | x)")
        self.x_pred_sigma_min = float(flow.get("x_pred_sigma_min", DEFAULT_X_PRED_SIGMA_MIN))

        te = raw.get("text_encoder", {}) or {}
        self.text_len = int(te.get("max_length", DEFAULT_TEXT_LEN))
        layers = te.get("hidden_layers", None)
        self.text_hidden_layers = tuple(int(i) for i in layers) if layers else DEFAULT_HIDDEN_LAYERS
        self.te_config_repo = str(te.get("pretrained", QWEN3_VL_REPO))

        model_raw = dict(raw.get("model", {}) or {})
        model_raw.update(self.model_config.model_kwargs.get("model_config", {}) or {})
        config = IrisConfig.from_dict(model_raw)
        if config.text_adapter == "lap_blocks2" and config.text_lap_num_layers != len(self.text_hidden_layers):
            raise ValueError(
                f"model.text_lap_num_layers={config.text_lap_num_layers} but text_encoder.hidden_layers "
                f"has {len(self.text_hidden_layers)} entries"
            )
        if config.text_len != self.text_len:
            raise ValueError(
                f"model.text_len={config.text_len} but text_encoder.max_length={self.text_len}"
            )
        return config

    def _load_transformer(self) -> IrisDiT:
        self.print_and_status_update("Loading Iris-3B transformer")
        weights, cfg_path = _resolve_checkpoint(
            self.model_config.name_or_path,
            self.model_config.model_kwargs.get("checkpoint_filename", None),
            hf_token=HF_TOKEN,
            status_fn=self.print_and_status_update,
        )
        config = self._read_checkpoint_config(cfg_path)
        self.patch_size = config.patch_size
        self.print_and_status_update(f"  - weights: {weights}")
        # as-shipped mode streams straight onto the gpu (no cpu staging);
        # low_vram keeps the legacy cpu parking for the phased placement.
        # Quantization / offload / final placement are the wrapper's
        # aitk_post_load, driven by model_config.
        dit_device = None if self.model_config.low_vram else self.device_torch
        transformer = IrisDiT.load_model(
            weights,
            dtype=self.torch_dtype,
            config=config,
            aitk_device=dit_device,
        )
        flush()
        return transformer

    def _load_text_encoder(self):
        dtype = self.torch_dtype
        # shared comfy-candidates door (same registered file Krea 2 resolves;
        # the fp8-scaled 4B repack). The model repo (iris) is not the TE's
        # comfy source repo, so the door is called with the repo that owns the
        # candidate. The text_encoder_path model_kwarg overrides.
        te_path = self.model_config.model_kwargs.get(
            "text_encoder_path",
            Qwen3VLTextEncoder.resolve_comfy_weights(
                TE_COMFY_REPO,
                subfolder="",
                hf_token=HF_TOKEN,
                status_fn=self.print_and_status_update,
                component="text_encoder",
            ),
        )
        if te_path is None:
            raise FileNotFoundError(
                "Qwen3-VL 4B comfy candidates resolved to nothing; set "
                "model.model_kwargs.text_encoder_path explicitly."
            )
        # the comfy file is weights-only; tokenizer + model config come from the
        # vendor repo unless the override is a real transformers folder
        tokenizer_src = te_path if os.path.isdir(te_path) else self.te_config_repo
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_src, token=HF_TOKEN)
        text_encoder = Qwen3VLTextEncoder.load_model(
            te_path,
            dtype=dtype,
            subfolder="",
            token=HF_TOKEN,
            config_path=self.te_config_repo,
            # stream straight onto the gpu in as-shipped mode (no cpu staging)
            aitk_device=None if self.model_config.low_vram else self.te_device_torch,
        )
        # text-only conditioning: the vision tower is dead weight
        text_encoder.drop_vision_tower()
        text_encoder.eval()
        text_encoder.requires_grad_(False)
        flush()
        return tokenizer, text_encoder, {}

    def _load_vae(self):
        # pixel space: the identity vae makes every latent/decode touchpoint a
        # pass-through. Cheap enough that the phase gates paying it is free,
        # and routing "latents" through encode_images keeps future latent
        # normalization in one place.
        self.print_and_status_update("Preparing pixel-space VAE (identity)")
        vae = FakeVAE(scaling_factor=1.0)
        vae.eval()
        vae.requires_grad_(False)
        return vae

    def _build_pipeline(self):
        return Iris3BPipeline(self)

    # ------------------------------------------------------------------
    # Sampling (training previews)
    # ------------------------------------------------------------------
    def get_generation_pipeline(self):
        return Iris3BPipeline(self)

    def generate_single_image(
        self,
        pipeline: Iris3BPipeline,
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

        mkw = self.model_config.model_kwargs
        cfg_interval = mkw.get("cfg_interval", (0.0, 1.0))
        img = pipeline(
            conditional_embeds=conditional_embeds,
            unconditional_embeds=unconditional_embeds,
            height=gen_config.height,
            width=gen_config.width,
            num_inference_steps=gen_config.num_inference_steps,
            guidance_scale=gen_config.guidance_scale,
            latents=gen_config.latents,
            generator=generator,
            order=int(mkw.get("sample_order", 2)),
            cfg_interval=(float(cfg_interval[0]), float(cfg_interval[1])),
        )[0]
        return img

    # ------------------------------------------------------------------
    # Training hooks
    # ------------------------------------------------------------------
    def model_velocity(
        self,
        x_t: torch.Tensor,
        t_model: torch.Tensor,
        text_feats: torch.Tensor,
        text_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Run the DiT and return the flow velocity ``noise - clean``. ``t_model``
        is Iris model time in [0, 1000] (= the toolkit timestep). An
        x-prediction checkpoint is converted via ``(x_t - x0) / max(sigma, s_min)``."""
        out = self.model(x_t, t_model, text_feats, y_mask=text_mask)
        if self.prediction == "x":
            sigma = (t_model.float() / self.num_train_timesteps).view(-1, 1, 1, 1).to(out.dtype)
            out = (x_t - out) / sigma.clamp(min=self.x_pred_sigma_min)
        return out

    def _stack_text_embeds(self, text_embeddings: AdvancedPromptEmbeds):
        """AdvancedPromptEmbeds -> ``(feats (B, text_len, L*D), mask (B, text_len))``
        on the compute device. Cached entries arrive per prompt; the fixed 300
        length means stacking needs no padding."""
        feats = text_embeddings.text_embeds
        if isinstance(feats, (list, tuple)):
            feats = torch.stack(list(feats))
        feats = feats.to(self.device_torch, self.torch_dtype)
        mask = text_embeddings["attention_mask"] if "attention_mask" in text_embeddings else None
        if mask is None:
            raise ValueError("Iris prompt embeds must carry an attention_mask")
        if isinstance(mask, (list, tuple)):
            mask = torch.stack(list(mask))
        return feats, mask.bool().to(self.device_torch)

    def get_noise_prediction(
        self,
        latent_model_input: torch.Tensor,  # (B, 3, H, W) pixels in [-1, 1] + noise
        timestep: torch.Tensor,  # 0..1000 scale, 1000 = pure noise
        text_embeddings: AdvancedPromptEmbeds,
        **kwargs,
    ):
        if self.model.device == torch.device("cpu"):
            self.model.to(self.device_torch)

        # toolkit timestep IS Iris model time (1000 * noise level); with
        # timestep_type=shift that is the shifted sigma the reference trained on
        t_model = timestep.to(self.device_torch, dtype=torch.float32)
        if t_model.dim() == 0:
            t_model = t_model.unsqueeze(0)
        if t_model.shape[0] != latent_model_input.shape[0]:
            t_model = t_model.expand(latent_model_input.shape[0])

        feats, mask = self._stack_text_embeds(text_embeddings)
        return self.model_velocity(
            latent_model_input.to(self.device_torch, self.torch_dtype), t_model, feats, mask
        )

    def get_prompt_embeds(self, prompt, control_images=None) -> AdvancedPromptEmbeds:
        """Fixed-length (text_len) stacked Qwen3-VL layer features.

        One ``(text_len, num_layers * 2560)`` tensor per prompt plus its
        ``(text_len,)`` mask -- the flattened layer axis is what the DiT
        restores, and rank-2-per-item is the format the caption/fixed caches,
        ``expand_to_batch`` and ``concat_prompt_embeds`` all expect (Krea 2
        convention). The empty prompt encodes to the reference CFG null."""
        if control_images is not None:
            raise NotImplementedError("Iris-3B conditioning is text-only; control images are not supported")
        if isinstance(prompt, str):
            prompt = [prompt]
        # encode_prompt wraps None (unset negatives arrive as [None]); the
        # reference CFG null is the empty prompt encoded through the template.
        prompt = ["" if p is None else str(p) for p in prompt]
        if self.text_encoder.device == torch.device("cpu"):
            self.text_encoder.to(self.device_torch)
        if self.prompt_encoder is None:
            self.prompt_encoder = IrisPromptEncoder(
                self.tokenizer, text_len=self.text_len, hidden_layers=self.text_hidden_layers
            )
        feats, masks = [], []
        for p in prompt:
            embeds, mask = self.prompt_encoder(self.text_encoder, [p], dtype=self.torch_dtype)
            feats.append(embeds[0])
            masks.append(mask[0])
        pe = AdvancedPromptEmbeds(text_embeds=feats, attention_mask=masks)
        # the mask is bool: never let a .to(dtype) pass recast it
        pe.frozen_dtype_keys = ["attention_mask"]
        return pe

    def get_model_has_grad(self):
        return False

    def get_te_has_grad(self):
        return False

    # ------------------------------------------------------------------
    # Saving / bookkeeping
    # ------------------------------------------------------------------
    def save_model(self, output_path, meta, save_dtype):
        # single safetensors of plain dequantized weights, the fork-wide
        # convention. LoRA training never calls this; the reference export's
        # config.yaml side-car is intentionally not reproduced (the loader
        # trains the shipped defaults; see the module docstring).
        from toolkit.util.quantize import dequantize_if_quantized

        if not output_path.endswith(".safetensors"):
            output_path = output_path + ".safetensors"
        transformer: IrisDiT = unwrap_model(self.model)
        state_dict = transformer.state_dict()
        save_dict = {}
        for k, v in state_dict.items():
            save_dict[k] = dequantize_if_quantized(v).clone().to("cpu", dtype=save_dtype)
        meta = get_meta_for_safetensors(meta, name=self.arch)
        save_file(save_dict, output_path, metadata=meta)

    def get_base_model_version(self):
        return "iris3b"

    def get_transformer_block_names(self) -> Optional[List[str]]:
        return ["blocks"]

    lora_keys_use_comfy_prefix = True
