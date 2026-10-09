from functools import partial
import json
import os
from typing import List, Optional

import torch
import torchaudio
from transformers import Gemma3Config
import yaml
from toolkit.config_modules import GenerateImageConfig, ModelConfig
from toolkit.data_transfer_object.data_loader import DataLoaderBatchDTO
from toolkit.dto import DTO
from toolkit.models.base_model import BaseModel
from toolkit.models.phased_load import PhasedLoadMixin
from .staged_sample import (
    decode_audio_latents,
    decode_video_latents,
    denoise_staged,
)
from toolkit.basic import flush
from toolkit.prompt_utils import PromptEmbeds
from toolkit.advanced_prompt_embeds import AdvancedPromptEmbeds
from toolkit.samplers.custom_flowmatch_sampler import (
    CustomFlowMatchEulerDiscreteScheduler,
)
from accelerate import init_empty_weights
from toolkit.accelerator import unwrap_model
from toolkit.paths import MODELS_PATH
from toolkit.util.mixed_precision import attach_per_op_casting, pin_stored_fp32
from safetensors.torch import load_file
from PIL import Image
import huggingface_hub

try:
    from diffusers import LTX2Pipeline, LTX2ImageToVideoPipeline
    from diffusers.pipelines.ltx2.export_utils import encode_video
    from transformers import (
        Gemma3ForConditionalGeneration,
        GemmaTokenizerFast,
    )
    from toolkit.models.v2.diffusion_models.ltx2 import (
        LTX2TextConnectors,
        LTX2VideoTransformer3DModel,
        LTX2Vocoder,
        LTX2VocoderWithBWE,
    )
    from toolkit.models.v2.vae.ltx2 import (
        LTX2AudioVAE as AutoencoderKLLTX2Audio,
        LTX2VideoVAE as AutoencoderKLLTX2Video,
    )
    from toolkit.models.v2.text_encoders.gemma3 import Gemma3TextEncoder
    from .convert_ltx2_to_diffusers import (
        CONNECTOR_KEY_PREFIXES,
        get_model_state_dict_from_combined_ckpt,
        get_ltx2_audio_vae_config,
        get_ltx2_connectors_config,
        get_ltx2_transformer_config,
        get_ltx2_vocoder_config,
        get_ltx2_video_vae_config,
        convert_ltx2_transformer,
        convert_ltx2_video_vae,
        convert_ltx2_audio_vae,
        convert_ltx2_vocoder,
        convert_ltx2_connectors,
        split_transformer_and_connector_state_dict,
        dequantize_state_dict,
        convert_comfy_gemma3_to_transformers,
        convert_lora_original_to_diffusers,
        convert_lora_diffusers_to_original,
    )
except ImportError as e:
    print("Diffusers import error:", e)
    raise ImportError(
        "Diffusers is out of date. Update diffusers to the latest version by doing pip uninstall diffusers and then pip install -r requirements.txt"
    )


scheduler_config = {
    "base_image_seq_len": 1024,
    "base_shift": 0.95,
    "invert_sigmas": False,
    "max_image_seq_len": 4096,
    "max_shift": 2.05,
    "num_train_timesteps": 1000,
    "shift": 1.0,
    "shift_terminal": 0.1,
    "stochastic_sampling": False,
    "time_shift_type": "exponential",
    "use_beta_sigmas": False,
    "use_dynamic_shifting": True,
    "use_exponential_sigmas": False,
    "use_karras_sigmas": False,
}

dit_prefix = "model.diffusion_model."
vae_prefix = "vae."
audio_vae_prefix = "audio_vae."
vocoder_prefix = "vocoder."
base_te_path = "Lightricks/gemma-3-12b-it-qat-q4_0-unquantized"

HF_TOKEN = os.getenv("HF_TOKEN", None)


def new_save_image_function(
    self: GenerateImageConfig,
    image,  # will contain a dict that can be dumped ditectly into encode_video, just add output_path to it.
    count: int = 0,
    max_count: int = 0,
    **kwargs,
):
    # this replaces gen image config save image function so we can save the video with sound from ltx2
    image["output_path"] = self.get_image_path(count, max_count)
    # make sample directory if it does not exist
    os.makedirs(os.path.dirname(image["output_path"]), exist_ok=True)
    encode_video(**image)
    flush()


def blank_log_image_function(self, *args, **kwargs):
    # todo handle wandb logging of videos with audio
    return


class ComboVae(torch.nn.Module):
    """Combines video and audio VAEs for joint encoding and decoding.

    The phased ltx2.5 path also carries the vocoder here so the whole
    audio/video decode bundle shares one lifecycle (H3 vae-bundle shape);
    the vocoder slot stays None on the legacy 2.0/2.3 path."""

    def __init__(
        self,
        vae: AutoencoderKLLTX2Video,
        audio_vae: AutoencoderKLLTX2Audio,
        vocoder: Optional[torch.nn.Module] = None,
    ) -> None:
        super().__init__()
        self.vae = vae
        self.audio_vae = audio_vae
        self.vocoder = vocoder

    @property
    def device(self):
        return self.vae.device

    @property
    def dtype(self):
        return self.vae.dtype

    @property
    def config(self):
        return self.vae.config

    def encode(
        self,
        *args,
        **kwargs,
    ):
        return self.vae.encode(*args, **kwargs)

    def decode(
        self,
        *args,
        **kwargs,
    ):
        return self.vae.decode(*args, **kwargs)


class AudioProcessor(torch.nn.Module):
    """Converts audio waveforms to log-mel spectrograms with optional resampling."""

    def __init__(
        self,
        sample_rate: int,
        mel_bins: int,
        mel_hop_length: int,
        n_fft: int,
    ) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=n_fft,
            hop_length=mel_hop_length,
            f_min=0.0,
            f_max=sample_rate / 2.0,
            n_mels=mel_bins,
            window_fn=torch.hann_window,
            center=True,
            pad_mode="reflect",
            power=1.0,
            mel_scale="slaney",
            norm="slaney",
        )

    def resample_waveform(
        self,
        waveform: torch.Tensor,
        source_rate: int,
        target_rate: int,
    ) -> torch.Tensor:
        """Resample waveform to target sample rate if needed."""
        if source_rate == target_rate:
            return waveform
        resampled = torchaudio.functional.resample(waveform, source_rate, target_rate)
        return resampled.to(device=waveform.device, dtype=waveform.dtype)

    def waveform_to_mel(
        self,
        waveform: torch.Tensor,
        waveform_sample_rate: int,
    ) -> torch.Tensor:
        """Convert waveform to log-mel spectrogram [batch, channels, time, n_mels]."""
        waveform = self.resample_waveform(
            waveform, waveform_sample_rate, self.sample_rate
        )

        mel = self.mel_transform(waveform)
        mel = torch.log(torch.clamp(mel, min=1e-5))

        mel = mel.to(device=waveform.device, dtype=waveform.dtype)
        return mel.permute(0, 1, 3, 2).contiguous()


class LTX2Model(BaseModel):
    arch = "ltx2"
    ltx_version = "2.0"
    ltx_te_path = None

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
        self.target_lora_modules = ["LTX2VideoTransformer3DModel"]
        # defines if the model supports model paths. Only some will
        self.supports_model_paths = True
        # use the new format on this new model by default
        self.use_old_lokr_format = False
        self.audio_processor = None
        # text-embedding connectors: loaded lazily in the embedding phase,
        # freed at the phase boundary (release_encode_phase_components)
        self.connectors = None

        # gemma needs left side padding
        self.te_padding_side = "left"

        # loss mask for i2v conditioning (1 = train, 0 = conditioned token), set per step in get_noise_prediction
        self._i2v_loss_mask = None

        # invalidate older caches
        self.latent_space_version = f"{self.arch}_v2"

    # static method to get the noise scheduler
    @staticmethod
    def get_train_scheduler():
        return CustomFlowMatchEulerDiscreteScheduler(**scheduler_config)

    def get_bucket_divisibility(self):
        return 32

    def load_model(self):
        dtype = self.torch_dtype
        self.print_and_status_update("Loading LTX2 model")
        model_path = self.model_config.name_or_path
        base_model_path = self.model_config.extras_name_or_path

        combined_state_dict = None

        self.print_and_status_update("Loading transformer")

        if not os.path.exists(model_path) and model_path.endswith(".safetensors"):
            # download the model from the Hugging Face Hub if it is not a local path
            splits = model_path.split("/")
            if len(splits) < 3:
                raise ValueError(
                    f"Invalid model path: {model_path}. Must be in the format 'repo_id/repo/filename.safetensors' or 'repo_id/repo/subfolder/filename.safetensors' to download from the Hugging Face Hub."
                )
            rel_path = "/".join(splits[2:])
            # use the file from the models folder if it is already there
            local_candidates = [
                os.path.join(MODELS_PATH, rel_path),
                os.path.join(MODELS_PATH, splits[-1]),
            ]
            for candidate in local_candidates:
                if os.path.exists(candidate):
                    model_path = candidate
                    break
            else:
                # download the model from the hub into the models folder
                model_path = huggingface_hub.hf_hub_download(
                    repo_id="/".join(splits[:2]),
                    filename=rel_path,
                    token=HF_TOKEN,
                    local_dir=MODELS_PATH,
                )

        # if we have a safetensors file it is a mono checkpoint
        if os.path.exists(model_path) and model_path.endswith(".safetensors"):
            combined_state_dict = load_file(model_path)
            combined_state_dict = dequantize_state_dict(combined_state_dict)

        if combined_state_dict is not None:
            original_dit_ckpt = get_model_state_dict_from_combined_ckpt(
                combined_state_dict, dit_prefix
            )
            transformer = convert_ltx2_transformer(
                original_dit_ckpt, version=self.ltx_version
            )
            transformer = transformer.to(dtype)
            # the transformer holds these tensors (assign=True); drop the dict refs
            # so each block's bf16 original frees as it quantizes instead of the
            # whole dit staying in RAM. Connector keys stay — converted later.
            transformer_sd, _ = split_transformer_and_connector_state_dict(
                original_dit_ckpt
            )
            for key in transformer_sd:
                combined_state_dict.pop(dit_prefix + key, None)
            del transformer_sd, original_dit_ckpt
            flush()
        else:
            if os.path.exists(model_path):
                # check if the path is a full checkpoint.
                te_folder_path = os.path.join(model_path, "text_encoder")
                # if we have the te, this folder is a full checkpoint, use it as the base
                if os.path.exists(te_folder_path):
                    base_model_path = model_path

            transformer = LTX2VideoTransformer3DModel.load_model(model_path, dtype=dtype)

        # quantize + offload + placement, all driven by model_config
        transformer.aitk_post_load(**self.component_load_kwargs("transformer"))

        flush()

        self.print_and_status_update("Loading text encoder")
        if (
            self.model_config.te_name_or_path is not None
            and self.model_config.te_name_or_path.endswith(".safetensors")
        ):
            # load from comfyui gemma3 checkpoint
            tokenizer = GemmaTokenizerFast.from_pretrained(base_te_path)

            with init_empty_weights():
                text_encoder = Gemma3TextEncoder(
                    Gemma3Config(
                        **{
                            "boi_token_index": 255999,
                            "bos_token_id": 2,
                            "eoi_token_index": 256000,
                            "eos_token_id": 106,
                            "image_token_index": 262144,
                            "initializer_range": 0.02,
                            "mm_tokens_per_image": 256,
                            "model_type": "gemma3",
                            "pad_token_id": 0,
                            "text_config": {
                                "attention_bias": False,
                                "attention_dropout": 0.0,
                                "attn_logit_softcapping": None,
                                "cache_implementation": "hybrid",
                                "final_logit_softcapping": None,
                                "head_dim": 256,
                                "hidden_activation": "gelu_pytorch_tanh",
                                "hidden_size": 3840,
                                "initializer_range": 0.02,
                                "intermediate_size": 15360,
                                "max_position_embeddings": 131072,
                                "model_type": "gemma3_text",
                                "num_attention_heads": 16,
                                "num_hidden_layers": 48,
                                "num_key_value_heads": 8,
                                "query_pre_attn_scalar": 256,
                                "rms_norm_eps": 1e-06,
                                "rope_local_base_freq": 10000,
                                "rope_scaling": {"factor": 8.0, "rope_type": "linear"},
                                "rope_theta": 1000000,
                                "sliding_window": 1024,
                                "sliding_window_pattern": 6,
                                "torch_dtype": "bfloat16",
                                "use_cache": True,
                                "vocab_size": 262208,
                            },
                            "torch_dtype": "bfloat16",
                            "transformers_version": "4.51.3",
                            "unsloth_fixed": True,
                            "vision_config": {
                                "attention_dropout": 0.0,
                                "hidden_act": "gelu_pytorch_tanh",
                                "hidden_size": 1152,
                                "image_size": 896,
                                "intermediate_size": 4304,
                                "layer_norm_eps": 1e-06,
                                "model_type": "siglip_vision_model",
                                "num_attention_heads": 16,
                                "num_channels": 3,
                                "num_hidden_layers": 27,
                                "patch_size": 14,
                                "torch_dtype": "bfloat16",
                                "vision_use_head": False,
                            },
                        }
                    )
                )
            te_state_dict = load_file(self.model_config.te_name_or_path)
            te_state_dict = convert_comfy_gemma3_to_transformers(te_state_dict)
            for key in te_state_dict:
                te_state_dict[key] = te_state_dict[key].to(dtype)

            text_encoder.load_state_dict(te_state_dict, assign=True, strict=True)
            del te_state_dict
            flush()
        elif self.model_config.te_name_or_path is not None:
            # a repo or folder
            tokenizer = GemmaTokenizerFast.from_pretrained(
                self.model_config.te_name_or_path
            )
            text_encoder = Gemma3TextEncoder.load_model(
                self.model_config.te_name_or_path, dtype=dtype, subfolder=""
            )
        elif self.ltx_te_path is not None:
            # pull from model specific te
            tokenizer = GemmaTokenizerFast.from_pretrained(self.ltx_te_path)
            text_encoder = Gemma3TextEncoder.load_model(
                self.ltx_te_path, dtype=dtype, subfolder=""
            )
        else:
            # using combo hf repo
            tokenizer = GemmaTokenizerFast.from_pretrained(
                self.model_config.name_or_path, subfolder="tokenizer"
            )
            text_encoder = Gemma3TextEncoder.load_model(
                self.model_config.name_or_path, dtype=dtype
            )

        # remove the vision tower
        text_encoder.model.vision_tower = None
        flush()
        
        # quantize + offload + placement, all driven by model_config
        text_encoder.aitk_post_load(**self.component_load_kwargs("te"))
        text_encoder.to(self.device_torch, dtype=dtype)
        flush()

        self.print_and_status_update("Loading VAEs and other components")
        if combined_state_dict is not None:
            original_vae_ckpt = get_model_state_dict_from_combined_ckpt(
                combined_state_dict, vae_prefix
            )
            vae = convert_ltx2_video_vae(
                original_vae_ckpt, version=self.ltx_version
            ).to(dtype)
            del original_vae_ckpt
            original_audio_vae_ckpt = get_model_state_dict_from_combined_ckpt(
                combined_state_dict, audio_vae_prefix
            )
            audio_vae = convert_ltx2_audio_vae(
                original_audio_vae_ckpt, version=self.ltx_version
            ).to(dtype)
            del original_audio_vae_ckpt
            original_connectors_ckpt = get_model_state_dict_from_combined_ckpt(
                combined_state_dict, dit_prefix
            )
            connectors = convert_ltx2_connectors(
                original_connectors_ckpt, version=self.ltx_version
            ).to(dtype)
            del original_connectors_ckpt
            original_vocoder_ckpt = get_model_state_dict_from_combined_ckpt(
                combined_state_dict, vocoder_prefix
            )
            vocoder = convert_ltx2_vocoder(
                original_vocoder_ckpt, version=self.ltx_version
            ).to(dtype)
            del original_vocoder_ckpt
            del combined_state_dict
            flush()
        else:
            vae = AutoencoderKLLTX2Video.load_model(base_model_path, dtype=dtype)
            audio_vae = AutoencoderKLLTX2Audio.load_model(base_model_path, dtype=dtype)

            connectors = LTX2TextConnectors.load_model(base_model_path, dtype=dtype)

            vocoder_cls = LTX2Vocoder
            if self.ltx_version in ("2.3", "2.5"):
                vocoder_cls = LTX2VocoderWithBWE

            vocoder = vocoder_cls.load_model(base_model_path, dtype=dtype)

        self.noise_scheduler = LTX2Model.get_train_scheduler()

        self.print_and_status_update("Making pipe")

        pipe: LTX2Pipeline = LTX2Pipeline(
            scheduler=self.noise_scheduler,
            vae=vae,
            audio_vae=audio_vae,
            text_encoder=None,
            tokenizer=tokenizer,
            connectors=connectors,
            transformer=None,
            vocoder=vocoder,
        )
        # for quantization, it works best to do these after making the pipe
        pipe.text_encoder = text_encoder
        pipe.transformer = transformer

        self.print_and_status_update("Preparing Model")

        text_encoder = [pipe.text_encoder]
        tokenizer = [pipe.tokenizer]

        # leave it on cpu for now
        if not self.low_vram:
            pipe.transformer = pipe.transformer.to(self.device_torch)

        flush()
        # low_vram: the text encoder stays on cpu; get_prompt_embeds moves it
        # to the gpu on demand
        if not self.low_vram:
            text_encoder[0].to(self.device_torch)
        text_encoder[0].requires_grad_(False)
        text_encoder[0].eval()
        flush()

        # save it to the model class
        self.vae = ComboVae(pipe.vae, pipe.audio_vae)
        self.text_encoder = text_encoder  # list of text encoders
        self.tokenizer = tokenizer  # list of tokenizers
        self.model = pipe.transformer
        self.pipeline = pipe

        self.audio_processor = AudioProcessor(
            sample_rate=pipe.audio_sampling_rate,
            mel_bins=audio_vae.config.mel_bins,
            mel_hop_length=pipe.audio_hop_length,
            n_fft=1024,  # todo get this from vae if we can, I couldnt find it.
        ).to(self.device_torch, dtype=torch.float32)

        self.print_and_status_update("Model Loaded")

    @torch.no_grad()
    def encode_images(self, image_list: List[torch.Tensor], device=None, dtype=None):
        if device is None:
            device = self.vae_device_torch
        if dtype is None:
            dtype = self.vae_torch_dtype

        if self.pipeline.vae.device == torch.device("cpu"):
            self.pipeline.vae.to(device)
        self.pipeline.vae.eval()
        self.pipeline.vae.requires_grad_(False)

        if self.model_config.low_vram:
            self.pipeline.vae.tile_sample_min_num_frames = 64
            self.pipeline.vae.tile_sample_stride_num_frames = 16
            # they check the wrong flat on encode currently so set both to future proof
            self.pipeline.vae.use_framewise_decoding = True
            self.pipeline.vae.use_framewise_encoding = True

        image_list = [image.to(device, dtype=dtype) for image in image_list]

        # Normalize shapes
        norm_images = []
        for image in image_list:
            if image.ndim == 3:
                # (C, H, W) -> (C, 1, H, W)
                norm_images.append(image.unsqueeze(1))
            elif image.ndim == 4:
                # (T, C, H, W) -> (C, T, H, W)
                norm_images.append(image.permute(1, 0, 2, 3))
            else:
                raise ValueError(f"Invalid image shape: {image.shape}")

        # Stack to (B, C, T, H, W)
        images = torch.stack(norm_images)

        latents = self.pipeline.vae.encode(images).latent_dist.sample()

        # Normalize latents across the channel dimension [B, C, F, H, W]
        scaling_factor = 1.0
        latents_mean = self.pipeline.vae.latents_mean.view(1, -1, 1, 1, 1).to(
            latents.device, latents.dtype
        )
        latents_std = self.pipeline.vae.latents_std.view(1, -1, 1, 1, 1).to(
            latents.device, latents.dtype
        )
        latents = (latents - latents_mean) * scaling_factor / latents_std

        if self.model_config.low_vram:
            self.pipeline.vae.use_framewise_decoding = False
            self.pipeline.vae.use_framewise_encoding = False

        return latents.to(device, dtype=dtype)

    def get_generation_pipeline(self):
        scheduler = LTX2Model.get_train_scheduler()

        pipeline: LTX2Pipeline = LTX2Pipeline(
            scheduler=scheduler,
            vae=unwrap_model(self.pipeline.vae),
            audio_vae=unwrap_model(self.pipeline.audio_vae),
            text_encoder=None,
            tokenizer=unwrap_model(self.pipeline.tokenizer),
            connectors=unwrap_model(self.pipeline.connectors),
            transformer=None,
            vocoder=unwrap_model(self.pipeline.vocoder),
        )
        pipeline.transformer = unwrap_model(self.model)
        if self.text_encoder is not None:
            pipeline.text_encoder = unwrap_model(self.text_encoder[0])
        else:
            # text encoder phase was skipped (every embed cached to disk,
            # gate: SDTrainer.needs_text_encoder_load). Sampling never
            # encodes: BaseModel.generate_images hands us embeds from the
            # fixed-embed cache and the pipeline call below only consumes
            # prompt_embeds, so no encoder is needed here.
            print(
                "generation pipeline: text encoder not resident "
                "(sample embeds come from the fixed-embed cache)"
            )

        pipeline = pipeline.to(self.device_torch)

        return pipeline

    def generate_single_image(
        self,
        pipeline: LTX2Pipeline,
        gen_config: GenerateImageConfig,
        conditional_embeds: PromptEmbeds,
        unconditional_embeds: PromptEmbeds,
        generator: torch.Generator,
        extra: dict,
    ):
        if self.model.device == torch.device("cpu"):
            self.model.to(self.device_torch)

        # handle control image
        if gen_config.ctrl_img is not None:
            # switch to image to video pipeline
            pipeline = LTX2ImageToVideoPipeline(
                scheduler=pipeline.scheduler,
                vae=pipeline.vae,
                audio_vae=pipeline.audio_vae,
                text_encoder=pipeline.text_encoder,
                tokenizer=pipeline.tokenizer,
                connectors=pipeline.connectors,
                transformer=pipeline.transformer,
                vocoder=pipeline.vocoder,
            )

        is_video = gen_config.num_frames > 1
        # override the generate single image to handle video + audio generation
        if is_video:
            gen_config._orig_save_image_function = gen_config.save_image
            gen_config.save_image = partial(new_save_image_function, gen_config)
            gen_config.log_image = partial(blank_log_image_function, gen_config)
            # set output extension to mp4
            gen_config.output_ext = "mp4"

        # reactivate progress bar since this is slooooow
        pipeline.set_progress_bar_config(disable=False)
        pipeline = pipeline.to(self.device_torch)

        # make sure dimensions are valid
        bd = self.get_bucket_divisibility()
        gen_config.height = (gen_config.height // bd) * bd
        gen_config.width = (gen_config.width // bd) * bd

        # handle control image
        if gen_config.ctrl_img is not None:
            control_img = Image.open(gen_config.ctrl_img).convert("RGB")
            # resize the control image
            control_img = control_img.resize(
                (gen_config.width, gen_config.height), Image.LANCZOS
            )
            # add the control image to the extra dict
            extra["image"] = control_img

        # frames must be divisible by 8 then + 1. so 1, 9, 17, 25, etc.
        if gen_config.num_frames != 1:
            if (gen_config.num_frames - 1) % 8 != 0:
                gen_config.num_frames = ((gen_config.num_frames - 1) // 8) * 8 + 1

        if self.low_vram:
            # set vae to tile decode
            # pipeline.vae.enable_tiling(
            #     tile_sample_min_height=256,
            #     tile_sample_min_width=256,
            #     tile_sample_min_num_frames=8,
            #     tile_sample_stride_height=224,
            #     tile_sample_stride_width=224,
            #     tile_sample_stride_num_frames=4,
            # )
            self.pipeline.vae.tile_sample_min_num_frames = 16
            self.pipeline.vae.tile_sample_stride_num_frames = 8
            self.pipeline.vae.use_framewise_decoding = True

        # We only encode and store the minimum prompt tokens, but need them padded to 1024 for LTX2
        conditional_embeds = self.pad_embeds(conditional_embeds)
        unconditional_embeds = self.pad_embeds(unconditional_embeds)


        if self.ltx_version in ("2.3", "2.5"):
            extra["stg_scale"] = 1.0
            extra["modality_scale"] = 3.0
            extra["guidance_rescale"] = 0.7
            extra["audio_guidance_scale"] = 7.0
            extra["audio_stg_scale"] = 1.0
            extra["audio_modality_scale"] = 3.0
            extra["audio_guidance_rescale"] = 0.7
            extra["spatio_temporal_guidance_blocks"] = [28]
            extra["use_cross_timestep"] = (
                True  # they dont set this in some examples in diffusers, but I believe it should always be true for 2.3
            )

        video, audio = pipeline(
            prompt_embeds=conditional_embeds.text_embeds.to(
                self.device_torch, dtype=self.torch_dtype
            ),
            prompt_attention_mask=conditional_embeds.attention_mask.to(
                self.device_torch
            ),
            negative_prompt_embeds=unconditional_embeds.text_embeds.to(
                self.device_torch, dtype=self.torch_dtype
            ),
            negative_prompt_attention_mask=unconditional_embeds.attention_mask.to(
                self.device_torch
            ),
            height=gen_config.height,
            width=gen_config.width,
            num_inference_steps=gen_config.num_inference_steps,
            guidance_scale=gen_config.guidance_scale,
            latents=gen_config.latents,
            num_frames=gen_config.num_frames,
            generator=generator,
            return_dict=False,
            output_type="np" if is_video else "pil",
            **extra,
        )
        if self.low_vram:
            # Restore no tiling
            # pipeline.vae.use_tiling = False
            self.pipeline.vae.use_framewise_decoding = False

        if is_video:
            # redurn as a dict, we will handle it with an override function
            video = (video * 255).round().astype("uint8")
            video = torch.from_numpy(video)
            return {
                "video": video[0],
                "fps": gen_config.fps,
                "audio": audio[0].float().cpu(),
                "audio_sample_rate": pipeline.vocoder.config.output_sampling_rate,  # should be 24000
                "output_path": None,
            }
        else:
            # shape = [1, frames, channels, height, width]
            # make sure this is right
            video = video[0]  # list of pil images
            audio = audio[0]  # tensor
            if gen_config.num_frames > 1:
                return video  # return the frames.
            else:
                # get just the first image
                img = video[0]
            return img

    def encode_audio(self, audio_data_list):
        # audio_date_list is a list of {"waveform": waveform[C, L], "sample_rate": int(sample_rate)}
        if self.pipeline.audio_vae.device == torch.device("cpu"):
            self.pipeline.audio_vae.to(self.device_torch)

        output_tensor = None
        audio_num_frames = None

        # do them seperatly for now
        for audio_data in audio_data_list:
            waveform = audio_data["waveform"].to(
                device=self.device_torch, dtype=torch.float32
            )
            sample_rate = audio_data["sample_rate"]

            # Add batch dimension if needed: [channels, samples] -> [batch, channels, samples]
            if waveform.dim() == 2:
                waveform = waveform.unsqueeze(0)

            if waveform.shape[1] == 1:
                # make sure it is stereo
                waveform = waveform.repeat(1, 2, 1)

            # Convert waveform to mel spectrogram using AudioProcessor
            mel_spectrogram = self.audio_processor.waveform_to_mel(
                waveform, waveform_sample_rate=sample_rate
            )
            mel_spectrogram = mel_spectrogram.to(dtype=self.torch_dtype)

            # Encode mel spectrogram to latents
            latents = self.pipeline.audio_vae.encode(
                mel_spectrogram.to(self.device_torch, dtype=self.torch_dtype)
            ).latent_dist.sample()

            if audio_num_frames is None:
                audio_num_frames = latents.shape[2]  # (latents is [B, C, T, F])

            packed_latents = self.pipeline._pack_audio_latents(
                latents,
                # patch_size=self.pipeline.transformer.config.audio_patch_size,
                # patch_size_t=self.pipeline.transformer.config.audio_patch_size_t,
            )  # [B, L, C * M]
            if output_tensor is None:
                output_tensor = packed_latents
            else:
                output_tensor = torch.cat([output_tensor, packed_latents], dim=0)

        # normalize latents, opposite of (latents * latents_std) + latents_mean
        latents_mean = self.pipeline.audio_vae.latents_mean
        latents_std = self.pipeline.audio_vae.latents_std
        output_tensor = (output_tensor - latents_mean) / latents_std
        return output_tensor

    def pad_embeds(self, embeds: PromptEmbeds):
        # ltx-2 connector requires 1024 tokens for good results. Any smaller and it degrades.
        target_length = 1024
        current_length = embeds.text_embeds.shape[1]
        if current_length < target_length:
            pad_length = target_length - current_length
            pad_tensor = torch.zeros(
                (embeds.text_embeds.shape[0], pad_length, embeds.text_embeds.shape[2]),
                device=embeds.text_embeds.device,
                dtype=embeds.text_embeds.dtype,
            )
            embeds.text_embeds = torch.cat([pad_tensor, embeds.text_embeds], dim=1)
            if embeds.attention_mask is not None:
                pad_mask = torch.zeros(
                    (embeds.attention_mask.shape[0], pad_length),
                    device=embeds.attention_mask.device,
                    dtype=embeds.attention_mask.dtype,
                )
                embeds.attention_mask = torch.cat(
                    [pad_mask, embeds.attention_mask], dim=1
                )
        return embeds

    def get_noise_prediction(
        self,
        latent_model_input: torch.Tensor,
        timestep: torch.Tensor,  # 0 to 1000 scale
        text_embeddings: PromptEmbeds,
        batch: "DataLoaderBatchDTO" = None,
        **kwargs,
    ):
        audio_target = None
        with torch.no_grad():
            if self.model.device == torch.device("cpu"):
                self.model.to(self.device_torch)

            # advanced (connector-space) payloads come straight from the
            # embed cache and were padded before the connectors ran; plain
            # payloads store the minimum tokens and need 1024 padding here
            advanced_embeds = (
                isinstance(text_embeddings, AdvancedPromptEmbeds)
                and "connector_prompt_embeds" in text_embeddings
            )
            if not advanced_embeds:
                # We only encode and store the minimum prompt tokens, but
                # need them padded to 1024 for LTX2
                text_embeddings = self.pad_embeds(text_embeddings)

            batch_size, C, latent_num_frames, latent_height, latent_width = (
                latent_model_input.shape
            )

            video_timestep = timestep.clone()
            self._i2v_loss_mask = None

            # i2v from first frame
            if batch.dataset_config.do_i2v and batch.num_frames > 1:
                # check to see if we had it cached
                if batch.first_frame_latents is not None:
                    init_latents = batch.first_frame_latents.to(
                        self.device_torch, dtype=self.torch_dtype
                    )
                else:
                    # extract the first frame and encode it
                    # videos come in (bs, num_frames, channels, height, width)
                    # images come in (bs, channels, height, width)
                    frames = batch.tensor
                    if len(frames.shape) == 4:
                        first_frames = frames
                    elif len(frames.shape) == 5:
                        first_frames = frames[:, 0]
                    else:
                        raise ValueError(f"Unknown frame shape {frames.shape}")
                    # first frame doesnt have time dim, add it back
                    init_latents = self.encode_images(
                        first_frames, device=self.device_torch, dtype=self.torch_dtype
                    )

                # expand the latents to match video frames
                init_latents = init_latents.repeat(1, 1, latent_num_frames, 1, 1)
                mask_shape = (
                    batch_size,
                    1,
                    latent_num_frames,
                    latent_height,
                    latent_width,
                )
                # First condition is image latents and those should be kept clean.
                conditioning_mask = torch.zeros(
                    mask_shape, device=self.device_torch, dtype=self.torch_dtype
                )
                conditioning_mask[:, :, 0] = 1.0

                # use conditioning mask to replace latents
                latent_model_input = (
                    init_latents * conditioning_mask
                    + latent_model_input * (1 - conditioning_mask)
                )

                # conditioned tokens are clean with timestep 0 and their prediction is
                # discarded at inference, so they must not contribute to the loss
                self._i2v_loss_mask = 1.0 - conditioning_mask

                packed_conditioning_mask = self.pipeline._pack_latents(
                    conditioning_mask,
                    patch_size=self.pipeline.transformer_spatial_patch_size,
                    patch_size_t=self.pipeline.transformer_temporal_patch_size,
                ).squeeze(-1)

                # set video timestep
                video_timestep = timestep.unsqueeze(-1) * (1 - packed_conditioning_mask)

            frame_rate = batch.dataset_config.fps
            # check frame dimension
            # Unpacked latents of shape are [B, C, F, H, W] are patched into tokens of shape [B, C, F // p_t, p_t, H // p, p, W // p, p].
            packed_latents = self.pipeline._pack_latents(
                latent_model_input,
                patch_size=self.pipeline.transformer_spatial_patch_size,
                patch_size_t=self.pipeline.transformer_temporal_patch_size,
            )

            # audio only trains for video batches from datasets that asked for
            # it. Cached latents can carry audio after do_audio was turned off,
            # and image (single frame) batches must never pick up a soundtrack.
            do_audio = (
                batch.dataset_config is not None
                and batch.dataset_config.do_audio
                and getattr(batch, "num_frames", 1) > 1
            )
            if do_audio and (
                batch.audio_latents is not None or batch.audio_tensor is not None
            ):
                if batch.audio_latents is not None:
                    # we have audio latents cached
                    raw_audio_latents = batch.audio_latents.to(
                        self.device_torch, dtype=self.torch_dtype
                    )
                else:
                    # we have audio waveforms to encode
                    # use audio from the batch if available
                    raw_audio_latents = self.encode_audio(batch.audio_data)

                audio_num_frames = raw_audio_latents.shape[1]
                # the audio noise is drawn once per step and shared by every
                # pass (prior, primary, cfg/guidance, preservation) so they all
                # see the same soundtrack and every pass's target matches. It
                # rides on the latents DTO.
                audio_noise = (
                    batch.latents.get("audio_noise")
                    if isinstance(batch.latents, DTO)
                    else None
                )
                if (
                    audio_noise is not None
                    and audio_noise.shape == raw_audio_latents.shape
                ):
                    audio_noise = audio_noise.to(
                        raw_audio_latents.device, dtype=raw_audio_latents.dtype
                    )
                else:
                    audio_noise = torch.randn_like(raw_audio_latents)
                    if batch.latents is not None:
                        batch.latents = DTO(batch.latents, audio_noise=audio_noise)
                audio_target = (audio_noise - raw_audio_latents).detach()
                audio_latents = self.add_noise(
                    raw_audio_latents,
                    audio_noise,
                    timestep,
                ).to(self.device_torch, dtype=self.torch_dtype)
            else:
                # no audio: the zero-stream shape comes from the same version
                # config table the audio VAE was built from, not from the
                # module — this branch runs with the vae bundle released
                # (cached-latents mode). prepare_audio_latents (latents=None)
                # and the pipeline's compression-rate constants are None-safe.
                audio_vae_config = get_ltx2_audio_vae_config(self.ltx_version)[0][
                    "diffusers_config"
                ]
                num_mel_bins = audio_vae_config["mel_bins"]
                # latent_mel_bins = num_mel_bins // self.audio_vae_mel_compression_ratio
                num_channels_latents_audio = audio_vae_config["latent_channels"]
                duration_s = batch.num_frames / frame_rate
                audio_latents_per_second = (
                    self.pipeline.audio_sampling_rate
                    / self.pipeline.audio_hop_length
                    / float(self.pipeline.audio_vae_temporal_compression_ratio)
                )
                audio_num_frames = round(duration_s * audio_latents_per_second)
                audio_latents = self.pipeline.prepare_audio_latents(
                    batch_size,
                    num_channels_latents=num_channels_latents_audio,
                    audio_latent_length=audio_num_frames,
                    num_mel_bins=num_mel_bins,
                    noise_scale=0.0,
                    dtype=torch.float32,
                    device=self.transformer.device,
                    generator=None,
                    latents=None,
                )

            # Padding side for default Gemma3-12B text encoder
            tokenizer_padding_side = "left"
            if getattr(self, "tokenizer", None) is not None:
                tokenizer_padding_side = getattr(self.tokenizer, "padding_side", "left")
            if advanced_embeds:
                # cached connector outputs; the connector module never needs
                # to load on this rail (see LTX25Model.get_prompt_embeds)
                connector_prompt_embeds = torch.cat(
                    text_embeddings["connector_prompt_embeds"], dim=0
                ).to(self.device_torch, self.torch_dtype)
                connector_audio_prompt_embeds = torch.cat(
                    text_embeddings["connector_audio_prompt_embeds"], dim=0
                ).to(self.device_torch, self.torch_dtype)
                connector_attention_mask = torch.cat(
                    text_embeddings["connector_attention_mask"], dim=0
                ).to(self.device_torch)
            else:
                if self.pipeline.connectors.device != self.transformer.device:
                    self.pipeline.connectors.to(self.transformer.device)
                (
                    connector_prompt_embeds,
                    connector_audio_prompt_embeds,
                    connector_attention_mask,
                ) = self.pipeline.connectors(
                    text_embeddings.text_embeds,
                    text_embeddings.attention_mask.to(self.torch_dtype),
                    padding_side=tokenizer_padding_side,
                )

            # compute video and audio positional ids
            video_coords = self.transformer.rope.prepare_video_coords(
                packed_latents.shape[0],
                latent_num_frames,
                latent_height,
                latent_width,
                packed_latents.device,
                fps=frame_rate,
            )
            audio_coords = self.transformer.audio_rope.prepare_audio_coords(
                audio_latents.shape[0], audio_num_frames, audio_latents.device
            )

        # use_cross_timestep - Whether to use the cross modality (audio is the cross modality of video, and vice versa) sigma when
        # calculating the cross attention modulation parameters. `True` is the newer (e.g. LTX-2.3) behavior;
        # `False` is the legacy LTX-2.0 behavior.
        use_cross_timestep = self.ltx_version in ("2.3", "2.5")

        # mask fast-path: an all-valid mask carries no information, but any
        # non-None mask routes every text cross-attn off flash and onto the
        # masked memory-efficient path (sm80 bprop kernels in the bs1
        # profile). With mask=None the dispatch picks flash; over an
        # all-valid mask the attention math is identical. One reduction per
        # pass; partial masks still take the normal masked path.
        if bool(connector_attention_mask.all()):
            connector_attention_mask = None

        noise_pred_video, noise_pred_audio = self.transformer(
            hidden_states=packed_latents,
            audio_hidden_states=audio_latents.to(self.torch_dtype),
            encoder_hidden_states=connector_prompt_embeds,
            audio_encoder_hidden_states=connector_audio_prompt_embeds,
            timestep=video_timestep,
            sigma=timestep,  # Used by LTX-2.3
            audio_timestep=timestep,
            encoder_attention_mask=connector_attention_mask,
            audio_encoder_attention_mask=connector_attention_mask,
            num_frames=latent_num_frames,
            height=latent_height,
            width=latent_width,
            fps=frame_rate,
            audio_num_frames=audio_num_frames,
            video_coords=video_coords,
            audio_coords=audio_coords,
            isolate_modalities=False,
            spatio_temporal_guidance_blocks=None,
            perturbation_mask=None,
            use_cross_timestep=use_cross_timestep,
            attention_kwargs=None,
            return_dict=False,
        )

        unpacked_output = self.pipeline._unpack_latents(
            latents=noise_pred_video,
            num_frames=latent_num_frames,
            height=latent_height,
            width=latent_width,
            patch_size=self.pipeline.transformer_spatial_patch_size,
            patch_size_t=self.pipeline.transformer_temporal_patch_size,
        )

        if audio_target is not None:
            # every pass's DTO carries its own audio stream and target
            return DTO(
                unpacked_output,
                audio=noise_pred_audio,
                audio_target=audio_target,
            )
        return unpacked_output

    def get_prompt_embeds(self, prompt: str) -> PromptEmbeds:
        text_encoder = (
            self.text_encoder[0] if isinstance(self.text_encoder, list)
            else self.text_encoder
        )
        if text_encoder.device != self.device_torch:
            text_encoder.to(self.device_torch)

        device = self.device_torch
        scale_factor = 8
        batch_size = len(prompt)
        # Gemma expects left padding for chat-style prompts
        self.tokenizer[0].padding_side = "left"
        if self.tokenizer[0].pad_token is None:
            self.tokenizer[0].pad_token = self.tokenizer[0].eos_token

        prompt = [p.strip() for p in prompt]
        text_inputs = self.tokenizer[0](
            prompt,
            # padding="max_length",
            padding="longest",
            max_length=1024,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        prompt_attention_mask = text_inputs.attention_mask

        text_input_ids = text_input_ids.to(device)
        prompt_attention_mask = prompt_attention_mask.to(device)

        text_encoder_outputs = text_encoder(
            input_ids=text_input_ids,
            attention_mask=prompt_attention_mask,
            output_hidden_states=True,
        )
        text_encoder_hidden_states = text_encoder_outputs.hidden_states
        text_encoder_hidden_states = torch.stack(text_encoder_hidden_states, dim=-1)
        prompt_embeds = text_encoder_hidden_states.flatten(2, 3).to(
            dtype=self.torch_dtype
        )  # Pack to 3D

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, 1, 1)
        prompt_embeds = prompt_embeds.view(batch_size * 1, seq_len, -1)

        prompt_attention_mask = prompt_attention_mask.view(batch_size, -1)
        prompt_attention_mask = prompt_attention_mask.repeat(1, 1)

        pe = PromptEmbeds([prompt_embeds, None])
        pe.attention_mask = prompt_attention_mask
        return pe

    def get_model_has_grad(self):
        return False

    def get_te_has_grad(self):
        return False

    def save_model(self, output_path, meta, save_dtype):
        transformer: LTX2VideoTransformer3DModel = unwrap_model(self.model)
        transformer.save_pretrained(
            save_directory=os.path.join(output_path, "transformer"),
            safe_serialization=True,
        )

        meta_path = os.path.join(output_path, "aitk_meta.yaml")
        with open(meta_path, "w") as f:
            yaml.dump(meta, f)

    def get_loss_target(self, *args, **kwargs):
        noise = kwargs.get("noise")
        batch = kwargs.get("batch")
        return (noise - batch.latents).detach()

    def scale_loss(self, loss):
        # zero out the loss on i2v conditioned tokens, renormalized so the loss
        # magnitude matches unconditioned batches (masked mean)
        if self._i2v_loss_mask is not None:
            loss_mask = self._i2v_loss_mask.to(loss.device, dtype=loss.dtype)
            loss = loss * loss_mask / loss_mask.mean().clamp(min=1e-8)
            self._i2v_loss_mask = None
        return loss

    def get_base_model_version(self):
        return "ltx2"

    def get_transformer_block_names(self) -> Optional[List[str]]:
        return ["transformer_blocks"]

    lora_keys_use_comfy_prefix = True

    def convert_lora_weights_before_save(self, state_dict):
        state_dict = super().convert_lora_weights_before_save(state_dict)
        return convert_lora_diffusers_to_original(state_dict, version=self.ltx_version)

    def convert_lora_weights_before_load(self, state_dict):
        state_dict = convert_lora_original_to_diffusers(
            state_dict, version=self.ltx_version
        )
        return super().convert_lora_weights_before_load(state_dict)


class LTX23Model(LTX2Model):
    arch = "ltx2.3"
    ltx_version = "2.3"
    ltx_te_path = base_te_path


class LTX25Model(PhasedLoadMixin, LTX2Model):
    arch = "ltx2.5"
    ltx_version = "2.5"
    ltx_te_path = None

    # ------------------------------------------------------------------
    # ComfyUI-style file resolution (H3-style comfy-candidates door)
    # ------------------------------------------------------------------
    # LTX-2.5 ships as ComfyUI-style split files (no diffusers folders, no
    # mono checkpoint). The candidate lists are registered on the component
    # classes (toolkit/models/v2/**/ltx2.py, gemma3.py) under the source repo
    # id, which is also the job's name_or_path. Files are used in place when
    # present under MODELS_PATH and downloaded only when missing, so the
    # models folder stays shareable with a ComfyUI install.
    def _comfy_component_cls(self, component: str):
        from toolkit.models.v2.text_encoders.gemma3 import Gemma4TextEncoder
        from toolkit.models.v2.vae.ltx2 import LTX2AudioVAE, LTX2VideoVAE

        return {
            "dit": LTX2VideoTransformer3DModel,
            "text_encoder": Gemma4TextEncoder,
            "video_vae": LTX2VideoVAE,
            "audio_vae": LTX2AudioVAE,
        }[component]

    def _resolve_comfy_file(self, component: str) -> str:
        component_cls = self._comfy_component_cls(component)
        name_or_path = self.model_config.name_or_path
        path = component_cls.resolve_comfy_weights(
            name_or_path,
            subfolder="",
            hf_token=HF_TOKEN,
            status_fn=self.print_and_status_update,
            component=component,
            model_kwargs=self.model_config.model_kwargs,
            override_path=self.model_config.model_kwargs.get(
                f"{component}_path", None
            ),
            # a local checkpoint dir is a search root for component files
            extra_roots=(
                [name_or_path]
                if name_or_path and os.path.isdir(name_or_path)
                else None
            ),
        )
        if path is None:
            raise FileNotFoundError(
                f"LTX-2.5: no comfy candidates registered for component "
                f"'{component}' under name_or_path '{name_or_path}'. Set "
                f"name_or_path to the source repo id, or point "
                f"'{component}_path' at an existing file."
            )
        return path

    def _resolve_named_file(self, path: str, component: str) -> str:
        from toolkit.models.v2.resolver import resolve_named_file

        return resolve_named_file(path, component=component, hf_token=HF_TOKEN)

    def _resolve_dit_path(self) -> str:
        name_or_path = self.model_config.name_or_path
        if name_or_path and name_or_path.endswith(".safetensors"):
            return self._resolve_named_file(name_or_path, "transformer")
        return self._resolve_comfy_file("dit")

    def _resolve_te_path(self) -> str:
        te_name_or_path = self.model_config.te_name_or_path
        if te_name_or_path is not None:
            return self._resolve_named_file(te_name_or_path, "text encoder")
        return self._resolve_comfy_file("text_encoder")

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def _load_quantized_module(self, module, state_dict, name: str) -> int:
        """Attach pre-quantized (int8 ConvRot) linears onto the toolkit's
        quantization backends and load the rest of the (meta-built) module
        from the state dict. Works unchanged for bf16 checkpoints, where no
        quant markers exist and everything strict-loads."""
        from toolkit.util.comfy_quant_import import import_comfy_quantized_layers
        from toolkit.models.v2._mixin import OstrisModelMixin

        state_dict, num_quantized = import_comfy_quantized_layers(
            module, state_dict, orig_dtype=self.torch_dtype
        )
        if num_quantized:
            self.print_and_status_update(
                f" - attached {num_quantized} pre-quantized ConvRot layers to {name}"
            )
        # whitelist for quantized weights + leftover-meta check
        OstrisModelMixin._load_state_dict_with_quantized(module, state_dict)
        return num_quantized

    def _gemma4_config(self, te_path: str) -> dict:
        from safetensors import safe_open

        with safe_open(te_path, framework="pt") as f:
            metadata = f.metadata() or {}
        return json.loads(metadata["gemma_config"])

    def _load_gemma4_text_encoder(self, te_sd: dict):
        """Build the Gemma-4 12B text stack from the comfy file's state dict
        (already resident on the target device when the stream loader read
        it). Only the text decoder is loaded — the unified checkpoint's
        vision/audio tower pieces and the connector projections are used
        elsewhere or dropped, matching how the Gemma-3 vision tower was
        discarded."""
        from transformers import Gemma4TextConfig
        from toolkit.models.v2.text_encoders.gemma3 import Gemma4TextEncoder

        gemma_config = self._gemma4_te_config
        text_config = {
            k: v for k, v in gemma_config["text_config"].items() if k != "dtype"
        }

        with init_empty_weights():
            text_encoder = Gemma4TextEncoder(Gemma4TextConfig(**text_config))

        def strip_model_prefix(key: str) -> str:
            return key[len("model.") :] if key.startswith("model.") else key

        te_sd = {
            strip_model_prefix(k): v
            for k, v in te_sd.items()
            if k.startswith("model.")
        }
        num_quantized = self._load_quantized_module(text_encoder, te_sd, "text encoder")
        return text_encoder, num_quantized

    def _load_gemma4_tokenizer(self, te_path: str):
        """The comfy file embeds the tokenizer and its configs as uint8
        tensors; extract them next to the file once and load from there."""
        from transformers import AutoTokenizer
        from toolkit.util.weight_source import WeightSource

        assets_dir = os.path.splitext(te_path)[0] + "_hf_assets"
        assets = {
            "tokenizer.json": "tokenizer_json",
            "tokenizer_config.json": "hf_asset__tokenizer_config.json",
            "chat_template.jinja": "hf_asset__chat_template.jinja",
        }
        os.makedirs(assets_dir, exist_ok=True)
        if any(
            not os.path.exists(os.path.join(assets_dir, filename))
            for filename in assets
        ):
            # only fault the embedded blobs when the extracted assets are new
            source = WeightSource.open(te_path)
            file_keys = set(source.keys())
            for filename, tensor_key in assets.items():
                out_path = os.path.join(assets_dir, filename)
                if os.path.exists(out_path) or tensor_key not in file_keys:
                    continue
                blob = source.get(tensor_key, "cpu")
                with open(out_path, "wb") as f:
                    f.write(bytes(blob.numpy().tobytes()))
        # the embedded tokenizer.json has an empty post-processor (ComfyUI
        # prepends BOS in its own wrapper); restore the standard Gemma
        # behavior so blank prompts still yield a token
        return AutoTokenizer.from_pretrained(assets_dir, add_bos_token=True)

    # ------------------------------------------------------------------
    # Phased loading (PhasedLoadMixin contract)
    # ------------------------------------------------------------------
    # No require_vae_during_training: every per-step vae touch is a cache
    # miss (first_frame/audio encode fallbacks), and the per-file latent
    # cache carries latent + first_frame_latent + audio_latent together, so
    # the process' completeness gate only skips the vae when no step needs
    # it. The no-audio zero-stream branch reads the version config table.
    display_name = "LTX-2.5"

    # denoise and decode are separate passes (staged_sample.py): the vae
    # bundle stays off the card for the whole denoise round, and nothing
    # but the transformer is needed to produce latents from cached
    # connector embeds. See BaseModel.generate_images staged dispatch.
    staged_sampling = True

    def generate_single_image(
        self,
        pipeline,
        gen_config: GenerateImageConfig,
        conditional_embeds: AdvancedPromptEmbeds,
        unconditional_embeds: AdvancedPromptEmbeds,
        generator: torch.Generator,
        extra: dict,
    ):
        # staged denoise pass: returns a latents payload, decoded later by
        # decode_staged_samples once the vae is resident
        if gen_config.ctrl_img is not None or gen_config.latents is not None:
            raise NotImplementedError(
                "LTX-2.5 staged sampling: control-image and latent-conditioned "
                "samples are not on this rail yet"
            )
        if gen_config.num_frames != 1 and (gen_config.num_frames - 1) % 8 != 0:
            gen_config.num_frames = ((gen_config.num_frames - 1) // 8) * 8 + 1
        is_video = gen_config.num_frames > 1
        if is_video:
            gen_config._orig_save_image_function = gen_config.save_image
            gen_config.save_image = partial(new_save_image_function, gen_config)
            gen_config.log_image = partial(blank_log_image_function, gen_config)
            gen_config.output_ext = "mp4"

        def cond(key):
            return conditional_embeds[key][0].to(
                self.device_torch, self.torch_dtype
            )

        def uncond_embeds(key):
            return unconditional_embeds[key][0].to(
                self.device_torch, self.torch_dtype
            )

        def uncond_mask(key):
            return unconditional_embeds[key][0].to(self.device_torch)

        latents, audio_latents, meta = denoise_staged(
            transformer=unwrap_model(self.model),
            scheduler=self.get_train_scheduler(),
            video_connector_embeds=cond("connector_prompt_embeds"),
            audio_connector_embeds=cond("connector_audio_prompt_embeds"),
            connector_attention_mask=conditional_embeds["connector_attention_mask"][
                0
            ].to(self.device_torch),
            neg_video_connector_embeds=uncond_embeds("connector_prompt_embeds"),
            neg_audio_connector_embeds=uncond_embeds("connector_audio_prompt_embeds"),
            neg_connector_attention_mask=uncond_mask("connector_attention_mask"),
            height=gen_config.height,
            width=gen_config.width,
            num_frames=gen_config.num_frames,
            frame_rate=24.0,
            num_inference_steps=gen_config.num_inference_steps,
            guidance_scale=gen_config.guidance_scale,
            # LTX-2.5 recommended guidance (the constants the inline path
            # sets in generate_single_image's extras)
            stg_scale=1.0,
            modality_scale=3.0,
            guidance_rescale=0.7,
            audio_guidance_scale=7.0,
            audio_stg_scale=1.0,
            audio_modality_scale=3.0,
            audio_guidance_rescale=0.7,
            spatio_temporal_guidance_blocks=[28],
            use_cross_timestep=True,
            generator=generator,
            device=self.device_torch,
            dtype=self.torch_dtype,
            video_config=get_ltx2_video_vae_config(self.ltx_version)[0][
                "diffusers_config"
            ],
            audio_config=get_ltx2_audio_vae_config(self.ltx_version)[0][
                "diffusers_config"
            ],
        )
        return {
            "latents": latents,
            "audio_latents": audio_latents,
            "meta": meta,
            "is_video": is_video,
            "fps": gen_config.fps,
        }

    def decode_staged_samples(self, staged_samples):
        """staged_sampling decode phase, one vae at a time (H3 shape): the
        video vae streams onto the gpu, decodes every payload, and is gone
        before the audio vae + vocoder arrive. Decoded pixels and waveforms
        park on the cpu; the mp4s are muxed at the very end."""
        dev = self.device_torch
        prior = self.vae
        if prior is not None:
            video_vae = prior.vae
            own_video = False
        else:
            video_vae = self._stream_video_vae(dev)
            own_video = True
        frames = []
        with torch.no_grad():
            for _gc, _idx, payload in staged_samples:
                f = decode_video_latents(
                    video_vae,
                    payload["latents"].to(dev),
                    device=dev,
                    dtype=self.torch_dtype,
                )
                # float (T,C,H,W) in [-1,1] -> uint8 (T,H,W,C) on cpu
                frames.append(
                    ((f.float() + 1) / 2)
                    .clamp(0, 1)
                    .mul(255)
                    .round()
                    .to(torch.uint8)
                    .permute(0, 2, 3, 1)
                    .cpu()
                )
        if own_video:
            del video_vae
            flush()

        if prior is not None and prior.audio_vae is not None:
            audio_vae, vocoder = prior.audio_vae, prior.vocoder
            own_audio = False
        else:
            audio_vae = self._stream_audio_vae(dev)
            vocoder = self._stream_vocoder(dev)
            own_audio = True
        sample_rate = int(vocoder.config.output_sampling_rate)
        waves = []
        with torch.no_grad():
            for _gc, _idx, payload in staged_samples:
                wave, _sr = decode_audio_latents(
                    audio_vae,
                    vocoder,
                    payload["audio_latents"].to(dev),
                    device=dev,
                    dtype=self.torch_dtype,
                )
                waves.append(wave.cpu())
        if own_audio:
            del audio_vae, vocoder
            flush()

        # mux and save, vae-s long gone
        for j, (gen_config, idx, payload) in enumerate(staged_samples):
            if payload["is_video"]:
                img = {
                    "video": frames[j],
                    "fps": payload["fps"],
                    "audio": waves[j],
                    "audio_sample_rate": sample_rate,
                    "output_path": None,
                }
            else:
                img = Image.fromarray(frames[j][0].numpy())
            gen_config.save_image_atomic(img, idx)
            gen_config.log_image(img, idx)
            self._after_sample_image(idx, len(staged_samples))
            flush()

    def _as_shipped_stream_target(self, role: str = "transformer"):
        """Device the stream loader reads straight onto, or None when the
        legacy batched cpu load applies (low_vram parking or a requested
        requantization — the requantizer needs the whole cpu state dict,
        same carve-out H3 keeps)."""
        want_requant = (
            self.model_config.quantize_te if role == "te" else self.model_config.quantize
        )
        if self.low_vram or want_requant:
            return None
        return self.device_torch

    # ------------------------------------------------------------------
    # Stream-to-module loading (as-shipped path)
    #
    # The batch converters only rename keys (verified: every rename_dict
    # entry and special handler is a pure key operation; none reads a tensor
    # value), so the same tables drive a per-key plan and every tensor is
    # faulted from the file straight onto the target device and assigned to
    # the meta shell one at a time. No whole-file state dict is ever
    # materialized, and nothing is ever copied or recast in the process.
    # ------------------------------------------------------------------
    @staticmethod
    def _remap_file_key(key, rename_dict, special_keys_remap):
        new_key = key
        for replace_key, rename_key in rename_dict.items():
            new_key = new_key.replace(replace_key, rename_key)
        for special_key, handler in special_keys_remap.items():
            if special_key not in new_key:
                continue
            # the converters' handlers are in-place dict key ops; emulate on
            # a one-key carrier (they never read the values)
            carrier = {new_key: None}
            handler(new_key, carrier)
            if not carrier:
                return None  # key dropped by the handler
            new_key = next(iter(carrier))
        return new_key

    def _stream_load(self, module, plan, device, name):
        """plan: {module_key: (WeightSource, file_key)}. Attaches each
        comfy-quant marker group through import_comfy_quantized_layers one
        module at a time, then assigns every remaining tensor individually.
        The closing meta scan plays the batch path's strict=True role."""
        from toolkit.util.comfy_quant_import import import_comfy_quantized_layers

        num_quantized = 0
        consumed = set()
        for marker_key in sorted(k for k in plan if k.endswith(".comfy_quant")):
            prefix = marker_key[: -len(".comfy_quant")]
            group = {}
            group_files = []
            for mod_key, (source, file_key) in plan.items():
                if mod_key == prefix or mod_key.startswith(prefix + "."):
                    group[mod_key] = source.get(file_key, device)
                    group_files.append(file_key)
            remaining, n = import_comfy_quantized_layers(
                module, group, orig_dtype=self.torch_dtype
            )
            if remaining:
                # unquantized siblings of a quantized module (bias, tables)
                module.load_state_dict(remaining, assign=True, strict=False)
            consumed.update(group_files)
            num_quantized += n
        assigned = set(consumed)
        for mod_key, (source, file_key) in plan.items():
            if file_key in assigned:
                continue
            module.load_state_dict(
                {mod_key: source.get(file_key, device)}, assign=True, strict=False
            )
            assigned.add(file_key)
        # persistent entries only (state_dict): quantized layers drop their
        # weight parameter / register non-persistent buffers, and computed
        # non-persistent buffers were meta in the batch path too
        still_meta = [
            n
            for n, t in module.state_dict(keep_vars=True).items()
            if torch.is_tensor(t) and t.device.type == "meta"
        ]
        if still_meta:
            raise ValueError(
                f"{name} stream load left {len(still_meta)} unfilled params/"
                f"buffers, first few: {still_meta[:8]}"
            )
        return num_quantized

    @staticmethod
    def _mixed_file_post_load(module, num_quantized):
        if not num_quantized:
            return
        # mixed-precision comfy file: comfy-style per-op input casting with
        # the stored-fp32 pieces pinned. No .to(dtype) anywhere: shipped
        # precision is kept as-is.
        attach_per_op_casting(module)
        pin_stored_fp32(module)
        # pre-quantized (ConvRot) comfy file; aitk_post_load skips quantize
        module.aitk_is_quantized = True

    def _load_transformer(self):
        # transformer streams from the comfy file; connectors are a separate
        # unit - they only ever run while embeds are produced (embedding
        # phase), see _ensure_connectors / release_encode_phase_components
        dit_path = self._resolve_dit_path()
        self.print_and_status_update(
            f"Loading transformer from {os.path.basename(dit_path)}"
        )
        stream_dev = self._as_shipped_stream_target()
        if stream_dev is None:
            return self._load_transformer_batched(dit_path)

        from toolkit.util.weight_source import WeightSource

        dit_source = WeightSource.open(dit_path)
        t_config, t_rename, t_special = get_ltx2_transformer_config(self.ltx_version)
        # same classification the batch split uses
        connector_prefixes = CONNECTOR_KEY_PREFIXES
        transformer_plan = {}
        for file_key in dit_source.keys():
            if not file_key.startswith(dit_prefix):
                continue
            key = file_key[len(dit_prefix):]
            if key.startswith(connector_prefixes):
                continue
            mapped = self._remap_file_key(key, t_rename, t_special)
            if mapped is not None:
                transformer_plan[mapped] = (dit_source, file_key)

        with init_empty_weights():
            transformer = LTX2VideoTransformer3DModel.from_config(
                t_config["diffusers_config"]
            )
        if self.ltx_version == "2.5":
            # 2.5 dropped the video feedforward biases and carries a learned
            # keyframe absolute-position embedding the pinned class predates
            # (mirrors the batch converter's meta surgery)
            for block in transformer.transformer_blocks:
                proj = block.ff.net[0].proj
                block.ff.net[0].proj = torch.nn.Linear(
                    proj.in_features, proj.out_features, bias=False, device="meta"
                )
                out = block.ff.net[2]
                block.ff.net[2] = torch.nn.Linear(
                    out.in_features, out.out_features, bias=False, device="meta"
                )
            keyframe_key = "keyframes_abs_pos_embedding"
            if keyframe_key in transformer_plan:
                source, file_key = transformer_plan[keyframe_key]
                shape = source.get(file_key, "cpu").shape
                transformer.keyframes_abs_pos_embedding = torch.nn.Parameter(
                    torch.empty(shape, device="meta"),
                    requires_grad=False,
                )

        num_quantized_dit = self._stream_load(
            transformer, transformer_plan, stream_dev, "transformer"
        )
        if num_quantized_dit:
            self.print_and_status_update(
                f" - attached {num_quantized_dit} pre-quantized ConvRot layers to transformer"
            )
        self._mixed_file_post_load(transformer, num_quantized_dit)
        if num_quantized_dit:
            self._bf16_modulation_tables(transformer)
            self._fuse_rmsnorms(transformer)
            self._fuse_split_rope()
        flush()
        return transformer

    def _bf16_modulation_tables(self, transformer):
        # The comfy file stores the adaLN scale_shift tables fp32; diffusers
        # get_mod_params adds table + train-dtype temb, and the fp32 table
        # promotes every modulation slice -> modulate/gate/residual keep the
        # full (B,T,dim) stream fp32 through the block (+165MB fp32 saved
        # tensors per block boundary at 1024, 2x elementwise traffic).
        # Official LTX casts the table to the timestep dtype at the add; the
        # tables are frozen constants, so one load-time cast is bitwise
        # equivalent and free. fp32 math stays where it belongs: norm
        # variance, rope tables, loss.
        n = 0
        for name, param in transformer.named_parameters():
            if name.endswith("scale_shift_table") and param.dtype == torch.float32:
                param.data = param.data.to(self.torch_dtype)
                n += 1
        print(
            f"LTX2.5: cast {n} fp32 scale_shift tables to {self.torch_dtype} "
            "(prevents fp32 promotion of the bf16 residual stream)"
        )
        return n

    def _fuse_rmsnorms(self, transformer):
        # diffusers RMSNorm with elementwise_affine=False runs an eager chain
        # (fp32 cast, pow, mean, fp32-promoted multiply, bf16 downcast) --
        # 4 full-size kernels per norm, 8 norms x 48 blocks, and grad
        # checkpointing replays them in backward. aten::rms_norm computes the
        # same fp32-variance formula in one kernel (the model already uses
        # this fused path for its q/k torch.nn.RMSNorm). Reduction order
        # differs (Welford vs mean-reduce): variance delta ~1e-6 rel, ~1000x
        # under a bf16 ulp; output differs only at rounding-boundary elements
        # by 1 ulp (parity test: docs/kernel_opt/tests/
        # ltx2_rmsnorm_fused_equiv.py). Weightless modules only; any affine
        # RMSNorm keeps the eager path.
        from diffusers.models.normalization import RMSNorm

        class _FusedRMSNorm(torch.nn.Module):
            def __init__(self, normalized_shape, eps):
                super().__init__()
                self.normalized_shape = tuple(normalized_shape)
                self.eps = eps

            def forward(self, x):
                return torch.nn.functional.rms_norm(
                    x, self.normalized_shape, weight=None, eps=self.eps)

        swaps = {}
        for name, mod in transformer.named_modules():
            if isinstance(mod, RMSNorm) and mod.weight is None and mod.bias is None:
                parent_path, _, leaf = name.rpartition(".")
                parent = transformer.get_submodule(parent_path) if parent_path else transformer
                swaps[(parent, leaf)] = _FusedRMSNorm(mod.dim, mod.eps)
        for (parent, leaf), fused in swaps.items():
            setattr(parent, leaf, fused)
        print(
            f"LTX2.5: fused {len(swaps)} weightless RMSNorms to aten::rms_norm "
            "(1 kernel vs 4 full-size eager ops per norm)"
        )
        return len(swaps)

    def _fuse_split_rope(self):
        # The eager split-rope chain costs ~10 launches per rope'd tensor
        # and checkpointing replays them in backward (measured: 4.73ms fwd+bwd
        # per video application vs 0.74ms fused; ~4.6k launches/step model-
        # wide). Head-to-head at this shape (bench: docs/kernel_opt/tests/
        # ltx2_rope_fused_bench.py): hand Triton 0.74ms < liger-triton 1.08
        # (+per-build freq materialization) < liger-cutedsl 2.33 (+cat cost),
        # eager 4.73. The kernel mirrors eager's fp32 rounding pattern
        # exactly (fwd mul+addcmul-fma; bwd unfused double-rounding): the
        # parity test is bitwise-equal forward AND backward, zero ulp flips.
        # This patches a diffusers module global; it is process-wide for
        # any transformer_ltx2 user in this process, and the function is
        # only reached for rope_type == "split" call sites.
        import diffusers.models.transformers.transformer_ltx2 as _t2
        from .rope_split_fused import fused_apply_split_rotary_emb
        if getattr(_t2.apply_split_rotary_emb, "_ltx25_fused", False):
            return 0
        fused_apply_split_rotary_emb._ltx25_fused = True
        _t2.apply_split_rotary_emb = fused_apply_split_rotary_emb
        print("LTX2.5: split-RoPE routed to fused Triton kernel "
              "(bitwise-eager fwd+bwd, 1 launch per direction)")
        return 1

    def _load_connectors_streamed(self, stream_dev):
        # connectors ride in the transformer file, plus the per-modality
        # text projections that live in the text encoder file (a load-time
        # file read, not a residency dependency)
        from toolkit.util.weight_source import WeightSource

        dit_source = WeightSource.open(self._resolve_dit_path())
        te_source = WeightSource.open(self._resolve_te_path())
        c_config, c_rename, c_special = get_ltx2_connectors_config(self.ltx_version)
        connector_prefixes = CONNECTOR_KEY_PREFIXES
        connector_plan = {}
        for file_key in dit_source.keys():
            if not file_key.startswith(dit_prefix):
                continue
            key = file_key[len(dit_prefix):]
            if not key.startswith(connector_prefixes):
                continue
            mapped = self._remap_file_key(key, c_rename, c_special)
            if mapped is not None:
                connector_plan[mapped] = (dit_source, file_key)
        for file_key in te_source.keys():
            if not file_key.startswith("text_embedding_projection."):
                continue
            mapped = self._remap_file_key(file_key, c_rename, c_special)
            if mapped is not None:
                connector_plan[mapped] = (te_source, file_key)

        with init_empty_weights():
            connectors = LTX2TextConnectors.from_config(c_config["diffusers_config"])
        num_quantized = self._stream_load(
            connectors, connector_plan, stream_dev, "connectors"
        )
        if num_quantized:
            self.print_and_status_update(
                f" - attached {num_quantized} pre-quantized ConvRot layers to connectors"
            )
        self._mixed_file_post_load(connectors, num_quantized)
        flush()
        return connectors

    def _ensure_connectors(self, reason: str):
        # connectors run only while embeds are produced (embedding phase).
        # They load on first use with a printed reason and stay out of every
        # other phase.
        if self.connectors is not None:
            return self.connectors
        self.print_and_status_update(f"Loading connectors - {reason}")
        stream_dev = self._as_shipped_stream_target()
        if stream_dev is None:
            connectors = self._load_connectors_batched(
                self._resolve_dit_path(), self._resolve_te_path()
            )
        else:
            connectors = self._load_connectors_streamed(stream_dev)
        self.connectors = connectors
        if self.pipeline is not None:
            self.pipeline.connectors = connectors
        flush()
        return connectors

    def release_encode_phase_components(self):
        # end of the embedding phase: nothing in training or staged sampling
        # consumes the connector module anymore - the embed caches carry
        # their outputs
        if self.connectors is None:
            return
        print("Connectors released - embed caches carry the connector outputs")
        self.connectors = None
        if self.pipeline is not None:
            self.pipeline.connectors = None
        flush()

    def _load_transformer_batched(self, dit_path):
        """low_vram / requested-requantization path: batched cpu state dict
        for the requantizer (same carve-out H3 keeps)."""
        combined = load_file(dit_path)
        dit_sd = get_model_state_dict_from_combined_ckpt(combined, dit_prefix)
        del combined

        transformer, transformer_sd = convert_ltx2_transformer(
            dit_sd, version=self.ltx_version, load=False
        )
        num_quantized_dit = self._load_quantized_module(
            transformer, transformer_sd, "transformer"
        )
        del transformer_sd
        trans_sd, _ = split_transformer_and_connector_state_dict(dit_sd)
        for key in trans_sd:
            dit_sd.pop(key, None)
        del trans_sd, dit_sd
        self._mixed_file_post_load(transformer, num_quantized_dit)
        if num_quantized_dit:
            self._bf16_modulation_tables(transformer)
            self._fuse_rmsnorms(transformer)
            self._fuse_split_rope()
        flush()
        return transformer

    def _load_connectors_batched(self, dit_path, te_path):
        combined = load_file(dit_path)
        dit_sd = get_model_state_dict_from_combined_ckpt(combined, dit_prefix)
        del combined
        te_state_dict = load_file(te_path)
        for key in te_state_dict:
            if key.startswith("text_embedding_projection."):
                dit_sd[key] = te_state_dict[key]
        del te_state_dict
        trans_sd, conn_sd = split_transformer_and_connector_state_dict(dit_sd)
        del trans_sd, dit_sd

        connectors, connectors_sd = convert_ltx2_connectors(
            conn_sd, version=self.ltx_version, load=False
        )
        del conn_sd
        num_quantized = self._load_quantized_module(
            connectors, connectors_sd, "connectors"
        )
        del connectors_sd
        self._mixed_file_post_load(connectors, num_quantized)
        flush()
        return connectors

    def get_prompt_embeds(self, prompt: str) -> AdvancedPromptEmbeds:
        # LTX-2.5 caches embeddings in CONNECTOR space. The raw encoder
        # output is padded and projected while the embedding phase is up
        # (connectors lazy-load here and are freed at the phase boundary);
        # nothing downstream ever sees the raw stack.
        raw = super().get_prompt_embeds(prompt)
        padded = self.pad_embeds(raw)
        connectors = self._ensure_connectors("embedding conversion")
        # tokenizer is up (we just tokenized through it); load sets padding
        # left for gemma
        padding_side = self.tokenizer[0].padding_side
        with torch.no_grad():
            video, audio, mask = connectors(
                padded.text_embeds.to(self.device_torch, self.torch_dtype),
                padded.attention_mask.to(self.device_torch, self.torch_dtype),
                padding_side=padding_side,
            )
        pe = AdvancedPromptEmbeds(
            connector_prompt_embeds=video.detach().cpu(),
            connector_audio_prompt_embeds=audio.detach().cpu(),
            connector_attention_mask=mask.detach().cpu(),
        )
        del raw, padded
        flush()
        return pe

    def embed_file_valid(self, path: str) -> bool:
        # LTX-2.5 embed files are connector-space AdvancedPromptEmbeds.
        # Anything else (plain raw-encoder caches from the old rail) fails
        # and gets re-encoded through the text encoder phase. Only damaged
        # or unreadable files are treated as invalid; both callers check
        # existence first, so anything other than a real file error must
        # propagate (a swallowed code bug here would silently re-encode
        # every cache, forever, and nobody would notice).
        from safetensors import _safetensors_rust, safe_open

        try:
            with safe_open(path, framework="pt") as f:
                if (f.metadata() or {}).get("class_name", "") != (
                    "AdvancedPromptEmbeds"
                ):
                    return False
                keys = set(f.keys())
        except (_safetensors_rust.SafetensorError, OSError):
            return False
        return {
            "connector_prompt_embeds",
            "connector_audio_prompt_embeds",
            "connector_attention_mask",
        } <= keys

    # unified-checkpoint keys in the TE file that are not the text stack:
    # connector projections (lifted by the transformer phase), embedded
    # tokenizer assets, and the vision/audio tower leftovers (dropped, as
    # the Gemma-3 vision tower always was)
    _TE_DROP_PREFIXES = (
        "text_embedding_projection.",
        "hf_asset__",
        "tokenizer_json",
        "vision_model",
        "audio_projector",
    )

    def _gemma4_shell(self):
        from transformers import Gemma4TextConfig
        from toolkit.models.v2.text_encoders.gemma3 import Gemma4TextEncoder

        text_config = {
            k: v
            for k, v in self._gemma4_te_config["text_config"].items()
            if k != "dtype"
        }
        with init_empty_weights():
            return Gemma4TextEncoder(Gemma4TextConfig(**text_config))

    def _load_text_encoder(self):
        te_path = self._resolve_te_path()
        self.print_and_status_update("Loading text encoder")
        self._gemma4_te_config = self._gemma4_config(te_path)
        tokenizer = self._load_gemma4_tokenizer(te_path)

        stream_dev = self._as_shipped_stream_target("te")
        if stream_dev is None:
            te_state_dict = {
                k: v
                for k, v in load_file(te_path).items()
                if not k.startswith(self._TE_DROP_PREFIXES)
            }
            text_encoder, num_quantized_te = self._load_gemma4_text_encoder(
                te_state_dict
            )
            del te_state_dict
        else:
            from toolkit.util.weight_source import WeightSource

            te_source = WeightSource.open(te_path)
            plan = {}
            for file_key in te_source.keys():
                if file_key.startswith(self._TE_DROP_PREFIXES):
                    continue
                mod_key = (
                    file_key[len("model."):]
                    if file_key.startswith("model.")
                    else file_key
                )
                plan[mod_key] = (te_source, file_key)
            text_encoder = self._gemma4_shell()
            num_quantized_te = self._stream_load(
                text_encoder, plan, stream_dev, "text encoder"
            )
            if num_quantized_te:
                self.print_and_status_update(
                    f" - attached {num_quantized_te} pre-quantized ConvRot layers to text encoder"
                )
        self._mixed_file_post_load(text_encoder, num_quantized_te)
        # no dtype cast: the stream loader kept the shipped precision
        text_encoder.requires_grad_(False)
        text_encoder.eval()
        flush()
        return tokenizer, text_encoder, {}

    def _stream_video_vae(self, device):
        from toolkit.util.weight_source import WeightSource

        self.print_and_status_update("Loading video VAE")
        video_config, video_rename, video_special = get_ltx2_video_vae_config(
            self.ltx_version
        )
        video_source = WeightSource.open(self._resolve_comfy_file("video_vae"))
        video_plan = {}
        for file_key in video_source.keys():
            mapped = self._remap_file_key(file_key, video_rename, video_special)
            if mapped is not None:
                video_plan[mapped] = (video_source, file_key)
        with init_empty_weights():
            vae = AutoencoderKLLTX2Video.from_config(video_config["diffusers_config"])
        self._stream_load(vae, video_plan, device, "video VAE")
        flush()
        return vae

    def _stream_audio_vae(self, device):
        from toolkit.util.weight_source import WeightSource

        self.print_and_status_update("Loading audio VAE")
        audio_config, audio_rename, audio_special = get_ltx2_audio_vae_config(
            self.ltx_version
        )
        audio_source = WeightSource.open(self._resolve_comfy_file("audio_vae"))
        audio_plan = {}
        for file_key in audio_source.keys():
            if not file_key.startswith(audio_vae_prefix):
                continue
            mapped = self._remap_file_key(
                file_key[len(audio_vae_prefix):], audio_rename, audio_special
            )
            if mapped is not None:
                audio_plan[mapped] = (audio_source, file_key)
        with init_empty_weights():
            audio_vae = AutoencoderKLLTX2Audio.from_config(
                audio_config["diffusers_config"]
            )
        self._stream_load(audio_vae, audio_plan, device, "audio VAE")
        flush()
        return audio_vae

    def _stream_vocoder(self, device):
        from toolkit.util.weight_source import WeightSource

        self.print_and_status_update("Loading vocoder")
        vocoder_config, vocoder_rename, vocoder_special = get_ltx2_vocoder_config(
            self.ltx_version
        )
        audio_source = WeightSource.open(self._resolve_comfy_file("audio_vae"))
        vocoder_plan = {}
        for file_key in audio_source.keys():
            if not file_key.startswith(vocoder_prefix):
                continue
            mapped = self._remap_file_key(
                file_key[len(vocoder_prefix):], vocoder_rename, vocoder_special
            )
            if mapped is not None:
                vocoder_plan[mapped] = (audio_source, file_key)
        vocoder_cls = LTX2Vocoder
        if self.ltx_version in ("2.3", "2.5"):
            vocoder_cls = LTX2VocoderWithBWE
        with init_empty_weights():
            vocoder = vocoder_cls.from_config(vocoder_config["diffusers_config"])
        self._stream_load(vocoder, vocoder_plan, device, "vocoder")
        flush()
        return vocoder

    def _load_vae(self):
        stream_dev = self._as_shipped_stream_target()
        if stream_dev is None:
            return self._load_vae_batched()
        # no dtype cast: files ship bf16 and stay bf16
        return ComboVae(
            self._stream_video_vae(stream_dev),
            self._stream_audio_vae(stream_dev),
            vocoder=self._stream_vocoder(stream_dev),
        )

    def _load_vae_batched(self):
        """low_vram / requantization carve-out: the batched cpu converters."""
        video_vae_path = self._resolve_comfy_file("video_vae")
        vae = convert_ltx2_video_vae(
            load_file(video_vae_path), version=self.ltx_version
        )
        flush()
        audio_vae_path = self._resolve_comfy_file("audio_vae")
        audio_combined = load_file(audio_vae_path)
        audio_sd = get_model_state_dict_from_combined_ckpt(
            audio_combined, audio_vae_prefix
        )
        audio_vae = convert_ltx2_audio_vae(audio_sd, version=self.ltx_version)
        vocoder_sd = get_model_state_dict_from_combined_ckpt(
            audio_combined, vocoder_prefix
        )
        del audio_combined, audio_sd
        vocoder = convert_ltx2_vocoder(vocoder_sd, version=self.ltx_version)
        del vocoder_sd
        flush()
        return ComboVae(vae, audio_vae, vocoder=vocoder)

    # The pipeline holds the three vae-family modules in its own slots, so
    # the plain holder attr would let `sd.vae = None` (the process' phase
    # boundary / post-sample free) drop nothing while the slots kept them
    # alive. One property keeps holder and slots in lockstep -- assign the
    # bundle to load it, assign None to truly free it.
    _vae = None

    @property
    def vae(self):
        return self._vae

    @vae.setter
    def vae(self, value):
        self._vae = value
        pipe = self.pipeline
        if pipe is not None:
            pipe.vae = getattr(value, "vae", None)
            pipe.audio_vae = getattr(value, "audio_vae", None)
            pipe.vocoder = getattr(value, "vocoder", None)

    def load_vae(self):
        # overrides the mixin's dtype-forcing placement line: the bundle
        # streams onto the vae device at shipped dtype
        if self.pipeline is None:
            # the pipeline is the shared container the cache passes
            # (encode_images/encode_audio) read their components from; in the
            # phased train order it must exist before phase 1's cache passes.
            # Slots are filled by each phase and re-synced by the vae
            # property; load_transformer rebuilds it with everything wired.
            if self.noise_scheduler is None:
                self.noise_scheduler = self.get_train_scheduler()
            self.pipeline = self._build_pipeline()
        bundle = self._load_vae()
        bundle.to(self.vae_device_torch)
        self.vae = bundle
        self._ensure_audio_processor()

    def _ensure_audio_processor(self):
        if self.audio_processor is not None:
            return self.audio_processor
        audio_config = get_ltx2_audio_vae_config(self.ltx_version)[0][
            "diffusers_config"
        ]
        self.audio_processor = AudioProcessor(
            sample_rate=audio_config["sample_rate"],
            mel_bins=audio_config["mel_bins"],
            mel_hop_length=audio_config["mel_hop_length"],
            n_fft=1024,  # todo get this from vae if we can, I couldnt find it.
        ).to(self.device_torch, dtype=torch.float32)
        return self.audio_processor

    def load_text_encoder(self):
        super().load_text_encoder()
        if self.pipeline is not None:
            self.pipeline.text_encoder = self.text_encoder
            self.pipeline.tokenizer = self.tokenizer
        # the shared 2.0/2.3 code consumes these as single-element lists
        if not isinstance(self.text_encoder, list):
            self.text_encoder = [self.text_encoder]
        if not isinstance(self.tokenizer, list):
            self.tokenizer = [self.tokenizer]

    def _build_pipeline(self):
        # slots are filled from whatever is resident at build time and
        # re-wired by the phases that follow (load order differs between
        # the composed all-at-once load and the phased train load, and a
        # fully-cached run may have no vae/te at all)
        bundle = self.vae
        text_encoder = self.text_encoder
        if isinstance(text_encoder, list):
            text_encoder = text_encoder[0] if text_encoder else None
        tokenizer = self.tokenizer
        if isinstance(tokenizer, list):
            tokenizer = tokenizer[0] if tokenizer else None
        pipe: LTX2Pipeline = LTX2Pipeline(
            scheduler=self.noise_scheduler,
            vae=bundle.vae if bundle is not None else None,
            audio_vae=bundle.audio_vae if bundle is not None else None,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            connectors=self.connectors,
            transformer=None,
            vocoder=bundle.vocoder if bundle is not None else None,
        )
        pipe.transformer = self.model
        return pipe
