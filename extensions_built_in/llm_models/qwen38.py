"""Qwen3.8-27B as a trainable text-generation model (arch ``qwen38``).

Image or video in (plus a .txt caption), text out. The frozen vision tower encodes
the media into embeddings [N, H] plus the processor's expanded placeholder span and
the MRoPE grid metadata (cached as the "latent" when latent caching is on). Training
runs the text stack on ``prefix + media span + instruction + assistant header +
caption`` with next-token cross-entropy on the caption (LoRA on the text stack;
towers, embeddings and lm_head stay frozen). ``is_llm`` routes the trainer to
``train_llm_accumulation`` -> ``get_llm_loss``: no noise, scheduler, VAE or prompt
encoding.

Weights: the ComfyUI single file (W4A8 codebook language model, int8 convrot
embeddings / lm_head / MTP / vision) is loaded with its quantization kept as-is
through the toolkit's convrotw4a8 / convrot8 backends, or an HF layout (bf16 shards)
is loaded and quantized to the requested qtype. See src/qwen38_model.py for the
model itself (static caches, CUDA-graph decode, MTP speculative decoding).

Samples: ``ctrl_img`` is the media file, the sample prompt is the instruction; the
generated text is written as a .txt.

model_kwargs:
  instruction            user-turn text used for every training item (default below)
  system_prompt          optional system message
  enable_thinking        keep the model's thinking mode on (default false: direct answers)
  max_pixels/min_pixels  image and video frame budget for the vision tower
  video_fps              fps the dataset extracted video frames at (dataset fps)
  max_new_tokens         sample generation budget (default 512)
  mtp_depth              MTP speculative draft depth for samples (default 3, 0 = off)
  cuda_graph             CUDA-graph the decode step (default true)
  vision_bf16            run the vision tower with bf16 activations on its dequantized int8
                         weights (default true; int8 activations drift the image embeddings ~6%)
  loss_chunk             positions per lm_head chunk in the CE (default 512)
"""

import os
import time
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from transformers import AutoConfig, AutoProcessor

from toolkit.basic import flush
from toolkit.config_modules import GenerateImageConfig, ModelConfig
from toolkit.dto import DTO
from toolkit.models.base_model import BaseModel
from toolkit.models.phased_load import PhasedLoadMixin
from toolkit.models.v2.resolver import resolve_component_file
from toolkit.print import print_acc

from .src.qwen38_model import Qwen38Generator, build_sequence_embeds, compute_rope_index

BASE_REPO = "Qwen/Qwen3.8-27B"
DEFAULT_CHECKPOINT = "Comfy-Org/Qwen3.8-27B/text_encoders/qwen3.8_27b_w4a8.safetensors"
DEFAULT_INSTRUCTION = "Describe this in detail."

KIND_IMAGE, KIND_VIDEO = 1, 2
VIDEO_EXTS = (".mp4", ".avi", ".mov", ".webm", ".mkv", ".wmv", ".m4v", ".flv")
AUDIO_EXTS = (".mp3", ".wav", ".flac", ".aac", ".ogg", ".m4a")


def _media_kind(path: str) -> int:
    ext = os.path.splitext(path)[1].lower()
    if ext in AUDIO_EXTS:
        raise ValueError(f"qwen38 takes images and videos, not audio: {path}")
    return KIND_VIDEO if ext in VIDEO_EXTS else KIND_IMAGE


class Qwen38LLM(PhasedLoadMixin, BaseModel):
    arch = "qwen38"
    is_llm = True
    # the vision tower is the vae: its embeddings are the cached "latents"
    # (a DTO with the media span + MRoPE grid), so phase 1 loads only the
    # tower and the iron rule frees it; phase 3 builds the language stack
    # without it (qwen38_vae_id.py registers the entry)
    vae_id = "qwen38"
    # the data loader accepts image and video files in one dataset
    is_multimodal_llm = True
    sample_rate = 16000

    # conversation datasets (the sidecar content array) are the only item
    # kind this arch trains on; the .json sidecar is parsed and validated at
    # scan time and assembled into the cache artifact in phase 1
    uses_conversation_sidecars = True

    def __init__(self, device, model_config: ModelConfig, dtype="bf16", custom_pipeline=None, noise_scheduler=None, **kwargs):
        super().__init__(device, model_config, dtype, custom_pipeline, noise_scheduler, **kwargs)
        self.is_transformer = True
        # LoRA goes on the text stack only (vision tower, embeddings and lm_head stay frozen)
        self.target_lora_modules = ["Qwen38TextModel"]
        kw = self.model_config.model_kwargs
        self.instruction = str(kw.get("instruction", DEFAULT_INSTRUCTION))
        self.system_prompt: Optional[str] = kw.get("system_prompt", None)
        self.enable_thinking = bool(kw.get("enable_thinking", False))
        self.max_pixels = int(kw.get("max_pixels", 1024 * 1024))
        self.min_pixels = int(kw.get("min_pixels", 64 * 32 * 32))
        self.video_fps = float(kw.get("video_fps", 24))
        self.max_new_tokens = int(kw.get("max_new_tokens", 512))
        self.mtp_depth = int(kw.get("mtp_depth", 3))
        self.cuda_graph = bool(kw.get("cuda_graph", True))
        self.vision_bf16 = bool(kw.get("vision_bf16", True))
        self.loss_chunk = int(kw.get("loss_chunk", 512))
        self.debug = bool(kw.get("debug", False))
        self.processor = None
        self.additional_loss_logs = {}

    @staticmethod
    def get_train_scheduler():
        # no diffusion: the trainer never samples timesteps for this model
        return None

    # ------------------------------------------------------------------
    # loading
    # ------------------------------------------------------------------
    def _config_source(self) -> str:
        extras = self.model_config.extras_name_or_path
        if extras and extras != self.model_config.name_or_path and not extras.endswith(".safetensors"):
            return extras
        return BASE_REPO

    def _cfg_src(self) -> str:
        """The config/processor source: an explicit extras path, else the
        checkpoint itself when it is a local dir or repo id (it ships the
        processor files), else the base repo."""
        cfg_src = self._config_source()
        name_or_path = self.model_config.name_or_path or DEFAULT_CHECKPOINT
        if not name_or_path.endswith(".safetensors"):
            if os.path.isdir(name_or_path) or "/" in name_or_path:
                cfg_src = name_or_path if self.model_config.extras_name_or_path in (None, "", name_or_path) else cfg_src
        return cfg_src

    def _checkpoint_path(self) -> str:
        """The qwen38 checkpoint through the MODELS_PATH-first door, shared by
        the vision-tower (vae) and transformer phases so both read the same
        file."""
        name_or_path = self.model_config.name_or_path or DEFAULT_CHECKPOINT
        if name_or_path.endswith(".safetensors"):
            return resolve_component_file(
                name_or_path, folder="text_encoders", component="qwen3.8"
            )
        return name_or_path

    def _load_processor(self, cfg_src: str):
        self.processor = AutoProcessor.from_pretrained(cfg_src)
        self.tokenizer = self.processor.tokenizer

    def load_vae(self):
        """Phase 1: the vision tower via the registry, plus the config,
        processor and chat template the encoding pass needs. The model is not
        built in this phase, so the template reads the config object."""
        cfg_src = self._cfg_src()
        self._config = AutoConfig.from_pretrained(cfg_src)
        super().load_vae()  # the registry entry streams the tower into self.vae
        self._load_processor(cfg_src)
        self._build_template()

    def load_model(self):
        dtype = self.torch_dtype
        device = self.device_torch
        name_or_path = self.model_config.name_or_path or DEFAULT_CHECKPOINT
        cfg_src = self._cfg_src()
        want_quant = self.model_config.quantize
        qtype = self.model_config.qtype
        if self.model_config.layer_offloading and self.model_config.layer_offloading_transformer_percent > 0:
            raise NotImplementedError("Layer offloading is not implemented for qwen38")

        from .src.qwen38_load import dequantize_vision_tower, load_qwen38

        source = self._checkpoint_path()
        if name_or_path.endswith(".safetensors"):
            self.print_and_status_update(f"Loading Qwen3.8 from {os.path.basename(source)}")
        else:
            self.print_and_status_update(f"Loading Qwen3.8 from {name_or_path}")
        config = AutoConfig.from_pretrained(cfg_src)
        self._config = config
        model, prequantized = load_qwen38(source, config, dtype, device, status_fn=self.print_and_status_update)
        if prequantized:
            model.aitk_is_quantized = True
            if want_quant and qtype not in ("convrotw4a8", "convrot8"):
                print_acc(f"Checkpoint is pre-quantized (W4A8 / int8 convrot); keeping it instead of {qtype}")
            want_quant = False
        if self.vision_bf16 and prequantized:
            n = dequantize_vision_tower(model.model.visual, dtype)
            print_acc(f"Vision tower: {n} int8 linears dequantized to {dtype} (model_kwargs.vision_bf16)")
        model.eval()
        model.requires_grad_(False)
        if want_quant:
            from toolkit.util.quantize import get_qtype, quantize

            self.print_and_status_update(f"Quantizing Qwen3.8 ({qtype})")
            quantize(model, weights=get_qtype(qtype))
            model.aitk_is_quantized = True
            flush()
        self.model = model
        self.model.generator = Qwen38Generator(model, use_cuda_graph=self.cuda_graph)
        self._load_processor(cfg_src)
        self.text_encoder = None
        self.vae = None
        self.pipeline = self
        self._build_template()
        flush()

    def _template_kwargs(self):
        return {"enable_thinking": self.enable_thinking}

    def _build_template(self):
        """Split the chat template once around the media span and the instruction so
        training sequences can be assembled from cached media spans without the processor."""
        mark = "<<AITK_INSTRUCTION>>"
        msgs = []
        if self.system_prompt is not None:
            msgs.append({"role": "system", "content": self.system_prompt})
        msgs.append({"role": "user", "content": [{"type": "image", "image": "x"}, {"type": "text", "text": mark}]})
        text = self.processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, **self._template_kwargs())
        span = "<|vision_start|><|image_pad|><|vision_end|>"
        if span not in text or mark not in text:
            raise RuntimeError("Unexpected Qwen3.8 chat template; cannot locate the media span")
        before, rest = text.split(span, 1)
        mid, after = rest.split(mark, 1)
        enc = lambda t: self.tokenizer(t, add_special_tokens=False).input_ids
        self._prefix_ids = enc(before)
        self._mid_ids = enc(mid)
        self._after_ids = enc(after)
        self._probe = {
            KIND_IMAGE: text.replace(mark, "x"),
            KIND_VIDEO: text.replace(span, "<|vision_start|><|video_pad|><|vision_end|>").replace(mark, "x"),
        }
        self._tail_ids = enc(mid + "x" + after)
        cfg = self._config
        self._placeholder = {KIND_IMAGE: cfg.image_token_id, KIND_VIDEO: cfg.video_token_id}
        self._im_end = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        self._eos_ids = [self._im_end, self.tokenizer.convert_tokens_to_ids("<|endoftext|>")]
        # job constant: tokenized once, spliced per step in _sequence
        self._instruction_ids = self.tokenizer(self.instruction, add_special_tokens=False).input_ids

    def get_transformer_block_names(self) -> Optional[List[str]]:
        return ["model.language_model.layers"]

    def get_model_has_grad(self):
        return False

    def get_te_has_grad(self):
        return False

    def get_bucket_divisibility(self):
        return 32

    def save_model(self, output_path, meta, save_dtype):
        raise NotImplementedError("qwen38: only LoRA training is supported (no full-model save)")

    # LoRA keys are model-relative (model.language_model.layers.N....), the HF layout
    def convert_lora_weights_before_save(self, state_dict):
        return {(k[len("transformer.") :] if k.startswith("transformer.") else k): v for k, v in state_dict.items()}

    def convert_lora_weights_before_load(self, state_dict):
        return {(k if k.startswith("transformer.") else "transformer." + k): v for k, v in state_dict.items()}

    # ------------------------------------------------------------------
    # media encoding (latent cache)
    # ------------------------------------------------------------------
    def _vision(self):
        """The tower: the registry vae when the phased path loaded it (the
        only owner there -- the model is built without the tower in phase 3),
        else the all-at-once model that still nests it."""
        if self.vae is not None:
            return self.vae
        if self.model is None:
            raise RuntimeError(
                "qwen38: no vision tower resident -- the phase gate decides "
                "when load_vae() runs; training on uncached latents needs "
                "vae_training_mode='keep'"
            )
        return self.model

    def encode_conversation(self, file_item):
        """The phase-1 artifact for one conversation sidecar: the full token
        sequence with media placeholders in place, the token-type vector, the
        MRoPE positions, the concatenated tower embeds and the loss-span start.
        Everything deterministic about the item -- the train step only gathers,
        splices, forwards and reads the loss span. Runs in phase 1 (the tower
        and processor are resident; the language stack is not). No grad policy
        here: the cache pass wraps its call site in no_grad, and a future
        tower-LoRA train step backprops through this."""
        from PIL import Image

        messages = file_item.conversation
        folder = os.path.dirname(file_item.path)
        if self.processor is None or getattr(self, "_config", None) is None:
            raise RuntimeError("qwen38: encode_conversation needs load_vae() to have run (the phase gate decides)")

        def entry_text(entry):
            if entry.get("file"):
                with open(os.path.join(folder, entry["file"]), "r") as f:
                    return f.read()
            return entry["text"]

        # the user turn: images collected in order, texts read; the processor
        # expands the placeholder spans and returns one packed pixel tensor
        user_content = []
        images = []
        for message in messages[:-1]:
            for entry in message["content"]:
                if entry["type"] == "image":
                    images.append(Image.open(os.path.join(folder, entry["file"])).convert("RGB"))
                    user_content.append({"type": "image"})
                else:
                    user_content.append({"type": "text", "text": entry_text(entry)})
        prompt_msgs = [{"role": m["role"], "content": user_content} for m in messages[:-1]]
        prompt_text = self.processor.apply_chat_template(
            prompt_msgs, add_generation_prompt=True, tokenize=False, **self._template_kwargs()
        )
        inputs = self.processor(
            text=[prompt_text], images=images, return_tensors="pt",
            min_pixels=self.min_pixels, max_pixels=self.max_pixels,
        )
        prompt_ids = inputs["input_ids"][0]
        prompt_types = inputs["mm_token_type_ids"][0]
        grid = inputs["image_grid_thw"]

        # the assistant turn: one text entry (scan-validated), tokenized raw
        # with the end-of-turn token appended; the loss span starts here
        output_text = entry_text(messages[-1]["content"][0])
        output_ids = torch.tensor(
            self.tokenizer(output_text, add_special_tokens=False).input_ids + [self._im_end],
            dtype=torch.long,
        )
        ids = torch.cat([prompt_ids, output_ids]).long()
        types = torch.cat([prompt_types, torch.zeros(output_ids.shape[0], dtype=prompt_types.dtype)])

        tower = self._vision()
        if tower.device == torch.device("cpu"):
            tower.to(self.device_torch)
        media = tower.get_image_features(
            inputs["pixel_values"].to(self.device_torch), grid.to(self.device_torch)
        )
        expected = int((ids == self._placeholder[KIND_IMAGE]).sum())
        if media.shape[0] != expected:
            raise RuntimeError(
                f"qwen38: tower embed count {media.shape[0]} != placeholder count {expected} for {file_item.path}"
            )
        position_ids = compute_rope_index(self._config, ids, types, image_grid_thw=grid)
        return DTO(
            media[None],
            ids=ids[None],
            types=types[None].to(torch.int32),
            position_ids=position_ids[None].to(torch.int64),
            grid_thw=grid.to(torch.int32),
            loss_start=torch.tensor([prompt_ids.shape[0]], dtype=torch.int32),
        )

    # ------------------------------------------------------------------
    # training
    # ------------------------------------------------------------------
    def _lm_loss(self, ids, media, position_ids, loss_start: int):
        model = self.model
        embeds = build_sequence_embeds(model, ids, media, [self._placeholder[KIND_IMAGE]])
        hidden = model.forward_hidden(embeds, position_ids)[0, loss_start - 1 : -1]
        targets = ids[loss_start:]
        lm_head = model.lm_head

        def chunk_ce(h, t):
            return F.cross_entropy(lm_head(h).float(), t, reduction="sum")

        total = hidden.new_zeros((), dtype=torch.float32)
        for s0 in range(0, targets.shape[0], self.loss_chunk):
            h, t = hidden[s0 : s0 + self.loss_chunk], targets[s0 : s0 + self.loss_chunk]
            if torch.is_grad_enabled():
                total = total + torch.utils.checkpoint.checkpoint(chunk_ce, h, t, use_reentrant=False)
            else:
                total = total + chunk_ce(h, t)
        return total / targets.shape[0]

    def get_llm_loss(self, batch) -> torch.Tensor:
        """Next-token cross-entropy on the output span, from the cached
        conversation artifact. Called by the trainer's train_llm_accumulation;
        logs ride in additional_loss_logs."""
        if self.model.device == torch.device("cpu"):
            self.model.to(self.device_torch)
        lat = batch.latents
        if not isinstance(lat, DTO) or lat.get("ids") is None:
            item = batch.file_items[0]
            if not getattr(item, "is_conversation", False):
                raise ValueError("qwen38 trains on conversation sidecar datasets")
            lat = self.encode_conversation(item)
        if lat.shape[0] != 1:
            raise ValueError("qwen38 trains with batch_size 1")
        media = lat.tensor[0].to(self.device_torch, self.torch_dtype)
        ids = lat.get("ids")[0].long().to(self.device_torch)
        position_ids = lat.get("position_ids")[0].long().to(self.device_torch)
        loss_start = int(lat.get("loss_start"))
        ce = self._lm_loss(ids, media, position_ids, loss_start)
        if self.debug:
            print_acc(
                f"qwen38: seq {ids.shape[0]} target {ids.shape[0] - loss_start} ce {ce.item():.4f} "
                f"peak {torch.cuda.max_memory_allocated() / 1e9:.1f} GB"
            )
        self.additional_loss_logs = {"loss/ce": ce.detach().float().item()}
        return ce

    # ------------------------------------------------------------------
    # sampling: ctrl_img is the media file, prompt is the instruction, output is text
    # ------------------------------------------------------------------
    def get_generation_pipeline(self):
        return self

    def _load_sample_media(self, path: str):
        kind = _media_kind(path)
        if kind == KIND_IMAGE:
            from PIL import Image

            return kind, {"type": "image", "image": "x"}, {"images": [Image.open(path).convert("RGB")]}
        from transformers.video_utils import load_video

        frames = load_video(path, fps=self.video_fps)
        if isinstance(frames, tuple):
            frames = frames[0]
        return kind, {"type": "video", "video": "x"}, {"videos": [frames], "fps": self.video_fps}

    def prepare_caption_inputs(self, media_path: Optional[str], instruction: Optional[str] = None) -> dict:
        """CPU half of caption(): media decode + processor. Safe in a worker thread."""
        instruction = instruction.strip() if instruction and instruction.strip() else self.instruction
        content = []
        media_kwargs = {}
        kind = None
        if media_path:
            kind, entry, media_kwargs = self._load_sample_media(media_path)
            content.append(entry)
        content.append({"type": "text", "text": instruction})
        msgs = ([{"role": "system", "content": self.system_prompt}] if self.system_prompt is not None else []) + [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, **self._template_kwargs())
        inputs = self.processor(text=text, return_tensors="pt", min_pixels=self.min_pixels, max_pixels=self.max_pixels, **media_kwargs)
        return {"kind": kind, "inputs": inputs}

    @torch.no_grad()
    def generate_from_inputs(self, prepared: dict, max_new_tokens: Optional[int] = None, mtp_depth: Optional[int] = None) -> List[int]:
        """GPU half of caption(): vision tower, prefill and decode. Returns token ids."""
        tower = self._vision()
        if tower.device == torch.device("cpu"):
            tower.to(self.device_torch)
        if self.model.device == torch.device("cpu"):
            self.model.to(self.device_torch)
        kind, inputs = prepared["kind"], prepared["inputs"]
        ids = inputs["input_ids"][0].to(self.device_torch)
        types = inputs["mm_token_type_ids"][0].to(self.device_torch)
        model = self.model
        media = None
        image_grid = video_grid = None
        if kind == KIND_IMAGE:
            image_grid = inputs["image_grid_thw"].to(self.device_torch)
            media = tower.get_image_features(inputs["pixel_values"].to(self.device_torch), image_grid)
        elif kind == KIND_VIDEO:
            video_grid = inputs["video_grid_thw"].to(self.device_torch)
            media = tower.get_image_features(inputs["pixel_values_videos"].to(self.device_torch), video_grid)
        embeds = build_sequence_embeds(model, ids, media, [self._placeholder[KIND_IMAGE], self._placeholder[KIND_VIDEO]])
        position_ids, delta = model.get_rope_index(ids, types, image_grid_thw=image_grid, video_grid_thw=video_grid)
        t0 = time.perf_counter()
        out_ids = model.generator.generate(
            embeds, ids[None], position_ids, delta,
            max_new_tokens=self.max_new_tokens if max_new_tokens is None else max_new_tokens,
            eos_ids=self._eos_ids, mtp_depth=self.mtp_depth if mtp_depth is None else mtp_depth,
        )
        self.last_generation_stats = dict(model.generator.last_stats, seconds=time.perf_counter() - t0, prompt_tokens=int(ids.shape[0]))
        return out_ids

    def decode_text(self, out_ids: List[int]) -> str:
        text = self.tokenizer.decode(out_ids, skip_special_tokens=True)
        # thinking mode: keep only the answer after the think block
        if "</think>" in text:
            text = text.split("</think>")[-1]
        return text.strip()

    @torch.no_grad()
    def caption(self, media_path: Optional[str], instruction: Optional[str] = None, max_new_tokens: Optional[int] = None, mtp_depth: Optional[int] = None) -> str:
        """Generate text for one media file (or none). Also the sample path."""
        return self.decode_text(self.generate_from_inputs(self.prepare_caption_inputs(media_path, instruction), max_new_tokens, mtp_depth))

    def generate_single_image(self, pipeline, gen_config: GenerateImageConfig, conditional_embeds, unconditional_embeds, generator, extra):
        # a str return is saved as <sample>.txt by GenerateImageConfig.save_image
        return self.caption(gen_config.ctrl_img, gen_config.prompt)
