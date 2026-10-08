"""Staged (split denoise/decode) sampling for LTX-2.x.

Vendored from the pinned diffusers ``LTX2Pipeline.__call__``
(diffusers @ c943837899, pipeline_ltx2.py), cut in half at the phase
boundary: :func:`denoise_staged` mirrors the call through the end of the
denoising loop and returns the raw latents, while the decode half runs
later through :func:`decode_video_latents` / :func:`decode_audio_latents`
with the vae resident (see LTX25Model.decode_staged_samples). This keeps
the vae bundle and the text connectors out of the denoise round - the same
staged-sampling concept the H3 sampler uses, on the ltx2 stack.

The guidance loop is transcribed so behavior matches the monolithic
pipeline; every helper it reaches for (``retrieve_timesteps``,
``calculate_shift``, ``rescale_noise_cfg``, the pack/unpack staticmethods,
``randn_tensor``) is the same diffusers code the real call uses.
"""

import copy
from typing import Optional

import numpy as np
import torch
from tqdm import tqdm

from diffusers.models.autoencoders.autoencoder_kl_ltx2_audio import (
    LATENT_DOWNSAMPLE_FACTOR,
)
from diffusers.pipelines.ltx2.pipeline_ltx2 import (
    LTX2Pipeline,
    calculate_shift,
    retrieve_timesteps,
    rescale_noise_cfg,
)
from diffusers.utils.torch_utils import randn_tensor


def _convert_velocity_to_x0(sample, velocity, step_idx, scheduler):
    # pipeline_ltx2.py convert_velocity_to_x0 (transcribed)
    return sample - velocity * scheduler.sigmas[step_idx]


def _convert_x0_to_velocity(sample, denoised_output, step_idx, scheduler):
    # pipeline_ltx2.py convert_x0_to_velocity (transcribed)
    return (sample - denoised_output) / scheduler.sigmas[step_idx]


def _prepare_video_latents(
    batch_size,
    num_channels_latents,
    height,
    width,
    num_frames,
    spatial_ratio,
    temporal_ratio,
    patch,
    patch_t,
    dtype,
    device,
    generator,
):
    # pipeline_ltx2.py prepare_latents, generation path (transcribed)
    height = height // spatial_ratio
    width = width // spatial_ratio
    num_frames = (num_frames - 1) // temporal_ratio + 1
    shape = (batch_size, num_channels_latents, num_frames, height, width)
    latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
    latents = LTX2Pipeline._pack_latents(latents, patch, patch_t)
    return latents


def _prepare_audio_latents(
    batch_size,
    num_channels_latents,
    audio_latent_length,
    num_mel_bins,
    mel_compression_ratio,
    dtype,
    device,
    generator,
):
    # pipeline_ltx2.py prepare_audio_latents, generation path (transcribed)
    latent_mel_bins = num_mel_bins // mel_compression_ratio
    shape = (batch_size, num_channels_latents, audio_latent_length, latent_mel_bins)
    latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
    latents = LTX2Pipeline._pack_audio_latents(latents)
    return latents


@torch.no_grad()
def denoise_staged(
    *,
    transformer,
    scheduler,
    video_connector_embeds: torch.Tensor,
    audio_connector_embeds: torch.Tensor,
    connector_attention_mask: torch.Tensor,
    neg_video_connector_embeds: Optional[torch.Tensor],
    neg_audio_connector_embeds: Optional[torch.Tensor],
    neg_connector_attention_mask: Optional[torch.Tensor],
    height: int,
    width: int,
    num_frames: int,
    frame_rate: float,
    num_inference_steps: int,
    guidance_scale: float,
    stg_scale: float,
    modality_scale: float,
    guidance_rescale: float,
    audio_guidance_scale: Optional[float],
    audio_stg_scale: Optional[float],
    audio_modality_scale: Optional[float],
    audio_guidance_rescale: Optional[float],
    spatio_temporal_guidance_blocks,
    use_cross_timestep: bool,
    generator: torch.Generator,
    device: torch.device,
    dtype: torch.dtype,
    video_config: dict,
    audio_config: dict,
    desc: str = "Denoising",
):
    """Denoise one staged sample.

    Consumes CONNECTOR outputs (the cached embedding form) and returns
    ``(video_latents, audio_latents, meta)``. The video latents are unpacked
    but still normalized (mean/std space); the audio latents stay packed and
    normalized - exactly the spaces ``decode_video_latents`` /
    ``decode_audio_latents`` below invert (the training-side latent writer
    stores the same spaces).
    """
    audio_guidance_scale = audio_guidance_scale or guidance_scale
    audio_stg_scale = audio_stg_scale or stg_scale
    audio_modality_scale = audio_modality_scale or modality_scale
    audio_guidance_rescale = audio_guidance_rescale or guidance_rescale

    do_cfg = (guidance_scale > 1.0) or (audio_guidance_scale > 1.0)
    do_stg = (stg_scale > 0.0) or (audio_stg_scale > 0.0)
    do_modality = (modality_scale > 1.0) or (audio_modality_scale > 1.0)

    batch_size = video_connector_embeds.shape[0]

    # CFG pair concatenated exactly as __call__ does after encode_prompt
    if do_cfg:
        c_video = torch.cat([neg_video_connector_embeds, video_connector_embeds], 0)
        c_audio = torch.cat([neg_audio_connector_embeds, audio_connector_embeds], 0)
        c_mask = torch.cat([neg_connector_attention_mask, connector_attention_mask], 0)
    else:
        c_video = video_connector_embeds
        c_audio = audio_connector_embeds
        c_mask = connector_attention_mask

    # pack/unpack patches belong to the TRANSFORMER (pipeline_ltx2.py
    # __init__ :254-259); the latent-channel latents are already in packed
    # space (patch_size 1). The pixel->latent grid division uses the vae's
    # own compression ratios (:241-246) - taken straight from the vae
    # config table since the vae module is deliberately not resident here.
    patch = transformer.config.patch_size
    patch_t = transformer.config.patch_size_t
    spatial_ratio = video_config["spatial_compression_ratio"]
    temporal_ratio = video_config["temporal_compression_ratio"]
    latent_num_frames = (num_frames - 1) // temporal_ratio + 1
    latent_height = height // spatial_ratio
    latent_width = width // spatial_ratio

    latents = _prepare_video_latents(
        batch_size,
        transformer.config.in_channels,
        height,
        width,
        num_frames,
        spatial_ratio,
        temporal_ratio,
        patch,
        patch_t,
        torch.float32,
        device,
        generator,
    )

    num_mel_bins = audio_config["mel_bins"]
    duration_s = num_frames / frame_rate
    audio_latents_per_second = (
        audio_config["sample_rate"]
        / audio_config["mel_hop_length"]
        / float(LATENT_DOWNSAMPLE_FACTOR)
    )
    audio_num_frames = round(duration_s * audio_latents_per_second)
    audio_latents = _prepare_audio_latents(
        batch_size,
        audio_config["latent_channels"],
        audio_num_frames,
        num_mel_bins,
        LATENT_DOWNSAMPLE_FACTOR,
        torch.float32,
        device,
        generator,
    )

    # 5. Prepare timesteps
    sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
    mu = calculate_shift(
        scheduler.config.get("max_image_seq_len", 4096),
        scheduler.config.get("base_image_seq_len", 1024),
        scheduler.config.get("max_image_seq_len", 4096),
        scheduler.config.get("base_shift", 0.95),
        scheduler.config.get("max_shift", 2.05),
    )
    audio_scheduler = copy.deepcopy(scheduler)
    retrieve_timesteps(
        audio_scheduler, num_inference_steps, device, None, sigmas=sigmas, mu=mu
    )
    timesteps, num_inference_steps = retrieve_timesteps(
        scheduler, num_inference_steps, device, None, sigmas=sigmas, mu=mu
    )
    scheduler.set_begin_index(0)
    audio_scheduler.set_begin_index(0)

    # 6. Prepare micro-conditions (pos ids are constant across steps)
    video_coords = transformer.rope.prepare_video_coords(
        latents.shape[0],
        latent_num_frames,
        latent_height,
        latent_width,
        latents.device,
        fps=frame_rate,
    )
    audio_coords = transformer.audio_rope.prepare_audio_coords(
        audio_latents.shape[0], audio_num_frames, audio_latents.device
    )
    if do_cfg:
        video_coords = video_coords.repeat((2,) + (1,) * (video_coords.ndim - 1))
        audio_coords = audio_coords.repeat((2,) + (1,) * (audio_coords.ndim - 1))

    # 7. Denoising loop (transcribed from pipeline_ltx2.py __call__)
    for i, t in enumerate(tqdm(timesteps, desc=desc, leave=False)):
        latent_model_input = torch.cat([latents] * 2) if do_cfg else latents
        latent_model_input = latent_model_input.to(dtype)
        audio_latent_model_input = (
            torch.cat([audio_latents] * 2) if do_cfg else audio_latents
        )
        audio_latent_model_input = audio_latent_model_input.to(dtype)

        timestep = t.expand(latent_model_input.shape[0])

        with transformer.cache_context("cond_uncond"):
            noise_pred_video, noise_pred_audio = transformer(
                hidden_states=latent_model_input,
                audio_hidden_states=audio_latent_model_input,
                encoder_hidden_states=c_video,
                audio_encoder_hidden_states=c_audio,
                timestep=timestep,
                sigma=timestep,  # Used by LTX-2.3
                encoder_attention_mask=c_mask,
                audio_encoder_attention_mask=c_mask,
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
        noise_pred_video = noise_pred_video.float()
        noise_pred_audio = noise_pred_audio.float()

        if do_cfg:
            noise_pred_video_uncond_text, noise_pred_video = noise_pred_video.chunk(2)
            noise_pred_video = _convert_velocity_to_x0(
                latents, noise_pred_video, i, scheduler
            )
            noise_pred_video_uncond_text = _convert_velocity_to_x0(
                latents, noise_pred_video_uncond_text, i, scheduler
            )
            video_cfg_delta = (guidance_scale - 1) * (
                noise_pred_video - noise_pred_video_uncond_text
            )

            noise_pred_audio_uncond_text, noise_pred_audio = noise_pred_audio.chunk(2)
            noise_pred_audio = _convert_velocity_to_x0(
                audio_latents, noise_pred_audio, i, audio_scheduler
            )
            noise_pred_audio_uncond_text = _convert_velocity_to_x0(
                audio_latents, noise_pred_audio_uncond_text, i, audio_scheduler
            )
            audio_cfg_delta = (audio_guidance_scale - 1) * (
                noise_pred_audio - noise_pred_audio_uncond_text
            )

            if do_stg or do_modality:
                if i == 0:
                    video_prompt_embeds = c_video.chunk(2, dim=0)[1]
                    audio_prompt_embeds = c_audio.chunk(2, dim=0)[1]
                    prompt_attn_mask = c_mask.chunk(2, dim=0)[1]
                    video_pos_ids = video_coords.chunk(2, dim=0)[0]
                    audio_pos_ids = audio_coords.chunk(2, dim=0)[0]

                timestep = timestep.chunk(2, dim=0)[0]
        else:
            video_cfg_delta = audio_cfg_delta = 0

            video_prompt_embeds = c_video
            audio_prompt_embeds = c_audio
            prompt_attn_mask = c_mask

            video_pos_ids = video_coords
            audio_pos_ids = audio_coords

            noise_pred_video = _convert_velocity_to_x0(
                latents, noise_pred_video, i, scheduler
            )
            noise_pred_audio = _convert_velocity_to_x0(
                audio_latents, noise_pred_audio, i, audio_scheduler
            )

        if do_stg:
            with transformer.cache_context("uncond_stg"):
                noise_pred_video_stg, noise_pred_audio_stg = transformer(
                    hidden_states=latents.to(dtype=dtype),
                    audio_hidden_states=audio_latents.to(dtype=dtype),
                    encoder_hidden_states=video_prompt_embeds,
                    audio_encoder_hidden_states=audio_prompt_embeds,
                    timestep=timestep,
                    sigma=timestep,  # Used by LTX-2.3
                    encoder_attention_mask=prompt_attn_mask,
                    audio_encoder_attention_mask=prompt_attn_mask,
                    num_frames=latent_num_frames,
                    height=latent_height,
                    width=latent_width,
                    fps=frame_rate,
                    audio_num_frames=audio_num_frames,
                    video_coords=video_pos_ids,
                    audio_coords=audio_pos_ids,
                    isolate_modalities=False,
                    # Use STG at given blocks to perturb model
                    spatio_temporal_guidance_blocks=spatio_temporal_guidance_blocks,
                    perturbation_mask=None,
                    use_cross_timestep=use_cross_timestep,
                    attention_kwargs=None,
                    return_dict=False,
                )
            noise_pred_video_stg = noise_pred_video_stg.float()
            noise_pred_audio_stg = noise_pred_audio_stg.float()
            noise_pred_video_stg = _convert_velocity_to_x0(
                latents, noise_pred_video_stg, i, scheduler
            )
            noise_pred_audio_stg = _convert_velocity_to_x0(
                audio_latents, noise_pred_audio_stg, i, audio_scheduler
            )

            video_stg_delta = stg_scale * (noise_pred_video - noise_pred_video_stg)
            audio_stg_delta = audio_stg_scale * (noise_pred_audio - noise_pred_audio_stg)
        else:
            video_stg_delta = audio_stg_delta = 0

        if do_modality:
            with transformer.cache_context("uncond_modality"):
                noise_pred_video_mod, noise_pred_audio_mod = transformer(
                    hidden_states=latents.to(dtype=dtype),
                    audio_hidden_states=audio_latents.to(dtype=dtype),
                    encoder_hidden_states=video_prompt_embeds,
                    audio_encoder_hidden_states=audio_prompt_embeds,
                    timestep=timestep,
                    sigma=timestep,  # Used by LTX-2.3
                    encoder_attention_mask=prompt_attn_mask,
                    audio_encoder_attention_mask=prompt_attn_mask,
                    num_frames=latent_num_frames,
                    height=latent_height,
                    width=latent_width,
                    fps=frame_rate,
                    audio_num_frames=audio_num_frames,
                    video_coords=video_pos_ids,
                    audio_coords=audio_pos_ids,
                    # Turn off A2V and V2A cross attn to isolate modalities
                    isolate_modalities=True,
                    spatio_temporal_guidance_blocks=None,
                    perturbation_mask=None,
                    use_cross_timestep=use_cross_timestep,
                    attention_kwargs=None,
                    return_dict=False,
                )
            noise_pred_video_mod = noise_pred_video_mod.float()
            noise_pred_audio_mod = noise_pred_audio_mod.float()
            noise_pred_video_mod = _convert_velocity_to_x0(
                latents, noise_pred_video_mod, i, scheduler
            )
            noise_pred_audio_mod = _convert_velocity_to_x0(
                audio_latents, noise_pred_audio_mod, i, audio_scheduler
            )

            video_modality_delta = (modality_scale - 1) * (
                noise_pred_video - noise_pred_video_mod
            )
            audio_modality_delta = (audio_modality_scale - 1) * (
                noise_pred_audio - noise_pred_audio_mod
            )
        else:
            video_modality_delta = audio_modality_delta = 0

        # Now apply all guidance terms
        noise_pred_video_g = (
            noise_pred_video + video_cfg_delta + video_stg_delta + video_modality_delta
        )
        noise_pred_audio_g = (
            noise_pred_audio + audio_cfg_delta + audio_stg_delta + audio_modality_delta
        )

        # Apply LTX-2.X guidance rescaling
        if guidance_rescale > 0:
            noise_pred_video = rescale_noise_cfg(
                noise_pred_video_g, noise_pred_video, guidance_rescale=guidance_rescale
            )
        else:
            noise_pred_video = noise_pred_video_g

        if audio_guidance_rescale > 0:
            noise_pred_audio = rescale_noise_cfg(
                noise_pred_audio_g,
                noise_pred_audio,
                guidance_rescale=audio_guidance_rescale,
            )
        else:
            noise_pred_audio = noise_pred_audio_g

        # Convert back to velocity for scheduler
        noise_pred_video = _convert_x0_to_velocity(
            latents, noise_pred_video, i, scheduler
        )
        noise_pred_audio = _convert_x0_to_velocity(
            audio_latents, noise_pred_audio, i, audio_scheduler
        )

        # compute the previous noisy sample x_t -> x_t-1
        # (scheduler.step may be wrapped by the toolkit step hooks - same
        # scheduler objects the inline pipeline path uses)
        latents = scheduler.step(noise_pred_video, t, latents, return_dict=False)[0]
        audio_latents = audio_scheduler.step(
            noise_pred_audio, t, audio_latents, return_dict=False
        )[0]

    # stop at the loop end - no vae, no denormalization. Video latents are
    # unpacked (5D, still mean/std normalized); audio stays packed/normed.
    latents = LTX2Pipeline._unpack_latents(
        latents, latent_num_frames, latent_height, latent_width, patch, patch_t
    )
    meta = {
        "latent_num_frames": latent_num_frames,
        "latent_height": latent_height,
        "latent_width": latent_width,
        "audio_num_frames": audio_num_frames,
    }
    return latents, audio_latents, meta


@torch.no_grad()
def decode_video_latents(
    video_vae: torch.nn.Module,
    latents: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Inverse of the normalized latent space; returns RGB frames
    (T, C, H, W) in [-1, 1]. Denormalizes with the resident vae's own
    stats - the vae module is loaded by the caller for this phase."""
    if video_vae.device.type == "cpu":
        video_vae.to(device)
    video_vae.eval()

    batch = latents.unsqueeze(0) if latents.ndim == 4 else latents
    batch = batch.to(device, dtype=dtype)
    latents_mean = video_vae.latents_mean.view(1, -1, 1, 1, 1).to(
        batch.device, batch.dtype
    )
    latents_std = video_vae.latents_std.view(1, -1, 1, 1, 1).to(
        batch.device, batch.dtype
    )
    denorm = batch * latents_std + latents_mean
    decoded = video_vae.decode(denorm).sample
    # (B, C, T, H, W) -> (T, C, H, W)
    return decoded.squeeze(0).permute(1, 0, 2, 3).contiguous()


@torch.no_grad()
def decode_audio_latents(
    audio_vae: torch.nn.Module,
    vocoder: torch.nn.Module,
    latents: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple:
    """Inverse of the packed audio latent space; returns stereo waveform
    (C, T) float32 and the vocoder output sample rate. Same diffusers
    staticmethods the monolithic pipeline call uses."""
    if audio_vae.device.type == "cpu":
        audio_vae.to(device)
    if next(vocoder.parameters()).device.type == "cpu":
        vocoder.to(device)
    audio_vae.eval()
    vocoder.eval()

    batch = latents.unsqueeze(0) if latents.ndim == 2 else latents
    batch = batch.to(device, dtype=dtype)
    denorm = LTX2Pipeline._denormalize_audio_latents(
        batch,
        audio_vae.latents_mean,
        audio_vae.latents_std,
    )
    latent_mel_bins = audio_vae.config.mel_bins // audio_vae.mel_compression_ratio
    unpacked = LTX2Pipeline._unpack_audio_latents(
        denorm,
        denorm.shape[1],
        num_mel_bins=latent_mel_bins,
    )
    mel = audio_vae.decode(unpacked.to(dtype=audio_vae.dtype), return_dict=False)[0]
    waveform = vocoder(mel)
    if waveform.ndim == 3:
        waveform = waveform.squeeze(0)
    sample_rate = int(vocoder.config.output_sampling_rate)
    return waveform.to(dtype=torch.float32), sample_rate
