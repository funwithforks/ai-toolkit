"""minimax_h3 vae_id: the video + audio VAE bundle behind one id.

Registration side of the H3 bundle: the load, and the encode/decode
primitives the id owns (video encode/decode, the seeded keyframe recipe, the
audio vae calls). What the latent caches store is exactly what
``encode_images`` returns. The model keeps the transformer's concerns: the
packed audio/video token layout around these, and borrowing the vae in and
out under staged sampling.

The load callable is the model's former ``_load_vaes`` body, verbatim:
each member streams straight onto the vae device through the comfy door
(component candidate metadata lives on the component classes), and the
shipped precisions (video fp16, audio fp32) survive because the vendored
loaders take no dtype at all -- the mixin's placement step only moves.

Cache identity is the ``vae_id`` itself: files hash the id, so every
model declaring the same id shares the same latents -- no version knob.
"""

import torch
from toolkit.basic import flush
from toolkit.models.vae_registry import VaeEntry, register_vae_entry

from .src import packing
from .src.audio_vae import MiniMaxH3AudioVAE
from .src.vae import MiniMaxH3VideoVAE


def _load_minimax_h3_vae(model):
    """-> MiniMaxH3VaeBundle(video_vae, audio_vae), both resident on
    model.vae_device_torch at their shipped dtypes."""
    # lazy: the bundle lives with the model, and the model imports this
    # module's primitives at module level
    from .minimax_h3 import MiniMaxH3VaeBundle

    model.print_and_status_update("Loading video VAE")
    video_vae = MiniMaxH3VideoVAE.load_to_device(
        model._resolve_comfy_file("video_vae", MiniMaxH3VideoVAE),
        model.vae_device_torch,
    )
    model.print_and_status_update("Loading audio VAE")
    audio_vae = MiniMaxH3AudioVAE.load_to_device(
        model._resolve_comfy_file("audio_vae", MiniMaxH3AudioVAE),
        model.vae_device_torch,
    )
    flush()
    return MiniMaxH3VaeBundle(video_vae, audio_vae)


# ----------------------------------------------------------------------
# encode / decode primitives (owned by the vae, shared with the model)
# ----------------------------------------------------------------------


@torch.no_grad()
def encode_images(model, image_list, device=None, dtype=None):
    """Images (C, H, W) or videos (T, C, H, W) in [-1, 1] -> normalized
    video latents (B, 24, t, h, w). Video frame counts are trimmed down to
    the VAE's 17n+5 grid when needed."""
    if device is None:
        device = model.vae_device_torch
    if dtype is None:
        dtype = model.vae_torch_dtype
    if model.vae.device == torch.device("cpu"):
        model.vae.to(model.vae_device_torch)

    items = []
    for image in image_list:
        if image.ndim == 3:
            items.append(image.unsqueeze(1))  # (C, 1, H, W)
        elif image.ndim == 4:
            items.append(image.permute(1, 0, 2, 3))  # (C, T, H, W)
        else:
            raise ValueError(f"Invalid image shape: {image.shape}")

    num_frames = items[0].shape[1]
    if num_frames > 1:
        aligned = packing.align_num_frames_down(num_frames)
        if aligned != num_frames and not model._warned_frame_trim:
            print(
                f"MiniMax-H3: trimming {num_frames}-frame clips to {aligned} "
                f"frames (the video VAE needs 17n+5: 5, 22, 39, 56, ...). Set "
                f"the dataset num_frames accordingly to avoid wasted decode."
            )
            model._warned_frame_trim = True
        items = [it[:, :aligned] for it in items]

    batch = torch.stack(items).to(model.vae_device_torch, model.video_vae.dtype)
    latents = model.video_vae.encode(batch, sample=True)
    return latents.to(device, dtype=dtype)


def decode_latents(model, latents, device=None, dtype=None):
    # differentiable: pixel-space losses backprop through the video VAE
    if model.vae.device == torch.device("cpu"):
        model.vae.to(model.vae_device_torch)
    video = model.video_vae.decode(
        latents.to(model.video_vae.device, model.video_vae.dtype)
    )
    if device is not None:
        video = video.to(device, dtype=dtype)
    return video


def encode_keyframe_latents(model, frames):
    """(B, 3, 1, H, W) in [-1, 1] -> normalized latents (B, 24, 1, h, w), with
    the released conditioning recipe: seeded posterior sample (seed 42,
    independent of the request seed) rounded to fp16 before normalization.

    This is an encode path of THIS vae, not a different vae: the residency
    dance for staged sampling stays with the model, the recipe lives here."""
    generator = torch.Generator(device="cpu").manual_seed(packing.KEYFRAME_ENCODE_SEED)
    return model.video_vae.encode(
        frames.to(model.vae_device_torch, model.video_vae.dtype),
        sample=True,
        generator=generator,
        fp16_round=True,
    ).float()


def decode_audio_latents(model, latents):
    # differentiable, like decode_latents
    """(B, 32, T) normalized -> waveform (B, 1, T*800) at 32 kHz."""
    if model.vae.device == torch.device("cpu"):
        model.vae.to(model.vae_device_torch)
    return model.audio_vae.decode(latents.to(model.audio_vae.device, torch.float32))


def encode_audio_latents(model, waveform):
    """Waveform (C, samples) fp32 at 32 kHz -> the mono vae's latents
    (2C, 32, T): each stereo channel rides as its own batch item."""
    return model.audio_vae.encode(waveform.unsqueeze(1))


register_vae_entry(
    VaeEntry(
        vae_id="minimax_h3",
        components={"video_vae": MiniMaxH3VideoVAE, "audio_vae": MiniMaxH3AudioVAE},
        load=_load_minimax_h3_vae,
        encode_images=encode_images,
        decode_latents=decode_latents,
        # members keep their shipped precisions (fp16 video / fp32 audio):
        # the loaders take no dtype, and placement only moves
        bundle=True,
    )
)
