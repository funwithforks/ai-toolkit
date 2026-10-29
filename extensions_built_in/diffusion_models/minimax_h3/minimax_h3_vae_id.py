"""minimax_h3 vae_id: the video + audio VAE bundle behind one id.

Registration side of the H3 bundle: the load. Everything the vae does --
video encode/decode, the seeded keyframe recipe, the audio vae calls --
lives on ``MiniMaxH3VaeBundle`` itself (minimax_h3.py), which is the object
this entry loads. What the latent caches store is exactly what the bundle's
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

from toolkit.basic import flush
from toolkit.models.vae_registry import VaeEntry, register_vae_entry

from .src.audio_vae import MiniMaxH3AudioVAE
from .src.vae import MiniMaxH3VideoVAE


def _load_minimax_h3_vae(model):
    """-> MiniMaxH3VaeBundle(video_vae, audio_vae), both resident on
    model.vae_device_torch at their shipped dtypes."""
    # lazy: the bundle lives with the model, and the model imports this
    # module for its entry registration at module level
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


register_vae_entry(
    VaeEntry(
        vae_id="minimax_h3",
        components={"video_vae": MiniMaxH3VideoVAE, "audio_vae": MiniMaxH3AudioVAE},
        load=_load_minimax_h3_vae,
        # members keep their shipped precisions (fp16 video / fp32 audio):
        # the loaders take no dtype, and placement only moves
        bundle=True,
    )
)
