"""minimax_h3 vae_id: the video + audio VAE bundle behind one id.

Registration side of the H3 bundle (loads, placement policy, cache-tag
pin). The encode/decode primitives stay on the model until the encode
migration; until then ``encode_images``/``decode_latents`` stay None and
the model methods are the call surface.

The load callable is the model's former ``_load_vaes`` body, verbatim:
each member streams straight onto the vae device through the comfy door
(component candidate metadata lives on the component classes), and the
shipped precisions (video fp16, audio fp32) survive placement because
the bundle pins ``keep_shipped`` -- the mixin's placement step only
moves it.

``latent_space_version`` is the explicit cache tag the model used to pin
in its own ``__init__``; from the id it applies at construction, so runs
that never load the vae (phases without 1) still hash the same tag into
cache filenames.
"""

from toolkit.basic import flush
from toolkit.models.vae_registry import VaeEntry, register_vae_entry

from .minimax_h3 import MiniMaxH3VaeBundle
from .src.audio_vae import MiniMaxH3AudioVAE
from .src.vae import MiniMaxH3VideoVAE


def _load_minimax_h3_vae(model):
    """-> MiniMaxH3VaeBundle(video_vae, audio_vae), both resident on
    model.vae_device_torch at their shipped dtypes."""
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
        # members ship at mixed precisions on purpose
        dtype_policy="keep_shipped",
        bundle=True,
        latent_space_version="minimax_h3_v1",
    )
)
