"""ltx2 vae_id: the LTX-2 video + audio VAE bundle behind one id.

Cache identity is the ``vae_id`` itself: latent cache files hash the id, so
every model declaring it shares one latent namespace and one set of files.
The video VAE (conv) and audio VAE are the LTX-2 family's own; the vocoder
is deliberately NOT loaded here -- it is decode-only, so the staged decode
path streams it on demand and frees it after (one vae at a time), and phase
1 (latent caching) never pays for it.

The vae does the vae: encode/decode live on the bundle (ltx2.ComboVae),
computed at the members' own shipped precision from their own normalization
buffers. Placement upstream only moves the bundle; there is no cast.
"""

import torch
from accelerate import init_empty_weights

from toolkit.models.vae_registry import VaeEntry, register_vae_entry
from toolkit.models.v2.vae.ltx2 import LTX2AudioVAE, LTX2VideoVAE

from .convert_ltx2_to_diffusers import (
    get_ltx2_audio_vae_config,
    get_ltx2_video_vae_config,
)
from .ltx23_files import (
    LTX2_VAE_ID,
    LTX23AudioVAE,
    LTX23VideoVAE,
    resolve_component_source,
)
from .ltx23_stream import build_stream_plan, stream_assign, stream_meta_check


def _load_ltx2_vae(model):
    from toolkit.util.weight_source import WeightSource

    from .ltx2 import AudioProcessor, ComboVae

    stream_dev = model.vae_device_torch
    # the pipeline is the shared container the cache passes read their
    # components from; in the phased train order it must exist before
    # phase 1's cache passes. Slots are filled by each phase and re-synced
    # by the model's vae property.
    model._ensure_phase_pipeline()

    video_path, video_prefix = resolve_component_source(model, "video_vae")
    v_config, v_rename, v_special = get_ltx2_video_vae_config("2.3")
    video_plan = build_stream_plan(
        WeightSource.open(video_path), video_prefix, v_rename, v_special
    )
    with init_empty_weights():
        video_vae = LTX2VideoVAE.from_config(v_config["diffusers_config"])
    stream_assign(video_vae, video_plan, stream_dev, "ltx2 video vae")
    stream_meta_check(video_vae, "ltx2 video vae")
    model.print_and_status_update(
        f" - video vae streamed from {video_path}"
    )
    del video_plan

    audio_path, audio_prefix = resolve_component_source(model, "audio_vae")
    a_config, a_rename, a_special = get_ltx2_audio_vae_config("2.3")
    audio_plan = build_stream_plan(
        WeightSource.open(audio_path), audio_prefix, a_rename, a_special
    )
    with init_empty_weights():
        audio_vae = LTX2AudioVAE.from_config(a_config["diffusers_config"])
    stream_assign(audio_vae, audio_plan, stream_dev, "ltx2 audio vae")
    stream_meta_check(audio_vae, "ltx2 audio vae")
    model.print_and_status_update(
        f" - audio vae streamed from {audio_path}"
    )
    del audio_plan

    bundle = ComboVae(video_vae, audio_vae, vocoder=None)
    bundle.low_vram_tiling = bool(getattr(model.model_config, "low_vram", False))
    audio_config = a_config["diffusers_config"]
    bundle.audio_processor = AudioProcessor(
        sample_rate=audio_config["sample_rate"],
        mel_bins=audio_config["mel_bins"],
        mel_hop_length=audio_config["mel_hop_length"],
        n_fft=1024,  # todo get this from vae if we can, I couldnt find it.
    ).to(stream_dev, dtype=torch.float32)
    return bundle


register_vae_entry(
    VaeEntry(
        vae_id=LTX2_VAE_ID,
        components={"video_vae": LTX23VideoVAE, "audio_vae": LTX23AudioVAE},
        load=_load_ltx2_vae,
        bundle=True,
    )
)
