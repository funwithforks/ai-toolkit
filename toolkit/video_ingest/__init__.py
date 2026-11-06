"""Float video/audio ingest for phased-loading latent caching.

Ported near-verbatim from the superdata pipeline (../projects/superdata),
where this exact chain is production-proven on LTX 2.3. The contract it
implements is the owner's: what hits the model must be pristine -- ffmpeg
float decode (gbrpf32le planar fp32, correct color management, lanczos
geometry) straight into a fp32 RGB CHW [-1, 1] tensor and from there into
the vae. No uint8 round trip, no PIL, no per-frame python image ops; the
only intermediate representation between the file and the vae is one fp32
tensor.

Module map (names kept from the source repo):

  probe.py         ffprobe VideoInfo (dims/fps/HDR/10-bit/color tags)
  pixel_tensor.py  the raw pipe -> fp32 training tensor
  decoder_spec.py  the float decode filter chains (ENCODER POLICY frozen)
  video_loader.py  frame-index planning + the select= float decode
  audio_span.py    seek-based fp32 audio span decode

Fork-specific adaptations, each documented at the site:
  - the frame-index planner reproduces the fork's own branch table
    (trim_auto_frame_count_tail, the random start, the duplicate-frame
    repetition) -- the fork is the authority on which frames train;
  - geometry order is scale-then-crop to match the fork's bucket bookkeeping
    (superdata's exports are pre-geometry, so its dataset path never faces
    this);
  - the encode seed is an optional config value, not a hard behavior.

This package is inert until the dataloader's video branch routes through it.
"""

from .probe import VideoInfo, probe_video
from .pixel_tensor import LoadedVideo, gbrpf32le_to_training_tensor
from .decoder_spec import (
    CropRect,
    SEEK_MARGIN_SEC,
    append_source_seek,
    build_float_decode_vf,
    reserved_color_tag_fixup,
    snap_rect,
)
from .video_loader import (
    LatentPixelMismatchError,
    assert_identity_bucket,
    bucket_is_identity,
    compute_auto_frame_count,
    load_dataset_float_tensors,
    plan_frame_indices,
)
from .audio_span import (
    load_audio_from_video,
    load_audio_span,
    load_clip_audio,
    set_encode_seed,
    waveform_to_stereo,
)
from .image_pixel_loader import (
    LoadedImage,
    _resize_lanczos,
    load_dataset_image_tensor,
    load_image_training_tensor,
)
