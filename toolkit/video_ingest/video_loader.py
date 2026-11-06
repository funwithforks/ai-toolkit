"""Frame-index planning + the select= float decode.

The frame-index planner reproduces the FORK's own branch table
(``get_auto_frame_count`` + the selection branches in
``load_and_process_video``) — the fork is the authority on which frames
train, so the ported planner carries the fork's trim branch, its random
start, and the duplicate-frame repetition that stretched clips need
(superdata's select= decode emits unique frames once). The decode itself is
superdata's verbatim: one ffmpeg pass, ``select='eq(n,..)'`` + the
fork-geometry float chain, gbrpf32le raw pipe — no uint8, no PIL, no cv2.
"""

from __future__ import annotations

import random
import subprocess

import torch

from .decoder_spec import build_float_decode_vf
from .pixel_tensor import LoadedVideo, gbrpf32le_to_training_tensor
from .probe import VideoInfo, probe_video


class LatentPixelMismatchError(ValueError):
    """Raised when video pixels do not already match the training bucket."""


def bucket_is_identity(
    bucket, width: int, height: int
) -> bool:
    """True when the source frame already matches the training bucket (no resize/crop).

    ``bucket`` is any object with the fork's six geometry fields
    (scale_to_width/height, crop_x/y, crop_width/height) -- the FileItemDTO
    qualifies as-is."""
    return (
        bucket.scale_to_width == width
        and bucket.scale_to_height == height
        and bucket.crop_x == 0
        and bucket.crop_y == 0
        and bucket.crop_width == width
        and bucket.crop_height == height
    )


def assert_identity_bucket(bucket, width: int, height: int) -> None:
    if not bucket_is_identity(bucket, width, height):
        raise LatentPixelMismatchError(
            f"Video {width}x{height} does not match training bucket "
            f"(scale {bucket.scale_to_width}x{bucket.scale_to_height}, "
            f"crop {bucket.crop_x},{bucket.crop_y} "
            f"{bucket.crop_width}x{bucket.crop_height}). "
            "Latent encode refuses resize/crop — fix export pipeline."
        )


def _probe_video_properties(video_path: str) -> tuple[int, int, float, int]:
    proc = subprocess.run(
        [
            "ffprobe",
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_streams",
            "-count_frames",
            video_path,
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    import json

    stream = next(
        s for s in json.loads(proc.stdout)["streams"] if s["codec_type"] == "video"
    )
    width = int(stream["width"])
    height = int(stream["height"])
    fps_raw = stream.get("avg_frame_rate") or stream.get("r_frame_rate") or "24/1"
    num, den = fps_raw.split("/")
    video_fps = float(num) / float(den) if float(den) else 24.0
    nb_frames = stream.get("nb_read_frames") or stream.get("nb_frames")
    if nb_frames:
        total_frames = int(nb_frames)
    else:
        duration = stream.get("duration")
        total_frames = int(round(float(duration) * video_fps)) if duration else 0
    if total_frames <= 0:
        raise RuntimeError(f"Could not determine frame count for {video_path}")
    return width, height, video_fps, total_frames


def compute_auto_frame_count(
    *,
    total_frames: int,
    video_fps: float,
    dataset_fps: float,
    temporal_compression: int = 8,
    frame_count_snapper=None,
    trim_auto_frame_count_tail: bool = False,
) -> int:
    """The fork's get_auto_frame_count, verbatim (A1 snapper / A2 trim / A3)."""
    vid_length_seconds = total_frames / video_fps
    desired_num_frames = int(vid_length_seconds * dataset_fps)

    if frame_count_snapper is not None:
        # model-specific valid-frame-count grid (e.g. minimax_h3's 17n+5)
        desired_num_frames = frame_count_snapper(desired_num_frames)
    else:
        # make sure it is divisible by temporal_compression
        if trim_auto_frame_count_tail:
            # snap to the largest valid count that fits inside the video (after the
            # key frame +1 below) so trim mode never overshoots the source, which
            # would freeze the last frame and pad the audio tail with silence
            desired_num_frames = (
                max(0, desired_num_frames - 1) // temporal_compression * temporal_compression
            )
        else:
            desired_num_frames = (
                desired_num_frames // temporal_compression * temporal_compression
            )

        # TODO, all models currently add a key frame, but future models may not, update here if this changes.
        desired_num_frames += 1  # add one for the key frame that is always added

    return desired_num_frames


def plan_frame_indices(
    *,
    total_frames: int,
    video_fps: float,
    dataset_fps: float,
    num_frames: int = 1,
    auto_frame_count: bool = False,
    shrink_video_to_frames: bool = True,
    trim_auto_frame_count_tail: bool = False,
    temporal_compression: int = 8,
    frame_count_snapper=None,
    start_frame_rng=None,
) -> tuple[list[int], int]:
    """The fork's selection branches, verbatim (B1 trim / B2 shrink / B3a
    fallback / B3b random start), with the auto count from
    ``auto_frame_count``. ``start_frame_rng`` defaults to ``random.randint``
    (the fork draws from the global RNG); tests inject a seeded callable."""
    max_frame_index = total_frames - 1
    rng = start_frame_rng if start_frame_rng is not None else random.randint

    if auto_frame_count:
        num_frames = compute_auto_frame_count(
            total_frames=total_frames,
            video_fps=video_fps,
            dataset_fps=dataset_fps,
            temporal_compression=temporal_compression,
            frame_count_snapper=frame_count_snapper,
            trim_auto_frame_count_tail=trim_auto_frame_count_tail,
        )

    frames_to_extract: list[int]
    if auto_frame_count and trim_auto_frame_count_tail:
        # preserve real time: pull frames at the dataset fps from the start of the
        # video and trim the tail that didn't fit the snapped frame count, instead of
        # shrinking the whole video to fit (which speeds up motion / chipmunks audio).
        # Critical for audio models (e.g. minimax_h3) where audio must stay in sync.
        fps_ratio = video_fps / dataset_fps if video_fps and video_fps > 0 else 1.0
        frames_to_extract = [
            min(round(i * fps_ratio), max_frame_index) for i in range(num_frames)
        ]
    elif shrink_video_to_frames or total_frames < num_frames:
        # Distribute frames evenly across the entire video
        interval = max_frame_index / (num_frames - 1) if num_frames > 1 else 0
        frames_to_extract = [
            min(int(round(i * interval)), max_frame_index) for i in range(num_frames)
        ]
    else:
        # Calculate frame interval based on FPS ratio
        fps_ratio = video_fps / dataset_fps
        frame_interval = max(1, int(round(fps_ratio)))

        # Calculate max consecutive frames we can extract at desired FPS
        max_consecutive_frames = total_frames // frame_interval

        if max_consecutive_frames < num_frames:
            # Not enough frames at desired FPS, so stretch instead
            interval = max_frame_index / (num_frames - 1) if num_frames > 1 else 0
            frames_to_extract = [
                min(int(round(i * interval)), max_frame_index) for i in range(num_frames)
            ]
        else:
            # Calculate max start frame to ensure we can get all num_frames
            max_start_frame = max_frame_index - ((num_frames - 1) * frame_interval)
            start_frame = rng(0, max(0, max_start_frame))

            # Generate list of frames to extract
            frames_to_extract = [
                start_frame + (i * frame_interval) for i in range(num_frames)
            ]

    # Final safety check - ensure no frame exceeds max valid index
    frames_to_extract = [
        min(frame_idx, max_frame_index) for frame_idx in frames_to_extract
    ]
    return frames_to_extract, num_frames


def _decode_dataset_float(
    video_path: str,
    *,
    frames_to_extract: list[int],
    vf: str,
    width: int,
    height: int,
) -> torch.Tensor:
    """One ffmpeg pass: select the requested frames, run the float chain, pipe
    gbrpf32le planar fp32. Duplicate indices (stretched clips) are selected
    once and repeated, matching the fork's decode-once-repeat-many."""
    if not frames_to_extract:
        raise RuntimeError(f"No frames to decode from {video_path}")

    unique_idx = sorted(set(frames_to_extract))
    if len(unique_idx) == 1:
        select_expr = f"eq(n\\,{unique_idx[0]})"
    else:
        select_expr = "+".join(f"eq(n\\,{idx})" for idx in unique_idx)

    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        video_path,
        "-vf",
        f"select='{select_expr}',setpts=N/FRAME_RATE/TB,{vf}",
        "-vsync",
        "vfr",
        "-frames:v",
        str(len(unique_idx)),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "gbrpf32le",
        "pipe:1",
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ffmpeg dataset float decode failed: {stderr}")
    tensor = gbrpf32le_to_training_tensor(
        proc.stdout,
        frames=len(unique_idx),
        height=height,
        width=width,
    )
    if len(unique_idx) == len(frames_to_extract):
        return tensor
    # repeat decoded rows for duplicate indices, in request order
    first_pos = {idx: i for i, idx in enumerate(unique_idx)}
    rows = tensor[[first_pos[idx] for idx in frames_to_extract]]
    return rows.contiguous()


def load_dataset_float_tensors(
    video_path: str,
    *,
    scale_to_width: int,
    scale_to_height: int,
    crop_x: int,
    crop_y: int,
    crop_width: int,
    crop_height: int,
    dataset_fps: float,
    num_frames: int = 1,
    auto_frame_count: bool = False,
    shrink_video_to_frames: bool = True,
    trim_auto_frame_count_tail: bool = False,
    temporal_compression: int = 8,
    frame_count_snapper=None,
    start_frame_rng=None,
    info: VideoInfo | None = None,
) -> LoadedVideo:
    """Float decode of a dataset clip into a fp32 [-1, 1] training tensor.

    Geometry comes from the fork's bucket bookkeeping (scale-to dims + a crop
    rect inside the scaled frame) and lands in the ffmpeg filter graph --
    scale-then-crop, per the fork's resize-then-crop PIL semantics this
    replaces. Pass ``info`` to reuse a probe."""
    if info is None:
        info = probe_video(video_path)
    width, height, video_fps, total_frames = (
        info.width,
        info.height,
        info.fps,
        _total_frames(info),
    )

    frames_to_extract, num_frames = plan_frame_indices(
        total_frames=total_frames,
        video_fps=video_fps,
        dataset_fps=dataset_fps,
        num_frames=num_frames,
        auto_frame_count=auto_frame_count,
        shrink_video_to_frames=shrink_video_to_frames,
        trim_auto_frame_count_tail=trim_auto_frame_count_tail,
        temporal_compression=temporal_compression,
        frame_count_snapper=frame_count_snapper,
        start_frame_rng=start_frame_rng,
    )

    vf = build_float_decode_vf(
        info,
        scale_to_width=scale_to_width,
        scale_to_height=scale_to_height,
        crop_x=crop_x,
        crop_y=crop_y,
        crop_width=crop_width,
        crop_height=crop_height,
    )
    # the crop is defined in scaled coords and is the LAST geometry op, so
    # the decoded frames are exactly crop_width x crop_height (== the source
    # or scaled dims whenever no crop filter was emitted)
    tensor = _decode_dataset_float(
        video_path,
        frames_to_extract=frames_to_extract,
        vf=vf,
        width=crop_width,
        height=crop_height,
    )
    return LoadedVideo(
        tensor=tensor,
        num_frames=num_frames,
        frames_to_extract=frames_to_extract,
        video_fps=video_fps,
        total_frames=total_frames,
        pixel_loader="dataset_float",
    )


def _total_frames(info: VideoInfo) -> int:
    """The frame count the planner uses: ffprobe's counted total when the
    probe carried one, else duration-based (matches _probe_video_properties)."""
    import subprocess as _sp

    proc = _sp.run(
        [
            "ffprobe",
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_streams",
            "-count_frames",
            str(info.path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    import json

    stream = next(
        s for s in json.loads(proc.stdout)["streams"] if s["codec_type"] == "video"
    )
    nb_frames = stream.get("nb_read_frames") or stream.get("nb_frames")
    if nb_frames:
        return int(nb_frames)
    duration = info.duration
    return int(round(float(duration) * info.fps)) if duration else 0
