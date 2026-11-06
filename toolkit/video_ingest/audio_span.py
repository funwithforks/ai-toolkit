"""Seek-based fp32 audio span decode.

Ported from superdata backend/vendor/aitk/ltx23_encode.py (the audio utils).
The span loaders decode only the requested span (ffmpeg pcm_f32le first,
torchaudio fast-seek fallback, full-decode last resort) -- replacing the
fork's full-file ``torchaudio.load`` in the video audio path. Waveforms are
stereo-normalized fp32 throughout; the mel/encode side stays on the vae
bundle (the fork's ComboVae.encode_audio), which already forces fp32
waveforms at its boundary.
"""

from __future__ import annotations

import json
import subprocess
from functools import lru_cache
from typing import Any, Callable

import torch
import torch.nn.functional as F
import torchaudio

from .decoder_spec import append_source_seek


def set_encode_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def waveform_to_stereo(waveform: torch.Tensor) -> torch.Tensor:
    channels = waveform.shape[0]
    if channels == 2:
        return waveform
    if channels == 1:
        return waveform.expand(2, -1)
    return waveform.mean(0, keepdim=True).expand(2, -1)


@lru_cache(maxsize=16)
def _probe_audio_stream(video_path: str) -> tuple[int, int]:
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=sample_rate,channels",
            "-of",
            "json",
            video_path,
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    stream = json.loads(probe.stdout)["streams"][0]
    sample_rate = int(stream["sample_rate"])
    channels = max(1, int(stream.get("channels") or 2))
    return sample_rate, channels


def _pcm_f32le_to_stereo(raw: bytes, *, channels: int) -> torch.Tensor:
    import numpy as np

    if not raw:
        raise RuntimeError("No audio decoded")
    audio = torch.from_numpy(np.frombuffer(raw, dtype=np.float32).copy())
    frame_count = audio.numel() // 2
    waveform = audio.reshape(frame_count, 2).transpose(0, 1).contiguous()
    if channels == 1:
        waveform = waveform.mean(0, keepdim=True).expand(2, -1)
    return waveform


def _load_audio_segment_torchaudio(
    video_path: str,
    *,
    start_sec: float,
    duration_sec: float,
) -> tuple[torch.Tensor, int]:
    """Decode only the requested span via torchaudio frame_offset (fast seek)."""
    sample_rate, _channels = _probe_audio_stream(video_path)
    start_frame = max(0, int(round(start_sec * sample_rate)))
    num_frames = max(1, int(round(duration_sec * sample_rate)))
    waveform, sample_rate = torchaudio.load(
        video_path,
        frame_offset=start_frame,
        num_frames=num_frames,
    )
    return waveform_to_stereo(waveform), sample_rate


def _load_audio_segment_ffmpeg(
    video_path: str,
    *,
    start_sec: float,
    duration_sec: float,
) -> tuple[torch.Tensor, int]:
    """Decode only the requested audio span (ffmpeg fallback)."""
    sample_rate, channels = _probe_audio_stream(video_path)
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
    ]
    from pathlib import Path

    append_source_seek(cmd, Path(video_path), start_sec)
    cmd += [
        "-t",
        f"{duration_sec:.6f}",
        "-vn",
        "-ac",
        "2",
        "-f",
        "f32le",
        "-acodec",
        "pcm_f32le",
        "pipe:1",
    ]
    proc = subprocess.run(cmd, capture_output=True, check=True)
    waveform = _pcm_f32le_to_stereo(proc.stdout, channels=channels)
    return waveform, sample_rate


def _load_audio_ffmpeg(video_path: str) -> tuple[torch.Tensor, int]:
    """Decode full audio via ffmpeg when torchcodec/torchaudio load is unavailable."""
    sample_rate, channels = _probe_audio_stream(video_path)
    proc = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            video_path,
            "-vn",
            "-ac",
            "2",
            "-f",
            "f32le",
            "-acodec",
            "pcm_f32le",
            "pipe:1",
        ],
        capture_output=True,
        check=True,
    )
    return _pcm_f32le_to_stereo(proc.stdout, channels=channels), sample_rate


def load_audio_from_video(video_path: str) -> tuple[torch.Tensor, int, str]:
    try:
        waveform, sample_rate = _load_audio_torchcodec(video_path)
        return waveform, sample_rate, "torchaudio"
    except Exception:
        waveform, sample_rate = _load_audio_ffmpeg(video_path)
        return waveform, sample_rate, "ffmpeg"


def load_audio_span(
    video_path: str,
    *,
    start_sec: float,
    duration_sec: float,
) -> tuple[torch.Tensor, int] | None:
    """Decode only the requested span, with the full fallback chain:
    ffmpeg pcm_f32le seek -> torchaudio fast-seek -> full decode + slice.
    Returns stereo fp32 waveform + sample rate, or None when the video has
    no usable audio stream."""
    try:
        return _load_audio_segment_ffmpeg(
            video_path, start_sec=start_sec, duration_sec=duration_sec
        )
    except Exception:
        pass
    try:
        return _load_audio_segment_torchaudio(
            video_path, start_sec=start_sec, duration_sec=duration_sec
        )
    except Exception:
        pass
    try:
        waveform, sample_rate, _backend = load_audio_from_video(video_path)
        waveform = waveform_to_stereo(waveform)
        start_sample = max(0, min(int(round(start_sec * sample_rate)), waveform.shape[-1]))
        end_sample = max(
            0,
            min(
                int(round((start_sec + duration_sec) * sample_rate)),
                waveform.shape[-1],
            ),
        )
        if end_sample <= start_sample:
            return None
        return waveform[..., start_sample:end_sample], sample_rate
    except Exception:
        return None


def load_clip_audio(
    video_path: str,
    *,
    frames_to_extract: list[int],
    video_fps: float,
    num_frames: int,
    dataset_fps: float,
    normalize: bool = False,
    preserve_pitch: bool = False,
    preserve_pitch_fn: Callable | None = None,
) -> dict[str, Any] | None:
    """Mirror ai-toolkit audio slice + stretch in load_and_process_video.

    ``preserve_pitch_fn`` is the fork's own time-stretch helper
    (time_stretch_preserve_pitch); superdata's version has no pitch
    preservation and raises without the hook, so the fork's capability stays
    reachable when the caller wires it."""
    if not frames_to_extract:
        return None

    waveform, sample_rate, _backend = load_audio_from_video(video_path)
    waveform = waveform_to_stereo(waveform)

    if normalize:
        peak = waveform.abs().amax()
        waveform = waveform * (0.999 / (peak + 1e-9))

    clip_start_frame = int(frames_to_extract[0])
    clip_end_frame = int(frames_to_extract[-1])
    clip_start_time = clip_start_frame / float(video_fps)
    clip_end_time = (clip_end_frame + 1) / float(video_fps)
    start_sample = int(round(clip_start_time * sample_rate))
    end_sample = int(round(clip_end_time * sample_rate))
    start_sample = max(0, min(start_sample, waveform.shape[-1]))
    end_sample = max(0, min(end_sample, waveform.shape[-1]))
    if end_sample <= start_sample:
        return None
    waveform = waveform[..., start_sample:end_sample]

    target_duration = float(num_frames) / float(dataset_fps)
    target_samples = int(round(target_duration * sample_rate))
    if target_samples > 0 and waveform.shape[-1] != target_samples:
        if preserve_pitch:
            if preserve_pitch_fn is None:
                raise NotImplementedError("audio_preserve_pitch not implemented yet")
            waveform = preserve_pitch_fn(waveform, sample_rate, target_samples)
        else:
            wf = waveform.unsqueeze(0)
            wf = F.interpolate(wf, size=target_samples, mode="linear", align_corners=False)
            waveform = wf.squeeze(0)

    return {
        "waveform": waveform,
        "sample_rate": int(sample_rate),
    }
