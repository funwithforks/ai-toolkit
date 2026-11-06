"""ffprobe video inspection -- ported verbatim from superdata
backend/app/probe.py (production-proven; do not "fix" the probe args)."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class VideoInfo:
    path: Path
    width: int
    height: int
    fps: float
    duration: float | None
    codec: str
    pix_fmt: str
    is_hdr: bool
    is_10bit: bool
    has_dovi: bool
    color_range: str | None
    color_transfer: str | None
    color_primaries: str | None
    color_space: str | None


def probe_video(path: Path) -> VideoInfo:
    cmd = [
        "ffprobe",
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_streams",
        "-show_format",
        str(path),
    ]
    raw = subprocess.check_output(cmd, text=True)
    data = json.loads(raw)
    stream = next(s for s in data["streams"] if s["codec_type"] == "video")
    fmt = data.get("format", {})

    fps_raw = stream.get("avg_frame_rate") or stream.get("r_frame_rate") or "24/1"
    num, den = fps_raw.split("/")
    fps = float(num) / float(den) if float(den) else 24.0

    duration = stream.get("duration") or fmt.get("duration")
    duration_f = float(duration) if duration else None

    pix_fmt = stream.get("pix_fmt", "yuv420p")
    transfer = stream.get("color_transfer")
    primaries = stream.get("color_primaries")
    is_10bit = "10" in pix_fmt or stream.get("profile", "").endswith("10")
    is_hdr = transfer in ("smpte2084", "arib-std-b67") or primaries == "bt2020"
    side_data = stream.get("side_data_list") or []
    has_dovi = any(
        item.get("side_data_type") == "DOVI configuration record" for item in side_data
    )

    return VideoInfo(
        path=path,
        width=int(stream["width"]),
        height=int(stream["height"]),
        fps=fps,
        duration=duration_f,
        codec=stream.get("codec_name", ""),
        pix_fmt=pix_fmt,
        is_hdr=is_hdr,
        is_10bit=is_10bit,
        has_dovi=has_dovi,
        color_range=stream.get("color_range"),
        color_transfer=transfer,
        color_primaries=primaries,
        color_space=stream.get("color_space"),
    )
