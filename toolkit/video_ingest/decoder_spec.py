"""The float decode filter chains.

Ported from superdata backend/app/encoder.py (the ingest half). The color
chains, the reserved-tag fixup and the policy are verbatim from the source
pipeline -- these exact filter graphs are tested and frozen:

  ENCODER POLICY — do not change without explicit user approval.
  (full text preserved below, with the export-only yuv420p tails omitted --
  this module stops at gbrpf32le, the latent-cache endpoint)

Deliberate fork adaptation, documented here once: geometry order. Superdata's
exports are already final-geometry so its dataset path runs no geometry
filters at all; its source path crops in source coords then scales. The
fork's bucket bookkeeping (setup_buckets) defines the crop INSIDE the scaled
frame, so the fork's graph is scale-then-crop -- and the reserved-tag fixup
must precede the scale, because ffmpeg 8's libswscale refuses to scale
reserved-tagged frames at all.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .probe import VideoInfo

# Fast demuxer seek, then fine decode only this many seconds before the clip start.
# Does not affect output pixels — avoids decoding from t=0 on long sources.
SEEK_MARGIN_SEC = 10.0

# ENCODER POLICY — do not change without explicit user approval.
# libx264 veryslow / qp 0 / gop 1 and the HDR tonemap decode chain are intentional
# for maximum detail preservation. Do NOT swap presets, CRF, or decode shortcuts for
# speed. Any proposed ffmpeg decoding or encoding change must be explained in terms
# of output quality impact; the default goal is preserving maximum detail above all.

# Explicit SDR tags — do not run HDR tonemap on these even when 10-bit.
_SDR_TRANSFERS = frozenset(
    {"bt709", "bt601", "smpte170m", "bt470m", "bt470bg", "iec61966-2-1"}
)
_SDR_PRIMARIES = frozenset({"bt709", "bt601", "smpte170m", "bt470bg", "smpte240m"})


@dataclass
class CropRect:
    x: int
    y: int
    w: int
    h: int


def snap_down(value: int, step: int) -> int:
    return max(step, (value // step) * step)


def snap_rect(x: int, y: int, w: int, h: int, spatial_snap: int) -> CropRect:
    w = snap_down(w, spatial_snap)
    h = snap_down(h, spatial_snap)
    return CropRect(x=x, y=y, w=w, h=h)


def needs_hdr_tonemap(info: VideoInfo) -> bool:
    """True for tagged HDR and untagged Dolby Vision (DOVI side data)."""
    return info.is_hdr or info.has_dovi


def reserved_color_tag_fixup(info: VideoInfo) -> str | None:
    """setparams filter replacing 'reserved' primaries/trc tags with bt709.

    Some sources carry H.273 values reserved for future study (ffprobe name
    "reserved") in color_primaries/color_transfer. ffmpeg 8's libswscale
    refuses ANY scaling or format conversion of such frames ("Unsupported
    input", AVERROR(-95)) — even identity yuv420p->yuv420p resizes — while
    older builds passed them through silently. libx264 encodes are unaffected
    (no swscale, output tags come from -color_* flags), so only chains that
    resize/convert need the retag. Returns None when nothing is reserved or
    when an HDR/10-bit chain already rewrites the tags itself.
    """
    if needs_hdr_tonemap(info) or info.is_10bit:
        return None
    opts = []
    if info.color_primaries == "reserved":
        opts.append("color_primaries=bt709")
    if info.color_transfer == "reserved":
        opts.append("color_trc=bt709")
    if not opts:
        return None
    return "setparams=" + ":".join(opts)


def _has_sdr_color_tags(info: VideoInfo) -> bool:
    transfer = info.color_transfer
    primaries = info.color_primaries
    return (
        transfer in _SDR_TRANSFERS
        or primaries in _SDR_PRIMARIES
    )


def _zscale_range_in(info: VideoInfo) -> str:
    if info.color_range == "pc" or "j420p" in info.pix_fmt:
        return "full"
    return "limited"


def _setparams_bt709(info: VideoInfo) -> str:
    """Tag decoded frames bt709 so libx264 -colorspace/-color_* output opts do not
    insert a second zscale conversion (fails on untagged 10-bit sources)."""
    color_range = "pc" if _zscale_range_in(info) == "full" else "tv"
    return (
        "setparams=color_primaries=bt709:color_trc=bt709:"
        f"colorspace=bt709:range={color_range}"
    )


def choose_hdr_float_chain(info: VideoInfo) -> str:
    """HDR tonemap to display-referred bt709 float RGB (no 8-bit quantize)."""
    if info.color_transfer or info.color_primaries:
        return (
            "zscale=transfer=linear:matrix=bt2020nc:primaries=bt2020:rangein=limited,format=gbrpf32le,"
            "tonemap=tonemap=hable:desat=0,format=gbrpf32le,"
            "zscale=transfer=bt709:matrix=bt709:primaries=bt709:range=full,format=gbrpf32le"
        )
    return (
        "zscale=transferin=smpte2084:transfer=linear:matrixin=bt2020nc:matrix=bt709:"
        "primariesin=bt2020:primaries=bt709,format=gbrpf32le,"
        "tonemap=tonemap=hable:desat=0,format=gbrpf32le,"
        "zscale=transfer=bt709:matrix=bt709:primaries=bt709:range=full,format=gbrpf32le"
    )


def choose_sdr_10bit_float_chain(info: VideoInfo) -> str:
    """10-bit SDR → bt709 float RGB without error-diffusion dither."""
    if _has_sdr_color_tags(info):
        transfer = info.color_transfer if info.color_transfer in _SDR_TRANSFERS else "bt709"
        primaries = (
            info.color_primaries if info.color_primaries in _SDR_PRIMARIES else "bt709"
        )
        matrix = info.color_space if info.color_space not in (None, "unknown") else "bt709"
        return (
            f"zscale=transfer={transfer}:matrix={matrix}:primaries={primaries}:"
            f"rangein={_zscale_range_in(info)},"
            "zscale=transfer=bt709:matrix=bt709:primaries=bt709:range=full,format=gbrpf32le"
        )
    # Untagged 10-bit: explicit bt709 zscale fails ("no path between colorspaces").
    # Native decode to float preserves precision; setparams tags bt709 for downstream.
    return f"format=gbrpf32le,{_setparams_bt709(info)}"


def _float_color_parts(info: VideoInfo) -> str:
    if needs_hdr_tonemap(info):
        return choose_hdr_float_chain(info)
    if info.is_10bit:
        return choose_sdr_10bit_float_chain(info)
    fixup = reserved_color_tag_fixup(info)
    return f"{fixup},format=gbrpf32le" if fixup else "format=gbrpf32le"


def build_float_decode_vf(
    info: VideoInfo,
    *,
    output_width: int,
    output_height: int,
) -> str:
    """Float decode chain for a dataset clip: color-float -> resize.

    Dataset clips are ALREADY cropped -- the export owns framing (the
    user's box in the player). The decode never crops: it color-converts
    to fp32 planes and resizes to the bucket dims with lanczos, so the
    full framed content survives. Filters are emitted only when they do
    something -- an identity bucket decodes with color management alone."""
    parts: list[str] = [_float_color_parts(info)]
    if output_width != info.width or output_height != info.height:
        parts.append(f"scale={output_width}:{output_height}:flags=lanczos")
    return ",".join(parts)


def append_source_seek(
    cmd: list[str],
    source: Path,
    start_sec: float,
    *,
    margin_sec: float = SEEK_MARGIN_SEC,
) -> None:
    """Seek before demux (fast) + short fine seek after (accurate). Same encode chain."""
    fast = max(0.0, start_sec - margin_sec)
    fine = start_sec - fast
    cmd.extend(["-ss", f"{fast:.6f}", "-i", str(source)])
    if fine > 1e-6:
        cmd.extend(["-ss", f"{fine:.6f}"])
