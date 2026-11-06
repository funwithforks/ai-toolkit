"""Latent verification: decode a cached LTX-2.3 latent and stack it beside
the resized source clip in one mp4 (original audio panned left, decoded
audio panned right -- desync is audible as an offset between the channels).

Usage:
  venv/bin/python tmp/latent_side_by_side.py \
      --video <dataset clip> --latent <cache .safetensors> [--out out.mp4]

The bucket geometry comes from the cache file's own metadata (the latent
info dict the caching pass wrote); the vaes stream from the mono
checkpoint via the registry machinery; the decoded audio goes through the
vocoder exactly as staged sampling does. Open the output in mpv.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile

os.environ.setdefault("MODELS_PATH", "/run/media/computer/240gb pny/huggingface")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from safetensors.torch import load_file

from extensions_built_in.diffusion_models.ltx2.ltx2 import LTX23Model
from extensions_built_in.diffusion_models.ltx2.staged_sample import (
    decode_audio_latents,
    decode_video_latents,
)
from toolkit.config_modules import ModelConfig
from toolkit.util.get_model import get_model_class


def run(cmd):
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"cmd failed: {' '.join(cmd)}\n{proc.stderr[-2000:]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="the dataset clip")
    ap.add_argument("--latent", required=True, help="the cached latent .safetensors")
    ap.add_argument("--out", default=None, help="output concat mp4")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    device = torch.device(args.device)
    state = load_file(args.latent)
    if "latent" not in state:
        raise SystemExit("no 'latent' key in the cache file")
    latent = state["latent"]  # [C, T, h, w] normalized
    if latent.ndim != 5:
        latent = latent.unsqueeze(0)
    # the bucket dims live in the safetensors HEADER metadata (load_file
    # returns tensors only) -- safe_open reads it
    from safetensors import safe_open

    with safe_open(args.latent, framework="pt") as f:
        raw_meta = f.metadata() or {}
    if "crop_width" not in raw_meta or "crop_height" not in raw_meta:
        raise SystemExit(
            f"cache metadata carries no bucket dims (keys: {sorted(raw_meta)})"
        )
    bucket_w = int(float(raw_meta["crop_width"]))
    bucket_h = int(float(raw_meta["crop_height"]))
    dataset_fps = float(raw_meta.get("fps", 24.0))
    print(f"latent {tuple(latent.shape)}; bucket {bucket_w}x{bucket_h} "
          f"@ {dataset_fps}fps from cache header", flush=True)

    mc = ModelConfig(**dict(
        name_or_path="Lightricks/LTX-2.3/ltx-2.3-22b-dev.safetensors",
        arch="ltx2.3", dtype="bf16",
        quantize=False, quantize_te=False,
        model_kwargs={}, low_vram=False, train_mode="lora",
    ))
    cls = get_model_class(mc)
    sd = cls(device, mc, dtype="bf16")

    print("loading the vae bundle (registry door)...", flush=True)
    sd.load_vae()
    bundle = sd.vae
    video_vae = bundle.vae
    audio_vae = bundle.audio_vae
    print("streaming the vocoder (decode-only)...", flush=True)
    vocoder = sd._stream_vocoder(device)

    # STRICT dim gates: the latent's spatial dims times the vae's
    # compression ratios must equal the bucket, or the cache and the
    # bucket disagree and nothing downstream is trustworthy.
    spatial_ratio = getattr(video_vae.config, "spatial_compression_ratio", None) or 32
    temporal_ratio = getattr(video_vae.config, "temporal_compression_ratio", None) or 8
    exp_h, exp_w = bucket_h, bucket_w
    lat_h, lat_w = latent.shape[-2], latent.shape[-1]
    if lat_h * spatial_ratio != exp_h or lat_w * spatial_ratio != exp_w:
        raise SystemExit(
            f"LATENT/BUCKET MISMATCH: latent spatial {lat_h}x{lat_w} x{spatial_ratio} "
            f"= {lat_h * spatial_ratio}x{lat_w * spatial_ratio}, "
            f"bucket is {exp_w}x{exp_h}"
        )
    expected_frames = (latent.shape[-3] - 1) * temporal_ratio + 1
    print(f"strict gates: latent {lat_h}x{lat_w} x{spatial_ratio} == {exp_w}x{exp_h}; "
          f"expecting {expected_frames} frames "
          f"(T={latent.shape[-3]} x{temporal_ratio})", flush=True)

    print("decoding the video latent...", flush=True)
    frames = decode_video_latents(
        video_vae, latent.to(device), device=device, dtype=torch.bfloat16,
    )  # (T, C, H, W) fp32 [-1, 1]
    print(f"decoded frames: {tuple(frames.shape)}", flush=True)
    if frames.shape[-2] != exp_h or frames.shape[-1] != exp_w:
        raise SystemExit(
            f"DECODE/BUCKET MISMATCH: decoded {frames.shape[-2]}x{frames.shape[-1]}, "
            f"bucket is {exp_w}x{exp_h}"
        )
    if frames.shape[0] != expected_frames:
        raise SystemExit(
            f"FRAME COUNT MISMATCH: decoded {frames.shape[0]}, "
            f"expected {expected_frames} from T={latent.shape[-3]} x{temporal_ratio}"
        )

    audio_wave = None
    audio_sr = None
    if "audio_latent" in state:
        print("decoding the audio latent...", flush=True)
        audio_latent = state["audio_latent"]
        if audio_latent.ndim == 2:
            audio_latent = audio_latent.unsqueeze(0)
        wave, sr = decode_audio_latents(
            audio_vae, vocoder, audio_latent.to(device),
            device=device, dtype=torch.bfloat16,
        )
        audio_wave = wave.detach().cpu().to(torch.float32)
        audio_sr = int(sr)
        print(f"decoded audio: {tuple(audio_wave.shape)} @ {audio_sr}Hz", flush=True)

    out = args.out or os.path.join(
        os.path.dirname(args.latent),
        os.path.splitext(os.path.basename(args.video))[0] + "_side_by_side.mp4",
    )
    tmpdir = tempfile.mkdtemp(prefix="latent_check_")
    resized = os.path.join(tmpdir, "resized.mp4")
    decoded = os.path.join(tmpdir, "decoded.mp4")

    # left half: the source clip resized to the bucket dims, original audio
    run([
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", args.video,
        "-vf", f"scale={bucket_w}:{bucket_h}:flags=lanczos",
        "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        resized,
    ])
    # right half: the decoded frames + the decoded audio, muxed via the
    # fork's own export helper
    video_u8 = (
        ((frames.float() + 1) / 2).clamp(0, 1).mul(255).round().to(torch.uint8)
        .permute(0, 2, 3, 1).cpu()
    )
    payload = {
        "video": video_u8,
        "fps": dataset_fps,
        "output_path": decoded,
    }
    if audio_wave is not None:
        payload["audio"] = audio_wave
        payload["audio_sample_rate"] = audio_sr
    from diffusers.pipelines.ltx2.export_utils import encode_video
    encode_video(**payload)

    # stack: original audio strictly in the left ear, decoded strictly in
    # the right ear (mono pans + amerge) -- desync is audible as the content
    # arriving at different times per ear, never as a both-ear mush
    if audio_wave is not None:
        afilter = (
            "[0:v][1:v]hstack=inputs=2[v];"
            "[0:a]pan=mono|c0=c0[a0];"
            "[1:a]pan=mono|c0=c0[a1];"
            "[a0][a1]amerge=inputs=2[a]"
        )
        run([
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", resized, "-i", decoded,
            "-filter_complex", afilter,
            "-map", "[v]", "-map", "[a]",
            "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            out,
        ])
    else:
        run([
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", resized, "-i", decoded,
            "-filter_complex", "[0:v][1:v]hstack=inputs=2[v]",
            "-map", "[v]",
            "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p",
            out,
        ])

    def duration(path):
        proc = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_streams", "-show_format", path],
            capture_output=True, text=True, check=True,
        )
        data = json.loads(proc.stdout)
        fmt_dur = float(data["format"]["duration"])
        vstream = next(s for s in data["streams"] if s["codec_type"] == "video")
        return fmt_dur, int(vstream.get("nb_frames") or 0)

    d_res, n_res = duration(resized)
    d_dec, n_dec = duration(decoded)
    print(f"\ndurations: resized {d_res:.3f}s ({n_res} frames) | "
          f"decoded {d_dec:.3f}s ({n_dec} frames)", flush=True)
    if abs(d_res - d_dec) > 0.05:
        print(f"*** MISMATCH: the halves differ by {abs(d_res - d_dec):.3f}s "
              "-- that is the defect this script exists to surface ***",
              flush=True)
    print(f"\nside-by-side written: {out}\nopen it in mpv.", flush=True)


if __name__ == "__main__":
    main()
