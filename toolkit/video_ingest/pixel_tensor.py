"""The raw float pipe -> fp32 training tensor.

Ported verbatim from superdata backend/vendor/aitk/pixel_tensor.py
(production-proven; the byte-count assert is the decode's correctness gate).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class LoadedVideo:
    tensor: torch.Tensor
    num_frames: int
    frames_to_extract: list[int]
    video_fps: float
    total_frames: int
    pixel_loader: str = "dataset_float"


def gbrpf32le_to_training_tensor(
    raw: bytes,
    *,
    frames: int,
    height: int,
    width: int,
) -> torch.Tensor:
    """gbrpf32le planar float -> RGB CHW in [-1, 1] (ai-toolkit ToTensor + RescaleTransform)."""
    plane_size = height * width
    expected = frames * plane_size * 3 * 4
    if len(raw) != expected:
        raise RuntimeError(
            f"gbrpf32le decode size mismatch: expected {expected} bytes, got {len(raw)}"
        )
    floats = np.frombuffer(raw, dtype="<f4").reshape(frames, 3, height, width)
    rgb = floats[:, [2, 0, 1], :, :]
    tensor = torch.from_numpy(rgb.copy()).float()
    return tensor.clamp_(0.0, 1.0).mul_(2.0).sub_(1.0)
