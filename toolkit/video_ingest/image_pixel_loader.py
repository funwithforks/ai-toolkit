"""Image-side float loading.

Ported verbatim from superdata backend/vendor/aitk/image_pixel_loader.py
(production-proven; the crop-then-resize order and the lanczos no-op guard
are the tested behavior). The fork's bucketed image branch calls
``_resize_lanczos`` for the bucket geometry; the crop coordinates and the
augment/control flows stay the fork's own.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

from .decoder_spec import CropRect


@dataclass(frozen=True)
class LoadedImage:
    tensor: torch.Tensor
    pixel_loader: str = "image_source"


def _open_rgb(path: Path) -> Image.Image:
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img)
        return img.convert("RGB")


def _resize_lanczos(img: Image.Image, width: int, height: int) -> Image.Image:
    if img.width == width and img.height == height:
        return img
    return img.resize((width, height), Image.Resampling.LANCZOS)


def _pil_to_training_tensor(img: Image.Image) -> torch.Tensor:
    """PIL RGB -> CHW float in [-1, 1] (ai-toolkit ToTensor + RescaleTransform)."""
    arr = np.asarray(img, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
    return tensor.mul_(2.0).sub_(1.0)


def load_image_training_tensor(
    source: Path,
    *,
    crop: CropRect,
    output_width: int,
    output_height: int,
) -> LoadedImage:
    """Decode source image with crop + optional scale only — no JPEG re-encode."""
    source = source.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)

    img = _open_rgb(source)
    if crop.w <= 0 or crop.h <= 0:
        raise ValueError(f"invalid crop: {crop}")
    if crop.x + crop.w > img.width or crop.y + crop.h > img.height:
        raise ValueError(
            f"crop {crop} exceeds image bounds {img.width}x{img.height}: {source}"
        )

    img = img.crop((crop.x, crop.y, crop.x + crop.w, crop.y + crop.h))
    img = _resize_lanczos(img, output_width, output_height)
    tensor = _pil_to_training_tensor(img)
    return LoadedImage(tensor=tensor)


def load_dataset_image_tensor(
    dataset_image: Path,
    *,
    bucket_width: int,
    bucket_height: int,
) -> LoadedImage:
    """Load shipped dataset image as-is (identity bucket — no loader resize/crop)."""
    dataset_image = dataset_image.resolve()
    if not dataset_image.is_file():
        raise FileNotFoundError(dataset_image)

    img = _open_rgb(dataset_image)
    if img.width != bucket_width or img.height != bucket_height:
        raise ValueError(
            f"dataset image {img.width}x{img.height} != bucket {bucket_width}x{bucket_height}: "
            f"{dataset_image}"
        )
    tensor = _pil_to_training_tensor(img)
    return LoadedImage(tensor=tensor, pixel_loader="dataset_image")
