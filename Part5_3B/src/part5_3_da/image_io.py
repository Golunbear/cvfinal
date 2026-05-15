"""Image IO helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def list_image_files(path: str | Path) -> list[Path]:
    root = Path(path)
    if not root.exists():
        return []
    return sorted([p for p in root.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES])


def read_image(path: str | Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def read_image_tensor(path: str | Path) -> torch.Tensor:
    image = read_image(path)
    array = np.asarray(image).astype(np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)


def pil_to_uint8(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"), dtype=np.uint8)


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    array = tensor.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    return Image.fromarray((array * 255.0).round().astype(np.uint8))


def write_image(image: Image.Image, path: str | Path) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)
