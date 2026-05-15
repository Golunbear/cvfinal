"""Tile processing helpers for flow matching inference."""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import torch
from PIL import Image


def iter_tile_windows(width: int, height: int, tile_size: int, overlap: int) -> list[tuple[int, int, int, int]]:
    if tile_size <= 0 or width <= tile_size and height <= tile_size:
        return [(0, 0, width, height)]
    if overlap < 0 or overlap >= tile_size:
        raise ValueError("tile_overlap must be >= 0 and smaller than tile_size")
    stride = tile_size - overlap

    def starts(size: int) -> list[int]:
        if size <= tile_size:
            return [0]
        values = list(range(0, max(size - tile_size, 0) + 1, stride))
        last = size - tile_size
        if values[-1] != last:
            values.append(last)
        return values

    windows = []
    for y in starts(height):
        for x in starts(width):
            windows.append((x, y, min(x + tile_size, width), min(y + tile_size, height)))
    return windows


def blend_mask(width: int, height: int, x0: int, y0: int, full_width: int, full_height: int, overlap: int) -> np.ndarray:
    if overlap <= 0:
        return np.ones((height, width, 1), dtype=np.float32)
    left = np.arange(width, dtype=np.float32)
    right = np.arange(width - 1, -1, -1, dtype=np.float32)
    top = np.arange(height, dtype=np.float32)
    bottom = np.arange(height - 1, -1, -1, dtype=np.float32)
    if x0 == 0:
        left = np.full_like(left, overlap)
    if x0 + width == full_width:
        right = np.full_like(right, overlap)
    if y0 == 0:
        top = np.full_like(top, overlap)
    if y0 + height == full_height:
        bottom = np.full_like(bottom, overlap)
    xx = np.minimum(left, right)
    yy = np.minimum(top, bottom)
    wx = np.clip(xx / float(overlap), 0.0, 1.0)
    wy = np.clip(yy / float(overlap), 0.0, 1.0)
    wx = 0.5 - 0.5 * np.cos(wx * math.pi)
    wy = 0.5 - 0.5 * np.cos(wy * math.pi)
    return (wy[:, None] * wx[None, :])[..., None].astype(np.float32)


def pad_to_multiple(tensor: torch.Tensor, multiple: int = 16) -> tuple[torch.Tensor, tuple[int, int]]:
    _, _, h, w = tensor.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h == 0 and pad_w == 0:
        return tensor, (0, 0)
    padded = torch.nn.functional.pad(tensor, (0, pad_w, 0, pad_h), mode="reflect")
    return padded, (pad_h, pad_w)


def crop_padding(tensor: torch.Tensor, pad: tuple[int, int]) -> torch.Tensor:
    pad_h, pad_w = pad
    if pad_h == 0 and pad_w == 0:
        return tensor
    h = tensor.shape[-2] - pad_h
    w = tensor.shape[-1] - pad_w
    return tensor[..., :h, :w]


def process_tiled_tensor(
    condition: torch.Tensor,
    tile_size: int,
    overlap: int,
    process_tile: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    _, c, h, w = condition.shape
    windows = iter_tile_windows(w, h, tile_size, overlap)
    accum = torch.zeros(1, c, h, w, dtype=condition.dtype, device=condition.device)
    weights = torch.zeros(1, 1, h, w, dtype=condition.dtype, device=condition.device)
    for x0, y0, x1, y1 in windows:
        tile = condition[:, :, y0:y1, x0:x1]
        out_tile = process_tile(tile)
        mask_np = blend_mask(x1 - x0, y1 - y0, x0, y0, w, h, overlap)
        mask = torch.from_numpy(mask_np).permute(2, 0, 1).unsqueeze(0).to(condition.device)
        accum[:, :, y0:y1, x0:x1] += out_tile * mask
        weights[:, :, y0:y1, x0:x1] += mask
    return accum / weights.clamp(min=1e-6)
