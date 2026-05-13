"""Tile processing helpers for diffusion inference."""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
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


def pad_to_multiple(image: Image.Image, multiple: int = 8) -> tuple[Image.Image, tuple[int, int]]:
    width, height = image.size
    pad_w = (multiple - width % multiple) % multiple
    pad_h = (multiple - height % multiple) % multiple
    if pad_w == 0 and pad_h == 0:
        return image, (0, 0)
    array = np.asarray(image.convert("RGB"), dtype=np.uint8)
    padded = np.pad(array, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")
    return Image.fromarray(padded), (pad_w, pad_h)


def crop_padding(image: Image.Image, pad: tuple[int, int]) -> Image.Image:
    pad_w, pad_h = pad
    if pad_w == 0 and pad_h == 0:
        return image
    width, height = image.size
    return image.crop((0, 0, width - pad_w, height - pad_h))


def process_tiled(
    image: Image.Image,
    tile_size: int,
    overlap: int,
    process_tile: Callable[[Image.Image, int, int], Image.Image],
) -> Image.Image:
    width, height = image.size
    windows = iter_tile_windows(width, height, tile_size, overlap)
    accum = np.zeros((height, width, 3), dtype=np.float32)
    weights = np.zeros((height, width, 1), dtype=np.float32)
    for x0, y0, x1, y1 in windows:
        tile = image.crop((x0, y0, x1, y1))
        out_tile = process_tile(tile, x0, y0).convert("RGB")
        if out_tile.size != tile.size:
            out_tile = out_tile.resize(tile.size, Image.Resampling.BICUBIC)
        tile_arr = np.asarray(out_tile, dtype=np.float32)
        mask = blend_mask(x1 - x0, y1 - y0, x0, y0, width, height, overlap)
        accum[y0:y1, x0:x1] += tile_arr * mask
        weights[y0:y1, x0:x1] += mask
    result = np.clip(accum / np.maximum(weights, 1e-6), 0, 255).round().astype(np.uint8)
    return Image.fromarray(result)
