"""Simple optical-flow temporal blending for generative outputs."""

from __future__ import annotations

import numpy as np
from PIL import Image


def temporal_blend_frame(
    generated: Image.Image,
    current_input: Image.Image,
    previous_output: Image.Image | None,
    previous_input: Image.Image | None,
    blend_strength: float,
) -> Image.Image:
    if blend_strength <= 0 or previous_output is None or previous_input is None:
        return generated

    import cv2

    gen = np.asarray(generated.convert("RGB"), dtype=np.float32)
    cur_in = np.asarray(current_input.convert("RGB").resize(generated.size), dtype=np.float32)
    prev_out = np.asarray(previous_output.convert("RGB").resize(generated.size), dtype=np.float32)
    prev_in = np.asarray(previous_input.convert("RGB").resize(generated.size), dtype=np.float32)

    cur_gray = cv2.cvtColor(cur_in.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    prev_gray = cv2.cvtColor(prev_in.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    flow = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM).calc(cur_gray, prev_gray, None)
    h, w = cur_gray.shape
    grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    map_x = grid_x + flow[..., 0]
    map_y = grid_y + flow[..., 1]
    warped_prev_out = cv2.remap(prev_out, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    warped_prev_in = cv2.remap(prev_in, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    diff = np.mean(np.abs(cur_in - warped_prev_in), axis=2, keepdims=True) / 255.0
    confidence = np.exp(-8.0 * diff)
    alpha = np.clip(blend_strength, 0.0, 1.0) * confidence
    blended = gen * (1.0 - alpha) + warped_prev_out * alpha
    return Image.fromarray(np.clip(blended, 0, 255).round().astype(np.uint8))
