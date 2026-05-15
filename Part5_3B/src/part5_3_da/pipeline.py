"""Load trained flow matching model and enhance images."""

from __future__ import annotations

from pathlib import Path

import torch

from .flow_matching import euler_solve, midpoint_solve
from .tiling import crop_padding, pad_to_multiple, process_tiled_tensor
from .unet import FlowMatchingUNet


def load_model(
    checkpoint: str | Path,
    channels: tuple[int, ...] = (128, 256, 384, 512),
    blocks_per_level: int = 2,
    attn_levels: tuple[int, ...] = (2, 3),
    device: torch.device | str = "cuda",
    use_ema: bool = True,
) -> FlowMatchingUNet:
    model = FlowMatchingUNet(
        in_channels=6,
        out_channels=3,
        channels=channels,
        blocks_per_level=blocks_per_level,
        attn_levels=attn_levels,
    )
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
    key = "ema_model" if use_ema and "ema_model" in ckpt else "model"
    model.load_state_dict(ckpt[key])
    model.to(device).eval()
    return model


@torch.inference_mode()
def enhance_tensor(
    model: FlowMatchingUNet,
    condition: torch.Tensor,
    steps: int = 25,
    solver: str = "euler",
    tile_size: int = 512,
    tile_overlap: int = 64,
    fast: bool = False,
) -> torch.Tensor:
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    cond = condition.unsqueeze(0).to(device=device, dtype=dtype)

    if tile_size > 0 and (cond.shape[-2] > tile_size or cond.shape[-1] > tile_size):

        def _process_tile(tile_cond: torch.Tensor) -> torch.Tensor:
            tile_cond, pad = pad_to_multiple(tile_cond, 16)
            if solver == "midpoint":
                result = midpoint_solve(model, tile_cond, steps=steps)
            else:
                result = euler_solve(model, tile_cond, steps=steps, fast=fast)
            return crop_padding(result, pad)

        result = process_tiled_tensor(cond, tile_size, tile_overlap, _process_tile)
    else:
        cond, pad = pad_to_multiple(cond, 16)
        if solver == "midpoint":
            result = midpoint_solve(model, cond, steps=steps)
        else:
            result = euler_solve(model, cond, steps=steps, fast=fast)
        result = crop_padding(result, pad)

    return result.squeeze(0).clamp(0, 1).cpu()
