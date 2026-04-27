"""Small architecture utilities adapted from BasicSR/MMagic.

The names and tensor conventions are kept close to the official projects so
official checkpoints can be loaded without pulling in the full training stack.
"""

from __future__ import annotations

import math
from typing import Callable

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init


@torch.no_grad()
def default_init_weights(module_list, scale: float = 1.0, bias_fill: float = 0.0) -> None:
    if not isinstance(module_list, list):
        module_list = [module_list]
    for module in module_list:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)


def make_layer(block: Callable[..., nn.Module], num_blocks: int, **kwargs) -> nn.Sequential:
    return nn.Sequential(*[block(**kwargs) for _ in range(num_blocks)])


class ResidualBlockNoBN(nn.Module):
    """Residual block without batch normalization."""

    def __init__(
        self,
        num_feat: int | None = None,
        mid_channels: int | None = None,
        res_scale: float = 1.0,
        pytorch_init: bool = False,
    ) -> None:
        super().__init__()
        channels = mid_channels if mid_channels is not None else num_feat
        if channels is None:
            channels = 64
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        if not pytorch_init:
            default_init_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv2(self.relu(self.conv1(x))) * self.res_scale


class PixelShufflePack(nn.Module):
    """Conv + pixel shuffle block used by MMagic BasicVSRNet."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        scale_factor: int,
        upsample_kernel: int = 3,
    ) -> None:
        super().__init__()
        padding = (upsample_kernel - 1) // 2
        self.upsample_conv = nn.Conv2d(
            in_channels,
            out_channels * scale_factor * scale_factor,
            upsample_kernel,
            1,
            padding,
        )
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pixel_shuffle(self.upsample_conv(x))


def flow_warp(
    x: torch.Tensor,
    flow: torch.Tensor,
    interp_mode: str = "bilinear",
    padding_mode: str = "zeros",
    align_corners: bool = True,
) -> torch.Tensor:
    """Warp an image or feature map with optical flow.

    Args:
        x: Tensor of shape (n, c, h, w).
        flow: Tensor of shape (n, h, w, 2), in pixel units.
    """

    if x.size()[-2:] != flow.size()[1:3]:
        raise ValueError(f"Feature shape {tuple(x.shape)} and flow shape {tuple(flow.shape)} do not match.")
    _, _, h, w = x.size()
    grid_y, grid_x = torch.meshgrid(
        torch.arange(0, h, device=x.device, dtype=x.dtype),
        torch.arange(0, w, device=x.device, dtype=x.dtype),
        indexing="ij",
    )
    grid = torch.stack((grid_x, grid_y), 2)
    vgrid = grid + flow
    vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(w - 1, 1) - 1.0
    vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(h - 1, 1) - 1.0
    vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
    return F.grid_sample(
        x,
        vgrid_scaled,
        mode=interp_mode,
        padding_mode=padding_mode,
        align_corners=align_corners,
    )


def pixel_unshuffle(x: torch.Tensor, scale: int) -> torch.Tensor:
    b, c, hh, hw = x.size()
    out_channel = c * (scale**2)
    if hh % scale != 0 or hw % scale != 0:
        raise ValueError(f"Input shape {tuple(x.shape)} is not divisible by scale {scale}.")
    h = hh // scale
    w = hw // scale
    x_view = x.view(b, c, h, scale, w, scale)
    return x_view.permute(0, 1, 3, 5, 2, 4).reshape(b, out_channel, h, w)


def pad_to_multiple(x: torch.Tensor, multiple: int) -> tuple[torch.Tensor, tuple[int, int]]:
    h, w = x.shape[-2:]
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h == 0 and pad_w == 0:
        return x, (0, 0)
    return F.pad(x, [0, pad_w, 0, pad_h], mode="reflect"), (pad_h, pad_w)


def crop_padding(x: torch.Tensor, pad_hw: tuple[int, int], scale: int = 1) -> torch.Tensor:
    pad_h, pad_w = pad_hw
    if pad_h:
        x = x[..., : -(pad_h * scale), :]
    if pad_w:
        x = x[..., : -(pad_w * scale)]
    return x


def scale_image_tensor(x: torch.Tensor, scale: int) -> torch.Tensor:
    if scale <= 0 or math.log2(scale) % 1:
        raise ValueError("Only positive power-of-two scales are supported.")
    for _ in range(int(math.log2(scale))):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
    return x
