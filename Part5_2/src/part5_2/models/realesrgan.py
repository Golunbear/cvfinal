"""Real-ESRGAN RRDBNet and tiled inference helper."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .arch_util import default_init_weights, make_layer, pixel_unshuffle


class ResidualDenseBlock(nn.Module):
    def __init__(self, num_feat: int = 64, num_grow_ch: int = 32) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        default_init_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    def __init__(self, num_feat: int, num_grow_ch: int = 32) -> None:
        super().__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.rdb3(self.rdb2(self.rdb1(x))) * 0.2 + x


class RRDBNet(nn.Module):
    """RRDBNet used by Real-ESRGAN x4plus checkpoints."""

    def __init__(
        self,
        num_in_ch: int = 3,
        num_out_ch: int = 3,
        scale: int = 4,
        num_feat: int = 64,
        num_block: int = 23,
        num_grow_ch: int = 32,
    ) -> None:
        super().__init__()
        self.scale = scale
        if scale == 2:
            num_in_ch *= 4
        elif scale == 1:
            num_in_ch *= 16
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = make_layer(RRDB, num_block, num_feat=num_feat, num_grow_ch=num_grow_ch)
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.scale == 2:
            feat = pixel_unshuffle(x, scale=2)
        elif self.scale == 1:
            feat = pixel_unshuffle(x, scale=4)
        else:
            feat = x
        feat = self.conv_first(feat)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest")))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest")))
        return self.conv_last(self.lrelu(self.conv_hr(feat)))


def _extract_state_dict(checkpoint: object) -> OrderedDict[str, torch.Tensor]:
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unsupported checkpoint type: {type(checkpoint)}")
    for key in ("params_ema", "params", "state_dict"):
        if key in checkpoint and isinstance(checkpoint[key], dict):
            checkpoint = checkpoint[key]
            break
    cleaned = OrderedDict()
    for key, value in checkpoint.items():
        if not torch.is_tensor(value):
            continue
        new_key = key
        for prefix in ("module.", "generator.", "model."):
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix) :]
        cleaned[new_key] = value
    return cleaned


def load_rrdbnet(weights_path: str | Path, device: torch.device, strict: bool = True) -> RRDBNet:
    model = RRDBNet(num_in_ch=3, num_out_ch=3, scale=4, num_feat=64, num_block=23, num_grow_ch=32)
    checkpoint = torch.load(str(weights_path), map_location="cpu")
    state = _extract_state_dict(checkpoint)
    missing, unexpected = model.load_state_dict(state, strict=strict)
    if strict is False and (missing or unexpected):
        print(f"[Real-ESRGAN] non-strict load: missing={len(missing)}, unexpected={len(unexpected)}")
    return model.to(device).eval()


class RealESRGANer:
    def __init__(
        self,
        weights_path: str | Path,
        device: torch.device,
        tile: int = 0,
        tile_pad: int = 10,
        pre_pad: int = 0,
        fp32: bool = False,
    ) -> None:
        self.scale = 4
        self.model = load_rrdbnet(weights_path, device=device, strict=True)
        self.device = device
        self.tile = tile
        self.tile_pad = tile_pad
        self.pre_pad = pre_pad
        self.fp32 = fp32

    @torch.inference_mode()
    def enhance_tensor(self, img: torch.Tensor, outscale: float = 4.0) -> torch.Tensor:
        """Enhance one RGB image tensor with shape (c,h,w) in [0,1]."""

        if img.ndim != 3:
            raise ValueError(f"Expected (c,h,w), got {tuple(img.shape)}")
        x = img.unsqueeze(0).to(self.device, non_blocking=True)
        if self.pre_pad > 0:
            x = F.pad(x, (0, self.pre_pad, 0, self.pre_pad), mode="reflect")
        mod_pad_h = (4 - x.shape[-2] % 4) % 4
        mod_pad_w = (4 - x.shape[-1] % 4) % 4
        if mod_pad_h or mod_pad_w:
            x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), mode="reflect")

        with torch.autocast(device_type=self.device.type, enabled=(self.device.type == "cuda" and not self.fp32)):
            if self.tile > 0:
                out = self._tile_process(x)
            else:
                out = self.model(x)

        if mod_pad_h:
            out = out[..., : -(mod_pad_h * self.scale), :]
        if mod_pad_w:
            out = out[..., : -(mod_pad_w * self.scale)]
        if self.pre_pad > 0:
            out = out[..., : -(self.pre_pad * self.scale), : -(self.pre_pad * self.scale)]
        if outscale != self.scale:
            out = F.interpolate(out, scale_factor=outscale / self.scale, mode="bilinear", align_corners=False)
        return out.squeeze(0).clamp(0, 1).float().cpu()

    def _tile_process(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        output = x.new_zeros(b, c, h * self.scale, w * self.scale)
        tiles_x = (w + self.tile - 1) // self.tile
        tiles_y = (h + self.tile - 1) // self.tile
        for y in range(tiles_y):
            for x_idx in range(tiles_x):
                input_start_x = x_idx * self.tile
                input_end_x = min(input_start_x + self.tile, w)
                input_start_y = y * self.tile
                input_end_y = min(input_start_y + self.tile, h)
                input_start_x_pad = max(input_start_x - self.tile_pad, 0)
                input_end_x_pad = min(input_end_x + self.tile_pad, w)
                input_start_y_pad = max(input_start_y - self.tile_pad, 0)
                input_end_y_pad = min(input_end_y + self.tile_pad, h)

                tile = x[:, :, input_start_y_pad:input_end_y_pad, input_start_x_pad:input_end_x_pad]
                output_tile = self.model(tile)
                output_start_x = input_start_x * self.scale
                output_end_x = input_end_x * self.scale
                output_start_y = input_start_y * self.scale
                output_end_y = input_end_y * self.scale
                output_start_x_tile = (input_start_x - input_start_x_pad) * self.scale
                output_end_x_tile = output_start_x_tile + (input_end_x - input_start_x) * self.scale
                output_start_y_tile = (input_start_y - input_start_y_pad) * self.scale
                output_end_y_tile = output_start_y_tile + (input_end_y - input_start_y) * self.scale
                output[:, :, output_start_y:output_end_y, output_start_x:output_end_x] = output_tile[
                    :, :, output_start_y_tile:output_end_y_tile, output_start_x_tile:output_end_x_tile
                ]
        return output
