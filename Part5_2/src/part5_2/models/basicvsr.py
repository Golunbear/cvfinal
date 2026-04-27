"""BasicVSR network and inference helpers.

The architecture mirrors MMagic's ``BasicVSRNet(mid_channels=64,
num_blocks=30)`` and keeps layer names compatible with the MMEditing/MMagic
``basicvsr_reds4_20120409-0e599677.pth`` checkpoint.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .arch_util import (
    PixelShufflePack,
    ResidualBlockNoBN,
    crop_padding,
    flow_warp,
    make_layer,
    pad_to_multiple,
)


class ResidualBlocksWithInputConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int = 64, num_blocks: int = 30) -> None:
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            make_layer(ResidualBlockNoBN, num_blocks, mid_channels=out_channels),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.main(feat)


class SPyNetConvModule(nn.Module):
    """Minimal ConvModule-compatible block used by MMagic SPyNet checkpoints."""

    def __init__(self, in_channels: int, out_channels: int, activate: bool = True) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 7, 1, 3)
        self.activate = nn.ReLU(inplace=True) if activate else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.activate is not None:
            x = self.activate(x)
        return x


class SPyNetBasicModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.basic_module = nn.Sequential(
            SPyNetConvModule(8, 32, activate=True),
            SPyNetConvModule(32, 64, activate=True),
            SPyNetConvModule(64, 32, activate=True),
            SPyNetConvModule(32, 16, activate=True),
            SPyNetConvModule(16, 2, activate=False),
        )

    def forward(self, tensor_input: torch.Tensor) -> torch.Tensor:
        return self.basic_module(tensor_input)


class SPyNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.basic_module = nn.ModuleList([SPyNetBasicModule() for _ in range(6)])
        self.register_buffer("mean", torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def compute_flow(self, ref: torch.Tensor, supp: torch.Tensor) -> torch.Tensor:
        n, _, h, w = ref.size()
        ref_pyramid = [(ref - self.mean) / self.std]
        supp_pyramid = [(supp - self.mean) / self.std]
        for _ in range(5):
            ref_pyramid.append(F.avg_pool2d(ref_pyramid[-1], 2, 2, count_include_pad=False))
            supp_pyramid.append(F.avg_pool2d(supp_pyramid[-1], 2, 2, count_include_pad=False))
        ref_pyramid = ref_pyramid[::-1]
        supp_pyramid = supp_pyramid[::-1]

        flow = ref.new_zeros(n, 2, h // 32, w // 32)
        for level, (ref_level, supp_level) in enumerate(zip(ref_pyramid, supp_pyramid)):
            if level == 0:
                flow_up = flow
            else:
                flow_up = F.interpolate(flow, scale_factor=2, mode="bilinear", align_corners=True) * 2.0
            flow = flow_up + self.basic_module[level](
                torch.cat(
                    [
                        ref_level,
                        flow_warp(supp_level, flow_up.permute(0, 2, 3, 1), padding_mode="border"),
                        flow_up,
                    ],
                    dim=1,
                )
            )
        return flow

    def forward(self, ref: torch.Tensor, supp: torch.Tensor) -> torch.Tensor:
        h, w = ref.shape[2:4]
        w_up = w if (w % 32) == 0 else 32 * (w // 32 + 1)
        h_up = h if (h % 32) == 0 else 32 * (h // 32 + 1)
        ref_up = F.interpolate(ref, size=(h_up, w_up), mode="bilinear", align_corners=False)
        supp_up = F.interpolate(supp, size=(h_up, w_up), mode="bilinear", align_corners=False)
        flow = F.interpolate(self.compute_flow(ref_up, supp_up), size=(h, w), mode="bilinear", align_corners=False)
        flow[:, 0, :, :] *= float(w) / float(w_up)
        flow[:, 1, :, :] *= float(h) / float(h_up)
        return flow


class BasicVSRNet(nn.Module):
    """BasicVSR x4 network compatible with MMagic BasicVSRNet checkpoints."""

    def __init__(self, mid_channels: int = 64, num_blocks: int = 30) -> None:
        super().__init__()
        self.mid_channels = mid_channels
        self.spynet = SPyNet()
        self.backward_resblocks = ResidualBlocksWithInputConv(mid_channels + 3, mid_channels, num_blocks)
        self.forward_resblocks = ResidualBlocksWithInputConv(mid_channels + 3, mid_channels, num_blocks)
        self.fusion = nn.Conv2d(mid_channels * 2, mid_channels, 1, 1, 0, bias=True)
        self.upsample1 = PixelShufflePack(mid_channels, mid_channels, 2, upsample_kernel=3)
        self.upsample2 = PixelShufflePack(mid_channels, 64, 2, upsample_kernel=3)
        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)
        self.img_upsample = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.is_mirror_extended = False

    def check_if_mirror_extended(self, lrs: torch.Tensor) -> None:
        self.is_mirror_extended = False
        if lrs.size(1) % 2 == 0:
            lrs_1, lrs_2 = torch.chunk(lrs, 2, dim=1)
            if torch.norm(lrs_1 - lrs_2.flip(1)) == 0:
                self.is_mirror_extended = True

    def compute_flow(self, lrs: torch.Tensor) -> tuple[torch.Tensor | None, torch.Tensor]:
        n, t, c, h, w = lrs.size()
        lrs_1 = lrs[:, :-1, :, :, :].reshape(-1, c, h, w)
        lrs_2 = lrs[:, 1:, :, :, :].reshape(-1, c, h, w)
        flows_backward = self.spynet(lrs_1, lrs_2).view(n, t - 1, 2, h, w)
        if self.is_mirror_extended:
            flows_forward = None
        else:
            flows_forward = self.spynet(lrs_2, lrs_1).view(n, t - 1, 2, h, w)
        return flows_forward, flows_backward

    def forward(self, lrs: torch.Tensor) -> torch.Tensor:
        n, t, _, h, w = lrs.size()
        self.check_if_mirror_extended(lrs)
        flows_forward, flows_backward = self.compute_flow(lrs)

        outputs = []
        feat_prop = lrs.new_zeros(n, self.mid_channels, h, w)
        for i in range(t - 1, -1, -1):
            if i < t - 1:
                feat_prop = flow_warp(feat_prop, flows_backward[:, i].permute(0, 2, 3, 1))
            feat_prop = self.backward_resblocks(torch.cat([lrs[:, i], feat_prop], dim=1))
            outputs.append(feat_prop)
        outputs = outputs[::-1]

        feat_prop = torch.zeros_like(feat_prop)
        for i in range(t):
            lr_curr = lrs[:, i]
            if i > 0:
                flow = flows_forward[:, i - 1] if flows_forward is not None else flows_backward[:, -i]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))
            feat_prop = self.forward_resblocks(torch.cat([lr_curr, feat_prop], dim=1))
            out = torch.cat([outputs[i], feat_prop], dim=1)
            out = self.lrelu(self.fusion(out))
            out = self.lrelu(self.upsample1(out))
            out = self.lrelu(self.upsample2(out))
            out = self.lrelu(self.conv_hr(out))
            out = self.conv_last(out)
            outputs[i] = out + self.img_upsample(lr_curr)
        return torch.stack(outputs, dim=1)


def _extract_state_dict(checkpoint: object) -> OrderedDict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "params_ema", "params"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                state = checkpoint[key]
                break
        else:
            state = checkpoint
    else:
        raise TypeError(f"Unsupported checkpoint type: {type(checkpoint)}")
    cleaned = OrderedDict()
    for key, value in state.items():
        if not torch.is_tensor(value):
            continue
        new_key = key
        for prefix in ("module.", "generator.", "model."):
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix) :]
        if new_key == "step_counter":
            continue
        cleaned[new_key] = value
    return cleaned


def load_basicvsr(weights_path: str | Path, device: torch.device, strict: bool = True) -> BasicVSRNet:
    model = BasicVSRNet(mid_channels=64, num_blocks=30)
    checkpoint = torch.load(str(weights_path), map_location="cpu")
    state = _extract_state_dict(checkpoint)
    missing, unexpected = model.load_state_dict(state, strict=strict)
    if strict is False and (missing or unexpected):
        print(f"[BasicVSR] non-strict load: missing={len(missing)}, unexpected={len(unexpected)}")
    return model.to(device).eval()


@torch.inference_mode()
def infer_basicvsr_sequence(
    model: BasicVSRNet,
    lq: torch.Tensor,
    chunk_size: int = 0,
    overlap: int = 2,
) -> torch.Tensor:
    """Run BasicVSR on a sequence tensor of shape (t, c, h, w)."""

    if lq.ndim != 4:
        raise ValueError(f"Expected (t,c,h,w), got {tuple(lq.shape)}")
    device = next(model.parameters()).device
    lq, pad_hw = pad_to_multiple(lq, 4)
    t = lq.shape[0]
    if chunk_size <= 0 or t <= chunk_size:
        lq = lq.to(device, non_blocking=True)
        output = model(lq.unsqueeze(0)).squeeze(0)
        return crop_padding(output, pad_hw, scale=4).clamp(0, 1).cpu()

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("--overlap must be >= 0 and smaller than --chunk_size")

    pieces: list[torch.Tensor] = []
    start = 0
    while start < t:
        end = min(t, start + chunk_size)
        chunk_start = max(0, start - overlap)
        chunk_end = min(t, end + overlap)
        chunk_lq = lq[chunk_start:chunk_end].to(device, non_blocking=True)
        chunk_out = model(chunk_lq.unsqueeze(0)).squeeze(0)
        keep_start = start - chunk_start
        keep_end = keep_start + (end - start)
        pieces.append(chunk_out[keep_start:keep_end].cpu())
        del chunk_lq, chunk_out
        if device.type == "cuda":
            torch.cuda.empty_cache()
        start = end
    output = torch.cat(pieces, dim=0)
    return crop_padding(output, pad_hw, scale=4).clamp(0, 1)
