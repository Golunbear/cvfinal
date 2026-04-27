"""IconVSR network and inference helpers.

The implementation mirrors MMagic's IconVSRNet defaults for REDS4 inference
while keeping the project self-contained. EDVR information-refill blocks are
implemented locally and use ``torchvision.ops.deform_conv2d`` instead of MMCV's
modulated deformable convolution extension.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.modules.utils import _pair
from torchvision.ops import deform_conv2d

from .arch_util import (
    PixelShufflePack,
    ResidualBlockNoBN,
    crop_padding,
    flow_warp,
    make_layer,
    pad_to_multiple,
)
from .basicvsr import ResidualBlocksWithInputConv, SPyNet, _extract_state_dict


class ConvModule(nn.Module):
    """Tiny ConvModule-compatible block with MMagic-style ``.conv`` keys."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        act: str | None = "leaky_relu",
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        if act == "leaky_relu":
            self.activate: nn.Module | None = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        elif act == "relu":
            self.activate = nn.ReLU(inplace=True)
        elif act is None:
            self.activate = None
        else:
            raise ValueError(f"Unsupported activation: {act}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.activate is not None:
            x = self.activate(x)
        return x


class ModulatedDCNPack(nn.Module):
    """MMCV ModulatedDeformConv2dPack-compatible wrapper.

    Checkpoints store regular convolution weights directly on this module and
    store offset/mask prediction weights under ``conv_offset.*``. The extra
    feature tensor predicts offsets and masks; the first tensor is sampled.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
        groups: int = 1,
        deform_groups: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.deform_groups = deform_groups
        self.kernel_size = _pair(kernel_size)

        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, *self.kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)
        offset_channels = deform_groups * 3 * self.kernel_size[0] * self.kernel_size[1]
        self.conv_offset = nn.Conv2d(
            in_channels,
            offset_channels,
            self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=0.1)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
        nn.init.zeros_(self.conv_offset.weight)
        nn.init.zeros_(self.conv_offset.bias)

    def forward(self, x: torch.Tensor, extra_feat: torch.Tensor) -> torch.Tensor:
        offset_mask = self.conv_offset(extra_feat)
        offset_1, offset_2, mask = torch.chunk(offset_mask, 3, dim=1)
        offset = torch.cat((offset_1, offset_2), dim=1)
        mask = torch.sigmoid(mask)
        return deform_conv2d(
            x,
            offset,
            self.weight,
            self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            mask=mask,
        )


class PCDAlignment(nn.Module):
    """Pyramid, cascading and deformable alignment module from EDVR."""

    def __init__(self, mid_channels: int = 64, deform_groups: int = 8) -> None:
        super().__init__()
        self.offset_conv1 = nn.ModuleDict()
        self.offset_conv2 = nn.ModuleDict()
        self.offset_conv3 = nn.ModuleDict()
        self.dcn_pack = nn.ModuleDict()
        self.feat_conv = nn.ModuleDict()

        for level in ["l3", "l2", "l1"]:
            self.offset_conv1[level] = ConvModule(mid_channels * 2, mid_channels, 3, padding=1)
            if level == "l3":
                self.offset_conv2[level] = ConvModule(mid_channels, mid_channels, 3, padding=1)
            else:
                self.offset_conv2[level] = ConvModule(mid_channels * 2, mid_channels, 3, padding=1)
                self.offset_conv3[level] = ConvModule(mid_channels, mid_channels, 3, padding=1)
            self.dcn_pack[level] = ModulatedDCNPack(
                mid_channels,
                mid_channels,
                3,
                padding=1,
                deform_groups=deform_groups,
            )
            if level in ["l2", "l1"]:
                act = "leaky_relu" if level == "l2" else None
                self.feat_conv[level] = ConvModule(mid_channels * 2, mid_channels, 3, padding=1, act=act)

        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.cas_offset_conv1 = ConvModule(mid_channels * 2, mid_channels, 3, padding=1)
        self.cas_offset_conv2 = ConvModule(mid_channels, mid_channels, 3, padding=1)
        self.cas_dcnpack = ModulatedDCNPack(mid_channels, mid_channels, 3, padding=1, deform_groups=deform_groups)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(
        self,
        neighbor_feats: dict[str, torch.Tensor],
        ref_feats: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        upsampled_offset: torch.Tensor | None = None
        upsampled_feat: torch.Tensor | None = None

        for level in ["l3", "l2", "l1"]:
            offset = torch.cat([neighbor_feats[level], ref_feats[level]], dim=1)
            offset = self.offset_conv1[level](offset)
            if level == "l3":
                offset = self.offset_conv2[level](offset)
            else:
                offset = torch.cat([offset, upsampled_offset], dim=1)
                offset = self.offset_conv2[level](offset)
                offset = self.offset_conv3[level](offset)

            feat = self.dcn_pack[level](neighbor_feats[level], offset)
            if level == "l3":
                feat = self.lrelu(feat)
            else:
                feat = self.feat_conv[level](torch.cat([feat, upsampled_feat], dim=1))
            if level != "l1":
                upsampled_offset = self.upsample(offset) * 2.0
                upsampled_feat = self.upsample(feat)

        offset = torch.cat([feat, ref_feats["l1"]], dim=1)
        offset = self.cas_offset_conv2(self.cas_offset_conv1(offset))
        return self.lrelu(self.cas_dcnpack(feat, offset))


class TSAFusion(nn.Module):
    """Temporal-spatial attention fusion used by EDVR."""

    def __init__(self, mid_channels: int = 64, num_frames: int = 5, center_frame_idx: int = 2) -> None:
        super().__init__()
        self.center_frame_idx = center_frame_idx
        self.temporal_attn1 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1)
        self.temporal_attn2 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1)
        self.feat_fusion = ConvModule(num_frames * mid_channels, mid_channels, 1, padding=0)

        self.spatial_attn1 = ConvModule(num_frames * mid_channels, mid_channels, 1, padding=0)
        self.spatial_attn2 = ConvModule(mid_channels * 2, mid_channels, 1, padding=0)
        self.spatial_attn3 = ConvModule(mid_channels, mid_channels, 3, padding=1)
        self.spatial_attn4 = ConvModule(mid_channels, mid_channels, 1, padding=0)
        self.spatial_attn5 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1)
        self.spatial_attn_l1 = ConvModule(mid_channels, mid_channels, 1, padding=0)
        self.spatial_attn_l2 = ConvModule(mid_channels * 2, mid_channels, 3, padding=1)
        self.spatial_attn_l3 = ConvModule(mid_channels, mid_channels, 3, padding=1)
        self.spatial_attn_add1 = ConvModule(mid_channels, mid_channels, 1, padding=0)
        self.spatial_attn_add2 = nn.Conv2d(mid_channels, mid_channels, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.max_pool = nn.MaxPool2d(3, stride=2, padding=1)
        self.avg_pool = nn.AvgPool2d(3, stride=2, padding=1)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

    def forward(self, aligned_feats: torch.Tensor) -> torch.Tensor:
        b, n, c, h, w = aligned_feats.size()
        embedding_ref = self.temporal_attn1(aligned_feats[:, self.center_frame_idx])
        embedding = self.temporal_attn2(aligned_feats.view(-1, c, h, w)).view(b, n, c, h, w)

        corr_l = []
        for i in range(n):
            corr_l.append(torch.sum(embedding[:, i] * embedding_ref, dim=1, keepdim=True))
        corr = torch.cat(corr_l, dim=1)
        corr_prob = torch.sigmoid(corr).view(b, n, 1, h, w)
        aligned_feats = aligned_feats * corr_prob

        feat = aligned_feats.view(b, -1, h, w)
        feat = self.feat_fusion(feat)

        attn = self.spatial_attn1(aligned_feats.view(b, -1, h, w))
        attn_max = self.max_pool(attn)
        attn_avg = self.avg_pool(attn)
        attn = self.spatial_attn2(torch.cat([attn_max, attn_avg], dim=1))
        attn_level = self.spatial_attn_l1(attn)
        attn_max = self.max_pool(attn_level)
        attn_avg = self.avg_pool(attn_level)
        attn_level = self.spatial_attn_l2(torch.cat([attn_max, attn_avg], dim=1))
        attn_level = self.spatial_attn_l3(attn_level)
        attn_level = self.upsample(attn_level)

        if attn_level.shape[-2:] != attn.shape[-2:]:
            attn_level = F.interpolate(attn_level, size=attn.shape[-2:], mode="bilinear", align_corners=False)
        attn = self.spatial_attn3(attn) + attn_level
        attn = self.spatial_attn4(attn)
        attn = self.upsample(attn)
        if attn.shape[-2:] != (h, w):
            attn = F.interpolate(attn, size=(h, w), mode="bilinear", align_corners=False)
        attn = self.spatial_attn5(attn)
        attn_add = self.spatial_attn_add2(self.spatial_attn_add1(attn))
        attn = torch.sigmoid(attn)
        return feat * attn * 2.0 + attn_add


class EDVRFeatureExtractor(nn.Module):
    """EDVR-M feature extractor used by IconVSR information refill."""

    def __init__(
        self,
        in_channels: int = 3,
        mid_channels: int = 64,
        num_frames: int = 5,
        deform_groups: int = 8,
        num_blocks_extraction: int = 5,
        center_frame_idx: int = 2,
    ) -> None:
        super().__init__()
        self.center_frame_idx = center_frame_idx
        self.conv_first = nn.Conv2d(in_channels, mid_channels, 3, 1, 1)
        self.feature_extraction = make_layer(
            ResidualBlockNoBN,
            num_blocks_extraction,
            mid_channels=mid_channels,
        )

        self.feat_l2_conv1 = ConvModule(mid_channels, mid_channels, 3, stride=2, padding=1)
        self.feat_l2_conv2 = ConvModule(mid_channels, mid_channels, 3, padding=1)
        self.feat_l3_conv1 = ConvModule(mid_channels, mid_channels, 3, stride=2, padding=1)
        self.feat_l3_conv2 = ConvModule(mid_channels, mid_channels, 3, padding=1)

        self.pcd_alignment = PCDAlignment(mid_channels=mid_channels, deform_groups=deform_groups)
        self.fusion = TSAFusion(mid_channels=mid_channels, num_frames=num_frames, center_frame_idx=center_frame_idx)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, c, h, w = x.size()
        if h % 4 != 0 or w % 4 != 0:
            raise ValueError(f"EDVR feature extractor expects spatial size divisible by 4, got {(h, w)}.")

        feat_l1 = self.lrelu(self.conv_first(x.view(-1, c, h, w)))
        feat_l1 = self.feature_extraction(feat_l1)
        feat_l2 = self.feat_l2_conv2(self.feat_l2_conv1(feat_l1))
        feat_l3 = self.feat_l3_conv2(self.feat_l3_conv1(feat_l2))

        feat_l1 = feat_l1.view(b, n, -1, h, w)
        feat_l2 = feat_l2.view(b, n, -1, h // 2, w // 2)
        feat_l3 = feat_l3.view(b, n, -1, h // 4, w // 4)
        ref_feats = {
            "l1": feat_l1[:, self.center_frame_idx],
            "l2": feat_l2[:, self.center_frame_idx],
            "l3": feat_l3[:, self.center_frame_idx],
        }

        aligned_feats = []
        for i in range(n):
            neighbor_feats = {"l1": feat_l1[:, i], "l2": feat_l2[:, i], "l3": feat_l3[:, i]}
            aligned_feats.append(self.pcd_alignment(neighbor_feats, ref_feats))
        aligned_feats = torch.stack(aligned_feats, dim=1)
        return self.fusion(aligned_feats)


class IconVSRNet(nn.Module):
    """IconVSR x4 network compatible with official REDS4 checkpoints."""

    def __init__(
        self,
        mid_channels: int = 64,
        num_blocks: int = 30,
        keyframe_stride: int = 5,
        padding: int = 2,
    ) -> None:
        super().__init__()
        self.mid_channels = mid_channels
        self.keyframe_stride = keyframe_stride
        self.padding = padding
        self.is_mirror_extended = False

        self.spynet = SPyNet()
        self.edvr = EDVRFeatureExtractor(mid_channels=mid_channels, num_frames=2 * padding + 1)
        self.backward_resblocks = ResidualBlocksWithInputConv(mid_channels + 3, mid_channels, num_blocks)
        self.backward_fusion = nn.Conv2d(mid_channels * 2, mid_channels, 3, 1, 1, bias=True)
        self.forward_fusion = nn.Conv2d(mid_channels * 2, mid_channels, 3, 1, 1, bias=True)
        self.forward_resblocks = ResidualBlocksWithInputConv(mid_channels * 2 + 3, mid_channels, num_blocks)
        self.upsample1 = PixelShufflePack(mid_channels, mid_channels, 2, upsample_kernel=3)
        self.upsample2 = PixelShufflePack(mid_channels, 64, 2, upsample_kernel=3)
        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)
        self.img_upsample = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

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

    def spatial_padding(self, lrs: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        _, _, _, h, w = lrs.size()
        mod_h = h % 4
        mod_w = w % 4
        pad_h = 0 if mod_h == 0 else 4 - mod_h
        pad_w = 0 if mod_w == 0 else 4 - mod_w
        if pad_h or pad_w:
            b, t, c, _, _ = lrs.size()
            lrs = lrs.view(-1, c, h, w)
            lrs = F.pad(lrs, [0, pad_w, 0, pad_h], mode="reflect")
            lrs = lrs.view(b, t, c, h + pad_h, w + pad_w)
        return lrs, pad_h, pad_w

    def _temporal_pad(self, lrs: torch.Tensor) -> torch.Tensor:
        t = lrs.size(1)
        p = self.padding
        if t >= 2 * p + 1:
            left = list(range(2 * p, p, -1))
            right = list(range(t - p - 2, t - 2 * p - 2, -1))
        else:
            left = [max(0, t - 1 - i) for i in range(p)]
            right = [min(t - 1, max(0, p - 1 - i)) for i in range(p)]
        left_tensor = lrs[:, left] if left else lrs[:, :0]
        right_tensor = lrs[:, right] if right else lrs[:, :0]
        return torch.cat([left_tensor, lrs, right_tensor], dim=1)

    def compute_refill_features(self, lrs: torch.Tensor) -> dict[int, torch.Tensor]:
        _, t, _, _, _ = lrs.size()
        keyframe_idx = list(range(0, t, self.keyframe_stride))
        if keyframe_idx[-1] != t - 1:
            keyframe_idx.append(t - 1)

        lrs_padded = self._temporal_pad(lrs)
        refill_features = {}
        for i in keyframe_idx:
            refill_features[i] = self.edvr(lrs_padded[:, i : i + 2 * self.padding + 1])
        return refill_features

    def forward(self, lrs: torch.Tensor) -> torch.Tensor:
        n, t, _, h_input, w_input = lrs.size()
        lrs, _, _ = self.spatial_padding(lrs)
        _, _, _, h, w = lrs.size()

        self.check_if_mirror_extended(lrs)
        flows_forward, flows_backward = self.compute_flow(lrs)
        refill_features = self.compute_refill_features(lrs)

        outputs = []
        feat_prop = lrs.new_zeros(n, self.mid_channels, h, w)
        for i in range(t - 1, -1, -1):
            lr_curr = lrs[:, i]
            if i < t - 1:
                feat_prop = flow_warp(feat_prop, flows_backward[:, i].permute(0, 2, 3, 1))
            if i in refill_features:
                feat_prop = self.backward_fusion(torch.cat([feat_prop, refill_features[i]], dim=1))
            feat_prop = self.backward_resblocks(torch.cat([lr_curr, feat_prop], dim=1))
            outputs.append(feat_prop)
        outputs = outputs[::-1]

        feat_prop = torch.zeros_like(feat_prop)
        for i in range(t):
            lr_curr = lrs[:, i]
            if i > 0:
                flow = flows_forward[:, i - 1] if flows_forward is not None else flows_backward[:, -i]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))
            if i in refill_features:
                feat_prop = self.forward_fusion(torch.cat([feat_prop, refill_features[i]], dim=1))
            feat_prop = self.forward_resblocks(torch.cat([lr_curr, outputs[i], feat_prop], dim=1))

            out = self.lrelu(self.upsample1(feat_prop))
            out = self.lrelu(self.upsample2(out))
            out = self.lrelu(self.conv_hr(out))
            out = self.conv_last(out)
            base = self.img_upsample(lr_curr)
            outputs[i] = out + base

        output = torch.stack(outputs, dim=1)
        return output[:, :, :, : 4 * h_input, : 4 * w_input]


def load_iconvsr(weights_path: str | Path, device: torch.device, strict: bool = True) -> IconVSRNet:
    model = IconVSRNet(mid_channels=64, num_blocks=30, keyframe_stride=5, padding=2)
    checkpoint = torch.load(str(weights_path), map_location="cpu")
    state = _extract_state_dict(checkpoint)
    missing, unexpected = model.load_state_dict(state, strict=strict)
    if strict is False and (missing or unexpected):
        print(f"[IconVSR] non-strict load: missing={len(missing)}, unexpected={len(unexpected)}")
    return model.to(device).eval()


@torch.inference_mode()
def infer_iconvsr_sequence(
    model: IconVSRNet,
    lq: torch.Tensor,
    chunk_size: int = 0,
    overlap: int = 2,
) -> torch.Tensor:
    """Run IconVSR on a sequence tensor of shape ``(t, c, h, w)``."""

    return _infer_recurrent_sequence(model, lq, chunk_size=chunk_size, overlap=overlap)


@torch.inference_mode()
def _infer_recurrent_sequence(
    model: nn.Module,
    lq: torch.Tensor,
    chunk_size: int,
    overlap: int,
) -> torch.Tensor:
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


SequenceInferFn = Callable[[nn.Module, torch.Tensor, int, int], torch.Tensor]
