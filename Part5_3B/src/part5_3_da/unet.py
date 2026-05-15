"""Lightweight U-Net for conditional flow matching super-resolution."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    dtype = timesteps.dtype
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=timesteps.device, dtype=torch.float32) / half
    )
    args = timesteps[:, None].float() * freqs[None, :]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1).to(dtype)


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_ch * 2)
        self.norm2 = nn.GroupNorm(32, out_ch)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.norm1(x))
        h = self.conv1(h)
        scale, shift = self.time_proj(F.silu(t_emb)).unsqueeze(-1).unsqueeze(-1).chunk(2, dim=1)
        h = self.norm2(h) * (1 + scale.to(h.dtype)) + shift.to(h.dtype)
        h = self.drop(F.silu(h))
        h = self.conv2(h)
        return h + self.skip(x)


class SelfAttention(nn.Module):
    def __init__(self, channels: int, num_heads: int = 8) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        self.num_heads = num_heads

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        normed = self.norm(x)
        qkv = self.qkv(normed).reshape(b, 3, self.num_heads, c // self.num_heads, h * w)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        attn = F.scaled_dot_product_attention(
            q.transpose(-1, -2), k.transpose(-1, -2), v.transpose(-1, -2),
        ).transpose(-1, -2)
        out = self.proj(attn.reshape(b, c, h, w))
        return x + out


class Downsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class FlowMatchingUNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 6,
        out_channels: int = 3,
        channels: tuple[int, ...] = (128, 256, 384, 512),
        blocks_per_level: int = 2,
        attn_levels: tuple[int, ...] = (2, 3),
        time_dim: int = 512,
        dropout: float = 0.0,
        num_heads: int = 8,
    ) -> None:
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim),
        )
        self.time_dim = time_dim
        self.conv_in = nn.Conv2d(in_channels, channels[0], 3, padding=1)

        self.encoder_blocks = nn.ModuleList()
        self.encoder_attns = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        prev_ch = channels[0]
        for level, ch in enumerate(channels):
            level_blocks = nn.ModuleList()
            level_attns = nn.ModuleList()
            for i in range(blocks_per_level):
                in_ch = prev_ch if i == 0 else ch
                level_blocks.append(ResBlock(in_ch, ch, time_dim, dropout))
                level_attns.append(
                    SelfAttention(ch, num_heads) if level in attn_levels else nn.Identity()
                )
                prev_ch = ch
            self.encoder_blocks.append(level_blocks)
            self.encoder_attns.append(level_attns)
            if level < len(channels) - 1:
                self.downsamples.append(Downsample(ch))
            else:
                self.downsamples.append(nn.Identity())

        self.mid_block1 = ResBlock(channels[-1], channels[-1], time_dim, dropout)
        self.mid_attn = SelfAttention(channels[-1], num_heads)
        self.mid_block2 = ResBlock(channels[-1], channels[-1], time_dim, dropout)

        self.decoder_blocks = nn.ModuleList()
        self.decoder_attns = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        prev_ch = channels[-1]
        rev = list(reversed(channels))
        for level, ch in enumerate(rev):
            orig_level = len(channels) - 1 - level
            level_blocks = nn.ModuleList()
            level_attns = nn.ModuleList()
            for i in range(blocks_per_level):
                if i == 0:
                    in_ch = prev_ch + ch
                else:
                    in_ch = ch
                level_blocks.append(ResBlock(in_ch, ch, time_dim, dropout))
                level_attns.append(
                    SelfAttention(ch, num_heads) if orig_level in attn_levels else nn.Identity()
                )
                prev_ch = ch
            self.decoder_blocks.append(level_blocks)
            self.decoder_attns.append(level_attns)
            if level < len(channels) - 1:
                self.upsamples.append(Upsample(ch))
            else:
                self.upsamples.append(nn.Identity())

        self.out_norm = nn.GroupNorm(32, channels[0])
        self.out_conv = nn.Conv2d(channels[0], out_channels, 3, padding=1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor, condition: torch.Tensor, fast: bool = False) -> torch.Tensor:
        t_emb = self.time_mlp(sinusoidal_embedding(t, self.time_dim))
        h = self.conv_in(torch.cat([x, condition], dim=1))

        skips = []
        for level, (blocks, attns, down) in enumerate(
            zip(self.encoder_blocks, self.encoder_attns, self.downsamples)
        ):
            for blk, attn in zip(blocks, attns):
                h = blk(h, t_emb)
                h = attn(h) if not fast else h
            skips.append(h)
            if level < len(self.encoder_blocks) - 1:
                h = down(h)

        h = self.mid_block1(h, t_emb)
        h = self.mid_attn(h) if not fast else h
        h = self.mid_block2(h, t_emb)

        for level, (blocks, attns, up) in enumerate(
            zip(self.decoder_blocks, self.decoder_attns, self.upsamples)
        ):
            skip = skips.pop()
            h = torch.cat([h, skip], dim=1)
            for i, (blk, attn) in enumerate(zip(blocks, attns)):
                if i == 0:
                    h = blk(h, t_emb)
                else:
                    h = blk(h, t_emb)
                h = attn(h) if not fast else h
            if level < len(self.decoder_blocks) - 1:
                h = up(h)

        return self.out_conv(F.silu(self.out_norm(h)))
