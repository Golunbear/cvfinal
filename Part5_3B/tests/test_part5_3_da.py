import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from pathlib import Path

import pytest
import torch

from part5_3_da.datasets import output_frames_dir, output_video_path
from part5_3_da.flow_matching import euler_solve, flow_matching_loss, sample_ot_path
from part5_3_da.temporal_metrics import temporal_lpips_from_deltas
from part5_3_da.tiling import blend_mask, iter_tile_windows, pad_to_multiple
from part5_3_da.unet import FlowMatchingUNet


def test_temporal_lpips_uses_relative_delta():
    assert temporal_lpips_from_deltas([0.1, 0.4], [0.05, 0.6]) == pytest.approx([0.05, 0.2])


def test_tile_windows_cover_edges():
    windows = iter_tile_windows(width=1000, height=700, tile_size=512, overlap=64)
    assert windows[0] == (0, 0, 512, 512)
    assert all(x1 > x0 and y1 > y0 for x0, y0, x1, y1 in windows)


def test_small_image_single_tile():
    assert iter_tile_windows(width=128, height=96, tile_size=512, overlap=64) == [(0, 0, 128, 96)]


def test_blend_mask_outer_border():
    mask = blend_mask(width=128, height=128, x0=0, y0=0, full_width=256, full_height=256, overlap=32)
    assert mask.shape == (128, 128, 1)
    assert float(mask[0, 0, 0]) == pytest.approx(1.0)
    assert float(mask[-1, -1, 0]) < 1.0


def test_output_paths():
    assert output_frames_dir("out", "fm_sr", "iconvsr", "reds_val", "000") == Path(
        "out/fm_sr/iconvsr/reds_val/000/frames"
    )
    assert output_video_path("out", "fm_sr", "iconvsr", "wild", "IMG_2175") == Path(
        "out/fm_sr/iconvsr/wild/IMG_2175.mp4"
    )


def test_unet_forward_shape():
    model = FlowMatchingUNet(
        in_channels=6, out_channels=3,
        channels=(32, 64), blocks_per_level=1, attn_levels=(1,),
        time_dim=64, num_heads=4,
    )
    x = torch.randn(1, 3, 64, 64)
    t = torch.tensor([0.5])
    cond = torch.randn(1, 3, 64, 64)
    out = model(x, t, cond)
    assert out.shape == (1, 3, 64, 64)


def test_sample_ot_path_shapes():
    x1 = torch.randn(2, 3, 16, 16)
    t = torch.tensor([0.0, 1.0])
    x_t, z, u_t = sample_ot_path(x1, t)
    assert x_t.shape == x1.shape
    assert z.shape == x1.shape
    assert u_t.shape == x1.shape


def test_ot_path_endpoints():
    x1 = torch.randn(2, 3, 16, 16)
    t = torch.tensor([0.0, 1.0])
    x_t, z, _ = sample_ot_path(x1, t, sigma_min=0.0)
    assert torch.allclose(x_t[0], z[0], atol=1e-5)
    assert torch.allclose(x_t[1], x1[1], atol=1e-5)


def test_flow_matching_loss_runs():
    model = FlowMatchingUNet(
        in_channels=6, out_channels=3,
        channels=(16, 32), blocks_per_level=1, attn_levels=(),
        time_dim=32, num_heads=2,
    )
    target = torch.randn(2, 3, 32, 32)
    cond = torch.randn(2, 3, 32, 32)
    loss = flow_matching_loss(model, target, cond)
    assert loss.ndim == 0
    assert loss.item() > 0


def test_pad_to_multiple():
    x = torch.randn(1, 3, 33, 47)
    padded, pad = pad_to_multiple(x, 16)
    assert padded.shape[-2] % 16 == 0
    assert padded.shape[-1] % 16 == 0
