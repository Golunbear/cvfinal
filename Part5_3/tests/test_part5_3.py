import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import pytest

from part5_3.datasets import part2_frames_dir, part3_frames_dir, part3_video_path
from part5_3.temporal_metrics import temporal_lpips_from_deltas
from part5_3.tiling import blend_mask, iter_tile_windows


def test_temporal_lpips_uses_relative_delta():
    assert temporal_lpips_from_deltas([0.1, 0.4], [0.05, 0.6]) == pytest.approx([0.05, 0.2])


def test_tile_windows_cover_edges_without_duplicate_full_frame():
    windows = iter_tile_windows(width=1000, height=700, tile_size=512, overlap=64)
    assert windows[0] == (0, 0, 512, 512)
    assert windows[-1] == (488, 188, 1000, 700)
    assert all(x1 > x0 and y1 > y0 for x0, y0, x1, y1 in windows)


def test_small_image_uses_single_tile():
    assert iter_tile_windows(width=128, height=96, tile_size=512, overlap=64) == [(0, 0, 128, 96)]


def test_blend_mask_keeps_outer_image_border_weighted():
    mask = blend_mask(width=128, height=128, x0=0, y0=0, full_width=256, full_height=256, overlap=32)
    assert mask.shape == (128, 128, 1)
    assert float(mask[0, 0, 0]) == pytest.approx(1.0)
    assert float(mask[-1, -1, 0]) < 1.0


def test_output_paths_are_method_and_input_model_scoped():
    assert part2_frames_dir("p2", "iconvsr", "reds_val", "000") == Path("p2/iconvsr/reds_val/000/frames")
    assert part3_frames_dir("out", "sd_tile", "iconvsr", "reds_val", "000") == Path(
        "out/sd_tile/iconvsr/reds_val/000/frames"
    )
    assert part3_video_path("out", "sd_tile", "iconvsr", "wild", "IMG_2175") == Path(
        "out/sd_tile/iconvsr/wild/IMG_2175.mp4"
    )
