import os

import pytest

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from part5_2.temporal_metrics import temporal_lpips_from_deltas
from part5_2.inference_options import MODEL_CHOICES, iter_basicvsr_chunk_windows, resolve_basicvsr_chunk_size


def test_temporal_lpips_uses_relative_delta():
    pred_deltas = [0.10, 0.40, 0.25]
    gt_deltas = [0.05, 0.35, 0.50]
    assert temporal_lpips_from_deltas(pred_deltas, gt_deltas) == pytest.approx([0.05, 0.05, 0.25])


def test_temporal_lpips_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        temporal_lpips_from_deltas([0.1], [0.1, 0.2])


def test_basicvsr_wild_uses_default_chunk_size():
    assert resolve_basicvsr_chunk_size("basicvsr", "wild", 0) == 5


def test_iconvsr_is_a_supported_model_choice():
    assert "iconvsr" in MODEL_CHOICES


def test_iconvsr_wild_uses_default_chunk_size():
    assert resolve_basicvsr_chunk_size("iconvsr", "wild", 0) == 5


def test_iconvsr_wild_respects_explicit_chunk_size():
    assert resolve_basicvsr_chunk_size("iconvsr", "wild", 30) == 30


def test_basicvsr_wild_respects_explicit_chunk_size():
    assert resolve_basicvsr_chunk_size("basicvsr", "wild", 50) == 50


def test_basicvsr_non_wild_keeps_whole_sequence_default():
    assert resolve_basicvsr_chunk_size("basicvsr", "sample_reds", 0) == 0
    assert resolve_basicvsr_chunk_size("basicvsr", "reds_val", 0) == 0


def test_basicvsr_streaming_windows_write_every_frame_once():
    windows = iter_basicvsr_chunk_windows(num_frames=10, chunk_size=5, overlap=2)
    written = []
    context_ranges = []
    for start, end, chunk_start, chunk_end, keep_start, keep_end in windows:
        written.extend(range(start, end))
        context_ranges.append((chunk_start, chunk_end))
        assert keep_end - keep_start == end - start

    assert written == list(range(10))
    assert context_ranges == [(0, 7), (3, 10)]


def test_iconvsr_state_dict_does_not_register_unused_l3_offset_conv3():
    from part5_2.models.iconvsr import IconVSRNet

    keys = set(IconVSRNet().state_dict())
    assert "edvr.pcd_alignment.offset_conv3.l3.conv.weight" not in keys
    assert "edvr.pcd_alignment.offset_conv3.l2.conv.weight" in keys
    assert "edvr.pcd_alignment.offset_conv3.l1.conv.weight" in keys


def test_extract_state_dict_filters_step_counter_after_prefix_cleanup():
    import torch

    from part5_2.models.basicvsr import _extract_state_dict

    tensor = torch.ones(1)
    state = _extract_state_dict(
        {
            "state_dict": {
                "step_counter": torch.tensor(3),
                "module.foo": tensor,
            }
        }
    )
    assert "step_counter" not in state
    assert state["foo"] is tensor
