"""Small helpers for inference option defaults."""

from __future__ import annotations

MODEL_CHOICES = ("basicvsr", "iconvsr", "realesrgan")
RECURRENT_VSR_MODELS = frozenset({"basicvsr", "iconvsr"})


def resolve_basicvsr_chunk_size(
    model_name: str,
    dataset: str,
    requested_chunk_size: int,
    default_wild_chunk_size: int = 5,
) -> int:
    if model_name in RECURRENT_VSR_MODELS and dataset == "wild" and requested_chunk_size <= 0:
        return default_wild_chunk_size
    return requested_chunk_size


def iter_basicvsr_chunk_windows(
    num_frames: int,
    chunk_size: int,
    overlap: int,
) -> list[tuple[int, int, int, int, int, int]]:
    """Return output and context windows for streaming BasicVSR/IconVSR inference."""

    if num_frames <= 0:
        return []
    if chunk_size <= 0:
        return [(0, num_frames, 0, num_frames, 0, num_frames)]
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("--overlap must be >= 0 and smaller than --chunk_size")

    windows = []
    start = 0
    while start < num_frames:
        end = min(num_frames, start + chunk_size)
        chunk_start = max(0, start - overlap)
        chunk_end = min(num_frames, end + overlap)
        keep_start = start - chunk_start
        keep_end = keep_start + (end - start)
        windows.append((start, end, chunk_start, chunk_end, keep_start, keep_end))
        start = end
    return windows
