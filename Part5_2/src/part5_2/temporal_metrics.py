"""Pure-Python temporal metric helpers."""

from __future__ import annotations


def temporal_lpips_from_deltas(pred_deltas: list[float], gt_deltas: list[float]) -> list[float]:
    """tLPIPS per Chu et al.: |LPIPS(I[t-1], I[t]) - LPIPS(G[t-1], G[t])|."""

    if len(pred_deltas) != len(gt_deltas):
        raise ValueError(f"Delta list lengths differ: {len(pred_deltas)} vs {len(gt_deltas)}")
    return [abs(float(pred_delta) - float(gt_delta)) for pred_delta, gt_delta in zip(pred_deltas, gt_deltas)]
