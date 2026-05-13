"""Temporal perceptual metric helpers."""

from __future__ import annotations


def temporal_lpips_from_deltas(pred_deltas: list[float], gt_deltas: list[float]) -> list[float]:
    if len(pred_deltas) != len(gt_deltas):
        raise ValueError("pred_deltas and gt_deltas must have the same length")
    return [abs(float(pred) - float(gt)) for pred, gt in zip(pred_deltas, gt_deltas)]
