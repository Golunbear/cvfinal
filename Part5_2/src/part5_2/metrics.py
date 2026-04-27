"""Evaluation metrics for paired video SR sequences."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from skimage.metrics import structural_similarity

from .image_io import list_image_files, read_image_tensor
from .temporal_metrics import temporal_lpips_from_deltas


@dataclass
class SequenceMetrics:
    frame_rows: list[dict]
    summary: dict


def crop_border_tensor(x: torch.Tensor, crop_border: int) -> torch.Tensor:
    if crop_border <= 0:
        return x
    return x[..., crop_border:-crop_border, crop_border:-crop_border]


def match_size(pred: torch.Tensor, gt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if pred.shape[-2:] == gt.shape[-2:]:
        return pred, gt
    h = min(pred.shape[-2], gt.shape[-2])
    w = min(pred.shape[-1], gt.shape[-1])
    return pred[..., :h, :w], gt[..., :h, :w]


def compute_psnr(pred: torch.Tensor, gt: torch.Tensor, crop_border: int = 0) -> float:
    pred, gt = match_size(pred, gt)
    pred = crop_border_tensor(pred, crop_border)
    gt = crop_border_tensor(gt, crop_border)
    mse = torch.mean((pred - gt) ** 2).item()
    if mse == 0:
        return float("inf")
    return 10.0 * math.log10(1.0 / mse)


def _tensor_to_uint8_hwc(x: torch.Tensor) -> np.ndarray:
    return (x.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255.0).round().astype(np.uint8)


def compute_ssim(pred: torch.Tensor, gt: torch.Tensor, crop_border: int = 0) -> float:
    pred, gt = match_size(pred, gt)
    pred = crop_border_tensor(pred, crop_border)
    gt = crop_border_tensor(gt, crop_border)
    return float(
        structural_similarity(
            _tensor_to_uint8_hwc(pred),
            _tensor_to_uint8_hwc(gt),
            channel_axis=2,
            data_range=255,
        )
    )


class LPIPSEvaluator:
    def __init__(self, device: torch.device, net: str = "alex") -> None:
        import lpips

        self.model = lpips.LPIPS(net=net).to(device).eval()
        self.device = device

    @torch.inference_mode()
    def __call__(self, a: torch.Tensor, b: torch.Tensor) -> float:
        a, b = match_size(a, b)
        a = a.unsqueeze(0).to(self.device) * 2.0 - 1.0
        b = b.unsqueeze(0).to(self.device) * 2.0 - 1.0
        return float(self.model(a, b).item())


def compute_fid(pred_dir: str | Path, gt_dir: str | Path, device: torch.device) -> float:
    from cleanfid import fid

    device_str = "cuda" if device.type == "cuda" else "cpu"
    return float(fid.compute_fid(str(pred_dir), str(gt_dir), mode="clean", device=device_str))


def evaluate_sequence(
    pred_dir: str | Path,
    gt_dir: str | Path,
    device: torch.device,
    crop_border: int = 0,
    compute_fid_score: bool = True,
    lpips_net: str = "alex",
) -> SequenceMetrics:
    pred_dir = Path(pred_dir)
    gt_dir = Path(gt_dir)
    pred_paths = list_image_files(pred_dir)
    gt_paths = list_image_files(gt_dir)
    if not pred_paths:
        raise ValueError(f"No predicted frames found in {pred_dir}")
    if not gt_paths:
        raise ValueError(f"No GT frames found in {gt_dir}")

    gt_by_name = {p.name: p for p in gt_paths}
    pairs = [(pred_path, gt_by_name[pred_path.name]) for pred_path in pred_paths if pred_path.name in gt_by_name]
    if not pairs:
        count = min(len(pred_paths), len(gt_paths))
        pairs = list(zip(pred_paths[:count], gt_paths[:count]))
    if not pairs:
        raise ValueError(f"No comparable pred/GT frames for {pred_dir} and {gt_dir}")

    lpips_eval = LPIPSEvaluator(device=device, net=lpips_net)
    frame_rows = []
    pred_tensors = []
    gt_tensors = []
    for idx, (pred_path, gt_path) in enumerate(pairs):
        pred = read_image_tensor(pred_path)
        gt = read_image_tensor(gt_path)
        pred_tensors.append(pred)
        gt_tensors.append(gt)
        frame_rows.append(
            {
                "frame_index": idx,
                "frame_name": pred_path.name,
                "psnr": compute_psnr(pred, gt, crop_border=crop_border),
                "ssim": compute_ssim(pred, gt, crop_border=crop_border),
                "lpips": lpips_eval(pred, gt),
            }
        )

    pred_deltas = []
    gt_deltas = []
    for idx in range(1, len(pred_tensors)):
        pred_deltas.append(lpips_eval(pred_tensors[idx - 1], pred_tensors[idx]))
        gt_deltas.append(lpips_eval(gt_tensors[idx - 1], gt_tensors[idx]))
    tlpips_rows = temporal_lpips_from_deltas(pred_deltas, gt_deltas)
    for idx, value in enumerate(tlpips_rows, start=1):
        frame_rows[idx]["tlpips"] = value
    if frame_rows:
        frame_rows[0]["tlpips"] = np.nan

    fid_score = np.nan
    if compute_fid_score:
        fid_score = compute_fid(pred_dir, gt_dir, device=device)

    summary = {
        "gt_available": True,
        "num_frames": len(pairs),
        "psnr_mean": _nanmean([row["psnr"] for row in frame_rows]),
        "ssim_mean": _nanmean([row["ssim"] for row in frame_rows]),
        "lpips_mean": _nanmean([row["lpips"] for row in frame_rows]),
        "tlpips_mean": _nanmean(tlpips_rows),
        "tlpips_std": _nanstd(tlpips_rows),
        "fid": fid_score,
    }
    return SequenceMetrics(frame_rows=frame_rows, summary=summary)


def no_gt_summary(num_frames: int) -> SequenceMetrics:
    return SequenceMetrics(
        frame_rows=[],
        summary={
            "gt_available": False,
            "num_frames": num_frames,
            "psnr_mean": np.nan,
            "ssim_mean": np.nan,
            "lpips_mean": np.nan,
            "tlpips_mean": np.nan,
            "tlpips_std": np.nan,
            "fid": np.nan,
        },
    )


def _nanmean(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(np.nanmean(np.asarray(values, dtype=np.float64)))


def _nanstd(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(np.nanstd(np.asarray(values, dtype=np.float64)))
