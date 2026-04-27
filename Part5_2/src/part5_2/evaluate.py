"""Evaluate saved VSR results."""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch

from .datasets import discover_sequences
from .image_io import list_image_files
from .inference_options import MODEL_CHOICES
from .metrics import compute_fid
from .metrics import evaluate_sequence, no_gt_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Part 5.2 outputs.")
    parser.add_argument("--models", nargs="+", required=True, choices=MODEL_CHOICES)
    parser.add_argument("--dataset", required=True, choices=["reds_val", "sample_reds", "vimeo_lr", "wild"])
    parser.add_argument("--data_root", type=Path, default=Path("../data"))
    parser.add_argument("--pred_root", type=Path, default=Path("outputs"))
    parser.add_argument("--wild_frames", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--crop_border", type=int, default=0)
    parser.add_argument("--skip_fid", action="store_true")
    parser.add_argument("--sequence_fid", action="store_true", help="Also compute per-sequence FID.")
    parser.add_argument("--max_sequences", type=int, default=0)
    parser.add_argument("--max_frames", type=int, default=0)
    parser.add_argument("--metrics_dir", type=Path, default=None)
    return parser.parse_args()


def _device_from_arg(value: str) -> torch.device:
    if value.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA requested but not available; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(value)


def main() -> None:
    args = parse_args()
    device = _device_from_arg(args.device)
    sequences = discover_sequences(
        args.dataset,
        data_root=args.data_root,
        wild_frames=args.wild_frames,
        max_sequences=args.max_sequences,
    )
    metrics_dir = args.metrics_dir or (args.pred_root / "metrics")
    metrics_dir.mkdir(parents=True, exist_ok=True)

    frame_rows = []
    summary_rows = []
    for model_name in args.models:
        model_frame_rows = []
        aggregate_pairs = []
        for item in sequences:
            pred_dir = args.pred_root / model_name / args.dataset / item.name / "frames"
            if not pred_dir.exists():
                print(f"missing predictions: {pred_dir}")
                continue
            if item.has_gt:
                metrics = evaluate_sequence(
                    pred_dir=pred_dir,
                    gt_dir=item.gt_dir,
                    device=device,
                    crop_border=args.crop_border,
                    compute_fid_score=(not args.skip_fid and args.sequence_fid),
                )
                aggregate_pairs.append((item.name, pred_dir, item.gt_dir))
            else:
                metrics = no_gt_summary(num_frames=len(list_image_files(pred_dir)))

            for row in metrics.frame_rows:
                row.update({"model": model_name, "dataset": args.dataset, "sequence": item.name})
                frame_rows.append(row)
                model_frame_rows.append(row)
            summary = metrics.summary
            summary.update({"model": model_name, "dataset": args.dataset, "sequence": item.name})
            summary_rows.append(summary)
            print(
                f"{model_name}/{args.dataset}/{item.name}: "
                f"PSNR={summary['psnr_mean']}, SSIM={summary['ssim_mean']}, "
                f"LPIPS={summary['lpips_mean']}, FID={summary['fid']}, tLPIPS={summary['tlpips_mean']}"
            )
        if aggregate_pairs:
            aggregate = _aggregate_model_summary(
                model_name=model_name,
                dataset=args.dataset,
                model_frame_rows=model_frame_rows,
                aggregate_pairs=aggregate_pairs,
                device=device,
                skip_fid=args.skip_fid,
                metrics_dir=metrics_dir,
            )
            summary_rows.append(aggregate)
            print(
                f"{model_name}/{args.dataset}/__all__: "
                f"PSNR={aggregate['psnr_mean']}, SSIM={aggregate['ssim_mean']}, "
                f"LPIPS={aggregate['lpips_mean']}, FID={aggregate['fid']}, tLPIPS={aggregate['tlpips_mean']}"
            )

    _write_csv(metrics_dir / "frame_metrics.csv", frame_rows)
    _write_csv(metrics_dir / "summary_metrics.csv", summary_rows)


def _write_csv(path: Path, rows: list[dict]) -> None:
    if rows:
        fieldnames = sorted({key for row in rows for key in row.keys()})
    else:
        fieldnames = [
            "model",
            "dataset",
            "sequence",
            "gt_available",
            "num_frames",
            "psnr_mean",
            "ssim_mean",
            "lpips_mean",
            "tlpips_mean",
            "tlpips_std",
            "fid",
        ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path}")


def _aggregate_model_summary(
    model_name: str,
    dataset: str,
    model_frame_rows: list[dict],
    aggregate_pairs: list[tuple[str, Path, Path]],
    device: torch.device,
    skip_fid: bool,
    metrics_dir: Path,
) -> dict:
    fid_score = np.nan
    if not skip_fid:
        with tempfile.TemporaryDirectory(dir=metrics_dir) as tmp:
            tmp_root = Path(tmp)
            pred_flat = tmp_root / "pred"
            gt_flat = tmp_root / "gt"
            pred_flat.mkdir()
            gt_flat.mkdir()
            for seq_name, pred_dir, gt_dir in aggregate_pairs:
                _flatten_images(pred_dir, pred_flat, prefix=seq_name)
                _flatten_images(gt_dir, gt_flat, prefix=seq_name)
            fid_score = compute_fid(pred_flat, gt_flat, device=device)

    tlpips_values = [row.get("tlpips") for row in model_frame_rows if not _is_nan(row.get("tlpips"))]
    return {
        "model": model_name,
        "dataset": dataset,
        "sequence": "__all__",
        "gt_available": True,
        "num_frames": len(model_frame_rows),
        "psnr_mean": _mean_from_rows(model_frame_rows, "psnr"),
        "ssim_mean": _mean_from_rows(model_frame_rows, "ssim"),
        "lpips_mean": _mean_from_rows(model_frame_rows, "lpips"),
        "tlpips_mean": float(np.nanmean(tlpips_values)) if tlpips_values else np.nan,
        "tlpips_std": float(np.nanstd(tlpips_values)) if tlpips_values else np.nan,
        "fid": fid_score,
    }


def _flatten_images(src_dir: Path, dst_dir: Path, prefix: str) -> None:
    for src in list_image_files(src_dir):
        dst = dst_dir / f"{prefix}_{src.name}"
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)


def _mean_from_rows(rows: list[dict], key: str) -> float:
    values = [row.get(key) for row in rows if not _is_nan(row.get(key))]
    return float(np.nanmean(values)) if values else np.nan


def _is_nan(value: object) -> bool:
    try:
        return bool(np.isnan(value))
    except TypeError:
        return False


if __name__ == "__main__":
    main()
