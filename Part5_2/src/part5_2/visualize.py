"""Create simple qualitative comparison grids."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw

from .datasets import discover_sequences
from .image_io import list_image_files, read_image
from .inference_options import MODEL_CHOICES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create side-by-side qualitative comparison images.")
    parser.add_argument("--models", nargs="+", required=True, choices=MODEL_CHOICES)
    parser.add_argument("--dataset", required=True, choices=["reds_val", "sample_reds", "vimeo_lr", "wild"])
    parser.add_argument("--data_root", type=Path, default=Path("../data"))
    parser.add_argument("--pred_root", type=Path, default=Path("outputs"))
    parser.add_argument("--wild_frames", type=Path, default=None)
    parser.add_argument("--out_dir", type=Path, default=Path("outputs/visuals"))
    parser.add_argument("--frame_index", type=int, default=0)
    parser.add_argument("--max_sequences", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sequences = discover_sequences(
        args.dataset,
        data_root=args.data_root,
        wild_frames=args.wild_frames,
        max_sequences=args.max_sequences,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for item in sequences:
        panels: list[tuple[str, Image.Image]] = []
        lr_frames = item.lr_frames()
        if not lr_frames:
            continue
        lr_frame = lr_frames[min(args.frame_index, len(lr_frames) - 1)]
        panels.append(("LR", read_image(lr_frame)))
        if item.has_gt:
            gt_frames = item.gt_frames()
            gt_frame = gt_frames[min(args.frame_index, len(gt_frames) - 1)]
            panels.append(("GT", read_image(gt_frame)))
        for model in args.models:
            pred_dir = args.pred_root / model / args.dataset / item.name / "frames"
            pred_frames = list_image_files(pred_dir) if pred_dir.exists() else []
            if pred_frames:
                pred_frame = pred_frames[min(args.frame_index, len(pred_frames) - 1)]
                panels.append((model, read_image(pred_frame)))
        if len(panels) >= 2:
            out = _make_grid(panels)
            out.save(args.out_dir / f"{args.dataset}_{item.name}_{args.frame_index:04d}.png")


def _make_grid(panels: list[tuple[str, Image.Image]]) -> Image.Image:
    target_h = max(image.height for _, image in panels)
    resized = []
    for label, image in panels:
        scale = target_h / image.height
        resized.append((label, image.resize((int(round(image.width * scale)), target_h), Image.Resampling.BICUBIC)))
    label_h = 28
    width = sum(image.width for _, image in resized)
    canvas = Image.new("RGB", (width, target_h + label_h), "white")
    draw = ImageDraw.Draw(canvas)
    x = 0
    for label, image in resized:
        canvas.paste(image, (x, label_h))
        draw.text((x + 8, 7), label, fill=(0, 0, 0))
        x += image.width
    return canvas


if __name__ == "__main__":
    main()
