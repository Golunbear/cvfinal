"""Create qualitative comparison grids."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw

from .constants import DATASET_CHOICES, DEFAULT_INPUT_MODEL
from .datasets import discover_sequences, output_frames_dir
from .image_io import list_image_files, read_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create Part 5.3 DA visual comparison grids.")
    parser.add_argument("--methods", nargs="+", required=True)
    parser.add_argument("--dataset", required=True, choices=DATASET_CHOICES)
    parser.add_argument("--data_root", type=Path, default=Path("../data"))
    parser.add_argument("--part2_root", type=Path, default=Path("../Part5_2/outputs"))
    parser.add_argument("--input_model", default=DEFAULT_INPUT_MODEL)
    parser.add_argument("--pred_root", type=Path, default=Path("outputs"))
    parser.add_argument("--wild_frames", type=Path, default=None)
    parser.add_argument("--out_dir", type=Path, default=Path("outputs/visuals"))
    parser.add_argument("--frame_index", type=int, default=0)
    parser.add_argument("--max_sequences", type=int, default=4)
    parser.add_argument("--crop", nargs=4, type=int, metavar=("X", "Y", "W", "H"), default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sequences = discover_sequences(
        dataset=args.dataset,
        data_root=args.data_root,
        part2_root=args.part2_root,
        input_model=args.input_model,
        wild_frames=args.wild_frames,
        max_sequences=args.max_sequences,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for item in sequences:
        panels: list[tuple[str, Image.Image]] = []
        lr_frames = item.lr_frames()
        if lr_frames:
            panels.append(("LR", read_image(lr_frames[min(args.frame_index, len(lr_frames) - 1)])))
        part2_frames = item.input_frames()
        if part2_frames:
            panels.append((f"Part2-{args.input_model}", read_image(part2_frames[min(args.frame_index, len(part2_frames) - 1)])))
        if item.has_gt:
            gt_frames = item.gt_frames()
            panels.append(("GT", read_image(gt_frames[min(args.frame_index, len(gt_frames) - 1)])))
        for method in args.methods:
            pred_dir = output_frames_dir(args.pred_root, method, args.input_model, args.dataset, item.name)
            pred_frames = list_image_files(pred_dir) if pred_dir.exists() else []
            if pred_frames:
                panels.append((method, read_image(pred_frames[min(args.frame_index, len(pred_frames) - 1)])))
        if len(panels) < 2:
            continue
        grid = _make_grid(panels)
        grid.save(args.out_dir / f"{args.dataset}_{item.name}_{args.frame_index:04d}.png")
        if args.crop is not None:
            crop_panels = [(label, _crop_image(image, args.crop)) for label, image in panels]
            _make_grid(crop_panels).save(args.out_dir / f"{args.dataset}_{item.name}_{args.frame_index:04d}_zoom.png")


def _crop_image(image: Image.Image, crop: list[int]) -> Image.Image:
    x, y, w, h = crop
    return image.crop((x, y, x + w, y + h))


def _make_grid(panels: list[tuple[str, Image.Image]]) -> Image.Image:
    target_h = max(image.height for _, image in panels)
    resized = []
    for label, image in panels:
        scale = target_h / image.height
        resized.append((label, image.resize((max(1, int(round(image.width * scale))), target_h), Image.Resampling.BICUBIC)))
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
