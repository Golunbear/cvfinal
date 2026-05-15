"""Run flow matching SR enhancement on Part2 VSR outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from .constants import DATASET_CHOICES, DEFAULT_INPUT_MODEL
from .datasets import discover_sequences, output_frames_dir, output_video_path
from .image_io import read_image_tensor, tensor_to_image, write_image
from .pipeline import enhance_tensor, load_model
from .temporal_consistency import temporal_blend_frame
from .video_io import encode_frames_to_video


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Part 5.3 Direction A flow matching inference.")
    p.add_argument("--dataset", required=True, choices=DATASET_CHOICES)
    p.add_argument("--data_root", type=Path, default=Path("../data"))
    p.add_argument("--part2_root", type=Path, default=Path("../Part5_2/outputs"))
    p.add_argument("--input_model", default=DEFAULT_INPUT_MODEL)
    p.add_argument("--wild_frames", type=Path, default=None)
    p.add_argument("--out_root", type=Path, default=Path("outputs"))
    p.add_argument("--run_name", default="fm_sr")
    p.add_argument("--checkpoint", type=Path, default=Path("weights/fm_sr/model.pt"))
    p.add_argument("--channels", type=int, nargs="+", default=[128, 256, 384, 512])
    p.add_argument("--blocks_per_level", type=int, default=2)
    p.add_argument("--attn_levels", type=int, nargs="+", default=[2, 3])
    p.add_argument("--use_ema", action="store_true", default=True)
    p.add_argument("--no_ema", action="store_true")
    p.add_argument("--device", default="cuda")
    p.add_argument("--fp32", action="store_true")
    p.add_argument("--steps", type=int, default=25)
    p.add_argument("--solver", choices=["euler", "midpoint"], default="euler")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tile_size", type=int, default=0)
    p.add_argument("--tile_overlap", type=int, default=64)
    p.add_argument("--temporal_blend", type=float, default=0.0)
    p.add_argument("--max_sequences", type=int, default=0)
    p.add_argument("--max_frames", type=int, default=0)
    p.add_argument("--fps", type=float, default=24.0)
    p.add_argument("--skip_existing", action="store_true")
    p.add_argument("--no_video", action="store_true")
    p.add_argument("--fast", action="store_true", help="Skip attention for faster inference")
    return p.parse_args()


def _device_from_arg(value: str) -> torch.device:
    if value.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA requested but not available; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(value)


def _should_encode_video(dataset: str, no_video: bool) -> bool:
    return (not no_video) and dataset != "reds_val"


def main() -> None:
    args = parse_args()
    device = _device_from_arg(args.device)
    if args.fp32:
        torch.set_default_dtype(torch.float32)

    sequences = discover_sequences(
        dataset=args.dataset,
        data_root=args.data_root,
        part2_root=args.part2_root,
        input_model=args.input_model,
        wild_frames=args.wild_frames,
        max_sequences=args.max_sequences,
    )
    if not sequences:
        raise RuntimeError(
            f"No Part2 input frames found for {args.input_model}/{args.dataset}. "
            "Run Part5_2 inference first or check --part2_root."
        )

    use_ema = args.use_ema and not args.no_ema
    model = load_model(
        checkpoint=args.checkpoint,
        channels=tuple(args.channels),
        blocks_per_level=args.blocks_per_level,
        attn_levels=tuple(args.attn_levels),
        device=device,
        use_ema=use_ema,
    )
    if not args.fp32 and device.type == "cuda":
        model = model.half()

    for item in sequences:
        frame_paths = item.input_frames(max_frames=args.max_frames)
        if not frame_paths:
            print(f"skip empty Part2 sequence: {item.input_dir}")
            continue
        out_dir = output_frames_dir(args.out_root, args.run_name, args.input_model, args.dataset, item.name)
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"[{args.run_name}] {args.input_model}/{args.dataset}/{item.name}: {len(frame_paths)} frames")

        prev_output = None
        prev_input = None
        for frame_path in tqdm(frame_paths, desc=f"{args.run_name} {item.name}"):
            out_path = out_dir / frame_path.name
            current_input_tensor = read_image_tensor(frame_path)
            current_input_pil = tensor_to_image(current_input_tensor)

            if args.skip_existing and out_path.exists():
                from .image_io import read_image
                prev_output = read_image(out_path)
                prev_input = current_input_pil
                continue

            output_tensor = enhance_tensor(
                model=model,
                condition=current_input_tensor,
                steps=args.steps,
                solver=args.solver,
                tile_size=args.tile_size,
                tile_overlap=args.tile_overlap,
                fast=args.fast,
            )
            output_pil = tensor_to_image(output_tensor)

            output_pil = temporal_blend_frame(
                generated=output_pil,
                current_input=current_input_pil,
                previous_output=prev_output,
                previous_input=prev_input,
                blend_strength=args.temporal_blend,
            )
            write_image(output_pil, out_path)
            prev_output = output_pil
            prev_input = current_input_pil

        if _should_encode_video(args.dataset, args.no_video):
            vid_path = output_video_path(args.out_root, args.run_name, args.input_model, args.dataset, item.name)
            encode_frames_to_video(out_dir, vid_path, fps=item.fps or args.fps)
            print(f"wrote video: {vid_path}")
        else:
            print(f"frames saved at {out_dir}")


if __name__ == "__main__":
    main()
