"""Run Stable Diffusion + ControlNet-Tile enhancement on Part2 VSR outputs."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from tqdm import tqdm

from .constants import (
    DATASET_CHOICES,
    DEFAULT_HF_ENDPOINT,
    DEFAULT_INPUT_MODEL,
    DEFAULT_NEGATIVE_PROMPT,
    DEFAULT_PROMPT,
)
from .datasets import discover_sequences, part3_frames_dir, part3_video_path
from .image_io import read_image, write_image
from .pipeline import enhance_image, load_pipeline
from .temporal_consistency import temporal_blend_frame
from .video_io import encode_frames_to_video


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Part 5.3 SD + ControlNet-Tile inference.")
    parser.add_argument("--dataset", required=True, choices=DATASET_CHOICES)
    parser.add_argument("--data_root", type=Path, default=Path("../data"))
    parser.add_argument("--part2_root", type=Path, default=Path("../Part5_2/outputs"))
    parser.add_argument("--input_model", default=DEFAULT_INPUT_MODEL)
    parser.add_argument("--wild_frames", type=Path, default=None)
    parser.add_argument("--out_root", type=Path, default=Path("outputs"))
    parser.add_argument("--run_name", default="sd_tile")
    parser.add_argument("--pretrained_model", default="weights/stable-diffusion-v1-5")
    parser.add_argument("--controlnet_model", default="weights/control_v11f1e_sd15_tile")
    parser.add_argument("--lora_path", type=Path, default=None)
    parser.add_argument("--hf_endpoint", default=DEFAULT_HF_ENDPOINT)
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--no_controlnet", action="store_true", help="Run SD img2img only for ablation.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fp32", action="store_true")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--negative_prompt", default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--strength", type=float, default=0.35)
    parser.add_argument("--control_scale", type=float, default=0.8)
    parser.add_argument("--guidance_scale", type=float, default=5.0)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tile_size", type=int, default=768)
    parser.add_argument("--tile_overlap", type=int, default=96)
    parser.add_argument("--temporal_blend", type=float, default=0.0)
    parser.add_argument("--max_sequences", type=int, default=0)
    parser.add_argument("--max_frames", type=int, default=0)
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--no_video", action="store_true")
    return parser.parse_args()


def _device_from_arg(value: str) -> torch.device:
    if value.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA requested but not available; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(value)


def _should_encode_video(dataset: str, no_video: bool) -> bool:
    return (not no_video) and dataset != "reds_val"


def main() -> None:
    args = parse_args()
    if args.hf_endpoint:
        os.environ.setdefault("HF_ENDPOINT", args.hf_endpoint)
    device = _device_from_arg(args.device)
    dtype = torch.float32 if args.fp32 or device.type == "cpu" else torch.float16
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

    pipe = load_pipeline(
        pretrained_model=args.pretrained_model,
        controlnet_model=args.controlnet_model,
        device=device,
        dtype=dtype,
        lora_path=args.lora_path,
        no_controlnet=args.no_controlnet,
        hf_endpoint=args.hf_endpoint,
        local_files_only=args.local_files_only,
    )

    for item in sequences:
        frame_paths = item.input_frames(max_frames=args.max_frames)
        if not frame_paths:
            print(f"skip empty Part2 sequence: {item.input_dir}")
            continue
        out_frames_dir = part3_frames_dir(args.out_root, args.run_name, args.input_model, args.dataset, item.name)
        out_frames_dir.mkdir(parents=True, exist_ok=True)
        print(f"[{args.run_name}] {args.input_model}/{args.dataset}/{item.name}: {len(frame_paths)} frames")

        prev_output = None
        prev_input = None
        for frame_idx, frame_path in enumerate(tqdm(frame_paths, desc=f"{args.run_name} {item.name}")):
            out_path = out_frames_dir / frame_path.name
            current_input = read_image(frame_path)
            if args.skip_existing and out_path.exists():
                prev_output = read_image(out_path)
                prev_input = current_input
                continue

            output = enhance_image(
                pipe=pipe,
                image=current_input,
                prompt=args.prompt,
                negative_prompt=args.negative_prompt,
                strength=args.strength,
                control_scale=args.control_scale,
                guidance_scale=args.guidance_scale,
                steps=args.steps,
                seed=args.seed,
                device=device,
                tile_size=args.tile_size,
                tile_overlap=args.tile_overlap,
                no_controlnet=args.no_controlnet,
            )
            output = temporal_blend_frame(
                generated=output,
                current_input=current_input,
                previous_output=prev_output,
                previous_input=prev_input,
                blend_strength=args.temporal_blend,
            )
            write_image(output, out_path)
            prev_output = output
            prev_input = current_input

        if _should_encode_video(args.dataset, args.no_video):
            video_path = part3_video_path(args.out_root, args.run_name, args.input_model, args.dataset, item.name)
            encode_frames_to_video(out_frames_dir, video_path, fps=item.fps or args.fps)
            print(f"wrote video: {video_path}")
        else:
            print(f"frames saved at {out_frames_dir}")


if __name__ == "__main__":
    main()
