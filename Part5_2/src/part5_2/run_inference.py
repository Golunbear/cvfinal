"""Run VSR baseline inference on supported datasets."""

from __future__ import annotations

import argparse
import gc
from pathlib import Path
from typing import Callable

import torch
from tqdm import tqdm

from .datasets import discover_sequences
from .image_io import list_image_files, read_image_tensor, read_sequence_tensor, write_sequence_tensors
from .inference_options import (
    MODEL_CHOICES,
    RECURRENT_VSR_MODELS,
    iter_basicvsr_chunk_windows,
    resolve_basicvsr_chunk_size,
)
from .models.basicvsr import infer_basicvsr_sequence, load_basicvsr
from .models.iconvsr import infer_iconvsr_sequence, load_iconvsr
from .models.realesrgan import RealESRGANer
from .video_io import encode_frames_to_video
from .weights import weight_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Part 5.2 VSR inference.")
    parser.add_argument("--model", required=True, choices=MODEL_CHOICES)
    parser.add_argument("--dataset", required=True, choices=["reds_val", "sample_reds", "vimeo_lr", "wild"])
    parser.add_argument("--data_root", type=Path, default=Path("../data"))
    parser.add_argument("--wild_frames", type=Path, default=None, help="Decoded frames folder for one wild video.")
    parser.add_argument("--weights_dir", type=Path, default=Path("weights"))
    parser.add_argument("--weights", type=Path, default=None, help="Explicit checkpoint path.")
    parser.add_argument("--out_root", type=Path, default=Path("outputs"))
    parser.add_argument("--device", default="cuda", help="cuda, cuda:0, or cpu.")
    parser.add_argument("--max_sequences", type=int, default=0)
    parser.add_argument("--max_frames", type=int, default=0)
    parser.add_argument("--chunk_size", type=int, default=0, help="BasicVSR/IconVSR chunk size; 0 means whole sequence.")
    parser.add_argument("--overlap", type=int, default=2, help="BasicVSR/IconVSR chunk overlap.")
    parser.add_argument("--tile", type=int, default=0, help="Real-ESRGAN tile size; 0 disables tiling.")
    parser.add_argument("--tile_pad", type=int, default=10)
    parser.add_argument("--fp32", action="store_true", help="Disable autocast for Real-ESRGAN.")
    parser.add_argument("--outscale", type=float, default=4.0)
    parser.add_argument("--fps", type=float, default=24.0, help="Fallback fps when metadata is unavailable.")
    return parser.parse_args()


def _device_from_arg(value: str) -> torch.device:
    if value.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA requested but not available; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(value)


def _should_encode_video(dataset: str) -> bool:
    return dataset != "reds_val"


def _is_cuda_oom(error: BaseException) -> bool:
    message = str(error).lower()
    return "cuda" in message and "out of memory" in message


def _clear_cuda_cache(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def _clear_output_frames(out_frames_dir: Path) -> None:
    if not out_frames_dir.exists():
        return
    for frame_path in list_image_files(out_frames_dir):
        frame_path.unlink()


def _run_recurrent_wild_streaming(
    display_name: str,
    model: torch.nn.Module,
    infer_sequence: Callable[[torch.nn.Module, torch.Tensor, int, int], torch.Tensor],
    lr_frames: list[Path],
    out_frames_dir: Path,
    names: list[str],
    chunk_size: int,
    overlap: int,
    device: torch.device,
) -> None:
    current_chunk_size = chunk_size
    while True:
        try:
            _clear_output_frames(out_frames_dir)
            windows = iter_basicvsr_chunk_windows(len(lr_frames), current_chunk_size, overlap)
            for start, end, chunk_start, chunk_end, keep_start, keep_end in tqdm(
                windows,
                desc=f"{display_name} wild chunks ({current_chunk_size})",
            ):
                lq = read_sequence_tensor(lr_frames[chunk_start:chunk_end])
                outputs = infer_sequence(model, lq, chunk_size=0, overlap=overlap)
                write_sequence_tensors(outputs[keep_start:keep_end], out_frames_dir, names=names[start:end])
                del lq, outputs
                _clear_cuda_cache(device)
            return
        except RuntimeError as error:
            if not _is_cuda_oom(error):
                raise
            _clear_cuda_cache(device)
            if current_chunk_size <= 3:
                _clear_output_frames(out_frames_dir)
                raise RuntimeError(
                    f"{display_name} wild inference still ran out of CUDA memory at chunk_size=3. "
                    "Try rerunning with --chunk_size 3, close other GPU processes, set "
                    "PYTORCH_ALLOC_CONF=expandable_segments:True, or downscale the wild input frames before inference."
                ) from error
            next_chunk_size = max(3, current_chunk_size // 2)
            print(f"CUDA OOM at chunk_size={current_chunk_size}; retrying with chunk_size={next_chunk_size}.")
            current_chunk_size = next_chunk_size


def main() -> None:
    args = parse_args()
    device = _device_from_arg(args.device)
    checkpoint = args.weights or weight_path(args.weights_dir, args.model)
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint}. Run python -m part5_2.download_weights --weights_dir {args.weights_dir}"
        )

    sequences = discover_sequences(
        args.dataset,
        data_root=args.data_root,
        wild_frames=args.wild_frames,
        max_sequences=args.max_sequences,
    )
    if not sequences:
        raise RuntimeError(f"No sequences found for dataset {args.dataset}")

    infer_sequence: Callable[[torch.nn.Module, torch.Tensor, int, int], torch.Tensor] | None = None
    recurrent_chunk_size = 0
    display_name = args.model
    if args.model in RECURRENT_VSR_MODELS:
        if args.model == "basicvsr":
            model = load_basicvsr(checkpoint, device=device, strict=True)
            infer_sequence = infer_basicvsr_sequence
            display_name = "BasicVSR"
        else:
            model = load_iconvsr(checkpoint, device=device, strict=True)
            infer_sequence = infer_iconvsr_sequence
            display_name = "IconVSR"
        recurrent_chunk_size = resolve_basicvsr_chunk_size(args.model, args.dataset, args.chunk_size)
        if recurrent_chunk_size > 0:
            print(f"{display_name} chunked inference: chunk_size={recurrent_chunk_size}, overlap={args.overlap}")
        else:
            print(f"{display_name} whole-sequence inference: chunk_size=0")
    else:
        model = RealESRGANer(checkpoint, device=device, tile=args.tile, tile_pad=args.tile_pad, fp32=args.fp32)

    for item in sequences:
        lr_frames = item.lr_frames(max_frames=args.max_frames)
        if not lr_frames:
            print(f"skip empty sequence: {item.lr_dir}")
            continue
        out_frames_dir = args.out_root / args.model / args.dataset / item.name / "frames"
        names = [path.name for path in lr_frames]
        print(f"[{args.model}] {args.dataset}/{item.name}: {len(lr_frames)} frames")

        if args.model in RECURRENT_VSR_MODELS and args.dataset == "wild":
            if infer_sequence is None:
                raise RuntimeError("Internal error: recurrent inference function was not initialized.")
            _run_recurrent_wild_streaming(
                display_name=display_name,
                model=model,
                infer_sequence=infer_sequence,
                lr_frames=lr_frames,
                out_frames_dir=out_frames_dir,
                names=names,
                chunk_size=recurrent_chunk_size,
                overlap=args.overlap,
                device=device,
            )
        elif args.model in RECURRENT_VSR_MODELS:
            if infer_sequence is None:
                raise RuntimeError("Internal error: recurrent inference function was not initialized.")
            lq = read_sequence_tensor(lr_frames)
            outputs = infer_sequence(model, lq, chunk_size=recurrent_chunk_size, overlap=args.overlap)
            write_sequence_tensors(outputs, out_frames_dir, names=names)
        else:
            out_frames_dir.mkdir(parents=True, exist_ok=True)
            for frame_path in tqdm(lr_frames, desc=f"Real-ESRGAN {item.name}"):
                image = read_image_tensor(frame_path)
                output = model.enhance_tensor(image, outscale=args.outscale)
                write_sequence_tensors(output.unsqueeze(0), out_frames_dir, names=[frame_path.name])

        if _should_encode_video(args.dataset):
            video_fps = item.fps or args.fps
            video_path = args.out_root / args.model / args.dataset / f"{item.name}.mp4"
            encode_frames_to_video(out_frames_dir, video_path, fps=video_fps)
            print(f"wrote video: {video_path}")
        else:
            print(f"skip mp4 encoding for reds_val; frames saved at {out_frames_dir}")


if __name__ == "__main__":
    main()
