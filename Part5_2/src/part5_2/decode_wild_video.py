"""Decode wild mp4 files once into reusable frame folders."""

from __future__ import annotations

import argparse
from pathlib import Path

from .video_io import decode_video_to_frames


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Decode data/wild-video/*.mp4 into decoded frame folders.")
    parser.add_argument("--wild_root", type=Path, required=True, help="Path to data/wild-video.")
    parser.add_argument("--overwrite", action="store_true", help="Re-decode videos even if frames already exist.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    wild_root = args.wild_root
    videos = sorted([p for p in wild_root.iterdir() if p.is_file() and p.suffix.lower() in {".mp4", ".mov", ".mkv"}])
    if not videos:
        raise FileNotFoundError(f"No video files found under {wild_root}")
    decoded_root = wild_root / "decoded_frames"
    for video in videos:
        out_dir = decoded_root / video.stem
        metadata = decode_video_to_frames(video, out_dir, overwrite=args.overwrite)
        status = "skipped" if metadata.get("skipped_existing") else "decoded"
        print(f"{status}: {video.name} -> {out_dir} ({metadata.get('num_frames', 0)} frames)")


if __name__ == "__main__":
    main()
