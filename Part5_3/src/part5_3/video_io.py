"""Video encoding and metadata helpers."""

from __future__ import annotations

import json
from pathlib import Path

from tqdm import tqdm

from .image_io import list_image_files


def read_video_metadata(frames_dir: str | Path) -> dict:
    metadata_path = Path(frames_dir) / "metadata.json"
    if not metadata_path.exists():
        return {}
    with metadata_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def encode_frames_to_video(
    frames_dir: str | Path,
    video_path: str | Path,
    fps: float = 24.0,
    codec: str = "mp4v",
) -> None:
    import cv2

    frame_paths = list_image_files(frames_dir)
    if not frame_paths:
        raise ValueError(f"No frames found in {frames_dir}")
    first = cv2.imread(str(frame_paths[0]), cv2.IMREAD_COLOR)
    if first is None:
        raise RuntimeError(f"Failed to read first frame: {frame_paths[0]}")

    height, width = first.shape[:2]
    out_path = Path(video_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(str(out_path), fourcc, fps if fps > 0 else 24.0, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to create video writer: {out_path}")
    try:
        for path in tqdm(frame_paths, desc=f"encode {out_path.name}"):
            frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if frame is None:
                raise RuntimeError(f"Failed to read frame: {path}")
            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            writer.write(frame)
    finally:
        writer.release()
