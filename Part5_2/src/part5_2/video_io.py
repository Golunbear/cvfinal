"""Video decoding and encoding helpers."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from .image_io import list_image_files


def decode_video_to_frames(video_path: str | Path, out_dir: str | Path, overwrite: bool = False) -> dict:
    import cv2

    video_path = Path(video_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = list_image_files(out_dir)
    metadata_path = out_dir / "metadata.json"
    if existing and metadata_path.exists() and not overwrite:
        with metadata_path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)
        metadata["skipped_existing"] = True
        return metadata

    if overwrite:
        for path in existing:
            path.unlink()
        if metadata_path.exists():
            metadata_path.unlink()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    frame_count_hint = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    count = 0
    progress = tqdm(total=frame_count_hint if frame_count_hint > 0 else None, desc=f"decode {video_path.name}")
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            Image.fromarray(frame_rgb).save(out_dir / f"{count:08d}.png")
            count += 1
            progress.update(1)
    finally:
        progress.close()
        cap.release()

    metadata = {
        "video_path": str(video_path.resolve()),
        "fps": fps,
        "width": width,
        "height": height,
        "num_frames": count,
        "frames_dir": str(out_dir.resolve()),
        "skipped_existing": False,
    }
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    return metadata


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
