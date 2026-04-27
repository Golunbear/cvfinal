"""Dataset discovery for REDS, sample clips, Vimeo-LR, and decoded wild videos."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .image_io import list_image_files
from .video_io import read_video_metadata


@dataclass(frozen=True)
class SequenceItem:
    name: str
    lr_dir: Path
    gt_dir: Path | None = None
    fps: float | None = None

    @property
    def has_gt(self) -> bool:
        return self.gt_dir is not None and self.gt_dir.exists()

    def lr_frames(self, max_frames: int = 0) -> list[Path]:
        frames = list_image_files(self.lr_dir)
        return frames[:max_frames] if max_frames and max_frames > 0 else frames

    def gt_frames(self, max_frames: int = 0) -> list[Path]:
        if self.gt_dir is None:
            return []
        frames = list_image_files(self.gt_dir)
        return frames[:max_frames] if max_frames and max_frames > 0 else frames


def discover_sequences(
    dataset: str,
    data_root: str | Path,
    wild_frames: str | Path | None = None,
    max_sequences: int = 0,
) -> list[SequenceItem]:
    root = Path(data_root)
    if dataset == "reds_val":
        items = _discover_reds_val(root)
    elif dataset == "sample_reds":
        items = _discover_sample_reds(root)
    elif dataset == "vimeo_lr":
        items = _discover_vimeo_lr(root)
    elif dataset == "wild":
        items = _discover_wild(root, wild_frames=wild_frames)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    return items[:max_sequences] if max_sequences and max_sequences > 0 else items


def _subdirs(path: Path) -> list[Path]:
    if not path.exists():
        return []
    return sorted([p for p in path.iterdir() if p.is_dir()])


def _discover_reds_val(root: Path) -> list[SequenceItem]:
    lr_root = root / "REDS" / "val" / "val_sharp_bicubic" / "X4"
    gt_root = root / "REDS" / "val" / "val_sharp"
    items = []
    for lr_dir in _subdirs(lr_root):
        gt_dir = gt_root / lr_dir.name
        if list_image_files(lr_dir):
            items.append(SequenceItem(name=lr_dir.name, lr_dir=lr_dir, gt_dir=gt_dir if gt_dir.exists() else None))
    return items


def _discover_sample_reds(root: Path) -> list[SequenceItem]:
    sample_root = root / "Sample_data" / "REDS-sample"
    return [
        SequenceItem(name=seq.name, lr_dir=seq)
        for seq in _subdirs(sample_root)
        if list_image_files(seq)
    ]


def _discover_vimeo_lr(root: Path) -> list[SequenceItem]:
    vimeo_root = root / "Sample_data" / "vimeo-RL"
    items = []
    for scene_dir in _subdirs(vimeo_root):
        for clip_dir in _subdirs(scene_dir):
            if list_image_files(clip_dir):
                items.append(SequenceItem(name=f"{scene_dir.name}_{clip_dir.name}", lr_dir=clip_dir))
    return items


def _discover_wild(root: Path, wild_frames: str | Path | None) -> list[SequenceItem]:
    if wild_frames:
        frames_dir = Path(wild_frames)
        if not frames_dir.exists():
            raise FileNotFoundError(f"Decoded wild frames directory does not exist: {frames_dir}")
        meta = read_video_metadata(frames_dir)
        return [SequenceItem(name=frames_dir.name, lr_dir=frames_dir, fps=meta.get("fps"))]

    decoded_root = root / "wild-video" / "decoded_frames"
    items = []
    for frames_dir in _subdirs(decoded_root):
        if list_image_files(frames_dir):
            meta = read_video_metadata(frames_dir)
            items.append(SequenceItem(name=frames_dir.name, lr_dir=frames_dir, fps=meta.get("fps")))
    if not items:
        raise FileNotFoundError(
            "No decoded wild frames found. Run: python -m part5_2.decode_wild_video --wild_root ../data/wild-video"
        )
    return items
