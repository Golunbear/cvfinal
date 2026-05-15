"""Dataset and Part2-output discovery for Part 5.3 Direction A."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .image_io import list_image_files
from .video_io import read_video_metadata


@dataclass(frozen=True)
class SequenceItem:
    name: str
    input_dir: Path
    lr_dir: Path | None = None
    gt_dir: Path | None = None
    fps: float | None = None

    @property
    def has_gt(self) -> bool:
        return self.gt_dir is not None and self.gt_dir.exists()

    def input_frames(self, max_frames: int = 0) -> list[Path]:
        frames = list_image_files(self.input_dir)
        return frames[:max_frames] if max_frames and max_frames > 0 else frames

    def lr_frames(self, max_frames: int = 0) -> list[Path]:
        if self.lr_dir is None:
            return []
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
    part2_root: str | Path,
    input_model: str,
    wild_frames: str | Path | None = None,
    max_sequences: int = 0,
) -> list[SequenceItem]:
    root = Path(data_root)
    part2_root = Path(part2_root)
    if dataset == "reds_val":
        items = _discover_reds_val(root, part2_root, input_model)
    elif dataset == "sample_reds":
        items = _discover_sample_reds(root, part2_root, input_model)
    elif dataset == "vimeo_lr":
        items = _discover_vimeo_lr(root, part2_root, input_model)
    elif dataset == "wild":
        items = _discover_wild(root, part2_root, input_model, wild_frames)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    return items[:max_sequences] if max_sequences and max_sequences > 0 else items


def output_frames_dir(out_root: str | Path, run_name: str, input_model: str, dataset: str, sequence_name: str) -> Path:
    return Path(out_root) / run_name / input_model / dataset / sequence_name / "frames"


def output_video_path(out_root: str | Path, run_name: str, input_model: str, dataset: str, sequence_name: str) -> Path:
    return Path(out_root) / run_name / input_model / dataset / f"{sequence_name}.mp4"


def _subdirs(path: Path) -> list[Path]:
    if not path.exists():
        return []
    return sorted([p for p in path.iterdir() if p.is_dir()])


def _part2_sequence_dirs(part2_root: Path, input_model: str, dataset: str) -> list[Path]:
    root = part2_root / input_model / dataset
    return [seq for seq in _subdirs(root) if list_image_files(seq / "frames")]


def _discover_reds_val(root: Path, part2_root: Path, input_model: str) -> list[SequenceItem]:
    gt_root = root / "REDS" / "val" / "val_sharp"
    lr_root = root / "REDS" / "val" / "val_sharp_bicubic" / "X4"
    return [
        SequenceItem(
            name=seq.name,
            input_dir=seq / "frames",
            lr_dir=lr_root / seq.name,
            gt_dir=gt_root / seq.name,
        )
        for seq in _part2_sequence_dirs(part2_root, input_model, "reds_val")
    ]


def _discover_sample_reds(root: Path, part2_root: Path, input_model: str) -> list[SequenceItem]:
    lr_root = root / "Sample_data" / "REDS-sample"
    return [
        SequenceItem(name=seq.name, input_dir=seq / "frames", lr_dir=lr_root / seq.name)
        for seq in _part2_sequence_dirs(part2_root, input_model, "sample_reds")
    ]


def _discover_vimeo_lr(root: Path, part2_root: Path, input_model: str) -> list[SequenceItem]:
    lr_map = {}
    vimeo_root = root / "Sample_data" / "vimeo-RL"
    for scene in _subdirs(vimeo_root):
        for clip in _subdirs(scene):
            lr_map[f"{scene.name}_{clip.name}"] = clip
    return [
        SequenceItem(name=seq.name, input_dir=seq / "frames", lr_dir=lr_map.get(seq.name))
        for seq in _part2_sequence_dirs(part2_root, input_model, "vimeo_lr")
    ]


def _discover_wild(
    root: Path,
    part2_root: Path,
    input_model: str,
    wild_frames: str | Path | None,
) -> list[SequenceItem]:
    if wild_frames:
        frames_dir = Path(wild_frames)
        input_dir = part2_root / input_model / "wild" / frames_dir.name / "frames"
        meta = read_video_metadata(frames_dir)
        return [SequenceItem(name=frames_dir.name, input_dir=input_dir, lr_dir=frames_dir, fps=meta.get("fps"))]

    decoded_root = root / "wild-video" / "decoded_frames"
    items = []
    for seq in _part2_sequence_dirs(part2_root, input_model, "wild"):
        lr_dir = decoded_root / seq.name
        meta = read_video_metadata(lr_dir)
        items.append(SequenceItem(name=seq.name, input_dir=seq / "frames", lr_dir=lr_dir, fps=meta.get("fps")))
    return items
