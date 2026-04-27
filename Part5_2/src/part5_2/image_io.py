"""Image tensor IO helpers."""

from __future__ import annotations

from pathlib import Path

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def list_image_files(path: str | Path) -> list[Path]:
    root = Path(path)
    return sorted([p for p in root.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS])


def read_image(path: str | Path) -> Image.Image:
    from PIL import Image

    return Image.open(path).convert("RGB")


def image_to_tensor(image: Image.Image) -> torch.Tensor:
    import numpy as np
    import torch

    arr = np.asarray(image).astype(np.float32) / 255.0
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"Expected RGB image, got array shape {arr.shape}")
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def read_image_tensor(path: str | Path) -> torch.Tensor:
    return image_to_tensor(read_image(path))


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    import numpy as np
    from PIL import Image

    if tensor.ndim != 3:
        raise ValueError(f"Expected (c,h,w), got {tuple(tensor.shape)}")
    array = tensor.detach().clamp(0, 1).cpu().permute(1, 2, 0).numpy()
    array = (array * 255.0).round().astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


def save_tensor_image(tensor: torch.Tensor, path: str | Path) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tensor_to_image(tensor).save(out_path)


def read_sequence_tensor(frame_paths: list[Path], max_frames: int = 0) -> torch.Tensor:
    import torch

    selected = frame_paths[:max_frames] if max_frames and max_frames > 0 else frame_paths
    if not selected:
        raise ValueError("No frames were provided.")
    return torch.stack([read_image_tensor(path) for path in selected], dim=0)


def write_sequence_tensors(frames: torch.Tensor, out_dir: str | Path, names: list[str] | None = None) -> list[Path]:
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    if names is None:
        names = [f"{idx:08d}.png" for idx in range(frames.shape[0])]
    if len(names) != frames.shape[0]:
        raise ValueError(f"Got {len(names)} names for {frames.shape[0]} frames.")
    out_paths = []
    for frame, name in zip(frames, names):
        out_path = out_root / name
        save_tensor_image(frame, out_path)
        out_paths.append(out_path)
    return out_paths
