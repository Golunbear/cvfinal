"""Pretrained weight metadata and downloader."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import requests
from tqdm import tqdm


@dataclass(frozen=True)
class WeightSpec:
    filename: str
    url: str
    description: str


WEIGHTS: dict[str, WeightSpec] = {
    "basicvsr": WeightSpec(
        filename="basicvsr_reds4_20120409-0e599677.pth",
        url="https://download.openmmlab.com/mmediting/restorers/basicvsr/basicvsr_reds4_20120409-0e599677.pth",
        description="MMEditing/MMagic BasicVSR REDS4 checkpoint",
    ),
    "iconvsr": WeightSpec(
        filename="iconvsr_reds4_20210413-9e09d621.pth",
        url="https://download.openmmlab.com/mmediting/restorers/iconvsr/iconvsr_reds4_20210413-9e09d621.pth",
        description="MMEditing/MMagic IconVSR REDS4 checkpoint",
    ),
    "realesrgan": WeightSpec(
        filename="RealESRGAN_x4plus.pth",
        url="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        description="Official Real-ESRGAN x4plus checkpoint",
    ),
}


def weight_path(weights_dir: str | Path, model_name: str) -> Path:
    return Path(weights_dir) / WEIGHTS[model_name].filename


def download_weight(model_name: str, weights_dir: str | Path, overwrite: bool = False) -> Path:
    if model_name not in WEIGHTS:
        raise KeyError(f"Unknown weight key {model_name}. Choices: {sorted(WEIGHTS)}")
    spec = WEIGHTS[model_name]
    out_dir = Path(weights_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / spec.filename
    if out_path.exists() and not overwrite:
        print(f"exists: {out_path}")
        return out_path

    with requests.get(spec.url, stream=True, timeout=30) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        tmp_path = out_path.with_suffix(out_path.suffix + ".part")
        with tmp_path.open("wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=spec.filename) as progress:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    progress.update(len(chunk))
        tmp_path.replace(out_path)
    print(f"downloaded: {out_path}")
    return out_path
