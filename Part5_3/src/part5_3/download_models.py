"""Download Stable Diffusion and ControlNet-Tile models through a Hugging Face mirror."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from .constants import DEFAULT_HF_ENDPOINT


MODEL_SPECS = {
    "sd15": ("runwayml/stable-diffusion-v1-5", "stable-diffusion-v1-5"),
    "controlnet_tile": ("lllyasviel/control_v11f1e_sd15_tile", "control_v11f1e_sd15_tile"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Part 5.3 diffusion models.")
    parser.add_argument("--models", nargs="+", default=list(MODEL_SPECS), choices=sorted(MODEL_SPECS))
    parser.add_argument("--weights_dir", type=Path, default=Path("weights"))
    parser.add_argument("--hf_endpoint", default=DEFAULT_HF_ENDPOINT)
    parser.add_argument("--local_dir_use_symlinks", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.hf_endpoint:
        os.environ.setdefault("HF_ENDPOINT", args.hf_endpoint)
    from huggingface_hub import snapshot_download

    args.weights_dir.mkdir(parents=True, exist_ok=True)
    for key in args.models:
        repo_id, folder = MODEL_SPECS[key]
        out_dir = args.weights_dir / folder
        snapshot_download(
            repo_id=repo_id,
            local_dir=out_dir,
            local_dir_use_symlinks=args.local_dir_use_symlinks,
        )
        print(f"downloaded {repo_id} -> {out_dir}")


if __name__ == "__main__":
    main()
