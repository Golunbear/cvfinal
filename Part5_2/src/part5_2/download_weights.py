"""Download pretrained model checkpoints."""

from __future__ import annotations

import argparse

from .weights import WEIGHTS, download_weight


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Part 5.2 pretrained checkpoints.")
    parser.add_argument("--weights_dir", default="weights", help="Directory where checkpoints are stored.")
    parser.add_argument("--models", nargs="+", default=list(WEIGHTS.keys()), choices=sorted(WEIGHTS.keys()))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for model_name in args.models:
        download_weight(model_name, args.weights_dir, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
