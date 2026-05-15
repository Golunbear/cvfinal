"""Train the flow matching SR model on REDS."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from .image_io import list_image_files


class REDSFlowDataset(Dataset):
    def __init__(self, data_root: str | Path, resolution: int = 256) -> None:
        self.resolution = resolution
        root = Path(data_root)
        hr_root = root / "REDS" / "train" / "train_sharp"
        lr_root = root / "REDS" / "train" / "train_sharp_bicubic" / "X4"
        self.pairs = self._collect(hr_root, lr_root)
        if not self.pairs:
            raise FileNotFoundError(f"No REDS train pairs under {hr_root}")

    def _collect(self, hr_root: Path, lr_root: Path) -> list[tuple[Path, Path]]:
        pairs = []
        if not hr_root.exists() or not lr_root.exists():
            return pairs
        for hr_seq in sorted(p for p in hr_root.iterdir() if p.is_dir()):
            lr_seq = lr_root / hr_seq.name
            if not lr_seq.exists():
                continue
            lr_map = {p.name: p for p in list_image_files(lr_seq)}
            for hr_path in list_image_files(hr_seq):
                lr_path = lr_map.get(hr_path.name)
                if lr_path is not None:
                    pairs.append((lr_path, hr_path))
        return pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict:
        lr_path, hr_path = self.pairs[index]
        hr = Image.open(hr_path).convert("RGB")
        lr = Image.open(lr_path).convert("RGB")
        condition = lr.resize(hr.size, Image.Resampling.BICUBIC)
        hr, condition = self._random_crop(hr, condition)
        return {
            "target": _img_to_tensor(hr),
            "condition": _img_to_tensor(condition),
        }

    def _random_crop(self, hr: Image.Image, cond: Image.Image) -> tuple[Image.Image, Image.Image]:
        w, h = hr.size
        s = self.resolution
        if w < s or h < s:
            scale = max(s / w, s / h)
            new_size = (max(s, int(w * scale)), max(s, int(h * scale)))
            hr = hr.resize(new_size, Image.Resampling.BICUBIC)
            cond = cond.resize(new_size, Image.Resampling.BICUBIC)
            w, h = hr.size
        x = random.randint(0, w - s)
        y = random.randint(0, h - s)
        box = (x, y, x + s, y + s)
        if random.random() < 0.5:
            hr = hr.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            cond = cond.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        return hr.crop(box), cond.crop(box)


def _img_to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Flow Matching SR on REDS.")
    p.add_argument("--data_root", type=Path, default=Path("../data"))
    p.add_argument("--output_dir", type=Path, default=Path("weights/fm_sr"))
    p.add_argument("--resolution", type=int, default=256)
    p.add_argument("--channels", type=int, nargs="+", default=[128, 256, 384, 512])
    p.add_argument("--blocks_per_level", type=int, default=2)
    p.add_argument("--attn_levels", type=int, nargs="+", default=[2, 3])
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--max_train_steps", type=int, default=100000)
    p.add_argument("--mixed_precision", choices=["no", "fp16", "bf16"], default="fp16")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--checkpoint_every", type=int, default=5000)
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--resume", type=Path, default=None)
    p.add_argument("--ema_decay", type=float, default=0.9999)
    return p.parse_args()


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _weight_dtype(mp: str | None) -> torch.dtype:
    if mp == "fp16":
        return torch.float16
    if mp == "bf16":
        return torch.bfloat16
    return torch.float32


@torch.no_grad()
def _ema_update(ema_model: torch.nn.Module, model: torch.nn.Module, decay: float) -> None:
    for ema_p, p in zip(ema_model.parameters(), model.parameters()):
        ema_p.data.mul_(decay).add_(p.data, alpha=1 - decay)


def main() -> None:
    args = parse_args()
    _set_seed(args.seed)

    from accelerate import Accelerator

    from .flow_matching import flow_matching_loss
    from .unet import FlowMatchingUNet

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=None if args.mixed_precision == "no" else args.mixed_precision,
    )

    model = FlowMatchingUNet(
        in_channels=6,
        out_channels=3,
        channels=tuple(args.channels),
        blocks_per_level=args.blocks_per_level,
        attn_levels=tuple(args.attn_levels),
    )

    ema_model = FlowMatchingUNet(
        in_channels=6,
        out_channels=3,
        channels=tuple(args.channels),
        blocks_per_level=args.blocks_per_level,
        attn_levels=tuple(args.attn_levels),
    )
    ema_model.load_state_dict(model.state_dict())
    ema_model.requires_grad_(False)

    global_step = 0
    if args.resume and args.resume.exists():
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt["model"])
        ema_model.load_state_dict(ckpt.get("ema_model", ckpt["model"]))
        global_step = ckpt.get("global_step", 0)
        if accelerator.is_main_process:
            print(f"resumed from {args.resume} at step {global_step}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)

    dataset = REDSFlowDataset(args.data_root, resolution=args.resolution)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    w_dtype = _weight_dtype(args.mixed_precision if args.mixed_precision != "no" else None)
    ema_model.to(accelerator.device)

    if accelerator.is_main_process:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"model params: {total_params / 1e6:.1f}M total, {trainable_params / 1e6:.1f}M trainable")
        print(f"dataset: {len(dataset)} pairs, batch={args.batch_size}, accum={args.gradient_accumulation_steps}")

    progress = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress.set_description("FM-SR")
    loss_accum = 0.0
    loss_count = 0

    while global_step < args.max_train_steps:
        for batch in dataloader:
            with accelerator.accumulate(model):
                target = batch["target"]
                condition = batch["condition"]
                loss = flow_matching_loss(model, target, condition)
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            loss_accum += loss.detach().item()
            loss_count += 1

            if accelerator.sync_gradients:
                global_step += 1
                progress.update(1)
                _ema_update(ema_model, accelerator.unwrap_model(model), args.ema_decay)

                if global_step % args.log_every == 0:
                    avg_loss = loss_accum / max(loss_count, 1)
                    progress.set_postfix(loss=f"{avg_loss:.4f}")
                    loss_accum = 0.0
                    loss_count = 0

                if accelerator.is_main_process and args.checkpoint_every > 0 and global_step % args.checkpoint_every == 0:
                    _save_checkpoint(
                        accelerator.unwrap_model(model), ema_model, global_step,
                        args.output_dir / f"checkpoint-{global_step}",
                    )
                if global_step >= args.max_train_steps:
                    break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        _save_checkpoint(accelerator.unwrap_model(model), ema_model, global_step, args.output_dir)
        print(f"saved final model: {args.output_dir}")


def _save_checkpoint(model: torch.nn.Module, ema_model: torch.nn.Module, step: int, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model": model.state_dict(), "ema_model": ema_model.state_dict(), "global_step": step},
        output_dir / "model.pt",
    )
    print(f"  checkpoint → {output_dir / 'model.pt'} (step {step})")


if __name__ == "__main__":
    main()
