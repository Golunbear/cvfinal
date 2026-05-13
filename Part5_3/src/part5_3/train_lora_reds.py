"""Train a REDS-domain LoRA for SD1.5 + ControlNet-Tile refinement."""

from __future__ import annotations

import argparse
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from .constants import DEFAULT_HF_ENDPOINT, DEFAULT_PROMPT
from .image_io import list_image_files


class REDSLoRADataset(Dataset):
    def __init__(
        self,
        data_root: str | Path,
        tokenizer,
        prompt: str,
        resolution: int = 512,
    ) -> None:
        self.data_root = Path(data_root)
        self.hr_root = self.data_root / "REDS" / "train" / "train_sharp"
        self.lr_root = self.data_root / "REDS" / "train" / "train_sharp_bicubic" / "X4"
        self.resolution = resolution
        self.tokenizer = tokenizer
        self.prompt = prompt
        self.pairs = self._collect_pairs()
        if not self.pairs:
            raise FileNotFoundError(f"No REDS train pairs found under {self.hr_root} and {self.lr_root}")

    def _collect_pairs(self) -> list[tuple[Path, Path]]:
        pairs = []
        if not self.hr_root.exists() or not self.lr_root.exists():
            return pairs
        for hr_seq in sorted([p for p in self.hr_root.iterdir() if p.is_dir()]):
            lr_seq = self.lr_root / hr_seq.name
            if not lr_seq.exists():
                continue
            lr_by_name = {p.name: p for p in list_image_files(lr_seq)}
            for hr_path in list_image_files(hr_seq):
                lr_path = lr_by_name.get(hr_path.name)
                if lr_path is not None:
                    pairs.append((lr_path, hr_path))
        return pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict:
        lr_path, hr_path = self.pairs[index]
        hr = Image.open(hr_path).convert("RGB")
        lr = Image.open(lr_path).convert("RGB")
        control = lr.resize(hr.size, Image.Resampling.BICUBIC)
        hr, control = self._random_pair_crop(hr, control)
        input_ids = self.tokenizer(
            self.prompt,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids[0]
        return {
            "pixel_values": _image_to_target_tensor(hr),
            "conditioning_pixel_values": _image_to_condition_tensor(control),
            "input_ids": input_ids,
        }

    def _random_pair_crop(self, hr: Image.Image, control: Image.Image) -> tuple[Image.Image, Image.Image]:
        width, height = hr.size
        if width < self.resolution or height < self.resolution:
            scale = max(self.resolution / width, self.resolution / height)
            new_size = (math.ceil(width * scale), math.ceil(height * scale))
            hr = hr.resize(new_size, Image.Resampling.BICUBIC)
            control = control.resize(new_size, Image.Resampling.BICUBIC)
            width, height = hr.size
        x = random.randint(0, width - self.resolution)
        y = random.randint(0, height - self.resolution)
        box = (x, y, x + self.resolution, y + self.resolution)
        if random.random() < 0.5:
            hr = hr.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            control = control.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        return hr.crop(box), control.crop(box)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a REDS LoRA for SD1.5 + ControlNet-Tile.")
    parser.add_argument("--data_root", type=Path, default=Path("../data"))
    parser.add_argument("--pretrained_model", default="weights/stable-diffusion-v1-5")
    parser.add_argument("--controlnet_model", default="weights/control_v11f1e_sd15_tile")
    parser.add_argument("--output_dir", type=Path, default=Path("weights/reds_sd15_tile_lora"))
    parser.add_argument("--hf_endpoint", default=DEFAULT_HF_ENDPOINT)
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--max_train_steps", type=int, default=10000)
    parser.add_argument("--mixed_precision", choices=["no", "fp16", "bf16"], default="fp16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--checkpoint_every", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.hf_endpoint:
        os.environ.setdefault("HF_ENDPOINT", args.hf_endpoint)

    from accelerate import Accelerator
    from diffusers import AutoencoderKL, ControlNetModel, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
    from diffusers.utils import convert_state_dict_to_diffusers
    from peft import LoraConfig, get_peft_model_state_dict
    from transformers import CLIPTextModel, CLIPTokenizer

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=None if args.mixed_precision == "no" else args.mixed_precision,
    )
    _set_seed(args.seed)

    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model,
        subfolder="tokenizer",
        local_files_only=args.local_files_only,
    )
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model,
        subfolder="text_encoder",
        local_files_only=args.local_files_only,
    )
    vae = AutoencoderKL.from_pretrained(args.pretrained_model, subfolder="vae", local_files_only=args.local_files_only)
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model, subfolder="unet", local_files_only=args.local_files_only)
    controlnet = ControlNetModel.from_pretrained(args.controlnet_model, local_files_only=args.local_files_only)
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model,
        subfolder="scheduler",
        local_files_only=args.local_files_only,
    )

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    controlnet.requires_grad_(False)
    unet.requires_grad_(False)
    unet.add_adapter(
        LoraConfig(
            r=args.rank,
            lora_alpha=args.rank,
            init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        )
    )
    trainable_params = [p for p in unet.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate)

    dataset = REDSLoRADataset(args.data_root, tokenizer=tokenizer, prompt=args.prompt, resolution=args.resolution)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    unet, optimizer, dataloader = accelerator.prepare(unet, optimizer, dataloader)
    weight_dtype = _weight_dtype(accelerator.mixed_precision)
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    controlnet.to(accelerator.device, dtype=weight_dtype)

    progress = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress.set_description("REDS LoRA")
    global_step = 0
    while global_step < args.max_train_steps:
        for batch in dataloader:
            with accelerator.accumulate(unet):
                pixel_values = batch["pixel_values"].to(accelerator.device, dtype=weight_dtype)
                cond_values = batch["conditioning_pixel_values"].to(accelerator.device, dtype=weight_dtype)
                input_ids = batch["input_ids"].to(accelerator.device)

                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (latents.shape[0],),
                    device=latents.device,
                    dtype=torch.long,
                )
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                encoder_hidden_states = text_encoder(input_ids)[0]
                down_res, mid_res = controlnet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    controlnet_cond=cond_values,
                    return_dict=False,
                )
                model_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    down_block_additional_residuals=[sample.to(dtype=weight_dtype) for sample in down_res],
                    mid_block_additional_residual=mid_res.to(dtype=weight_dtype),
                    return_dict=False,
                )[0]
                target = noise
                if noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                loss = torch.nn.functional.mse_loss(model_pred.float(), target.float(), reduction="mean")
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                global_step += 1
                progress.update(1)
                progress.set_postfix(loss=f"{loss.detach().item():.4f}")
                if accelerator.is_main_process and args.checkpoint_every > 0 and global_step % args.checkpoint_every == 0:
                    _save_lora(accelerator.unwrap_model(unet), args.output_dir / f"checkpoint-{global_step}", StableDiffusionPipeline, convert_state_dict_to_diffusers, get_peft_model_state_dict)
                if global_step >= args.max_train_steps:
                    break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        _save_lora(accelerator.unwrap_model(unet), args.output_dir, StableDiffusionPipeline, convert_state_dict_to_diffusers, get_peft_model_state_dict)
        print(f"saved LoRA: {args.output_dir}")


def _image_to_target_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1) * 2.0 - 1.0


def _image_to_condition_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)


def _weight_dtype(mixed_precision: str | None) -> torch.dtype:
    if mixed_precision == "fp16":
        return torch.float16
    if mixed_precision == "bf16":
        return torch.bfloat16
    return torch.float32


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _save_lora(unet, output_dir: Path, pipeline_cls, convert_fn, peft_state_fn) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    state_dict = convert_fn(peft_state_fn(unet))
    pipeline_cls.save_lora_weights(save_directory=output_dir, unet_lora_layers=state_dict)


if __name__ == "__main__":
    main()
