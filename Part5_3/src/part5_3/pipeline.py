"""Stable Diffusion + ControlNet-Tile pipeline helpers."""

from __future__ import annotations

import os
from pathlib import Path

import torch
from PIL import Image

from .constants import DEFAULT_HF_ENDPOINT
from .tiling import crop_padding, pad_to_multiple, process_tiled


def load_pipeline(
    pretrained_model: str | Path,
    controlnet_model: str | Path,
    device: torch.device,
    dtype: torch.dtype,
    lora_path: str | Path | None = None,
    no_controlnet: bool = False,
    hf_endpoint: str = DEFAULT_HF_ENDPOINT,
    local_files_only: bool = False,
):
    if hf_endpoint:
        os.environ.setdefault("HF_ENDPOINT", hf_endpoint)
    if no_controlnet:
        from diffusers import StableDiffusionImg2ImgPipeline, UniPCMultistepScheduler

        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            str(pretrained_model),
            torch_dtype=dtype,
            safety_checker=None,
            requires_safety_checker=False,
            local_files_only=local_files_only,
        )
    else:
        from diffusers import ControlNetModel, StableDiffusionControlNetImg2ImgPipeline, UniPCMultistepScheduler

        controlnet = ControlNetModel.from_pretrained(
            str(controlnet_model),
            torch_dtype=dtype,
            local_files_only=local_files_only,
        )
        pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
            str(pretrained_model),
            controlnet=controlnet,
            torch_dtype=dtype,
            safety_checker=None,
            requires_safety_checker=False,
            local_files_only=local_files_only,
        )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    if lora_path:
        pipe.load_lora_weights(str(lora_path))
    pipe.to(device)
    _enable_memory_savers(pipe)
    return pipe


def enhance_image(
    pipe,
    image: Image.Image,
    prompt: str,
    negative_prompt: str,
    strength: float,
    control_scale: float,
    guidance_scale: float,
    steps: int,
    seed: int,
    device: torch.device,
    tile_size: int,
    tile_overlap: int,
    no_controlnet: bool = False,
) -> Image.Image:
    image = image.convert("RGB")
    if tile_size > 0 and max(image.size) > tile_size:
        return process_tiled(
            image,
            tile_size=tile_size,
            overlap=tile_overlap,
            process_tile=lambda tile, x, y: _enhance_single(
                pipe=pipe,
                image=tile,
                prompt=prompt,
                negative_prompt=negative_prompt,
                strength=strength,
                control_scale=control_scale,
                guidance_scale=guidance_scale,
                steps=steps,
                seed=_tile_seed(seed, x, y),
                device=device,
                no_controlnet=no_controlnet,
            ),
        )
    return _enhance_single(
        pipe=pipe,
        image=image,
        prompt=prompt,
        negative_prompt=negative_prompt,
        strength=strength,
        control_scale=control_scale,
        guidance_scale=guidance_scale,
        steps=steps,
        seed=seed,
        device=device,
        no_controlnet=no_controlnet,
    )


def _enhance_single(
    pipe,
    image: Image.Image,
    prompt: str,
    negative_prompt: str,
    strength: float,
    control_scale: float,
    guidance_scale: float,
    steps: int,
    seed: int,
    device: torch.device,
    no_controlnet: bool,
) -> Image.Image:
    padded, pad = pad_to_multiple(image, multiple=8)
    generator_device = "cuda" if device.type == "cuda" else "cpu"
    generator = torch.Generator(device=generator_device).manual_seed(int(seed))
    kwargs = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "image": padded,
        "strength": strength,
        "guidance_scale": guidance_scale,
        "num_inference_steps": steps,
        "generator": generator,
        "output_type": "pil",
    }
    if not no_controlnet:
        kwargs["control_image"] = padded
        kwargs["controlnet_conditioning_scale"] = control_scale
    with torch.inference_mode():
        result = pipe(**kwargs).images[0]
    return crop_padding(result, pad)


def _tile_seed(seed: int, x: int, y: int) -> int:
    return int(seed + x * 73856093 + y * 19349663) % (2**31 - 1)


def _enable_memory_savers(pipe) -> None:
    try:
        pipe.enable_vae_tiling()
    except Exception:
        pass
    try:
        pipe.enable_attention_slicing()
    except Exception:
        pass
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        pass
