# Part 5.3 - Stable Diffusion + ControlNet-Tile + REDS LoRA

This folder implements Direction B, "Consistent Enhancement". It uses Part 2 VSR outputs as structure-preserving inputs, then refines them with Stable Diffusion v1.5 and ControlNet-Tile. A REDS-domain LoRA can be trained to make generated details better match video super-resolution data.

The default pipeline is:

```text
LR video
  -> Part 2 VSR output, recommended iconvsr
  -> Stable Diffusion img2img
  -> ControlNet-Tile structural conditioning
  -> optional REDS LoRA
  -> optional optical-flow temporal blending
  -> Part 3 enhanced frames/video
```

## Environment

Run commands from `Part5_3/`.

```bash
conda create -n CVfinal_part3 python=3.11 -y
conda activate CVfinal_part3
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install -r requirements.txt
```

The server cannot connect to Hugging Face directly, so use the mirror endpoint:

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

Download models with either the bundled helper:

```bash
python -m part5_3.download_models --weights_dir weights
```

or with the Hugging Face CLI:

```bash
huggingface-cli download runwayml/stable-diffusion-v1-5 --local-dir weights/stable-diffusion-v1-5
huggingface-cli download lllyasviel/control_v11f1e_sd15_tile --local-dir weights/control_v11f1e_sd15_tile
```

Expected folders:

```text
weights/stable-diffusion-v1-5/
weights/control_v11f1e_sd15_tile/
```

## Inputs And Outputs

Part 3 reads Part 2 restored frames, not raw LR frames:

```text
../Part5_2/outputs/<input_model>/<dataset>/<sequence>/frames/*.png
```

Outputs are written to:

```text
outputs/<run_name>/<input_model>/<dataset>/<sequence>/frames/*.png
outputs/<run_name>/<input_model>/<dataset>/<sequence>.mp4
```

`reds_val` saves frames for evaluation. `sample_reds`, `vimeo_lr`, and `wild` also save mp4 videos.

## Train REDS LoRA

The LoRA script uses REDS train HR frames as targets and bicubic LR frames resized to HR as ControlNet-Tile conditions.

```bash
accelerate config
accelerate launch -m part5_3.train_lora_reds \
  --data_root ../data \
  --pretrained_model weights/stable-diffusion-v1-5 \
  --controlnet_model weights/control_v11f1e_sd15_tile \
  --output_dir weights/reds_sd15_tile_lora \
  --resolution 512 \
  --rank 16 \
  --learning_rate 1e-4 \
  --batch_size 1 \
  --gradient_accumulation_steps 4 \
  --max_train_steps 10000 \
  --mixed_precision fp16
```

Smoke test:

```bash
accelerate launch -m part5_3.train_lora_reds \
  --data_root ../data \
  --pretrained_model weights/stable-diffusion-v1-5 \
  --controlnet_model weights/control_v11f1e_sd15_tile \
  --output_dir weights/lora_smoke \
  --max_train_steps 10 \
  --resolution 512
```

## Inference

SD img2img only ablation:

```bash
python -m part5_3.run_inference \
  --dataset reds_val \
  --data_root ../data \
  --part2_root ../Part5_2/outputs \
  --input_model iconvsr \
  --out_root outputs \
  --run_name sd_img2img \
  --no_controlnet \
  --device cuda
```

Stable Diffusion + ControlNet-Tile:

```bash
python -m part5_3.run_inference \
  --dataset reds_val \
  --data_root ../data \
  --part2_root ../Part5_2/outputs \
  --input_model iconvsr \
  --out_root outputs \
  --run_name sd_tile \
  --device cuda
```

ControlNet-Tile + REDS LoRA:

```bash
python -m part5_3.run_inference \
  --dataset reds_val \
  --data_root ../data \
  --part2_root ../Part5_2/outputs \
  --input_model iconvsr \
  --out_root outputs \
  --run_name sd_tile_lora \
  --lora_path weights/reds_sd15_tile_lora \
  --device cuda
```

ControlNet-Tile + REDS LoRA + temporal blending:

```bash
python -m part5_3.run_inference \
  --dataset reds_val \
  --data_root ../data \
  --part2_root ../Part5_2/outputs \
  --input_model iconvsr \
  --out_root outputs \
  --run_name sd_tile_lora_temporal \
  --lora_path weights/reds_sd15_tile_lora \
  --temporal_blend 0.2 \
  --device cuda
```

Mandatory datasets:

```bash
python -m part5_3.run_inference --dataset sample_reds --data_root ../data --part2_root ../Part5_2/outputs --input_model iconvsr --out_root outputs --run_name sd_tile_lora --lora_path weights/reds_sd15_tile_lora --device cuda

python -m part5_3.run_inference --dataset vimeo_lr --data_root ../data --part2_root ../Part5_2/outputs --input_model iconvsr --out_root outputs --run_name sd_tile_lora --lora_path weights/reds_sd15_tile_lora --device cuda

python -m part5_3.run_inference --dataset wild --wild_frames ../data/wild-video/decoded_frames/IMG_2175 --data_root ../data --part2_root ../Part5_2/outputs --input_model iconvsr --out_root outputs --run_name sd_tile_lora --lora_path weights/reds_sd15_tile_lora --device cuda
```

Useful controls:

```text
--strength 0.35
--control_scale 0.8
--guidance_scale 5.0
--steps 25
--tile_size 768
--tile_overlap 96
--seed 42
```

If memory is tight, use `--tile_size 512`. If details drift too much, lower `--strength` to `0.25` or increase `--control_scale` to `1.0`.

## Evaluation

Evaluate REDS val:

```bash
python -m part5_3.evaluate \
  --methods sd_img2img sd_tile sd_tile_lora sd_tile_lora_temporal \
  --dataset reds_val \
  --data_root ../data \
  --part2_root ../Part5_2/outputs \
  --input_model iconvsr \
  --pred_root outputs \
  --device cuda
```

Fast smoke evaluation:

```bash
python -m part5_3.evaluate \
  --methods sd_tile \
  --dataset reds_val \
  --max_sequences 1 \
  --max_frames 2 \
  --data_root ../data \
  --part2_root ../Part5_2/outputs \
  --input_model iconvsr \
  --pred_root outputs_smoke \
  --device cuda \
  --skip_fid
```

CSV outputs:

```text
outputs/metrics/frame_metrics.csv
outputs/metrics/summary_metrics.csv
```

Per-sequence FID is skipped by default; aggregate FID is reported in the `sequence=__all__` row. Add `--sequence_fid` if per-sequence FID is needed.

## Visuals

```bash
python -m part5_3.visualize \
  --methods sd_tile sd_tile_lora sd_tile_lora_temporal \
  --dataset reds_val \
  --data_root ../data \
  --part2_root ../Part5_2/outputs \
  --input_model iconvsr \
  --pred_root outputs \
  --out_dir outputs/visuals \
  --frame_index 0 \
  --crop 300 200 256 256
```

## Recommended Experiments

Use one Part 2 baseline as input, preferably the best REDS val performer from Part 2, such as IconVSR.

Quantitative table:

```text
Method | PSNR | SSIM | LPIPS | FID | tLPIPS
IconVSR
IconVSR + SD img2img
IconVSR + SD + ControlNet-Tile
IconVSR + SD + ControlNet-Tile + REDS LoRA
IconVSR + SD + ControlNet-Tile + REDS LoRA + temporal blending
```

Parameter sweep:

```text
strength: 0.25, 0.35, 0.45
control_scale: 0.6, 0.8, 1.0
fixed: guidance_scale=5.0, steps=25, seed=42
```

## Tests

```bash
python -m compileall src tests
PYTHONPATH=src pytest -q
```

## References

Model APIs and training conventions follow the projects below.

## Stable Diffusion / Latent Diffusion

- Paper: Robin Rombach et al., "High-Resolution Image Synthesis with Latent Diffusion Models", CVPR 2022.
- Repository: https://github.com/CompVis/stable-diffusion
- Default model: `runwayml/stable-diffusion-v1-5`

## ControlNet / ControlNet-Tile

- Paper: Lvmin Zhang et al., "Adding Conditional Control to Text-to-Image Diffusion Models", ICCV 2023.
- Repository: https://github.com/lllyasviel/ControlNet
- Default ControlNet-Tile model: `lllyasviel/control_v11f1e_sd15_tile`

## LoRA

- Paper: Edward J. Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models", ICLR 2022.
- Repository: https://github.com/microsoft/LoRA

## Diffusers

- Repository: https://github.com/huggingface/diffusers
- Used for Stable Diffusion img2img, ControlNet img2img, and LoRA adapter training utilities.


