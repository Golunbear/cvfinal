# AIAA 3201 Computer Vision Final Project

This project focuses on Video Super-Resolution (VSR): reconstructing high-resolution videos from low-resolution inputs while preserving both spatial detail and temporal consistency. The full project follows a three-stage experimental route from classical baselines to modern VSR reproduction and generative enhancement.

The runnable parts documented here are:

- `Part5_2/`: Part 2 SOTA reproduction with BasicVSR, IconVSR, and Real-ESRGAN.
- `Part5_3/`: Part 3 Exploration A, using Stable Diffusion + ControlNet-Tile + optional REDS LoRA + temporal blending.

Reserved sections for future code:

- Part 1 baseline: to be completed after the Part 1 code is added.
- Part 3 Exploration B: Conditional Flow Matching / Flow-Matching Generative VSR from the report, to be completed after the code is added.

## Recommended Workflow

1. Run `Part5_2/` first to generate restored frames from BasicVSR, IconVSR, and Real-ESRGAN.
2. Run `Part5_3/` next. It reads restored frames from `Part5_2/outputs` and applies the Stable Diffusion + ControlNet-Tile enhancement pipeline.
3. Run the evaluation and visualization scripts to generate metric CSV files and qualitative comparison images.

The commands below are written for Windows PowerShell. Each module assumes that you start from the repository root before entering the corresponding folder.

## Repository Layout

Recommended data layout:

```text
cvfinal/
  data/
    REDS/
      val/
        val_sharp/
        val_sharp_bicubic/
          X4/
      train/
        train_sharp/
        train_sharp_bicubic/
          X4/
    Sample_data/
      REDS-sample/
      vimeo-RL/
    wild-video/
      IMG_2175.mp4
  Part5_2/
  Part5_3/
```

Dataset notes:

- `reds_val`: paired REDS validation data for PSNR, SSIM, LPIPS, FID, and tLPIPS.
- `sample_reds`: LR-only REDS sample clips. Results and videos are saved, but paired metrics are skipped.
- `vimeo_lr`: LR-only Vimeo clips. Results and videos are saved, but paired metrics are skipped.
- `wild`: a self-captured video. The mp4 must be decoded into frames before inference.
- REDS train split is only needed for Part 3 LoRA training. It is not required for inference.

## Part 1 Placeholder

Part 1 baseline code will be documented here after it is added:

- detailed subdirectory README link
- environment setup
- data preparation
- baseline method commands
- evaluation commands

## Part 2: SOTA Reproduction

Part 2 lives in `Part5_2/` and includes three methods:

For more implementation details, see the subdirectory README: [Part5_2/README.md](Part5_2/README.md).

- `basicvsr`: bidirectional recurrent propagation with optical-flow alignment.
- `iconvsr`: BasicVSR-style recurrent propagation with keyframe information refill.
- `realesrgan`: frame-by-frame perceptual SR baseline.

### Environment

```powershell
conda create -n CVfinal_part52 python=3.12 -y
conda activate CVfinal_part52
Set-Location .\Part5_2
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install -r requirements.txt
```

`requirements.txt` uses the CUDA 13.0 PyTorch wheel index by default. If your CUDA version is different, install the matching `torch`, `torchvision`, and `torchaudio` first, then install the remaining dependencies.

### Download Weights

Download all official checkpoints:

```powershell
python -m part5_2.download_weights --weights_dir weights
```

Download only one model, for example IconVSR:

```powershell
python -m part5_2.download_weights --weights_dir weights --models iconvsr
```

Expected files:

```text
Part5_2/weights/basicvsr_reds4_20120409-0e599677.pth
Part5_2/weights/iconvsr_reds4_20210413-9e09d621.pth
Part5_2/weights/RealESRGAN_x4plus.pth
```

### Decode Wild Video

The `wild` dataset reads decoded frames rather than mp4 files directly:

```powershell
python -m part5_2.decode_wild_video --wild_root ..\data\wild-video
```

For example, `IMG_2175.mp4` will produce:

```text
data/wild-video/decoded_frames/IMG_2175/*.png
data/wild-video/decoded_frames/IMG_2175/metadata.json
```

### Smoke Test

Run a small test first:

```powershell
python -m part5_2.run_inference --model basicvsr --dataset sample_reds --data_root ..\data --out_root outputs --max_sequences 1 --max_frames 5 --device cuda
python -m part5_2.run_inference --model iconvsr --dataset sample_reds --data_root ..\data --out_root outputs --max_sequences 1 --max_frames 5 --device cuda
python -m part5_2.run_inference --model realesrgan --dataset sample_reds --data_root ..\data --out_root outputs --max_sequences 1 --max_frames 5 --tile 512 --device cuda
```

If no GPU is available, change `--device cuda` to `--device cpu` for a small smoke test. CPU inference will be slow.

### Inference

REDS validation:

```powershell
python -m part5_2.run_inference --model basicvsr --dataset reds_val --data_root ..\data --out_root outputs --device cuda
python -m part5_2.run_inference --model iconvsr --dataset reds_val --data_root ..\data --out_root outputs --device cuda
python -m part5_2.run_inference --model realesrgan --dataset reds_val --data_root ..\data --out_root outputs --tile 512 --device cuda
```

Mandatory sample datasets:

```powershell
python -m part5_2.run_inference --model basicvsr --dataset sample_reds --data_root ..\data --out_root outputs --device cuda
python -m part5_2.run_inference --model iconvsr --dataset sample_reds --data_root ..\data --out_root outputs --device cuda
python -m part5_2.run_inference --model realesrgan --dataset sample_reds --data_root ..\data --out_root outputs --tile 512 --device cuda

python -m part5_2.run_inference --model basicvsr --dataset vimeo_lr --data_root ..\data --out_root outputs --device cuda
python -m part5_2.run_inference --model iconvsr --dataset vimeo_lr --data_root ..\data --out_root outputs --device cuda
python -m part5_2.run_inference --model realesrgan --dataset vimeo_lr --data_root ..\data --out_root outputs --tile 512 --device cuda
```

Wild video:

```powershell
python -m part5_2.run_inference --model basicvsr --dataset wild --wild_frames ..\data\wild-video\decoded_frames\IMG_2175 --data_root ..\data --out_root outputs --device cuda
python -m part5_2.run_inference --model iconvsr --dataset wild --wild_frames ..\data\wild-video\decoded_frames\IMG_2175 --data_root ..\data --out_root outputs --device cuda
python -m part5_2.run_inference --model realesrgan --dataset wild --wild_frames ..\data\wild-video\decoded_frames\IMG_2175 --data_root ..\data --out_root outputs --tile 512 --device cuda
```

Outputs:

```text
Part5_2/outputs/<model>/<dataset>/<sequence>/frames/*.png
Part5_2/outputs/<model>/<dataset>/<sequence>.mp4
```

`reds_val` is mainly used for evaluation and saves restored frames. `sample_reds`, `vimeo_lr`, and `wild` also save mp4 videos.

### Evaluation

```powershell
python -m part5_2.evaluate --models basicvsr iconvsr realesrgan --dataset reds_val --data_root ..\data --pred_root outputs --device cuda
```

Metric outputs:

```text
Part5_2/outputs/metrics/frame_metrics.csv
Part5_2/outputs/metrics/summary_metrics.csv
```

Implemented metrics:

- PSNR
- SSIM
- LPIPS
- FID
- tLPIPS

### Visualization

```powershell
python -m part5_2.visualize --models basicvsr iconvsr realesrgan --dataset reds_val --data_root ..\data --pred_root outputs --out_dir outputs\visuals --frame_index 0
```

Example output:

```text
Part5_2/outputs/visuals/reds_val_000_0000.png
```

### Tests

```powershell
$env:PYTHONPATH = "src"
pytest -q
```

## Part 3 Exploration A: Stable Diffusion + ControlNet-Tile

Part 3 Exploration A lives in `Part5_3/`. Following the naming in the final report, Exploration A refers to the consistent enhancement pipeline with Stable Diffusion + ControlNet-Tile + REDS LoRA.

For more implementation details, see the subdirectory README: [Part5_3/README.md](Part5_3/README.md).

Default pipeline:

```text
LR video
  -> Part 2 VSR output, recommended input_model=iconvsr
  -> Stable Diffusion img2img
  -> ControlNet-Tile structural conditioning
  -> optional REDS LoRA
  -> optional temporal blending
  -> enhanced frames/video
```

Before running Part 3 Exploration A, run Part 2 inference first and make sure the input frames exist:

```text
Part5_2/outputs/iconvsr/<dataset>/<sequence>/frames/*.png
```

### Environment

```powershell
conda create -n CVfinal_part53 python=3.11 -y
conda activate CVfinal_part53
Set-Location .\Part5_3
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install -r requirements.txt
```

If Hugging Face cannot be reached directly, use the mirror endpoint:

```powershell
$env:HF_ENDPOINT = "https://hf-mirror.com"
```

### Download Models

```powershell
python -m part5_3.download_models --weights_dir weights
```

Expected directories:

```text
Part5_3/weights/stable-diffusion-v1-5/
Part5_3/weights/control_v11f1e_sd15_tile/
```

You may also download them manually with the Hugging Face CLI into the same directories.

### Smoke Test

```powershell
python -m part5_3.run_inference --dataset sample_reds --data_root ..\data --part2_root ..\Part5_2\outputs --input_model iconvsr --out_root outputs --run_name sd_tile --max_sequences 1 --max_frames 5 --device cuda
```

If GPU memory is limited, reduce tile size or frame count:

```powershell
python -m part5_3.run_inference --dataset sample_reds --data_root ..\data --part2_root ..\Part5_2\outputs --input_model iconvsr --out_root outputs --run_name sd_tile_small --tile_size 512 --max_sequences 1 --max_frames 3 --device cuda
```

### Inference

Stable Diffusion img2img only:

```powershell
python -m part5_3.run_inference --dataset reds_val --data_root ..\data --part2_root ..\Part5_2\outputs --input_model iconvsr --out_root outputs --run_name sd_img2img --no_controlnet --device cuda
```

Stable Diffusion + ControlNet-Tile:

```powershell
python -m part5_3.run_inference --dataset reds_val --data_root ..\data --part2_root ..\Part5_2\outputs --input_model iconvsr --out_root outputs --run_name sd_tile --device cuda
```

ControlNet-Tile + REDS LoRA:

```powershell
python -m part5_3.run_inference --dataset reds_val --data_root ..\data --part2_root ..\Part5_2\outputs --input_model iconvsr --out_root outputs --run_name sd_tile_lora --lora_path weights\reds_sd15_tile_lora --device cuda
```

ControlNet-Tile + REDS LoRA + temporal blending:

```powershell
python -m part5_3.run_inference --dataset reds_val --data_root ..\data --part2_root ..\Part5_2\outputs --input_model iconvsr --out_root outputs --run_name sd_tile_lora_temporal --lora_path weights\reds_sd15_tile_lora --temporal_blend 0.2 --device cuda
```

Mandatory sample datasets:

```powershell
python -m part5_3.run_inference --dataset sample_reds --data_root ..\data --part2_root ..\Part5_2\outputs --input_model iconvsr --out_root outputs --run_name sd_tile --device cuda
python -m part5_3.run_inference --dataset vimeo_lr --data_root ..\data --part2_root ..\Part5_2\outputs --input_model iconvsr --out_root outputs --run_name sd_tile --device cuda
python -m part5_3.run_inference --dataset wild --wild_frames ..\data\wild-video\decoded_frames\IMG_2175 --data_root ..\data --part2_root ..\Part5_2\outputs --input_model iconvsr --out_root outputs --run_name sd_tile --device cuda
```

Useful controls:

```text
--strength 0.25
--control_scale 0.8
--guidance_scale 5.0
--steps 25
--tile_size 768
--tile_overlap 96
--seed 42
```

Outputs:

```text
Part5_3/outputs/<run_name>/<input_model>/<dataset>/<sequence>/frames/*.png
Part5_3/outputs/<run_name>/<input_model>/<dataset>/<sequence>.mp4
```

`reds_val` is mainly used for evaluation and saves frames. `sample_reds`, `vimeo_lr`, and `wild` also save mp4 videos.

### Train REDS LoRA

LoRA training uses REDS train HR frames as targets and bicubic LR frames resized to HR as ControlNet-Tile conditions.

```powershell
accelerate config
accelerate launch -m part5_3.train_lora_reds --data_root ..\data --pretrained_model weights\stable-diffusion-v1-5 --controlnet_model weights\control_v11f1e_sd15_tile --output_dir weights\reds_sd15_tile_lora --resolution 512 --rank 16 --learning_rate 1e-4 --batch_size 1 --gradient_accumulation_steps 4 --max_train_steps 10000 --mixed_precision fp16
```

LoRA smoke test:

```powershell
accelerate launch -m part5_3.train_lora_reds --data_root ..\data --pretrained_model weights\stable-diffusion-v1-5 --controlnet_model weights\control_v11f1e_sd15_tile --output_dir weights\lora_smoke --max_train_steps 10 --resolution 512
```

### Evaluation

Only include generated run names in `--methods`.

```powershell
python -m part5_3.evaluate --methods sd_img2img sd_tile sd_tile_lora sd_tile_lora_temporal --dataset reds_val --data_root ..\data --part2_root ..\Part5_2\outputs --input_model iconvsr --pred_root outputs --device cuda
```

Fast evaluation:

```powershell
python -m part5_3.evaluate --methods sd_tile --dataset reds_val --max_sequences 1 --max_frames 2 --data_root ..\data --part2_root ..\Part5_2\outputs --input_model iconvsr --pred_root outputs --device cuda --skip_fid
```

Metric outputs:

```text
Part5_3/outputs/metrics/frame_metrics.csv
Part5_3/outputs/metrics/summary_metrics.csv
```

### Visualization

```powershell
python -m part5_3.visualize --methods sd_tile sd_tile_lora sd_tile_lora_temporal --dataset reds_val --data_root ..\data --part2_root ..\Part5_2\outputs --input_model iconvsr --pred_root outputs --out_dir outputs\visuals --frame_index 0 --crop 300 200 256 256
```

### Tests

```powershell
python -m compileall src tests
$env:PYTHONPATH = "src"
pytest -q
```

## Part 3 Exploration B Placeholder

Part 3 Exploration B corresponds to the Conditional Flow Matching / Flow-Matching Generative VSR method in the final report. The current repository does not include this code yet. This section should be completed after the code is added:

- detailed subdirectory README link
- environment setup
- model weights / checkpoints
- training commands
- inference commands
- evaluation commands

## Notes

- `Part5_3` uses `iconvsr` as the default input model because the final report identifies IconVSR as the strongest Part 2 recurrent VSR input.
- If Part 2 outputs were generated only for `basicvsr` or `realesrgan`, change `--input_model iconvsr` in Part 3 commands accordingly.
- Wild video x4 outputs can become 8K. Some players may fail to play the generated mp4 files. In that case, check `frames/*.png` first or transcode the mp4 to H.264 with `ffmpeg`.
- More implementation details are available in `Part5_2/README.md` and `Part5_3/README.md`.
