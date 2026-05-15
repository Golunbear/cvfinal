# AIAA 3201 Computer Vision Final Project: Video Super-Resolution

![VSR overview](Figures/vsr.png)

This project focuses on Video Super-Resolution (VSR): reconstructing high-resolution videos from low-resolution inputs while preserving both spatial detail and temporal consistency. The full project follows a three-stage experimental route from classical baselines to modern VSR reproduction and generative enhancement.

The runnable parts are:

- `Part5_2/`: Part 2 SOTA reproduction with BasicVSR, IconVSR, and Real-ESRGAN.
- `Part5_3/`: Part 3 Exploration A, using Stable Diffusion + ControlNet-Tile + optional REDS LoRA + temporal blending.
- `Part5_3B/`: Part 3 Exploration B, using Conditional Flow Matching / Flow-Matching Generative VSR.

## Recommended Workflow

1. Run `Part5_2/` first to generate restored frames from BasicVSR, IconVSR, and Real-ESRGAN.
2. Run `Part5_3/` for Exploration A. It reads restored frames from `Part5_2/outputs` and applies the Stable Diffusion + ControlNet-Tile enhancement pipeline.
3. Run `Part5_3B/` for Exploration B. It also reads restored frames from `Part5_2/outputs`, then refines them with a task-specific Flow Matching U-Net.
4. Run the evaluation and visualization scripts to generate metric CSV files and qualitative comparison images.

The commands below are written for Linux/bash.

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
      xxx.mp4
  Part5_2/
  Part5_3/
  Part5_3B/
```

## Part 1



## Part 2: SOTA Reproduction

Part 2 lives in `Part5_2/` and includes three methods:

- `basicvsr`: bidirectional recurrent propagation with optical-flow alignment.
- `iconvsr`: BasicVSR-style recurrent propagation with keyframe information refill.
- `realesrgan`: frame-by-frame perceptual SR baseline.

For more implementation details, see the subdirectory README: [Part5_2/README.md](Part5_2/README.md).

### Environment

```bash
conda create -n CVfinal_part52 python=3.12 -y
conda activate CVfinal_part52
cd Part5_2
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install -r requirements.txt
```

`requirements.txt` uses the CUDA 13.0 PyTorch wheel index by default. If your CUDA version is different, install the matching `torch`, `torchvision`, and `torchaudio` first, then install the remaining dependencies.

### Download Weights

Download all official checkpoints:

```bash
python -m part5_2.download_weights --weights_dir weights
```

Download only one model, for example IconVSR:

```bash
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

```bash
python -m part5_2.decode_wild_video --wild_root ../data/wild-video
```

For example, `IMG_2175.mp4` will produce:

```text
data/wild-video/decoded_frames/IMG_2175/00000000.png
data/wild-video/decoded_frames/IMG_2175/00000001.png
...
data/wild-video/decoded_frames/IMG_2175/metadata.json
```

### Smoke Test

Run a small test first:

```bash
python -m part5_2.run_inference --model basicvsr --dataset sample_reds --data_root ../data --out_root outputs --max_sequences 1 --max_frames 5 --device cuda
python -m part5_2.run_inference --model iconvsr --dataset sample_reds --data_root ../data --out_root outputs --max_sequences 1 --max_frames 5 --device cuda
python -m part5_2.run_inference --model realesrgan --dataset sample_reds --data_root ../data --out_root outputs --max_sequences 1 --max_frames 5 --tile 512 --device cuda
```

### Inference

Full REDS_val:

```bash
python -m part5_2.run_inference --model basicvsr --dataset reds_val --data_root ../data --out_root outputs --device cuda
python -m part5_2.run_inference --model iconvsr --dataset reds_val --data_root ../data --out_root outputs --device cuda
python -m part5_2.run_inference --model realesrgan --dataset reds_val --data_root ../data --out_root outputs --tile 512 --device cuda
```

Mandatory sample datasets:

```bash
python -m part5_2.run_inference --model basicvsr --dataset sample_reds --data_root ../data --out_root outputs --device cuda
python -m part5_2.run_inference --model iconvsr --dataset sample_reds --data_root ../data --out_root outputs --device cuda
python -m part5_2.run_inference --model realesrgan --dataset sample_reds --data_root ../data --out_root outputs --tile 512 --device cuda

python -m part5_2.run_inference --model basicvsr --dataset vimeo_lr --data_root ../data --out_root outputs --device cuda
python -m part5_2.run_inference --model iconvsr --dataset vimeo_lr --data_root ../data --out_root outputs --device cuda
python -m part5_2.run_inference --model realesrgan --dataset vimeo_lr --data_root ../data --out_root outputs --tile 512 --device cuda
```

Wild video:

```bash
python -m part5_2.run_inference --model basicvsr --dataset wild --wild_frames ../data/wild-video/decoded_frames/IMG_2175 --data_root ../data --out_root outputs --device cuda
python -m part5_2.run_inference --model iconvsr --dataset wild --wild_frames ../data/wild-video/decoded_frames/IMG_2175 --data_root ../data --out_root outputs --device cuda
python -m part5_2.run_inference --model realesrgan --dataset wild --wild_frames ../data/wild-video/decoded_frames/IMG_2175 --data_root ../data --out_root outputs --tile 512 --device cuda
```

Outputs:

```text
Part5_2/outputs/<model>/<dataset>/<sequence>/frames/*.png
Part5_2/outputs/<model>/<dataset>/<sequence>.mp4
```

`reds_val` is mainly used for evaluation and saves restored frames. `sample_reds`, `vimeo_lr`, and `wild` also save mp4 videos.

### Wild Video Playback Compatibility

The current video writer uses OpenCV `mp4v` encoding. This is usually fine for the smaller sample outputs, but wild videos can become very large after x4 super-resolution. For example, a 1920x1080 wild video becomes 7680x4320 after x4 upscaling. Some players, including Windows Media Player, Movies & TV, and Bilibili player, may fail to play an `mp4v` 8K mp4. More tolerant players may still open it, such as Quark player.

The full-resolution restored frames are still valid:

```text
outputs/<model>/wild/<sequence>/frames/*.png
```

Only the generated mp4 may need transcoding for broader playback compatibility. For better compatibility, convert the wild mp4 to a 4K H.264 file:

```bash
ffmpeg -y -i outputs/basicvsr/wild/IMG_2175.mp4 -vf "scale=3840:2160" -c:v libx264 -pix_fmt yuv420p -crf 18 -preset slow outputs/basicvsr/wild/IMG_2175_h264_4k.mp4
ffmpeg -y -i outputs/iconvsr/wild/IMG_2175.mp4 -vf "scale=3840:2160" -c:v libx264 -pix_fmt yuv420p -crf 18 -preset slow outputs/iconvsr/wild/IMG_2175_h264_4k.mp4
ffmpeg -y -i outputs/realesrgan/wild/IMG_2175.mp4 -vf "scale=3840:2160" -c:v libx264 -pix_fmt yuv420p -crf 18 -preset slow outputs/realesrgan/wild/IMG_2175_h264_4k.mp4
```

### Evaluation

```bash
python -m part5_2.evaluate --models basicvsr iconvsr realesrgan --dataset reds_val --data_root ../data --pred_root outputs --device cuda
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

```bash
python -m part5_2.visualize --models basicvsr iconvsr realesrgan --dataset reds_val --data_root ../data --pred_root outputs --out_dir outputs/visuals --frame_index 0
```

Example output:

```text
Part5_2/outputs/visuals/reds_val_000_0000.png
```

### Tests

```bash
PYTHONPATH=src pytest -q
```

### Visual Results

The following visualizations show, from left to right: LR input, GT, BasicVSR, IconVSR, and Real-ESRGAN.

![Part 2 visual result 000](Figures/part2_visual/reds_val_000_0000.png)

![Part 2 visual result 001](Figures/part2_visual/reds_val_001_0000.png)

![Part 2 visual result 002](Figures/part2_visual/reds_val_002_0000.png)

![Part 2 visual result 003](Figures/part2_visual/reds_val_003_0000.png)


## Part 3 Exploration A: Stable Diffusion + ControlNet-Tile

Part 3 Exploration A lives in `Part5_3/`. It refers to the consistent enhancement pipeline with Stable Diffusion + ControlNet-Tile + REDS LoRA.

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

```bash
conda create -n CVfinal_part53 python=3.11 -y
conda activate CVfinal_part53
cd Part5_3
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install -r requirements.txt
```

If Hugging Face cannot be reached directly, use the mirror endpoint:

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

### Download Models

```bash
python -m part5_3.download_models --weights_dir weights
```

Expected directories:

```text
Part5_3/weights/stable-diffusion-v1-5/
Part5_3/weights/control_v11f1e_sd15_tile/
```

### Train REDS LoRA

LoRA training uses REDS train HR frames as targets and bicubic LR frames resized to HR as ControlNet-Tile conditions.

```bash
accelerate config
accelerate launch -m part5_3.train_lora_reds --data_root ../data --pretrained_model weights/stable-diffusion-v1-5 --controlnet_model weights/control_v11f1e_sd15_tile --output_dir weights/reds_sd15_tile_lora --resolution 512 --rank 16 --learning_rate 1e-4 --batch_size 1 --gradient_accumulation_steps 4 --max_train_steps 10000 --mixed_precision fp16
```

LoRA smoke test:

```bash
accelerate launch -m part5_3.train_lora_reds --data_root ../data --pretrained_model weights/stable-diffusion-v1-5 --controlnet_model weights/control_v11f1e_sd15_tile --output_dir weights/lora_smoke --max_train_steps 10 --resolution 512
```

### Inference

Smoke Test

```bash
python -m part5_3.run_inference --dataset sample_reds --data_root ../data --part2_root ../Part5_2/outputs --input_model iconvsr --out_root outputs --run_name sd_tile --max_sequences 1 --max_frames 5 --device cuda
```

Stable Diffusion img2img only:

```bash
python -m part5_3.run_inference --dataset reds_val --data_root ../data --part2_root ../Part5_2/outputs --input_model iconvsr --out_root outputs --run_name sd_img2img --no_controlnet --device cuda
```

Stable Diffusion + ControlNet-Tile:

```bash
python -m part5_3.run_inference --dataset reds_val --data_root ../data --part2_root ../Part5_2/outputs --input_model iconvsr --out_root outputs --run_name sd_tile --device cuda
```

SD + ControlNet-Tile + REDS LoRA:

```bash
python -m part5_3.run_inference --dataset reds_val --data_root ../data --part2_root ../Part5_2/outputs --input_model iconvsr --out_root outputs --run_name sd_tile_lora --lora_path weights/reds_sd15_tile_lora --device cuda
```

SD + ControlNet-Tile + REDS LoRA + temporal blending:

```bash
python -m part5_3.run_inference --dataset reds_val --data_root ../data --part2_root ../Part5_2/outputs --input_model iconvsr --out_root outputs --run_name sd_tile_lora_temporal --lora_path weights/reds_sd15_tile_lora --temporal_blend 0.2 --device cuda
```

Mandatory sample datasets:

```bash
python -m part5_3.run_inference --dataset sample_reds --data_root ../data --part2_root ../Part5_2/outputs --input_model iconvsr --out_root outputs --run_name sd_tile --device cuda
python -m part5_3.run_inference --dataset vimeo_lr --data_root ../data --part2_root ../Part5_2/outputs --input_model iconvsr --out_root outputs --run_name sd_tile --device cuda
python -m part5_3.run_inference --dataset wild --wild_frames ../data/wild-video/decoded_frames/IMG_2175 --data_root ../data --part2_root ../Part5_2/outputs --input_model iconvsr --out_root outputs --run_name sd_tile --device cuda
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

### Evaluation

Only include generated run names in `--methods`.

```bash
python -m part5_3.evaluate --methods sd_img2img sd_tile sd_tile_lora sd_tile_lora_temporal --dataset reds_val --data_root ../data --part2_root ../Part5_2/outputs --input_model iconvsr --pred_root outputs --device cuda
```

Fast evaluation:

```bash
python -m part5_3.evaluate --methods sd_tile --dataset reds_val --max_sequences 1 --max_frames 2 --data_root ../data --part2_root ../Part5_2/outputs --input_model iconvsr --pred_root outputs --device cuda --skip_fid
```

Metric outputs:

```text
Part5_3/outputs/metrics/frame_metrics.csv
Part5_3/outputs/metrics/summary_metrics.csv
```

### Visualization

```bash
python -m part5_3.visualize --methods sd_tile sd_tile_lora sd_tile_lora_temporal --dataset reds_val --data_root ../data --part2_root ../Part5_2/outputs --input_model iconvsr --pred_root outputs --out_dir outputs/visuals --frame_index 0 --crop 300 200 256 256
```

### Tests

```bash
python -m compileall src tests
PYTHONPATH=src pytest -q
```

### Visual Results

The following visualizations show, from left to right: LR input, IconVSR, GT, SD img2img, SD + ControlNet-Tile, and SD + ControlNet-Tile + LoRA.

![Part 3A visual result 000](Figures/part3_visual/reds_val_000_0000.png)

![Part 3A visual result 001](Figures/part3_visual/reds_val_001_0000.png)

![Part 3A visual result 002](Figures/part3_visual/reds_val_002_0000.png)

![Part 3A visual result 003](Figures/part3_visual/reds_val_003_0000.png)

## Part 3 Exploration B: Flow Matching Generative VSR

Part 3 Exploration B lives in `Part5_3B/`. It implements the Conditional Flow Matching / Flow-Matching Generative VSR method from the final report. Instead of using a pre-trained text-to-image diffusion prior, this module trains a task-specific U-Net to predict a residual velocity field that refines Part 2 VSR outputs.

For more implementation details, see the subdirectory README: [Part5_3B/README.md](Part5_3B/README.md).

Default pipeline:

```text
LR video
  -> Part 2 VSR output, recommended input_model=iconvsr
  -> Flow Matching U-Net velocity prediction
  -> ODE solver, usually Euler with 4 steps
  -> refined HR frames/video
```

Before running Part 3 Exploration B, run Part 2 inference first and make sure the input frames exist:

```text
Part5_2/outputs/iconvsr/<dataset>/<sequence>/frames/*.png
```

### Environment

```bash
conda create -n CVfinal_part53b python=3.11 -y
conda activate CVfinal_part53b
cd Part5_3B
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install -r requirements.txt
```

### Training

Train the Flow Matching model on REDS train:

```bash
python -m part5_3_da.train --data_root ../data --output_dir weights/fm_sr --resolution 256 --batch_size 4 --max_train_steps 10000 --checkpoint_every 2000 --mixed_precision fp16
```

Resume from a checkpoint:

```bash
python -m part5_3_da.train --data_root ../data --output_dir weights/fm_sr --resolution 256 --batch_size 4 --max_train_steps 10000 --resume weights/fm_sr/checkpoint-2000/model.pt
```

Expected checkpoints:

```text
Part5_3B/weights/fm_sr/checkpoint-2000/model.pt
Part5_3B/weights/fm_sr/checkpoint-4000/model.pt
Part5_3B/weights/fm_sr/model.pt
```

### Smoke Test

Use an existing checkpoint and a few frames first:

```bash
python -m part5_3_da.run_inference --dataset sample_reds --data_root ../data --part2_root ../Part5_2/outputs --checkpoint weights/fm_sr/model.pt --steps 4 --tile_size 512 --max_sequences 1 --max_frames 5 --device cuda
```

### Inference

REDS validation:

```bash
python -m part5_3_da.run_inference --dataset reds_val --data_root ../data --part2_root ../Part5_2/outputs --checkpoint weights/fm_sr/model.pt --steps 4 --tile_size 512 --device cuda
```

Mandatory sample datasets:

```bash
python -m part5_3_da.run_inference --dataset sample_reds --data_root ../data --part2_root ../Part5_2/outputs --checkpoint weights/fm_sr/model.pt --steps 4 --tile_size 512 --device cuda
python -m part5_3_da.run_inference --dataset vimeo_lr --data_root ../data --part2_root ../Part5_2/outputs --checkpoint weights/fm_sr/model.pt --steps 4 --tile_size 512 --device cuda
python -m part5_3_da.run_inference --dataset wild --wild_frames ../data/wild-video/decoded_frames/IMG_2175 --data_root ../data --part2_root ../Part5_2/outputs --checkpoint weights/fm_sr/model.pt --steps 4 --tile_size 512 --device cuda
```

Useful inference options:

```text
--run_name fm_sr
--input_model iconvsr
--steps 4
--solver euler
--tile_size 512
--tile_overlap 64
--temporal_blend 0.2
--skip_existing
--max_sequences N
--max_frames N
```

Outputs:

```text
Part5_3B/outputs/<run_name>/<input_model>/<dataset>/<sequence>/frames/*.png
Part5_3B/outputs/<run_name>/<input_model>/<dataset>/<sequence>.mp4
```

`reds_val` saves frames for evaluation. `sample_reds`, `vimeo_lr`, and `wild` also save mp4 videos.

### Evaluation

```bash
python -m part5_3_da.evaluate --methods fm_sr --dataset reds_val --data_root ../data --part2_root ../Part5_2/outputs --pred_root outputs --device cuda --crop_border 4
```

Fast evaluation:

```bash
python -m part5_3_da.evaluate --methods fm_sr --dataset reds_val --max_sequences 1 --max_frames 5 --data_root ../data --part2_root ../Part5_2/outputs --pred_root outputs --device cuda --skip_fid
```

Metric outputs:

```text
Part5_3B/outputs/metrics/frame_metrics.csv
Part5_3B/outputs/metrics/summary_metrics.csv
```

### Visualization

```bash
python -m part5_3_da.visualize --methods fm_sr --dataset reds_val --data_root ../data --part2_root ../Part5_2/outputs --pred_root outputs --out_dir outputs/visuals --frame_index 50 --max_sequences 4
```

### Tests

```bash
PYTHONPATH=src pytest -q
```

## Contribution Statement

- **Yuzhe Zhuang**: Part 2 and Part 3A
- **Ziyu Liang**: Part 1 and Part 3B