# Part 5.2 - SOTA Reproduction: BasicVSR, IconVSR, and Real-ESRGAN

Three independent baselines are provided:

- BasicVSR: video SR with bidirectional recurrent propagation and SPyNet flow alignment.
- IconVSR: BasicVSR-style recurrent propagation with EDVR-M information refill at keyframes.
- Real-ESRGAN: frame-by-frame perceptual SR with an RRDBNet generator.

## Conda Environment
Create a clean conda environment, then install this package and its requirements from `Part5_2/`:

```bash
conda create -n CVfinal python=3.12 -y
conda activate CVfinal
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install -r requirements.txt
```

## Data Layout

Run commands from `Part5_2/`. The expected project layout is:

```text
CVfinal/
  data/
    REDS/
      val/
        val_sharp/
        val_sharp_bicubic/X4/
    Sample_data/
      REDS-sample/
      vimeo-RL/
    wild-video/
  Part5_2/
```

Supported dataset names:

- `reds_val`: paired REDS validation data. Metrics are computed here; inference saves frames only.
- `sample_reds`: LR-only sample REDS clips. Results and videos are saved, metrics are skipped.
- `vimeo_lr`: LR-only Vimeo sample clips. Results and videos are saved, metrics are skipped.
- `wild`: decoded frames from `data/wild-video/decoded_frames`. Results and videos are saved, metrics are skipped.

## Weights

Download all official checkpoints:

```bash
python -m part5_2.download_weights --weights_dir weights
```

Download only IconVSR:

```bash
python -m part5_2.download_weights --weights_dir weights --models iconvsr
```

Expected files:

```text
weights/basicvsr_reds4_20120409-0e599677.pth
weights/iconvsr_reds4_20210413-9e09d621.pth
weights/RealESRGAN_x4plus.pth
```

Manual URLs are listed in `src/part5_2/weights.py` if the server cannot download directly.

## Decode Wild Video Once

Wild videos are decoded in a separate step. Inference never reads the mp4 directly.

```bash
python -m part5_2.decode_wild_video --wild_root ../data/wild-video
```

For `IMG_2175.mp4`, this creates:

```text
../data/wild-video/decoded_frames/IMG_2175/
  00000000.png
  00000001.png
  ...
  metadata.json
```

The script skips existing decoded frames by default. Use `--overwrite` to decode again.

## Quick Smoke Test

```
python -m part5_2.run_inference \
  --model basicvsr \
  --dataset sample_reds \
  --data_root ../data \
  --out_root outputs \
  --max_sequences 1 \
  --max_frames 5 \
  --device cuda
```

```
python -m part5_2.run_inference \
  --model iconvsr \
  --dataset sample_reds \
  --data_root ../data \
  --out_root outputs \
  --max_sequences 1 \
  --max_frames 5 \
  --device cuda
```

```
python -m part5_2.run_inference \
  --model realesrgan \
  --dataset sample_reds \
  --data_root ../data \
  --out_root outputs \
  --max_sequences 1 \
  --max_frames 5 \
  --tile 512 \
  --device cuda
```

## Inference

Mandatory sample data:
```
# REDS-sample
python -m part5_2.run_inference --model basicvsr --dataset sample_reds --data_root ../data --out_root outputs --device cuda

python -m part5_2.run_inference --model iconvsr --dataset sample_reds --data_root ../data --out_root outputs --device cuda

python -m part5_2.run_inference --model realesrgan --dataset sample_reds --data_root ../data --out_root outputs --tile 512 --device cuda

# Vimeo-sample
python -m part5_2.run_inference --model basicvsr --dataset vimeo_lr --data_root ../data --out_root outputs --device cuda

python -m part5_2.run_inference --model iconvsr --dataset vimeo_lr --data_root ../data --out_root outputs --device cuda

python -m part5_2.run_inference --model realesrgan --dataset vimeo_lr --data_root ../data --out_root outputs --tile 512 --device cuda
```

Full REDS Dataset, for quantitative metrics:

```
python -m part5_2.run_inference --model basicvsr --dataset reds_val --data_root ../data --out_root outputs --device cuda

python -m part5_2.run_inference --model iconvsr --dataset reds_val --data_root ../data --out_root outputs --device cuda

python -m part5_2.run_inference --model realesrgan --dataset reds_val --data_root ../data --out_root outputs --tile 512 --device cuda
```

Wild video, using decoded frames:
```
python -m part5_2.run_inference --model basicvsr --dataset wild --wild_frames ../data/wild-video/decoded_frames/IMG_2175 --data_root ../data --out_root outputs --device cuda

python -m part5_2.run_inference --model iconvsr --dataset wild --wild_frames ../data/wild-video/decoded_frames/IMG_2175 --data_root ../data --out_root outputs --device cuda

python -m part5_2.run_inference --model realesrgan --dataset wild --wild_frames ../data/wild-video/decoded_frames/IMG_2175 --data_root ../data --out_root outputs --tile 512 --device cuda
```

For `wild` + BasicVSR/IconVSR, `--chunk_size` defaults to `5` automatically and streams chunk
outputs directly to disk to avoid CUDA OOM on long or high-resolution videos. If memory is still
tight, the script retries with smaller chunks down to `3`; if memory allows, use `--chunk_size 20`
or `--chunk_size 50` for faster inference.

## Outputs

Inference writes:

```text
outputs/<model>/<dataset>/<sequence>/frames/*.png
outputs/<model>/<dataset>/<sequence>.mp4
```

Examples:

```text
outputs/basicvsr/reds_val/000/frames/00000000.png
outputs/iconvsr/sample_reds/000.mp4
```

`reds_val` writes restored frames only. `sample_reds`, `vimeo_lr`, and `wild` also write mp4
videos.

## Wild Video Playback Compatibility

The current video writer uses OpenCV `mp4v` encoding. This is usually fine for the smaller sample outputs, but wild videos can become very large after x4 super-resolution. For example, a 1920x1080 wild video becomes 7680x4320 after x4 upscaling. Some players, including Windows Media Player, Movies & TV, and Bilibili player, may fail to play an `mp4v` 8K mp4. More tolerant players may still open it, such as Quark player.

The full-resolution restored frames are still valid:

```text
outputs/<model>/wild/<sequence>/frames/*.png
```

Only the generated mp4 may need transcoding for broader playback compatibility. Convert an existing wild mp4 to H.264 without resizing:

```bash
ffmpeg -y -i outputs/basicvsr/wild/IMG_2175.mp4 -c:v libx264 -pix_fmt yuv420p -crf 18 -preset slow outputs/basicvsr/wild/IMG_2175_h264.mp4
```

(Recommended) For better compatibility, convert the wild mp4 to a 4K H.264 file:

```bash
ffmpeg -y -i outputs/basicvsr/wild/IMG_2175.mp4 -vf "scale=3840:2160" -c:v libx264 -pix_fmt yuv420p -crf 18 -preset slow outputs/basicvsr/wild/IMG_2175_h264_4k.mp4
ffmpeg -y -i outputs/iconvsr/wild/IMG_2175.mp4 -vf "scale=3840:2160" -c:v libx264 -pix_fmt yuv420p -crf 18 -preset slow outputs/iconvsr/wild/IMG_2175_h264_4k.mp4
ffmpeg -y -i outputs/realesrgan/wild/IMG_2175.mp4 -vf "scale=3840:2160" -c:v libx264 -pix_fmt yuv420p -crf 18 -preset slow outputs/realesrgan/wild/IMG_2175_h264_4k.mp4
```

## Evaluation

Metrics are computed only when GT exists. For LR-only sample/wild data, the CSV summary records `gt_available=false` and leaves metric columns empty.

Run evaluation on REDS validation:

```bash
python -m part5_2.evaluate \
  --models basicvsr iconvsr realesrgan \
  --dataset reds_val \
  --data_root ../data \
  --pred_root outputs
```

CSV outputs:

```text
outputs/metrics/frame_metrics.csv
outputs/metrics/summary_metrics.csv
```

Implemented metrics:

- PSNR and SSIM: RGB, full frame by default, configurable through `--crop_border`.
- LPIPS: `alex` backbone.
- FID: clean-fid over all predicted frames and all GT frames for each dataset/model pair.
  Per-sequence FID is disabled by default; add `--sequence_fid` if you also want it.
- tLPIPS: temporal perceptual relative error:

```text
tLP_t = abs(LPIPS(I[t-1], I[t]) - LPIPS(G[t-1], G[t]))
```

The summary CSV stores `tlpips_mean` and `tlpips_std`.

## Qualitative Visuals

Create simple side-by-side frame comparisons:

```bash
python -m part5_2.visualize \
  --models basicvsr iconvsr realesrgan \
  --dataset reds_val \
  --data_root ../data \
  --pred_root outputs \
  --out_dir outputs/visuals \
  --frame_index 0
```

In each image, the results at frame_index=0 of LR, GT, basicvsr, iconvsr, and realesrgan will be horizontally concatenated. `--max_sequences` is set to 4 by default. This creates:

```
outputs/visuals/reds_val_000_0000.png
outputs/visuals/reds_val_001_0000.png
outputs/visuals/reds_val_002_0000.png
outputs/visuals/reds_val_003_0000.png
```


## Tests

Run unit tests:

```bash
PYTHONPATH=src pytest -q
```

The current tests verify the tLPIPS relative-delta formula.

# Third-Party Sources and Citations

This implementation reuses architecture ideas and checkpoint conventions from the official
super-resolution projects below. The code in this folder is written locally for the course
pipeline, but class names and tensor conventions are aligned with these sources.

## BasicVSR

- Paper: Kelvin C. K. Chan et al., "BasicVSR: The Search for Essential Components in Video
  Super-Resolution and Beyond", CVPR 2021.
- Official BasicVSR/IconVSR repository: https://github.com/ckkelvinchan/BasicVSR-IconVSR
- Project page: https://ckkelvinchan.github.io/projects/BasicVSR/
- MMagic docs: https://mmagic.readthedocs.io/
- Checkpoint: `basicvsr_reds4_20120409-0e599677.pth`
- Checkpoint URL:
  https://download.openmmlab.com/mmediting/restorers/basicvsr/basicvsr_reds4_20120409-0e599677.pth

## IconVSR

- Paper: Kelvin C. K. Chan et al., "BasicVSR: The Search for Essential Components in Video
  Super-Resolution and Beyond", CVPR 2021.
- Official BasicVSR/IconVSR repository: https://github.com/ckkelvinchan/BasicVSR-IconVSR
- MMagic IconVSR docs: https://mmagic.readthedocs.io/
- Checkpoint: `iconvsr_reds4_20210413-9e09d621.pth`
- Checkpoint URL:
  https://download.openmmlab.com/mmediting/restorers/iconvsr/iconvsr_reds4_20210413-9e09d621.pth

## Real-ESRGAN

- Paper: Xintao Wang et al., "Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure
  Synthetic Data", ICCVW 2021.
- Official repository: https://github.com/xinntao/Real-ESRGAN
- Checkpoint: `RealESRGAN_x4plus.pth`
- Checkpoint URL:
  https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth

## LPIPS

- Paper: Richard Zhang et al., "The Unreasonable Effectiveness of Deep Features as a Perceptual
  Metric", CVPR 2018.
- Repository: https://github.com/richzhang/PerceptualSimilarity

## tLPIPS

- Paper: Mengyu Chu et al., "Temporally Coherent GANs for Video Super-Resolution",
  ACM Transactions on Graphics 2020.
- Formula used here:
  `abs(LPIPS(I[t-1], I[t]) - LPIPS(G[t-1], G[t]))`

## FID

- This project uses `clean-fid` for Frechet Inception Distance computation.
- Repository: https://github.com/GaParmar/clean-fid
