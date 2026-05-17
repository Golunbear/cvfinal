# Part 1 Baseline: Hand-crafted Approach

This folder provides a minimal runnable pipeline for Section 5.1:
- Bicubic and Lanczos interpolation baseline
- SRCNN (3-layer CNN) baseline
- Temporal weighted averaging baseline
- Optional unsharp masking
- PSNR/SSIM evaluation and CSV export

## 1) Install dependencies

```bash
pip install torch torchvision opencv-python pillow numpy
```

## 2) Suggested folder structure

```text
data/
  input.mp4
  hr_frames/
  lr_x4/
outputs/
  bicubic_x4/
  lanczos_x4/
  srcnn_x4/
  tempavg_bicubic/
  tempavg_bicubic_unsharp/
  videos/
metrics/
```

## 3) Example workflow (x4)

### Step A: Extract HR frames from a source video

```bash
python baseline_part1.py extract_frames \
  --video data/input.mp4 \
  --out_dir data/hr_frames
```

### Step B: Create synthetic LR frames from HR (for paired evaluation)

```bash
python baseline_part1.py make_lr \
  --hr_dir data/hr_frames \
  --out_dir data/lr_x4 \
  --scale 4
```

### Step C: Interpolation baselines

```bash
python baseline_part1.py interpolate \
  --lr_dir data/lr_x4 \
  --out_dir outputs/bicubic_x4 \
  --scale 4 --method bicubic

python baseline_part1.py interpolate \
  --lr_dir data/lr_x4 \
  --out_dir outputs/lanczos_x4 \
  --scale 4 --method lanczos
```

### Step D: Train SRCNN on HR frames

```bash
python baseline_part1.py train_srcnn \
  --hr_dir data/hr_frames \
  --ckpt checkpoints/srcnn_x4.pth \
  --scale 4 --epochs 50 --batch_size 8 --lr 1e-4
```

### Step E: Infer SRCNN from LR frames

```bash
python baseline_part1.py infer_srcnn \
  --lr_dir data/lr_x4 \
  --out_dir outputs/srcnn_x4 \
  --ckpt checkpoints/srcnn_x4.pth \
  --scale 4
```

### Step F: Temporal baseline (weighted average)

```bash
python baseline_part1.py temporal_avg \
  --in_dir outputs/bicubic_x4 \
  --out_dir outputs/tempavg_bicubic \
  --weights 0.1 0.2 0.4 0.2 0.1
```

Optional unsharp mask:

```bash
python baseline_part1.py temporal_avg \
  --in_dir outputs/bicubic_x4 \
  --out_dir outputs/tempavg_bicubic_unsharp \
  --weights 0.1 0.2 0.4 0.2 0.1 \
  --unsharp --unsharp_amount 0.6 --unsharp_sigma 1.2
```

### Step G: Evaluate PSNR/SSIM and export CSV

Fill average runtime ms/frame from your command logs:

```bash
python baseline_part1.py evaluate \
  --gt_dir data/hr_frames \
  --out_csv metrics/part1_metrics.csv \
  --items \
  bicubic:outputs/bicubic_x4:1.8 \
  lanczos:outputs/lanczos_x4:2.1 \
  srcnn:outputs/srcnn_x4:7.6 \
  tempavg:outputs/tempavg_bicubic:2.5 \
  tempavg_unsharp:outputs/tempavg_bicubic_unsharp:2.8
```

### Step H: Export result videos

```bash
python baseline_part1.py frames_to_video \
  --frames_dir outputs/bicubic_x4 \
  --out_video outputs/videos/bicubic_x4.mp4 \
  --fps 24
```

Repeat for each method to prepare mandatory demo videos.

## Notes

- Evaluation assumes predicted frames and GT frames share the same filenames.
- If you evaluate real-world clips without GT, use qualitative comparison and report this separately.
- For the report table, keep: method, PSNR, SSIM, runtime (ms/frame).
