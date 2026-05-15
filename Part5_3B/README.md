# Part 5.3 Exploration B — Flow Matching Generative VSR

This folder implements Direction A, "Generative VSR" with Conditional Flow Matching. Instead of
borrowing a pre-trained text-to-image diffusion prior, we train a task-specific U-Net from
scratch using the Flow Matching framework. The model learns to predict a
residual velocity field that refines Part~2 VSR outputs along straight Optimal Transport (OT)
trajectories.

The default pipeline is:

```text
LR video
  -> Part 2 VSR output (recommended: iconvsr)
  -> Flow Matching U-Net (velocity field prediction)
  -> ODE solver (Euler, 4-8 steps)
  -> refined HR frames/video
```

**Key insight:** The ODE starts from the Part~2 output, not from Gaussian noise. This
``conservative refinement'' strategy preserves the baseline PSNR/SSIM (31.15 dB / 0.892)
while improving temporal consistency (tLPIPS 0.012). In contrast, initializing from noise
causes the model to ignore the conditioning signal entirely (Pearson $r=0.07$ between
output and input).

## Environment

Run commands from `Part5_3_DA/`.

```bash
conda create -n CVfinal_part3 python=3.11 -y
conda activate CVfinal_part3
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install -r requirements.txt
```

The only heavy dependency is PyTorch (2.10.0 with CUDA 13.0). The flow matching model is
a standalone U-Net (~73M parameters) with no external diffusion model or Hugging Face
dependencies.

## Data Layout

The expected project layout follows Part 5.2 conventions:

```text
CVfinal/
  data/                          # shared data root
    REDS/
      train/
        train_sharp/             # HR training frames (240 seq x 100)
        train_sharp_bicubic/X4/  # LR training frames
      val/
        val_sharp/               # GT for evaluation
        val_sharp_bicubic/X4/    # LR validation frames
    Sample_data/
      REDS-sample/               # 10 sequences, no GT
      vimeo-RL/                  # 265 clips x 7 frames
    wild-video/
      decoded_frames/            # pre-extracted wild video frames
  Part5_2/outputs/               # Part 2 inference results
    iconvsr/<dataset>/<seq>/frames/*.png
  Part5_3_DA/                    # this directory
```

## Training

The model trains on REDS train with random 256x256 crops. LR frames are bicubic-upsampled
to HR size as the condition. The CFM objective minimizes the MSE between the predicted
velocity field and the constant OT velocity $u_t = x_{\text{HR}} - x_{\text{LR}}$.

```bash
# Full training (10,000 steps, ~15.2h on RTX A6000)
python -m part5_3_da.train \
  --data_root ../data \
  --output_dir weights/fm_sr \
  --resolution 256 \
  --batch_size 4 \
  --max_train_steps 10000 \
  --checkpoint_every 2000 \
  --mixed_precision fp16

# Resume from a checkpoint
python -m part5_3_da.train \
  --data_root ../data \
  --output_dir weights/fm_sr \
  --resolution 256 \
  --batch_size 4 \
  --max_train_steps 10000 \
  --resume weights/fm_sr/checkpoint-2000/model.pt
```

Training saves checkpoints every `--checkpoint_every` steps:

```text
weights/fm_sr/checkpoint-2000/model.pt
weights/fm_sr/checkpoint-4000/model.pt
...
weights/fm_sr/model.pt           # final (std step)
```

## Inference

Inference reads Part~2 restored frames and writes enhanced frames/videos.

```bash
# REDS validation (full, for quantitative evaluation)
python -m part5_3_da.run_inference \
  --dataset reds_val \
  --data_root ../data \
  --part2_root ../Part5_2/outputs \
  --checkpoint weights/fm_sr/checkpoint-2000/model.pt \
  --steps 4 --tile_size 512 --device cuda

# Sample REDS (mandatory dataset)
python -m part5_3_da.run_inference \
  --dataset sample_reds \
  --data_root ../data \
  --part2_root ../Part5_2/outputs \
  --checkpoint weights/fm_sr/checkpoint-2000/model.pt \
  --steps 4 --tile_size 512 --device cuda

# Vimeo LR
python -m part5_3_da.run_inference \
  --dataset vimeo_lr \
  --data_root ../data \
  --part2_root ../Part5_2/outputs \
  --checkpoint weights/fm_sr/checkpoint-2000/model.pt \
  --steps 4 --tile_size 512 --device cuda

# Wild video
python -m part5_3_da.run_inference \
  --dataset wild \
  --wild_frames ../data/wild-video/decoded_frames/IMG_2175 \
  --data_root ../data \
  --part2_root ../Part5_2/outputs \
  --checkpoint weights/fm_sr/checkpoint-2000/model.pt \
  --steps 4 --tile_size 512 --device cuda
```

Key inference options:

```text
--steps 4 | 8          ODE integration steps (4 is sufficient)
--solver euler | midpoint
--tile_size 512        max tile dimension for GPU memory
--temporal_blend 0.2   optical-flow temporal blending (0 = disabled)
--skip_existing        skip already-processed frames
--max_sequences N      limit number of sequences
--max_frames N         limit frames per sequence
```

## Outputs

```text
outputs/<run_name>/<input_model>/<dataset>/<sequence>/frames/*.png
outputs/<run_name>/<input_model>/<dataset>/<sequence>.mp4
```

Examples:

```text
outputs/fm_sr/iconvsr/reds_val/000/frames/00000000.png
outputs/fm_sr/iconvsr/sample_reds/002.mp4
outputs/fm_sr/iconvsr/wild/IMG_2175.mp4
```

`reds_val` saves frames only. `sample_reds`, `vimeo_lr`, and `wild` also produce mp4 videos.

## Evaluation

Metrics are computed against GT for `reds_val`.

```bash
# Full evaluation with FID
python -m part5_3_da.evaluate \
  --methods fm_sr \
  --dataset reds_val \
  --data_root ../data \
  --part2_root ../Part5_2/outputs \
  --crop_border 4

# Fast smoke test (skip FID)
python -m part5_3_da.evaluate \
  --methods fm_sr \
  --dataset reds_val \
  --max_sequences 1 --max_frames 5 \
  --skip_fid
```

CSV outputs:

```text
outputs/metrics/frame_metrics.csv
outputs/metrics/summary_metrics.csv
```

Implemented metrics:

- **PSNR / SSIM**: RGB, full frame, configurable `--crop_border`.
- **LPIPS**: AlexNet backbone.
- **FID**: clean-fid over all predicted vs GT frames.
- **tLPIPS**: $|\text{LPIPS}(I_{t-1}, I_t) - \text{LPIPS}(G_{t-1}, G_t)|$.

## Visuals

```bash
python -m part5_3_da.visualize \
  --methods fm_sr \
  --dataset reds_val \
  --data_root ../data \
  --part2_root ../Part5_2/outputs \
  --out_dir outputs/visuals \
  --frame_index 50 \
  --max_sequences 4
```

## Model Architecture

The Flow Matching U-Net (~73M parameters):

```text
Input: concat(x_t, condition) 6ch
  -> Encoder: [128, 256, 384, 512] ch, 2 ResBlocks/level
     Self-Attention at 64x64 and 32x32
  -> Bottleneck: 512ch + Self-Attention
  -> Decoder: [512, 384, 256, 128] ch with skip connections
  -> Output: 3ch velocity field
FiLM time conditioning at every ResBlock.
Output convolution zero-initialized.
```

## Key Design Insight: LR-to-HR vs Noise-to-HR

| Initialization | PSNR | $r$(out, inp) | Behavior |
|---|---|---|---|
| Noise-to-HR | $\sim$0 | 0.07 | Model ignores condition, generates random textures |
| LR-to-HR (ours) | 31.15 | 0.9999 | Model predicts residuals, preserves input fidelity |

## Tests

```bash
PYTHONPATH=src pytest -q
```

## References

### Flow Matching

- Paper: Yaron Lipman et al., "Flow Matching for Generative Modeling", ICLR 2023.
- Repository: https://github.com/facebookresearch/flow_matching
- The Conditional Flow Matching (CFM) objective and Optimal Transport path formulation
  used in this work follow Sections 3 and 4 of the paper.

### BasicVSR / IconVSR

- Paper: Kelvin C. K. Chan et al., "BasicVSR: The Search for Essential Components in
  Video Super-Resolution and Beyond", CVPR 2021.
- Our flow matching model uses IconVSR output as the starting point for ODE integration.
  The Part~2 pipeline is in `../Part5_2/`.

### Evaluation Metrics

- **LPIPS**: Richard Zhang et al., "The Unreasonable Effectiveness of Deep Features as a
  Perceptual Metric", CVPR 2018.
- **tLPIPS**: Mengyu Chu et al., "Temporally Coherent GANs for Video Super-Resolution",
  TOG 2020.
- **FID**: clean-fid, https://github.com/GaParmar/clean-fid
- **SSIM**: scikit-image implementation.
