import os
import cv2
import argparse
import subprocess
import numpy as np
import torch
import torch.nn as nn


class SRCNN(nn.Module):
    """
    Classic SRCNN.
    Default is 1-channel (Y channel only), which is the standard usage.
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(32, 1, kernel_size=5, padding=2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x


def parse_args():
    parser = argparse.ArgumentParser(description="Apply SRCNN to video frame-by-frame.")

    parser.add_argument("--input", type=str, required=True, help="Path to input video.")
    parser.add_argument("--output", type=str, required=True, help="Path to final output video.")
    parser.add_argument("--weights", type=str, required=True, help="Path to SRCNN .pth weights.")

    parser.add_argument("--scale", type=int, default=2, help="Upscaling factor, e.g. 2.")
    parser.add_argument("--fps", type=float, default=None, help="Target processing/output fps.")
    parser.add_argument("--max_seconds", type=float, default=None, help="Only process the first N seconds.")
    parser.add_argument("--resize_width", type=int, default=None, help="Resize input width before SRCNN.")
    parser.add_argument("--resize_height", type=int, default=None, help="Resize input height before SRCNN.")
    parser.add_argument("--frame_stride", type=int, default=1, help="Process every Nth frame. >=1")

    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"],
                        help="Inference device.")
    parser.add_argument("--keep_audio", action="store_true",
                        help="Try to merge original audio back after processing.")
    parser.add_argument("--temp_dir", type=str, default="temp_srcnn_outputs",
                        help="Directory for temporary files.")

    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(weights_path: str, device: torch.device) -> nn.Module:
    model = SRCNN().to(device)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def compute_resized_dims(width: int, height: int, resize_width: int = None, resize_height: int = None):
    if resize_width is None and resize_height is None:
        return width, height

    if resize_width is not None and resize_height is not None:
        return resize_width, resize_height

    if resize_width is not None:
        new_height = int(round(height * resize_width / width))
        return resize_width, new_height

    new_width = int(round(width * resize_height / height))
    return new_width, resize_height


def run_ffmpeg(cmd):
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    args = parse_args()

    if args.frame_stride < 1:
        raise ValueError("--frame_stride must be >= 1")

    ensure_dir(args.temp_dir)

    device = resolve_device(args.device)
    print(f"[Info] Using device: {device}")

    model = load_model(args.weights, device)
    print("[Info] Model loaded successfully.")

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open input video: {args.input}")

    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    if orig_fps <= 0:
        orig_fps = 30.0

    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"[Info] Original video: {orig_width}x{orig_height}, fps={orig_fps:.3f}, frames={total_frames}")

    # Resize before SR to reduce computation
    proc_width, proc_height = compute_resized_dims(
        orig_width, orig_height, args.resize_width, args.resize_height
    )

    # Output size after SRCNN upscale
    out_width = proc_width * args.scale
    out_height = proc_height * args.scale

    # Decide effective processing/output fps
    # Case A: user explicitly sets fps -> use that
    # Case B: no fps set -> orig_fps / frame_stride
    if args.fps is not None:
        target_fps = args.fps
    else:
        target_fps = orig_fps / args.frame_stride

    if target_fps <= 0:
        raise ValueError("Computed target fps must be > 0")

    print(f"[Info] Processing size before SR: {proc_width}x{proc_height}")
    print(f"[Info] Output size after SR x{args.scale}: {out_width}x{out_height}")
    print(f"[Info] Output fps: {target_fps:.3f}")
    print(f"[Info] Frame stride: {args.frame_stride}")

    max_input_frames = None
    if args.max_seconds is not None:
        max_input_frames = int(orig_fps * args.max_seconds)
        print(f"[Info] Will process at most first {args.max_seconds} seconds "
              f"(~{max_input_frames} input frames).")

    temp_video = os.path.join(args.temp_dir, "temp_no_audio.mp4")

    # Use mp4v for temp export, then ffmpeg transcode to H.264 yuv420p
    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    writer = cv2.VideoWriter(temp_video, fourcc, target_fps, (out_width, out_height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Failed to open VideoWriter for temporary output.")

    frame_idx = 0
    written_frames = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if max_input_frames is not None and frame_idx >= max_input_frames:
                break

            # Skip frames according to stride
            if frame_idx % args.frame_stride != 0:
                frame_idx += 1
                continue

            # Optional resize BEFORE SR, to reduce per-frame cost
            if (proc_width, proc_height) != (orig_width, orig_height):
                frame = cv2.resize(frame, (proc_width, proc_height), interpolation=cv2.INTER_AREA)

            # Convert BGR -> YCrCb
            ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
            y, cr, cb = cv2.split(ycrcb)

            # Bicubic upsample to target SR size
            y_up = cv2.resize(y, (out_width, out_height), interpolation=cv2.INTER_CUBIC)
            cr_up = cv2.resize(cr, (out_width, out_height), interpolation=cv2.INTER_CUBIC)
            cb_up = cv2.resize(cb, (out_width, out_height), interpolation=cv2.INTER_CUBIC)

            # Normalize Y for SRCNN
            y_input = y_up.astype(np.float32) / 255.0
            y_input = torch.from_numpy(y_input).unsqueeze(0).unsqueeze(0).to(device)

            with torch.no_grad():
                y_sr = model(y_input).clamp(0.0, 1.0)

            y_sr = y_sr.squeeze().cpu().numpy()
            y_sr = (y_sr * 255.0).round().astype(np.uint8)

            # Merge channels back
            sr_ycrcb = cv2.merge([y_sr, cr_up, cb_up])
            sr_bgr = cv2.cvtColor(sr_ycrcb, cv2.COLOR_YCrCb2BGR)

            writer.write(sr_bgr)
            written_frames += 1

            if written_frames % 20 == 0:
                print(f"[Progress] input_frame={frame_idx}, written_frames={written_frames}")

            frame_idx += 1

    finally:
        cap.release()
        writer.release()

    print(f"[Info] Temporary video saved: {temp_video}")
    print(f"[Info] Written frames: {written_frames}")

    # Transcode to a more compatible mp4
    final_video = args.output
    transcoded_video = os.path.join(args.temp_dir, "temp_h264.mp4")

    run_ffmpeg([
        "ffmpeg", "-y",
        "-i", temp_video,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        transcoded_video
    ])

    if args.keep_audio:
        # Merge original audio if possible
        try:
            run_ffmpeg([
                "ffmpeg", "-y",
                "-i", transcoded_video,
                "-i", args.input,
                "-c:v", "copy",
                "-map", "0:v:0",
                "-map", "1:a:0?",
                "-c:a", "aac",
                "-shortest",
                final_video
            ])
            print(f"[Done] Output video with audio: {final_video}")
        except subprocess.CalledProcessError:
            print("[Warn] Failed to merge audio. Falling back to video-only output.")
            os.replace(transcoded_video, final_video)
            print(f"[Done] Output video (no audio): {final_video}")
    else:
        os.replace(transcoded_video, final_video)
        print(f"[Done] Output video: {final_video}")


if __name__ == "__main__":
    main()