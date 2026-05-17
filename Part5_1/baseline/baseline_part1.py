import argparse
import csv
import glob
import os
import time
from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset


IMG_EXTS = ("*.png", "*.jpg", "*.jpeg", "*.bmp")


def list_images(folder: str) -> List[str]:
    files: List[str] = []
    for ext in IMG_EXTS:
        files.extend(glob.glob(os.path.join(folder, ext)))
    return sorted(files)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def read_img_rgb(path: str) -> np.ndarray:
    img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def write_img_rgb(path: str, img_rgb: np.ndarray) -> None:
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, img_bgr)


def np_to_tensor(img_rgb: np.ndarray) -> torch.Tensor:
    arr = img_rgb.astype(np.float32) / 255.0
    chw = np.transpose(arr, (2, 0, 1))
    return torch.from_numpy(chw)


def np_gray_to_tensor(img_gray: np.ndarray) -> torch.Tensor:
    arr = img_gray.astype(np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


def tensor_to_np(t: torch.Tensor) -> np.ndarray:
    t = t.detach().cpu().clamp(0.0, 1.0)
    arr = t.numpy()
    hwc = np.transpose(arr, (1, 2, 0))
    return (hwc * 255.0 + 0.5).astype(np.uint8)


def tensor_to_np_gray(t: torch.Tensor) -> np.ndarray:
    t = t.detach().cpu().clamp(0.0, 1.0)
    arr = t.numpy()
    return (arr * 255.0 + 0.5).astype(np.uint8)


def center_crop_pair(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    h = min(a.shape[0], b.shape[0])
    w = min(a.shape[1], b.shape[1])
    a = a[:h, :w]
    b = b[:h, :w]
    return a, b


def psnr_uint8(pred: np.ndarray, gt: np.ndarray) -> float:
    pred, gt = center_crop_pair(pred, gt)
    mse = np.mean((pred.astype(np.float32) - gt.astype(np.float32)) ** 2)
    if mse <= 1e-12:
        return 99.0
    return float(20.0 * np.log10(255.0 / np.sqrt(mse)))


def ssim_uint8(pred: np.ndarray, gt: np.ndarray) -> float:
    pred, gt = center_crop_pair(pred, gt)
    pred = pred.astype(np.float64)
    gt = gt.astype(np.float64)

    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2

    vals = []
    for ch in range(3):
        p = pred[:, :, ch]
        g = gt[:, :, ch]

        mu_p = cv2.GaussianBlur(p, (11, 11), 1.5)
        mu_g = cv2.GaussianBlur(g, (11, 11), 1.5)

        mu_p2 = mu_p * mu_p
        mu_g2 = mu_g * mu_g
        mu_pg = mu_p * mu_g

        sigma_p2 = cv2.GaussianBlur(p * p, (11, 11), 1.5) - mu_p2
        sigma_g2 = cv2.GaussianBlur(g * g, (11, 11), 1.5) - mu_g2
        sigma_pg = cv2.GaussianBlur(p * g, (11, 11), 1.5) - mu_pg

        numerator = (2 * mu_pg + c1) * (2 * sigma_pg + c2)
        denominator = (mu_p2 + mu_g2 + c1) * (sigma_p2 + sigma_g2 + c2)
        ssim_map = numerator / (denominator + 1e-12)
        vals.append(float(np.mean(np.asarray(ssim_map, dtype=np.float64))))

    return float(np.mean(vals))


def pil_resize(img_rgb: np.ndarray, size_hw: Tuple[int, int], method: str) -> np.ndarray:
    h, w = size_hw
    pil = Image.fromarray(img_rgb)
    resampling = getattr(Image, "Resampling", Image)
    if method == "bicubic":
        res = pil.resize((w, h), resampling.BICUBIC)
    elif method == "lanczos":
        res = pil.resize((w, h), resampling.LANCZOS)
    else:
        raise ValueError(f"Unsupported method: {method}")
    return np.array(res)


class Srcnn(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, padding=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, kernel_size=5, padding=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SrcnnClassicY(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(32, 1, kernel_size=5, padding=2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x


class SrcnnTrainDataset(Dataset):
    def __init__(self, hr_dir: str, scale: int, patch_size: int = 128) -> None:
        self.hr_paths = list_images(hr_dir)
        if not self.hr_paths:
            raise ValueError(f"No images found in {hr_dir}")
        self.scale = scale
        self.patch_size = patch_size

    def __len__(self) -> int:
        return len(self.hr_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        hr = read_img_rgb(self.hr_paths[idx])
        h, w = hr.shape[:2]

        h = (h // self.scale) * self.scale
        w = (w // self.scale) * self.scale
        hr = hr[:h, :w]

        lr_h = h // self.scale
        lr_w = w // self.scale

        lr = pil_resize(hr, (lr_h, lr_w), method="bicubic")
        bic = pil_resize(lr, (h, w), method="bicubic")

        if h >= self.patch_size and w >= self.patch_size:
            top = np.random.randint(0, h - self.patch_size + 1)
            left = np.random.randint(0, w - self.patch_size + 1)
            hr_patch = hr[top : top + self.patch_size, left : left + self.patch_size]
            bic_patch = bic[top : top + self.patch_size, left : left + self.patch_size]
        else:
            hr_patch = hr
            bic_patch = bic

        x = np_to_tensor(bic_patch)
        y = np_to_tensor(hr_patch)
        return x, y


@dataclass
class EvalResult:
    name: str
    psnr: float
    ssim: float
    avg_time_ms: float


def cmd_extract_frames(args: argparse.Namespace) -> None:
    ensure_dir(args.out_dir)
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")

    idx = 0
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        out_path = os.path.join(args.out_dir, f"{idx:06d}.png")
        write_img_rgb(out_path, frame_rgb)
        idx += 1

    cap.release()
    print(f"Extracted {idx} frames to {args.out_dir}")


def cmd_make_lr(args: argparse.Namespace) -> None:
    ensure_dir(args.out_dir)
    paths = list_images(args.hr_dir)
    if not paths:
        raise ValueError(f"No images found in {args.hr_dir}")

    for p in paths:
        name = os.path.basename(p)
        hr = read_img_rgb(p)
        h, w = hr.shape[:2]
        h = (h // args.scale) * args.scale
        w = (w // args.scale) * args.scale
        hr = hr[:h, :w]

        lr = pil_resize(hr, (h // args.scale, w // args.scale), method="bicubic")
        write_img_rgb(os.path.join(args.out_dir, name), lr)

    print(f"Generated LR frames in {args.out_dir}")


def cmd_interpolate(args: argparse.Namespace) -> None:
    ensure_dir(args.out_dir)
    paths = list_images(args.lr_dir)
    if not paths:
        raise ValueError(f"No images found in {args.lr_dir}")

    times = []
    for p in paths:
        name = os.path.basename(p)
        lr = read_img_rgb(p)
        h, w = lr.shape[:2]

        t0 = time.perf_counter()
        sr = pil_resize(lr, (h * args.scale, w * args.scale), method=args.method)
        dt = (time.perf_counter() - t0) * 1000.0
        times.append(dt)

        write_img_rgb(os.path.join(args.out_dir, name), sr)

    print(f"Interpolation {args.method} done: {len(paths)} frames")
    print(f"Average runtime: {np.mean(times):.3f} ms/frame")


def cmd_train_srcnn(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    ds = SrcnnTrainDataset(args.hr_dir, scale=args.scale, patch_size=args.patch_size)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0)

    model = Srcnn().to(device)
    opt = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    ensure_dir(os.path.dirname(args.ckpt) or ".")

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for x, y in dl:
            x = x.to(device)
            y = y.to(device)

            pred = model(x)
            loss = criterion(pred, y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            losses.append(float(loss.item()))

        avg_loss = float(np.mean(losses))
        print(f"Epoch {epoch}/{args.epochs} - loss: {avg_loss:.6f}")

    torch.save({"model": model.state_dict(), "scale": args.scale}, args.ckpt)
    print(f"Saved checkpoint to {args.ckpt}")


def load_srcnn(ckpt_path: str, device: torch.device) -> Tuple[nn.Module, str]:
    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict):
        state_dict = None
        for key in ("model", "state_dict", "params", "params_ema", "net"):
            candidate = ckpt.get(key)
            if isinstance(candidate, dict):
                state_dict = candidate
                break
        if state_dict is None:
            if all(torch.is_tensor(value) for value in ckpt.values()):
                state_dict = ckpt
            else:
                raise KeyError(
                    "Unsupported checkpoint format: expected a raw state_dict or a dict with one of the keys "
                    "'model', 'state_dict', 'params', 'params_ema', 'net'."
                )
    else:
        state_dict = ckpt

    normalized_state_dict = {}
    for key, value in state_dict.items():
        normalized_key = key
        for prefix in ("module.", "model."):
            if normalized_key.startswith(prefix):
                normalized_key = normalized_key[len(prefix):]
        normalized_state_dict[normalized_key] = value

    # Auto-detect classic SRCNN (Y channel, conv1/2/3 naming).
    is_classic_y = (
        "conv1.weight" in normalized_state_dict
        and "conv2.weight" in normalized_state_dict
        and "conv3.weight" in normalized_state_dict
    )

    if is_classic_y:
        model = SrcnnClassicY().to(device)
        model.load_state_dict(normalized_state_dict, strict=True)
        model.eval()
        return model, "y"

    rgb_state_dict = {}
    for key, value in normalized_state_dict.items():
        mapped_key = key
        if mapped_key.startswith("conv1."):
            mapped_key = mapped_key.replace("conv1.", "net.0.", 1)
        elif mapped_key.startswith("conv2."):
            mapped_key = mapped_key.replace("conv2.", "net.2.", 1)
        elif mapped_key.startswith("conv3."):
            mapped_key = mapped_key.replace("conv3.", "net.4.", 1)
        rgb_state_dict[mapped_key] = value

    model = Srcnn().to(device)
    model.load_state_dict(rgb_state_dict, strict=True)
    model.eval()
    return model, "rgb"


def cmd_infer_srcnn(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    ensure_dir(args.out_dir)

    model, infer_mode = load_srcnn(args.ckpt, device)
    paths = list_images(args.lr_dir)
    if not paths:
        raise ValueError(f"No images found in {args.lr_dir}")

    times = []
    with torch.no_grad():
        for p in paths:
            name = os.path.basename(p)
            lr = read_img_rgb(p)
            h, w = lr.shape[:2]
            bic = pil_resize(lr, (h * args.scale, w * args.scale), method="bicubic")

            if infer_mode == "rgb":
                x = np_to_tensor(bic).unsqueeze(0).to(device)
                t0 = time.perf_counter()
                pred = model(x)[0]
                sr = tensor_to_np(pred)
            else:
                ycrcb = cv2.cvtColor(bic, cv2.COLOR_RGB2YCrCb)
                y = ycrcb[:, :, 0]
                x = np_gray_to_tensor(y).unsqueeze(0).to(device)
                t0 = time.perf_counter()
                pred_y = model(x)[0, 0]
                out_y = tensor_to_np_gray(pred_y)
                ycrcb[:, :, 0] = out_y
                sr = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)

            dt = (time.perf_counter() - t0) * 1000.0
            times.append(dt)
            write_img_rgb(os.path.join(args.out_dir, name), sr)

    print(f"SRCNN inference done: {len(paths)} frames")
    print(f"Average runtime: {np.mean(times):.3f} ms/frame")


def cmd_temporal_avg(args: argparse.Namespace) -> None:
    ensure_dir(args.out_dir)
    paths = list_images(args.in_dir)
    if not paths:
        raise ValueError(f"No images found in {args.in_dir}")

    radius = len(args.weights) // 2
    w = np.array(args.weights, dtype=np.float32)
    w = w / np.sum(w)

    frames = [read_img_rgb(p).astype(np.float32) for p in paths]

    for i, p in enumerate(paths):
        acc = np.zeros_like(frames[i], dtype=np.float32)
        wsum = 0.0
        for k in range(-radius, radius + 1):
            j = i + k
            if 0 <= j < len(frames):
                wk = w[k + radius]
                acc += wk * frames[j]
                wsum += wk

        out = np.clip(acc / max(wsum, 1e-8), 0, 255).astype(np.uint8)

        if args.unsharp:
            blur = cv2.GaussianBlur(out, (0, 0), sigmaX=args.unsharp_sigma)
            out = cv2.addWeighted(out, 1 + args.unsharp_amount, blur, -args.unsharp_amount, 0)

        name = os.path.basename(p)
        write_img_rgb(os.path.join(args.out_dir, name), out)

    print(f"Temporal averaging done: {len(paths)} frames")


def eval_folder(pred_dir: str, gt_dir: str) -> Tuple[float, float]:
    pred_paths = list_images(pred_dir)
    if not pred_paths:
        raise ValueError(f"No images found in {pred_dir}")

    psnr_vals = []
    ssim_vals = []
    for p in pred_paths:
        name = os.path.basename(p)
        gt_path = os.path.join(gt_dir, name)
        if not os.path.exists(gt_path):
            continue
        pred = read_img_rgb(p)
        gt = read_img_rgb(gt_path)
        psnr_vals.append(psnr_uint8(pred, gt))
        ssim_vals.append(ssim_uint8(pred, gt))

    if not psnr_vals:
        raise ValueError("No matched filenames found between pred_dir and gt_dir")

    return float(np.mean(psnr_vals)), float(np.mean(ssim_vals))


def cmd_evaluate(args: argparse.Namespace) -> None:
    rows = []
    for item in args.items:
        name, pred_dir, time_ms = item.split(":")
        psnr, ssim = eval_folder(pred_dir, args.gt_dir)
        rows.append(EvalResult(name=name, psnr=psnr, ssim=ssim, avg_time_ms=float(time_ms)))

    ensure_dir(os.path.dirname(args.out_csv) or ".")
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "psnr", "ssim", "avg_time_ms"])
        for r in rows:
            writer.writerow([r.name, f"{r.psnr:.4f}", f"{r.ssim:.4f}", f"{r.avg_time_ms:.4f}"])

    print(f"Saved metrics to {args.out_csv}")


def cmd_frames_to_video(args: argparse.Namespace) -> None:
    paths = list_images(args.frames_dir)
    if not paths:
        raise ValueError(f"No images found in {args.frames_dir}")

    first = read_img_rgb(paths[0])
    h, w = first.shape[:2]
    fourcc = getattr(cv2, "VideoWriter_fourcc")(*"mp4v")
    writer = cv2.VideoWriter(args.out_video, fourcc, args.fps, (w, h))

    for p in paths:
        frame_rgb = read_img_rgb(p)
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        writer.write(frame_bgr)

    writer.release()
    print(f"Saved video to {args.out_video}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Part 1 baseline pipeline for video super-resolution")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("extract_frames")
    p.add_argument("--video", required=True)
    p.add_argument("--out_dir", required=True)
    p.set_defaults(func=cmd_extract_frames)

    p = sub.add_parser("make_lr")
    p.add_argument("--hr_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--scale", type=int, default=4)
    p.set_defaults(func=cmd_make_lr)

    p = sub.add_parser("interpolate")
    p.add_argument("--lr_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--scale", type=int, default=4)
    p.add_argument("--method", choices=["bicubic", "lanczos"], required=True)
    p.set_defaults(func=cmd_interpolate)

    p = sub.add_parser("train_srcnn")
    p.add_argument("--hr_dir", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--scale", type=int, default=4)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--patch_size", type=int, default=128)
    p.add_argument("--cpu", action="store_true")
    p.set_defaults(func=cmd_train_srcnn)

    p = sub.add_parser("infer_srcnn")
    p.add_argument("--lr_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--scale", type=int, default=4)
    p.add_argument("--cpu", action="store_true")
    p.set_defaults(func=cmd_infer_srcnn)

    p = sub.add_parser("temporal_avg")
    p.add_argument("--in_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--weights", type=float, nargs="+", default=[0.1, 0.2, 0.4, 0.2, 0.1])
    p.add_argument("--unsharp", action="store_true")
    p.add_argument("--unsharp_amount", type=float, default=0.6)
    p.add_argument("--unsharp_sigma", type=float, default=1.2)
    p.set_defaults(func=cmd_temporal_avg)

    p = sub.add_parser("evaluate")
    p.add_argument("--gt_dir", required=True)
    p.add_argument("--out_csv", required=True)
    p.add_argument(
        "--items",
        required=True,
        nargs="+",
        help="Format: method_name:pred_dir:avg_time_ms",
    )
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("frames_to_video")
    p.add_argument("--frames_dir", required=True)
    p.add_argument("--out_video", required=True)
    p.add_argument("--fps", type=float, default=24.0)
    p.set_defaults(func=cmd_frames_to_video)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
