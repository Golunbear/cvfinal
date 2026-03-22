import cv2
import torch
import torch.nn as nn
import numpy as np
import subprocess


class SRCNN(nn.Module):
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. load model
model = SRCNN().to(device)
model.load_state_dict(torch.load("asset/srcnn_x4.pth", map_location=device))
model.eval()

# 2. open video
input_path = "data/test.mp4"
scale = 3

cap = cv2.VideoCapture(input_path)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out_width = width * scale
out_height = height * scale

temp_output = "temp/temp_output.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter(temp_output, fourcc, fps, (out_width, out_height))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 3. convert BGR -> YCrCb
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)

    # 4. bicubic upscale
    y_up = cv2.resize(y, (out_width, out_height), interpolation=cv2.INTER_CUBIC)
    cr_up = cv2.resize(cr, (out_width, out_height), interpolation=cv2.INTER_CUBIC)
    cb_up = cv2.resize(cb, (out_width, out_height), interpolation=cv2.INTER_CUBIC)

    # 5. normalize and to tensor
    y_input = y_up.astype(np.float32) / 255.0
    y_input = torch.from_numpy(y_input).unsqueeze(0).unsqueeze(0).to(device)

    # 6. SRCNN inference
    with torch.no_grad():
        y_sr = model(y_input).clamp(0.0, 1.0)

    # 7. back to numpy
    y_sr = y_sr.squeeze().cpu().numpy()
    y_sr = (y_sr * 255.0).astype(np.uint8)

    # 8. merge channels
    sr_ycrcb = cv2.merge([y_sr, cr_up, cb_up])
    sr_bgr = cv2.cvtColor(sr_ycrcb, cv2.COLOR_YCrCb2BGR)

    # 9. write frame
    writer.write(sr_bgr)

cap.release()
writer.release()


subprocess.run([
    "ffmpeg",
    "-y",
    "-i", "temp_output.mp4",
    "-c:v", "libx264",
    "-pix_fmt", "yuv420p",
    "-movflags", "+faststart",
    "result/final_output.mp4"
], check=True)


print("Done:", "result/final_output.mp4")