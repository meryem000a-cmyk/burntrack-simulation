#!/usr/bin/env python3
"""
Run both models (YOLO ONNX FP16 + CNN ONNX FP32) on an image.
Output: [genus, dryness]

Fixes data issues in the original training:
  - "baobab" and "adansonia" are the same → both output as "adansonia"
  - "vachellia" doesn't exist in YOLO classes → mapped to "acacia"
  - Dryness CNN uses temperature-scaled sigmoid for better calibration
"""

import sys, warnings
import numpy as np
from pathlib import Path

warnings.filterwarnings("ignore")

# ── YOLO class names (from best(6).pt) ──────────────────────
YOLO_CLASSES = [
    "acacia", "adansonia", "aloe", "andropogon", "baobab",
    "brachystegia", "colophospermum", "combretum", "erica",
    "euphorbia", "ficus", "khaya", "macaranga", "protea",
    "senegalia", "tamarix", "themeda",
]

# ── DATA FIX: merge duplicate classes ────────────────────────
# The dataset has botanical/common name duplicates that we resolve here:
#   1. "baobab" and "adansonia" are the same plant (Baobab tree / Adansonia genus).
#   2. "senegalia" and "vachellia" are both grouped under "acacia" for fuel modeling.
CLASS_FIX = {
    "baobab": "adansonia",
    "senegalia": "acacia",
    "vachellia": "acacia",
}

# Closest YOLO class fallback map for evaluation purposes
VACHELLIA_MAP = "acacia"

# ── Load models once ────────────────────────────────────────
PROJECT = Path(__file__).resolve().parents[1]
YOLO_ONNX = PROJECT / "models" / "yolo_fp16.onnx"
CNN_ONNX = PROJECT / "models" / "cnn_fp32.onnx"

if not CNN_ONNX.exists():
    CNN_ONNX = PROJECT / "models" / "best_dryness_cnn.pt"

import onnxruntime as ort
from PIL import Image
from torchvision import transforms

yolo_sess = None
cnn_sess = None
cnn_pt = None

def _load_yolo():
    global yolo_sess
    if yolo_sess is not None:
        return yolo_sess
    yolo_sess = ort.InferenceSession(str(YOLO_ONNX), providers=["CPUExecutionProvider"])
    return yolo_sess

def _load_cnn():
    global cnn_sess, cnn_pt
    if cnn_sess is not None:
        return cnn_sess, None

    if CNN_ONNX.suffix == ".onnx":
        cnn_sess = ort.InferenceSession(str(CNN_ONNX), providers=["CPUExecutionProvider"])
        return cnn_sess, None

    # Fallback to PyTorch if ONNX not available
    import torch, torch.nn as nn

    class DrynessCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2))
            self.conv2 = nn.Sequential(nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2))
            self.conv3 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2))
            self.conv4 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2))
            self.conv5 = nn.Sequential(nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.AdaptiveAvgPool2d(1))
            self.fc = nn.Linear(256, 1)
        def forward(self, x):
            x = self.conv1(x); x = self.conv2(x); x = self.conv3(x)
            x = self.conv4(x); x = self.conv5(x).flatten(1)
            return self.fc(x).squeeze(1)

    cnn_pt = DrynessCNN()
    cnn_pt.load_state_dict(torch.load(str(CNN_ONNX), map_location="cpu", weights_only=True))
    cnn_pt.eval()
    return None, cnn_pt


# ── Image preprocessing ─────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


def predict(image_path: str) -> tuple[str, str]:
    """Run inference. Returns (genus, "dry"|"alive")."""

    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0)  # [1, 3, 224, 224]

    # ── YOLO prediction (genus) ──────────────────────────────
    sess = _load_yolo()
    inp_name = sess.get_inputs()[0].name
    out = sess.run(None, {inp_name: x.numpy().astype(np.float32)})[0][0]
    raw_class = YOLO_CLASSES[out.argmax()]

    # Confidence = softmax probability of top prediction
    exp = np.exp(out - out.max())
    probs = exp / exp.sum()
    confidence = float(probs[out.argmax()])

    # Apply data fix: baobab → adansonia
    genus = CLASS_FIX.get(raw_class, raw_class)

    # ── CNN prediction (dryness) ─────────────────────────────
    sess_cnn, pt_model = _load_cnn()

    if sess_cnn is not None:
        inp_cnn = sess_cnn.get_inputs()[0].name
        logit = sess_cnn.run(None, {inp_cnn: x.numpy()})[0][0]
        # Temperature scaling at 0.5 → sharper/safer calibration
        prob = 1.0 / (1.0 + np.exp(-logit / 0.5))
    else:
        import torch
        with torch.no_grad():
            logit = pt_model(x)
            prob = torch.sigmoid(logit / 0.5).item()

    dryness = "dry" if prob > 0.5 else "alive"

    return genus, dryness


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 scripts/infer.py <image_path> [image_path2 ...]")
        sys.exit(1)

    for path in sys.argv[1:]:
        if not Path(path).exists():
            print(f"  {path}: file not found", file=sys.stderr)
            continue
        genus, dryness = predict(path)
        print(f"  {Path(path).name:35s} [{genus:20s}, {dryness}]")


if __name__ == "__main__":
    main()
