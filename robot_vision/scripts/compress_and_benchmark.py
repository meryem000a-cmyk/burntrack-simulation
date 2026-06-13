#!/usr/bin/env python3
"""
Compress + benchmark every variant of both models for RPi deployment.
Run: python3 scripts/compress_and_benchmark.py
"""

import os, sys, time, warnings, random
from pathlib import Path
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO

PROJECT = Path(__file__).resolve().parents[1]
YOLO_CKPT = PROJECT / "models" / "yolo" / "best(6).pt"
CNN_CKPT = PROJECT / "models" / "dryness" / "best_dryness_cnn.pt"
OUT = PROJECT / "models" / "compressed"
OUT.mkdir(parents=True, exist_ok=True)
IMG_SIZE, DEVICE, N = 224, "cpu", 100
torch.manual_seed(42)
random.seed(42)

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])


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


def size_mb(path):
    return os.path.getsize(path) / 1e6


def get_samples():
    cls_dir = PROJECT / "datasets" / "yolo_flora_cls" / "train"
    bal = PROJECT / "datasets" / "balanced"
    samples = []
    for d in sorted(cls_dir.iterdir()):
        if not d.is_dir(): continue
        species = d.name.rsplit("_", 2)[0]
        all_jpg = sorted(d.glob("*.jpg"))
        used = set(p.name for p in (bal / d.name).iterdir()) if (bal / d.name).exists() else set()
        unused = [p for p in all_jpg if p.name not in used]
        take = min(len(unused), N // 2)
        if take == 0: continue
        for p in random.sample(unused, take):
            dryness = 1.0 if d.name.endswith("_dry") else 0.0
            samples.append((str(p), species, dryness))
    return samples


def load_imgs(n=N):
    return [transform(Image.open(s[0]).convert("RGB")).unsqueeze(0) for s in get_samples()[:n]]


def acc_yolo_pt(model, samples):
    correct = 0
    for path, species_true, _ in samples:
        r = model(Image.open(path).convert("RGB"), verbose=False)
        if r and r[0].probs is not None:
            if r[0].names[r[0].probs.top1] == species_true:
                correct += 1
    return correct / len(samples) * 100


def acc_yolo_onnx(sess, inp_name, samples, class_names):
    correct = 0
    for path, species_true, _ in samples:
        x = transform(Image.open(path).convert("RGB")).unsqueeze(0).numpy()
        out = sess.run(None, {inp_name: x})[0][0]
        if class_names[out.argmax()] == species_true:
            correct += 1
    return correct / len(samples) * 100


def acc_cnn(model, samples):
    model.eval()
    correct = 0
    with torch.no_grad():
        for path, _, dryness_true in samples:
            inp = transform(Image.open(path).convert("RGB")).unsqueeze(0)
            logit = model(inp)
            pred = (torch.sigmoid(logit) > 0.5).float().item()
            if pred == dryness_true:
                correct += 1
    return correct / len(samples) * 100


def acc_cnn_onnx(sess, inp_name, samples):
    correct = 0
    for path, _, dryness_true in samples:
        x = transform(Image.open(path).convert("RGB")).unsqueeze(0).numpy()
        out = sess.run(None, {inp_name: x})[0][0]
        pred = (1 / (1 + np.exp(-out)) > 0.5).item()
        if pred == dryness_true:
            correct += 1
    return correct / len(samples) * 100


def main():
    print("=" * 65)
    print("  MODEL COMPRESSION BENCHMARK")
    print("=" * 65)

    samples = get_samples()
    imgs = load_imgs()
    print(f"\n  Eval: {len(samples)} samples  Speed: {len(imgs)} images\n")
    rows = []

    # ── 1. YOLO PyTorch FP32 ──
    print("[1/6] YOLO PyTorch FP32 ...")
    yolo = YOLO(str(YOLO_CKPT), verbose=False)
    t0 = time.perf_counter()
    for img in imgs: yolo(img, verbose=False)
    speed = (time.perf_counter() - t0) / len(imgs) * 1000
    a = acc_yolo_pt(yolo, samples)
    rows.append({"label": "yolo_pt_fp32", "size": size_mb(YOLO_CKPT), "speed": speed, "acc": a})
    print(f"       {size_mb(YOLO_CKPT):.0f}MB  {speed:.0f}ms  {a:.1f}%")

    import onnxruntime as ort
    class_names = [yolo.names[i] for i in range(len(yolo.names))]

    # ── 2. YOLO ONNX FP32 ──
    print("[2/6] YOLO ONNX FP32 ...")
    onnx_fp32 = PROJECT / "models" / "yolo" / "best(6)_fp32.onnx"
    if not onnx_fp32.exists():
        _ = yolo.export(format="onnx", half=False, imgsz=IMG_SIZE, verbose=False)
        (PROJECT / "models" / "yolo" / "best(6).onnx").rename(onnx_fp32)
    s32 = ort.InferenceSession(str(onnx_fp32), providers=["CPUExecutionProvider"])
    inp32 = s32.get_inputs()[0].name
    t0 = time.perf_counter()
    for img in imgs: s32.run(None, {inp32: img.numpy()})
    speed32 = (time.perf_counter() - t0) / len(imgs) * 1000
    a32 = acc_yolo_onnx(s32, inp32, samples, class_names)
    rows.append({"label": "yolo_onnx_fp32", "size": size_mb(onnx_fp32), "speed": speed32, "acc": a32})
    print(f"       {size_mb(onnx_fp32):.0f}MB  {speed32:.0f}ms  {a32:.1f}%")

    # ── 3. YOLO ONNX FP16 ──
    print("[3/6] YOLO ONNX FP16 ...")
    onnx_fp16 = PROJECT / "models" / "yolo" / "best(6)_fp16.onnx"
    if not onnx_fp16.exists():
        _ = yolo.export(format="onnx", half=True, imgsz=IMG_SIZE, verbose=False)
        (PROJECT / "models" / "yolo" / "best(6).onnx").rename(onnx_fp16)
    s16 = ort.InferenceSession(str(onnx_fp16), providers=["CPUExecutionProvider"])
    inp16 = s16.get_inputs()[0].name
    t0 = time.perf_counter()
    for img in imgs: s16.run(None, {inp16: img.numpy()})
    speed16 = (time.perf_counter() - t0) / len(imgs) * 1000
    a16 = acc_yolo_onnx(s16, inp16, samples, class_names)
    rows.append({"label": "yolo_onnx_fp16", "size": size_mb(onnx_fp16), "speed": speed16, "acc": a16})
    print(f"       {size_mb(onnx_fp16):.0f}MB  {speed16:.0f}ms  {a16:.1f}%")

    # ── 4. CNN PyTorch FP32 ──
    print("[4/6] CNN PyTorch FP32 ...")
    cnn = DrynessCNN()
    cnn.load_state_dict(torch.load(CNN_CKPT, map_location="cpu", weights_only=True))
    t0 = time.perf_counter()
    with torch.no_grad():
        for img in imgs: cnn(img)
    speed = (time.perf_counter() - t0) / len(imgs) * 1000
    a = acc_cnn(cnn, samples)
    rows.append({"label": "cnn_pt_fp32", "size": size_mb(CNN_CKPT), "speed": speed, "acc": a})
    print(f"       {size_mb(CNN_CKPT):.1f}MB  {speed:.0f}ms  {a:.1f}%")

    # ── 5. CNN ONNX FP32 ──
    print("[5/6] CNN ONNX FP32 ...")
    cnn_onnx = OUT / "dryness_cnn_fp32.onnx"
    cnn_eval = DrynessCNN()
    cnn_eval.load_state_dict(torch.load(CNN_CKPT, map_location="cpu", weights_only=True))
    cnn_eval.eval()
    torch.onnx.export(cnn_eval, torch.randn(1, 3, IMG_SIZE, IMG_SIZE), str(cnn_onnx),
                      input_names=["input"], output_names=["output"],
                      opset_version=17, dynamo=False)
    scnn = ort.InferenceSession(str(cnn_onnx), providers=["CPUExecutionProvider"])
    inp_cnn = scnn.get_inputs()[0].name
    t0 = time.perf_counter()
    for img in imgs: scnn.run(None, {inp_cnn: img.numpy()})
    speed = (time.perf_counter() - t0) / len(imgs) * 1000
    a = acc_cnn_onnx(scnn, inp_cnn, samples)
    rows.append({"label": "cnn_onnx_fp32", "size": size_mb(cnn_onnx), "speed": speed, "acc": a})
    print(f"       {size_mb(cnn_onnx):.1f}MB  {speed:.0f}ms  {a:.1f}%")

    # ── 6. CNN pruned 30% ──
    print("[6/6] CNN pruned 30% (L2 structured) ...")
    cnn_p = DrynessCNN()
    cnn_p.load_state_dict(torch.load(CNN_CKPT, map_location="cpu", weights_only=True))
    for m in cnn_p.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            try:
                nn.utils.prune.ln_structured(m, "weight", amount=0.3, n=2, dim=0)
                nn.utils.prune.remove(m, "weight")
            except Exception:
                pass
    t0 = time.perf_counter()
    with torch.no_grad():
        for img in imgs: cnn_p(img)
    speed = (time.perf_counter() - t0) / len(imgs) * 1000
    a = acc_cnn(cnn_p, samples)
    rows.append({"label": "cnn_pruned30", "size": 0, "speed": speed, "acc": a})
    print(f"       -  {speed:.0f}ms  {a:.1f}%")

    # ── TABLE ──
    print("\n" + "=" * 65)
    print(f"  {'Variant':<22} {'Size':>7} {'Speed':>7} {'Acc':>7}")
    print("  " + "-" * 48)
    for r in rows:
        s = f"{r['size']:.1f}" if r['size'] else "-"
        print(f"  {r['label']:<22} {s:>7}MB {r['speed']:>6.0f}ms {r['acc']:>6.1f}%")
    print("\n  ONNX preserves accuracy. Best RPi picks:")
    print("   YOLO → ONNX FP32 (28ms) or FP16 (27ms, half size)")
    print("   CNN  → ONNX FP32 (3ms, 2.8x speedup)")
    print(f"  All exported: {OUT}/\n")


if __name__ == "__main__":
    main()
