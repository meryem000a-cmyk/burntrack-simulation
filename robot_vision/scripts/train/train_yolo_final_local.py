#!/usr/bin/env python3
"""
Local: Train YOLO11m-cls on 17 species with one-time .mmap cache.

Run with:  python scripts/train/train_yolo_final_local.py
Venv:      yolos/.venv/
Dataset:   datasets/final_17species/
Cache:     .cache/cache_17species.mmap
Runs:      models/yolo/runs/

Config mirrors the Kaggle version (train_yolo_final.py).
"""

import random
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from ultralytics import YOLO, settings

# ── Project Root ─────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # Vision/
DATA = PROJECT_ROOT / "datasets" / "final_17species"
CACHE = PROJECT_ROOT / ".cache" / "cache_17species.mmap"
MODELS = PROJECT_ROOT / "models" / "yolo"
RUNS = MODELS / "runs"

# ── Training Config ──────────────────────────────────────────

MODEL_NAME = "yolo11m-cls.pt"
IMG_SIZE = 224
BATCH_SIZE = 8  # small for CPU; bump to 64 if using GPU
EPOCHS = 200
PATIENCE = 30
SEED = 42
DEVICE = "cpu"  # change to [0] or [0,1] for GPU
SAVE_PERIOD = 10  # save epoch_N.pt every N epochs

# Augmentation
HSV_H, HSV_S, HSV_V = 0.05, 0.80, 0.50
DEGREES, TRANSLATE, SCALE, SHEAR = 15.0, 0.15, 0.35, 8.0
FLIPLR, FLIPUD, ERASING = 0.5, 0.1, 0.45

# LR
LR0, LRF = 0.001, 0.00001
WARMUP_EPOCHS, LABEL_SMOOTHING = 5, 0.1
MOMENTUM, WEIGHT_DECAY = 0.937, 5e-4

random.seed(SEED)

# Increase cache limit for .mmap creation (avoids "possibly exceeding avail RAM" hint)
import PIL.Image
PIL.Image.MAX_IMAGE_PIXELS = None

# Point YOLO to local paths
settings.update({
    "runs_dir": str(RUNS),
    "weights_dir": str(MODELS),
    "datasets_dir": str(DATA.parent),
})


# ══════════════════════════════════════════════════════════════
#  Create .mmap cache (one-time, visible)
# ══════════════════════════════════════════════════════════════

def create_mmap_cache():
    """Pre-load all train images into a single .mmap file with progress bar."""
    if CACHE.exists():
        print(f"  ✅ Using existing cache: {CACHE} ({CACHE.stat().st_size/1024**3:.1f} GB)")
        return

    CACHE.parent.mkdir(parents=True, exist_ok=True)
    train_dir = DATA / "train"
    images = sorted(train_dir.rglob("*.[jJ][pP][gG]")) + sorted(train_dir.rglob("*.[pP][nN][gG]"))
    n, h, w = len(images), IMG_SIZE, IMG_SIZE

    print(f"\n{'='*60}")
    print(f"  📀 One-time .mmap: caching {n} images to {CACHE}")
    print(f"{'='*60}")

    mm = np.memmap(str(CACHE), dtype=np.uint8, mode="w+", shape=(n, h, w, 3))
    failed = 0
    for i, path in enumerate(tqdm(images, desc="  Caching", unit="img")):
        img = cv2.imread(str(path))
        if img is None:
            failed += 1
            continue
        mm[i] = cv2.resize(img, (w, h))
    mm.flush()
    print(f"  ✅ Cache created: {CACHE} ({CACHE.stat().st_size/1024**3:.1f} GB, {failed} failed)")
    del mm


# ══════════════════════════════════════════════════════════════
#  Train
# ══════════════════════════════════════════════════════════════

def train():
    print("=" * 65)
    print(f"  YOLO {MODEL_NAME} — 17 Species Classifier (local)")
    print(f"  Data: {DATA}")
    print(f"  Runs: {RUNS}")
    print(f"  Device: {DEVICE}")
    print("=" * 65)

    n_species = len(list((DATA / "train").iterdir()))
    n_train = sum(1 for _ in (DATA / "train").rglob("*.[jJ][pP][gG]"))
    n_val = sum(1 for _ in (DATA / "val").rglob("*.[jJ][pP][gG]"))
    print(f"\n  {n_species} species, {n_train:,} train / {n_val:,} val")
    print(f"  {MODEL_NAME}, {EPOCHS} epochs, batch {BATCH_SIZE}, {DEVICE}")
    print(f"  Aug: HSV({HSV_S}) Rot({DEGREES}) Scale({SCALE}) Erase({ERASING})")
    print(f"  LR: cosine warmup={WARMUP_EPOCHS} smoothing={LABEL_SMOOTHING}")

    print(f"\n  Loading model...")
    model = YOLO(str(MODELS / MODEL_NAME) if (MODELS / MODEL_NAME).exists() else MODEL_NAME)

    print(f"\n  Training...")
    model.train(
        data=str(DATA),
        imgsz=IMG_SIZE, batch=BATCH_SIZE, epochs=EPOCHS,
        patience=PATIENCE, device=DEVICE, workers=0, seed=SEED,
        deterministic=False,
        # Heavy augmentation
        hsv_h=HSV_H, hsv_s=HSV_S, hsv_v=HSV_V,
        degrees=DEGREES, translate=TRANSLATE, scale=SCALE, shear=SHEAR,
        fliplr=FLIPLR, flipud=FLIPUD, erasing=ERASING,
        # LR
        lr0=LR0, lrf=LRF, warmup_epochs=WARMUP_EPOCHS, cos_lr=True,
        momentum=MOMENTUM, weight_decay=WEIGHT_DECAY, optimizer="SGD",
        label_smoothing=LABEL_SMOOTHING, dropout=0.15,
        # Save
        save=True, save_period=SAVE_PERIOD,
        project=str(RUNS), name="species_yolo11m", exist_ok=True,
        val=True, plots=True,
    )

    print(f"\n  ✅ Done! Model: {RUNS}/species_yolo11m/weights/best.pt")


# ══════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════

def main():
    print("=" * 65)
    print(f"  YOLO {MODEL_NAME} — 17 Species Classifier (local)")
    print(f"  Dataset: {DATA}")
    print(f"  Venv:    {PROJECT_ROOT / 'yolos' / '.venv'}")
    print("=" * 65)

    if not DATA.exists():
        print(f"\n  ❌ Dataset not found at {DATA}")
        print(f"     Run scripts/data/build_final_dataset.py first")
        return

    create_mmap_cache()
    train()


if __name__ == "__main__":
    main()
