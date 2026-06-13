#!/usr/bin/env python3
"""
Kaggle: Train YOLO11m-cls on 17 species with one-time .mmap cache.

Paste this entire file into a Kaggle notebook cell.
Requires dataset: final-17species uploaded to Kaggle datasets.
"""

import random
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

try:
    from ultralytics import YOLO, settings
except ImportError:
    print("  ultralytics not found — installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "ultralytics"])
    from ultralytics import YOLO, settings

# ── Config ──────────────────────────────────────────────────

DATA_SRC = Path("/kaggle/input/datasets/anwarmounir67/final-17species/final_17species")
WORKING = Path("/kaggle/working")
DATA_DST = WORKING / "final_17species"
CACHE = WORKING / "cache_17species.mmap"
PROJECT = WORKING / "yolo_runs"

MODEL_NAME = "yolo11m-cls.pt"
IMG_SIZE = 224
BATCH_SIZE = 64
EPOCHS = 200
PATIENCE = 30
SEED = 42
DEVICE = [0, 1]
SAVE_PERIOD = 10  # save epoch_N.pt every N epochs (best.pt + last.pt always saved each epoch)

# Augmentation
HSV_H, HSV_S, HSV_V = 0.05, 0.80, 0.50
DEGREES, TRANSLATE, SCALE, SHEAR = 15.0, 0.15, 0.35, 8.0
FLIPLR, FLIPUD, ERASING = 0.5, 0.1, 0.45

# LR
LR0, LRF = 0.001, 0.00001
WARMUP_EPOCHS, LABEL_SMOOTHING = 5, 0.1
MOMENTUM, WEIGHT_DECAY = 0.937, 5e-4

random.seed(SEED)

# Point YOLO's cache/runs to writable /kaggle/working
settings.update({
    "runs_dir": str(WORKING / "runs"),
    "weights_dir": str(WORKING / "weights"),
    "datasets_dir": str(WORKING / "datasets"),
})


# ══════════════════════════════════════════════════════════════
#  Step 1 — Copy dataset to working dir (writable, SSD)
# ══════════════════════════════════════════════════════════════

def copy_dataset():
    """Copy dataset symlinks to /kaggle/working/ so YOLO can write cache files."""
    if DATA_DST.exists():
        print(f"  ✅ Data already at {DATA_DST}")
        return

    print(f"\n  Copying dataset to {DATA_DST} ...")
    shutil.copytree(DATA_SRC, DATA_DST, symlinks=False, ignore_dangling_symlinks=True)
    n_train = sum(1 for _ in (DATA_DST / "train").rglob("*.jpg"))
    n_val = sum(1 for _ in (DATA_DST / "val").rglob("*.jpg"))
    print(f"  ✅ Copied {n_train:,} train + {n_val:,} val images")


# ══════════════════════════════════════════════════════════════
#  Step 2 — Create .mmap cache (one-time, visible)
# ══════════════════════════════════════════════════════════════

def create_mmap_cache():
    """Pre-load all train images into a single .mmap file with progress bar."""
    if CACHE.exists():
        print(f"  ✅ Using existing cache: {CACHE} ({CACHE.stat().st_size/1024**3:.1f} GB)")
        return

    train_dir = DATA_DST / "train"
    images = sorted(train_dir.rglob("*.jpg"))
    n, h, w = len(images), IMG_SIZE, IMG_SIZE

    print(f"\n{'='*60}")
    print(f"  📀 One-time .mmap: caching {n} images to {CACHE.name}")
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
    del mm  # release file handle


# ══════════════════════════════════════════════════════════════
#  Step 3 — Train
# ══════════════════════════════════════════════════════════════

def train():
    print("=" * 65)
    print(f"  YOLO {MODEL_NAME} — 17 Species Classifier")
    print(f"  Data: {DATA_DST}")
    print(f"  Cache: {CACHE}")
    print("=" * 65)

    n_species = len(list((DATA_DST / "train").iterdir()))
    n_train = sum(1 for _ in (DATA_DST / "train").rglob("*.jpg"))
    n_val = sum(1 for _ in (DATA_DST / "val").rglob("*.jpg"))
    print(f"\n  {n_species} species, {n_train:,} train / {n_val:,} val")
    print(f"  {MODEL_NAME}, {EPOCHS} epochs, batch {BATCH_SIZE}, {DEVICE}")
    print(f"  Aug: HSV({HSV_S}) Rot({DEGREES}) Scale({SCALE}) Erase({ERASING})")
    print(f"  LR: cosine warmup={WARMUP_EPOCHS} smoothing={LABEL_SMOOTHING}")

    print(f"\n  Loading model...")
    model = YOLO(MODEL_NAME)

    print(f"\n  Training...")
    model.train(
        data=str(DATA_DST),
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
        project=str(PROJECT), name="species_yolo11m", exist_ok=True,
        val=True, plots=True,
    )

    print(f"\n  ✅ Done! Model: {PROJECT}/species_yolo11m/weights/best.pt")


# ══════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════

def main():
    print("=" * 65)
    print(f"  YOLO {MODEL_NAME} — 17 Species Classifier")
    print(f"  Source: {DATA_SRC}")
    print("=" * 65)

    copy_dataset()
    create_mmap_cache()
    train()


if __name__ == "__main__":
    main()
