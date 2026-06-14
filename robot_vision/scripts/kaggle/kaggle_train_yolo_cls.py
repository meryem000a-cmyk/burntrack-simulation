#!/usr/bin/env python3
"""
Kaggle Notebook: Train YOLOv8n-cls on 17 species (no dryness).

Paste this entire file into a Kaggle notebook cell.

Required Kaggle Dataset: flora_balanced_vlm.zip (balanced/)
"""

import random
import shutil
from pathlib import Path

from ultralytics import YOLO

SEED = 42
random.seed(SEED)

DATA = Path("/kaggle/input/datasets/anwarmounir67/vlm-nano")
BALANCED_DIR = DATA / "balanced"
WORKING = Path("/kaggle/working")
YOLO_DIR = WORKING / "yolo_cls"

VAL_RATIO = 0.1
IMG_SIZE = 224
BATCH_SIZE = 64
EPOCHS = 100
MODEL_NAME = "yolov8n-cls.pt"
DEVICE = [0, 1]

SPECIES_MAP = {}
for cls_dir in sorted(BALANCED_DIR.iterdir()):
    if not cls_dir.is_dir():
        continue
    parts = cls_dir.name.rsplit("_", 2)
    if len(parts) != 3 or parts[2] not in ("dry", "not_dry"):
        continue
    species = parts[0]
    if species not in SPECIES_MAP:
        SPECIES_MAP[species] = []
    for img_path in cls_dir.glob("*.jpg"):
        SPECIES_MAP[species].append(img_path)

SPECIES = sorted(SPECIES_MAP.keys())
print(f"  Species: {len(SPECIES)}")
for s in SPECIES:
    print(f"    {s}: {len(SPECIES_MAP[s])} images")

print(f"\n  Creating YOLO dataset structure...")
if YOLO_DIR.exists():
    print(f"  Cleaning previous run: {YOLO_DIR}")
    shutil.rmtree(YOLO_DIR)
train_dir = YOLO_DIR / "train"
val_dir = YOLO_DIR / "val"
train_dir.mkdir(parents=True, exist_ok=True)
val_dir.mkdir(parents=True, exist_ok=True)

total_train, total_val = 0, 0
for species in SPECIES:
    images = SPECIES_MAP[species]
    random.shuffle(images)
    split = max(1, int(len(images) * VAL_RATIO))
    val_imgs, train_imgs = images[:split], images[split:]

    species_train = train_dir / species
    species_val = val_dir / species
    species_train.mkdir(exist_ok=True)
    species_val.mkdir(exist_ok=True)

    for i, src in enumerate(train_imgs):
        shutil.copy2(src, species_train / f"{i:04d}.jpg")
    for i, src in enumerate(val_imgs):
        shutil.copy2(src, species_val / f"{i:04d}.jpg")

    total_train += len(train_imgs)
    total_val += len(val_imgs)
    print(f"    {species}: {len(train_imgs)} train, {len(val_imgs)} val")

print(f"\n  Total: {total_train} train, {total_val} val")

print(f"\n  Loading YOLOv8n-cls...")
model = YOLO(MODEL_NAME)

print(f"\n  Training...")
results = model.train(
    data=str(YOLO_DIR),
    imgsz=IMG_SIZE,
    batch=BATCH_SIZE,
    epochs=EPOCHS,
    device=DEVICE,
    workers=4,
    seed=SEED,
    patience=20,
    save=True,
    project=str(WORKING / "yolo_cls_runs"),
    name="species_cls",
    exist_ok=True,
)
