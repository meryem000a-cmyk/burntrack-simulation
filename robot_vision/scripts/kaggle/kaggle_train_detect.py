#!/usr/bin/env python3
"""
Flora Detection Training — Kaggle Script (Fixed)
==================================================
Two-phase pipeline:
  Phase 1: Auto-annotate images with real bounding boxes (YOLO-World)
  Phase 2: Train YOLO11n detection on properly annotated data

Fixes over previous run:
  - Real bounding boxes instead of full-frame (0.5 0.5 1.0 1.0)
  - AMP disabled to prevent float16 overflow → no more inf box_loss
  - Conservative learning rate for stable convergence
  - Proper optimizer settings

Upload yolo_flora_640.zip to Kaggle, then run this as a notebook.
"""

import shutil
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────
# 0. Setup paths
# ──────────────────────────────────────────────────────────────────────

KAGGLE_INPUT = Path("/kaggle/input")
WORKING_DIR = Path("/kaggle/working")
DATASET_DIR = WORKING_DIR / "yolo_flora"

# ──────────────────────────────────────────────────────────────────────
# 1. Unpack dataset to working dir (we need write access for labels)
# ──────────────────────────────────────────────────────────────────────

print("=" * 65)
print("  Phase 0: Preparing Dataset")
print("=" * 65)

# Kaggle auto-extracts uploaded zips into /kaggle/input/<dataset-name>/
# We need to copy to /kaggle/working/ for write access (to fix labels)
KAGGLE_SOURCE = Path("/kaggle/input/datasets/anwarmounir67/cokaiscool/yolo_flora")

if DATASET_DIR.exists() and (DATASET_DIR / "images").exists():
    print(f"✅ Dataset already at {DATASET_DIR}")
else:
    if not KAGGLE_SOURCE.exists():
        raise FileNotFoundError(
            f"Dataset not found at {KAGGLE_SOURCE}. "
            "Make sure the dataset is attached to the notebook."
        )

    print(f"📁 Copying {KAGGLE_SOURCE} → {DATASET_DIR} (need write access for labels)...")
    shutil.copytree(KAGGLE_SOURCE, DATASET_DIR)
    print(f"✅ Copied to {DATASET_DIR}")

# Verify structure
assert (DATASET_DIR / "images" / "train").exists(), "Missing images/train"
assert (DATASET_DIR / "labels" / "train").exists(), "Missing labels/train"

train_imgs = len(list((DATASET_DIR / "images" / "train").glob("*")))
val_imgs = len(list((DATASET_DIR / "images" / "val").glob("*")))
print(f"   Train: {train_imgs:,} images")
print(f"   Val:   {val_imgs:,} images")

# ──────────────────────────────────────────────────────────────────────
# 2. Write data.yaml with correct paths for Kaggle
# ──────────────────────────────────────────────────────────────────────

DATA_YAML = WORKING_DIR / "data.yaml"

# 34 classes: 17 species × 2 curing states
# Using the ordering from data.yaml (the one used in the original Kaggle run)
CLASS_NAMES = {
    0: "adansonia_not_dry", 1: "adansonia_dry",
    2: "acacia_not_dry", 3: "acacia_dry",
    4: "vachellia_not_dry", 5: "vachellia_dry",
    6: "senegalia_not_dry", 7: "senegalia_dry",
    8: "combretum_not_dry", 9: "combretum_dry",
    10: "brachystegia_not_dry", 11: "brachystegia_dry",
    12: "colophospermum_not_dry", 13: "colophospermum_dry",
    14: "ficus_not_dry", 15: "ficus_dry",
    16: "khaya_not_dry", 17: "khaya_dry",
    18: "macaranga_not_dry", 19: "macaranga_dry",
    20: "euphorbia_not_dry", 21: "euphorbia_dry",
    22: "aloe_not_dry", 23: "aloe_dry",
    24: "protea_not_dry", 25: "protea_dry",
    26: "erica_not_dry", 27: "erica_dry",
    28: "themeda_not_dry", 29: "themeda_dry",
    30: "andropogon_not_dry", 31: "andropogon_dry",
    32: "tamarix_not_dry", 33: "tamarix_dry",
}

yaml_lines = [
    f"path: {DATASET_DIR}",
    "train: images/train",
    "val: images/val",
    "",
    "names:",
]
for cid, name in CLASS_NAMES.items():
    yaml_lines.append(f"  {cid}: {name}")

DATA_YAML.write_text("\n".join(yaml_lines) + "\n")
print(f"\n📝 Wrote {DATA_YAML}")

# ──────────────────────────────────────────────────────────────────────
# 3. Phase 1: Auto-annotate with YOLO-World (real bounding boxes)
# ──────────────────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("  Phase 1: Auto-Annotating with YOLO-World")
print("=" * 65)

from ultralytics import YOLO
import time
import numpy as np
import gc

PLANT_PROMPTS = [
    "plant", "tree", "shrub", "flower", "grass",
    "succulent", "cactus", "bush", "leaf", "vegetation",
]
MIN_BBOX_AREA_RATIO = 0.02
CONFIDENCE_THRESHOLD = 0.15
FALLBACK_BBOX = "0.5 0.5 0.85 0.85"

print("📦 Loading YOLO-World (yolov8x-worldv2)...")
world_model = YOLO("yolov8x-worldv2.pt")
world_model.set_classes(PLANT_PROMPTS)

total_detected = 0
total_fallback = 0
total_processed = 0

for split in ("train", "val"):
    img_dir = DATASET_DIR / "images" / split
    lbl_dir = DATASET_DIR / "labels" / split
    
    all_image_files = sorted([
        f for f in img_dir.iterdir()
        if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    ])
    
    # Check for already-processed labels to allow resuming
    image_files = []
    for f in all_image_files:
        lbl_file = lbl_dir / (f.stem + ".txt")
        try:
            text = lbl_file.read_text().strip()
            if text:
                parts = text.split("\n")[0].split()
                if len(parts) == 5:
                    # Check if it is a bad dummy box (full picture: "0.5 0.5 1.0 1.0" or "0.5 0.5 0.85 0.85")
                    cx, cy, w, h = map(float, parts[1:])
                    is_dummy = (
                        (abs(cx - 0.5) < 1e-3 and abs(cy - 0.5) < 1e-3 and abs(w - 1.0) < 1e-3 and abs(h - 1.0) < 1e-3) or
                        (abs(cx - 0.5) < 1e-3 and abs(cy - 0.5) < 1e-3 and abs(w - 0.85) < 1e-3 and abs(h - 0.85) < 1e-3)
                    )
                    if not is_dummy:
                        continue  # Skip auto-annotation only if it is a real, high-quality YOLO-World box!
            image_files.append(f)
        except Exception:
            pass

    print(f"\n🔍 Auto-annotating {split}: {len(image_files)} pending images (out of {len(all_image_files)} total)...")
    if not image_files:
        print("✅ Split already fully annotated.")
        continue
    start = time.time()
    
    batch_size = 32
    for i in range(0, len(image_files), batch_size):
        batch = image_files[i:i+batch_size]
        
        results = world_model.predict(
            [str(p) for p in batch],
            conf=CONFIDENCE_THRESHOLD,
            device="0",
            verbose=False,
            imgsz=640,
            stream=True,
        )
        
        for img_path, result in zip(batch, results):
            label_path = lbl_dir / (img_path.stem + ".txt")
            
            # Read existing class ID
            try:
                text = label_path.read_text().strip()
                if not text:
                    continue
                class_id = int(text.split("\n")[0].split()[0])
            except (FileNotFoundError, ValueError):
                continue
            
            img_h, img_w = result.orig_shape
            boxes = result.boxes
            
            if boxes is not None and len(boxes) > 0:
                # Pick largest detection
                best_idx, best_area = -1, 0
                for j in range(len(boxes)):
                    xyxy = boxes.xyxy[j].cpu().numpy()
                    area = (xyxy[2]-xyxy[0]) * (xyxy[3]-xyxy[1])
                    if area / (img_w*img_h) >= MIN_BBOX_AREA_RATIO and area > best_area:
                        best_area = area
                        best_idx = j
                
                if best_idx >= 0:
                    x1, y1, x2, y2 = boxes.xyxy[best_idx].cpu().numpy()
                    cx = max(0, min(1, ((x1+x2)/2) / img_w))
                    cy = max(0, min(1, ((y1+y2)/2) / img_h))
                    w = max(0.01, min(1, (x2-x1) / img_w))
                    h = max(0.01, min(1, (y2-y1) / img_h))
                    bbox_str = f"{cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
                    total_detected += 1
                else:
                    bbox_str = FALLBACK_BBOX
                    total_fallback += 1
            else:
                bbox_str = FALLBACK_BBOX
                total_fallback += 1
            
            label_path.write_text(f"{class_id} {bbox_str}\n")
            total_processed += 1
        
        done = min(i+batch_size, len(image_files))
        elapsed = time.time() - start
        rate = done / elapsed if elapsed > 0 else 0
        print(f"   {done:>6}/{len(image_files)} ({rate:.0f} img/s)", end="\r", flush=True)
        
        # Clean up memory to prevent Kaggle RAM spikes
        del results
        del batch
        gc.collect()
    
    print()

print(f"\n✅ Auto-annotation complete!")
print(f"   Processed:  {total_processed:,}")
print(f"   Real bbox:  {total_detected:,}")
print(f"   Fallback:   {total_fallback:,}")

# Clean up YOLO-World to free GPU memory
del world_model
import torch
torch.cuda.empty_cache()

# ──────────────────────────────────────────────────────────────────────
# 4. Phase 2: Train YOLO11n Detection
# ──────────────────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("  Phase 2: Training YOLO11n Detection")
print("=" * 65)

model = YOLO("yolo11n.pt")

results = model.train(
    data=str(DATA_YAML),
    epochs=150,
    imgsz=640,
    batch=64,
    device="0",                 # Single GPU
    amp=False,                  # ← KEY FIX: disable AMP to prevent inf loss spikes
    cos_lr=True,
    lr0=0.005,                  # ← Lower than default 0.01 for stability
    lrf=0.01,
    optimizer="SGD",            # SGD is proven for YOLO detection
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=5,            # Longer warmup for stability
    warmup_bias_lr=0.05,
    patience=30,
    save_period=10,
    close_mosaic=15,
    workers=4,
    deterministic=True,
    seed=42,
    cache="disk",               # ← Disk cache: deterministic + avoids RAM issues
    augment=True,
    mosaic=1.0,
    fliplr=0.5,
    scale=0.5,
    erasing=0.4,
    auto_augment="randaugment",
    project=str(WORKING_DIR / "runs" / "detect"),
    name="flora-640-v2",
    exist_ok=True,
    plots=True,
    verbose=True,
)

# ──────────────────────────────────────────────────────────────────────
# 5. Export for Raspberry Pi
# ──────────────────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("  Exporting for Raspberry Pi")
print("=" * 65)

best_path = WORKING_DIR / "runs" / "detect" / "flora-640-v2" / "weights" / "best.pt"

if best_path.exists():
    best_model = YOLO(str(best_path))
    
    # Validate
    metrics = best_model.val(data=str(DATA_YAML), imgsz=640, batch=64, device="0")
    print(f"\n  mAP50:    {metrics.box.map50:.4f}")
    print(f"  mAP50-95: {metrics.box.map:.4f}")
    
    # ONNX
    onnx_path = best_model.export(format="onnx", imgsz=640, simplify=True)
    print(f"  ONNX: {onnx_path}")
    
    # TFLite INT8
    try:
        tflite_path = best_model.export(
            format="tflite", imgsz=320, int8=True, data=str(DATA_YAML)
        )
        print(f"  TFLite INT8: {tflite_path}")
    except Exception as e:
        print(f"  ⚠️  TFLite failed: {e}")
    
    # Copy to easy download location
    exports_dir = WORKING_DIR / "exports"
    exports_dir.mkdir(exist_ok=True)
    shutil.copy2(best_path, exports_dir / "best.pt")
    for ext in ("*.onnx", "*.tflite"):
        for f in (best_path.parent.parent).rglob(ext):
            shutil.copy2(f, exports_dir / f.name)
    print(f"\n  Exports copied to: {exports_dir}")

print("\n✅ Pipeline complete!")
