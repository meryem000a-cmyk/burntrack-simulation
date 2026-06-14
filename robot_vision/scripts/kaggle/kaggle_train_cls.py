#!/usr/bin/env python3
"""
Flora Classification Training Script for Kaggle
=================================================
Trains YOLO11n-cls on the flora classification dataset.
Designed for Kaggle's dual Tesla T4 GPU environment.

Upload yolo_flora_cls.zip to Kaggle as a dataset, then run this script.

Key fixes over the previous detection training:
  - Uses classification mode (no box regression → no inf loss spikes)
  - Properly matches the data format (whole-image = classification, not detection)
  - Disables AMP to prevent float16 overflow on T4
  - Uses cosine LR schedule for stable convergence
"""

from pathlib import Path
import shutil

# ──────────────────────────────────────────────────────────────────────
# 1. Setup: Unzip dataset if running on Kaggle
# ──────────────────────────────────────────────────────────────────────

KAGGLE_INPUT = Path("/kaggle/input")
WORKING_DIR = Path("/kaggle/working")

# Find the dataset — adapt this to your Kaggle dataset name
dataset_candidates = [
    KAGGLE_INPUT / "yolo-flora-cls" / "yolo_flora_cls",
    KAGGLE_INPUT / "yolo-flora-cls",
    KAGGLE_INPUT / "flora-cls" / "yolo_flora_cls",
]

data_path = None
for candidate in dataset_candidates:
    if candidate.exists() and (candidate / "train").exists():
        data_path = candidate
        break

if data_path is None:
    # Try to find it dynamically
    for d in KAGGLE_INPUT.glob("*/yolo_flora_cls"):
        if (d / "train").exists():
            data_path = d
            break

if data_path is None:
    # Fallback: check if we're running locally
    local_path = Path(__file__).parent / "datasets" / "yolo_flora_cls"
    if local_path.exists():
        data_path = local_path
    else:
        raise FileNotFoundError(
            "Could not find yolo_flora_cls dataset. "
            "Make sure it's uploaded as a Kaggle dataset or available locally."
        )

print(f"📂 Dataset found at: {data_path}")

# Count classes and images
train_classes = sorted([d.name for d in (data_path / "train").iterdir() if d.is_dir()])
train_count = sum(len(list((data_path / "train" / c).glob("*"))) for c in train_classes)
val_count = sum(len(list((data_path / "val" / c).glob("*"))) for c in train_classes if (data_path / "val" / c).exists())

print(f"   Classes: {len(train_classes)}")
print(f"   Train images: {train_count:,}")
print(f"   Val images: {val_count:,}")

# ──────────────────────────────────────────────────────────────────────
# 2. Train YOLO11n-cls
# ──────────────────────────────────────────────────────────────────────

from ultralytics import YOLO

model = YOLO("yolo11n-cls.pt")  # Pretrained classification backbone

results = model.train(
    data=str(data_path),
    epochs=100,
    imgsz=224,                # Standard classification input size
    batch=128,                # Classification is lighter than detection
    device="0,1",             # Dual T4 GPUs
    amp=False,                # Disable AMP — prevents float16 overflow on T4
    cos_lr=True,              # Cosine LR annealing for stable convergence
    lr0=0.001,                # Conservative learning rate
    lrf=0.01,                 # Final LR = lr0 * lrf = 0.00001
    optimizer="AdamW",        # Better than SGD for classification
    weight_decay=0.01,        # Standard AdamW regularization
    warmup_epochs=5,          # Longer warmup for stability
    patience=20,              # Early stopping patience
    dropout=0.2,              # Regularization for 33-class problem
    save_period=10,           # Checkpoint every 10 epochs
    workers=4,
    deterministic=True,
    seed=42,
    project=str(WORKING_DIR / "runs" / "classify"),
    name="flora-cls-v1",
    exist_ok=True,
    plots=True,
    verbose=True,
)

# ──────────────────────────────────────────────────────────────────────
# 3. Evaluate
# ──────────────────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("  VALIDATION RESULTS")
print("=" * 65)

best_model_path = WORKING_DIR / "runs" / "classify" / "flora-cls-v1" / "weights" / "best.pt"
if best_model_path.exists():
    best_model = YOLO(str(best_model_path))
    metrics = best_model.val(data=str(data_path), imgsz=224, batch=128, device="0")
    print(f"\n  Top-1 Accuracy: {metrics.top1:.4f}")
    print(f"  Top-5 Accuracy: {metrics.top5:.4f}")

# ──────────────────────────────────────────────────────────────────────
# 4. Export for Raspberry Pi (INT8 TFLite)
# ──────────────────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("  EXPORTING FOR RASPBERRY PI")
print("=" * 65)

if best_model_path.exists():
    best_model = YOLO(str(best_model_path))

    # ONNX export (universal, good for testing)
    onnx_path = best_model.export(format="onnx", imgsz=224, simplify=True)
    print(f"  ONNX exported: {onnx_path}")

    # TFLite INT8 export (for Pi deployment)
    try:
        tflite_path = best_model.export(
            format="tflite",
            imgsz=224,
            int8=True,
            data=str(data_path),
        )
        print(f"  TFLite INT8 exported: {tflite_path}")
    except Exception as e:
        print(f"  ⚠️  TFLite export failed (may need tensorflow): {e}")
        print(f"  You can export locally with: yolo export model=best.pt format=tflite int8=True imgsz=224")

    # Copy exports to working dir root for easy download
    exports_dir = WORKING_DIR / "exports"
    exports_dir.mkdir(exist_ok=True)

    for ext in ["*.onnx", "*.tflite"]:
        for f in (best_model_path.parent.parent).rglob(ext):
            shutil.copy2(f, exports_dir / f.name)
            print(f"  Copied to exports: {f.name}")

print("\n✅ Training pipeline complete!")
print(f"   Best model: {best_model_path}")
print(f"   Exports: {WORKING_DIR / 'exports'}")
