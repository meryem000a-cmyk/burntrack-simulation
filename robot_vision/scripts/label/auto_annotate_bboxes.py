#!/usr/bin/env python3
"""
Flora Auto-Annotator: Generate Real Bounding Boxes (Local Version)
===================================================================
Uses YOLO-World (open-vocabulary detector) to find actual plant regions
in each image, then assigns the existing VLM-derived class labels
(species + curing state) to those bounding boxes.

Replaces dummy full-frame annotations (0.5 0.5 1.0 1.0) with
real, localized bounding boxes.

After annotation, re-zips the dataset so you can upload the improved
version to Colab/Kaggle.

Usage:
    # Using the flora_env venv:
    ./flora_env/bin/python auto_annotate_bboxes.py

    # Custom paths:
    ./flora_env/bin/python auto_annotate_bboxes.py \
        --data datasets/yolo_flora \
        --output yolo_flora_640_annotated.zip \
        --device cpu
"""

import argparse
import gc
import os
import shutil
import sys
import time
from pathlib import Path
from collections import defaultdict

# ──────────────────────────────────────────────────────────────────────
# Plant detection prompts for YOLO-World
# ──────────────────────────────────────────────────────────────────────
PLANT_PROMPTS = [
    "plant", "tree", "shrub", "flower", "grass",
    "succulent", "cactus", "bush", "leaf", "vegetation",
]

# Minimum bbox area as fraction of image area — skip tiny detections
MIN_BBOX_AREA_RATIO = 0.02
# Confidence threshold for YOLO-World detections
CONFIDENCE_THRESHOLD = 0.15
# If YOLO-World finds nothing, fall back to a padded center crop
FALLBACK_BBOX = "0.5 0.5 0.85 0.85"


def is_dummy_bbox(parts: list[str]) -> bool:
    """Check if a label line has a dummy full-frame or fallback bbox."""
    if len(parts) < 5:
        return True
    try:
        cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
    except ValueError:
        return True

    # Full-frame dummy: 0.5 0.5 1.0 1.0
    if (abs(cx - 0.5) < 1e-3 and abs(cy - 0.5) < 1e-3
            and abs(w - 1.0) < 1e-3 and abs(h - 1.0) < 1e-3):
        return True
    # Fallback dummy: 0.5 0.5 0.85 0.85
    if (abs(cx - 0.5) < 1e-3 and abs(cy - 0.5) < 1e-3
            and abs(w - 0.85) < 1e-3 and abs(h - 0.85) < 1e-3):
        return True
    return False


def read_existing_label(label_path: Path) -> int | None:
    """Read the class ID from an existing YOLO label file."""
    try:
        text = label_path.read_text().strip()
        if not text:
            return None
        first_line = text.split("\n")[0]
        class_id = int(first_line.split()[0])
        return class_id
    except (ValueError, IndexError, OSError):
        return None


def needs_annotation(label_path: Path) -> bool:
    """Return True if this label needs re-annotation (dummy or missing bbox)."""
    try:
        text = label_path.read_text().strip()
        if not text:
            return False  # No label at all, skip
        first_line = text.split("\n")[0]
        parts = first_line.split()
        return is_dummy_bbox(parts)
    except OSError:
        return False


def auto_annotate(
    data_root: str,
    batch_size: int = 4,
    device: str = "cpu",
    model_name: str = "yolov8s-worldv2",
    output_zip: str | None = None,
):
    """Run YOLO-World on all images with dummy labels and write real bboxes."""
    from ultralytics import YOLO

    data_root = Path(data_root)

    print("=" * 65)
    print("  Flora Auto-Annotator (Local)")
    print("=" * 65)
    print(f"  Dataset:  {data_root}")
    print(f"  Device:   {device}")
    print(f"  Model:    {model_name}")
    print(f"  Batch:    {batch_size}")
    if output_zip:
        print(f"  Output:   {output_zip}")
    print("=" * 65)

    # ── Load YOLO-World ──
    print("\n📦 Loading YOLO-World model (will download if needed)...")
    world_model = YOLO(f"{model_name}.pt")
    world_model.set_classes(PLANT_PROMPTS)
    print(f"   Prompts: {PLANT_PROMPTS}")

    stats = defaultdict(int)

    for split in ("train", "val"):
        img_dir = data_root / "images" / split
        lbl_dir = data_root / "labels" / split

        if not img_dir.exists():
            print(f"\n⚠️  {img_dir} not found, skipping")
            continue

        # Get all images
        all_image_files = sorted([
            f for f in img_dir.iterdir()
            if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
        ])

        # Filter to only those needing annotation (dummy bboxes)
        image_files = []
        for f in all_image_files:
            lbl_file = lbl_dir / (f.stem + ".txt")
            if lbl_file.exists() and needs_annotation(lbl_file):
                image_files.append(f)

        already_done = len(all_image_files) - len(image_files)
        print(f"\n📂 {split}: {len(image_files)} pending | "
              f"{already_done} already annotated | "
              f"{len(all_image_files)} total")

        if not image_files:
            print("   ✅ All labels already have real bboxes.")
            continue

        start_time = time.time()

        # Process in batches
        for batch_start in range(0, len(image_files), batch_size):
            batch_paths = image_files[batch_start:batch_start + batch_size]
            batch_str_paths = [str(p) for p in batch_paths]

            # Run YOLO-World inference
            results = world_model.predict(
                batch_str_paths,
                conf=CONFIDENCE_THRESHOLD,
                device=device,
                verbose=False,
                imgsz=640,
                stream=True,  # memory-efficient streaming
            )

            for img_path, result in zip(batch_paths, results):
                label_path = lbl_dir / (img_path.stem + ".txt")

                # Read existing class ID
                class_id = read_existing_label(label_path)
                if class_id is None:
                    stats["skipped_no_label"] += 1
                    continue

                img_h, img_w = result.orig_shape

                # Get YOLO-World detections
                boxes = result.boxes
                if boxes is not None and len(boxes) > 0:
                    # Pick the detection with the largest area
                    best_idx = -1
                    best_area = 0

                    for i in range(len(boxes)):
                        xyxy = boxes.xyxy[i].cpu().numpy()
                        x1, y1, x2, y2 = xyxy
                        area = (x2 - x1) * (y2 - y1)
                        img_area = img_w * img_h

                        if area / img_area < MIN_BBOX_AREA_RATIO:
                            continue

                        if area > best_area:
                            best_area = area
                            best_idx = i

                    if best_idx >= 0:
                        xyxy = boxes.xyxy[best_idx].cpu().numpy()
                        x1, y1, x2, y2 = xyxy

                        cx = max(0, min(1, ((x1 + x2) / 2) / img_w))
                        cy = max(0, min(1, ((y1 + y2) / 2) / img_h))
                        w = max(0.01, min(1, (x2 - x1) / img_w))
                        h = max(0.01, min(1, (y2 - y1) / img_h))

                        bbox_str = f"{cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
                        stats["detected"] += 1
                    else:
                        bbox_str = FALLBACK_BBOX
                        stats["fallback"] += 1
                else:
                    bbox_str = FALLBACK_BBOX
                    stats["fallback"] += 1

                label_path.write_text(f"{class_id} {bbox_str}\n")
                stats["total"] += 1

            # Progress
            done = min(batch_start + batch_size, len(image_files))
            elapsed = time.time() - start_time
            rate = done / elapsed if elapsed > 0 else 0
            eta_s = (len(image_files) - done) / rate if rate > 0 else 0
            eta_m = eta_s / 60

            print(
                f"   {done:>6}/{len(image_files)} "
                f"({rate:.1f} img/s) "
                f"[det: {stats['detected']}, fb: {stats['fallback']}] "
                f"ETA: {eta_m:.0f}m",
                end="\r", flush=True
            )

            # Free memory periodically
            del results
            gc.collect()

        print()  # newline after progress

    # ── Summary ──
    print("\n" + "=" * 65)
    print("  AUTO-ANNOTATION SUMMARY")
    print("=" * 65)
    print(f"  Total labels written        : {stats['total']:,}")
    print(f"  Plant detected (real bbox)  : {stats['detected']:,}")
    print(f"  Fallback (center crop)      : {stats['fallback']:,}")
    print(f"  Skipped (no existing label) : {stats['skipped_no_label']:,}")
    print("=" * 65)

    # ── Clean up model ──
    del world_model
    gc.collect()

    # ── Re-zip the dataset ──
    if output_zip:
        rezip_dataset(data_root, output_zip)

    print("\n✅ Done! Labels now have real bounding boxes.")


def rezip_dataset(data_root: Path, output_zip: str):
    """Re-zip the dataset folder into a new zip file."""
    data_root = Path(data_root)
    output_path = Path(output_zip)

    print(f"\n📦 Re-zipping dataset → {output_path}")
    print("   This may take a while for large datasets...")

    # Remove .zip extension if present (shutil adds it)
    archive_base = str(output_path).removesuffix(".zip")

    start = time.time()
    result = shutil.make_archive(
        base_name=archive_base,
        format="zip",
        root_dir=data_root.parent,
        base_dir=data_root.name,
    )
    elapsed = time.time() - start

    size_gb = Path(result).stat().st_size / (1024**3)
    print(f"   ✅ Created {result} ({size_gb:.2f} GB) in {elapsed:.0f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Auto-annotate flora images with real bounding boxes using YOLO-World"
    )
    parser.add_argument(
        "--data", type=str,
        default="datasets/yolo_flora",
        help="Path to YOLO dataset root (with images/ and labels/ subdirs)",
    )
    parser.add_argument(
        "--output", type=str,
        default="yolo_flora_640.zip",
        help="Output zip file path (overwrites existing). Set to '' to skip zipping.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=4,
        help="Batch size for inference (keep low on CPU)",
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="Device: 'cpu' or '0' for CUDA GPU",
    )
    parser.add_argument(
        "--model", type=str, default="yolov8s-worldv2",
        help="YOLO-World model variant (s/m/l/x)",
    )
    parser.add_argument(
        "--no-zip", action="store_true",
        help="Skip re-zipping after annotation",
    )
    args = parser.parse_args()

    output = None if args.no_zip else args.output
    auto_annotate(
        data_root=args.data,
        batch_size=args.batch_size,
        device=args.device,
        model_name=args.model,
        output_zip=output,
    )
