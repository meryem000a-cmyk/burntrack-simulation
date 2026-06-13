#!/usr/bin/env python3
"""
Flora Dataset In-Place Resizer
==============================
Resizes all images in the YOLO flora dataset so the longest side is at most
TARGET_SIZE pixels. Operates IN-PLACE to conserve disk space.

Features:
  - Parallel processing (uses all CPU cores)
  - Skips images already at or below target size
  - Detects and removes corrupt images + their matching label files
  - Progress bar via tqdm
  - Dry-run mode to preview what would happen

Usage:
  python resize_dataset.py                  # Run the resize
  python resize_dataset.py --dry-run        # Preview without changing anything
  python resize_dataset.py --target 1024    # Use 1024px instead of 640px
"""

import argparse
import os
import sys
import time
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, ImageFile

# Allow opening huge images and truncated files (for detection)
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────
DATASET_ROOT = Path(__file__).parent / "datasets" / "yolo_flora"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}
JPEG_QUALITY = 85  # Good balance of quality vs size


@dataclass
class Stats:
    """Accumulator for processing statistics."""
    total: int = 0
    resized: int = 0
    skipped: int = 0
    corrupt: int = 0
    errors: int = 0
    bytes_before: int = 0
    bytes_after: int = 0
    corrupt_files: list = field(default_factory=list)
    error_files: list = field(default_factory=list)


def get_label_path(image_path: Path) -> Path:
    """Given an image path, return the corresponding label .txt path."""
    # images/train/foo.jpg -> labels/train/foo.txt
    parts = list(image_path.parts)
    idx = parts.index("images")
    parts[idx] = "labels"
    label_path = Path(*parts).with_suffix(".txt")
    return label_path


def process_single_image(args: tuple) -> dict:
    """
    Process a single image. Returns a dict with results.
    Runs in a worker process.
    """
    image_path_str, target_size, dry_run = args
    image_path = Path(image_path_str)
    result = {
        "path": image_path_str,
        "status": "unknown",
        "bytes_before": 0,
        "bytes_after": 0,
    }

    try:
        result["bytes_before"] = image_path.stat().st_size

        with Image.open(image_path) as img:
            # Force load to detect truncation
            img.load()

            w, h = img.size
            longest = max(w, h)

            if longest <= target_size:
                result["status"] = "skipped"
                result["bytes_after"] = result["bytes_before"]
                return result

            if dry_run:
                # Estimate new size (rough: proportional to pixel count reduction)
                ratio = target_size / longest
                result["status"] = "would_resize"
                result["bytes_after"] = int(result["bytes_before"] * ratio * ratio * 0.8)
                return result

            # Resize maintaining aspect ratio
            img.thumbnail((target_size, target_size), Image.LANCZOS)

            # Convert to RGB if needed (removes alpha, handles palette mode)
            if img.mode not in ("RGB", "L"):
                img = img.convert("RGB")

            # Save in-place as JPEG
            output_path = image_path.with_suffix(".jpg")
            img.save(output_path, "JPEG", quality=JPEG_QUALITY, optimize=True)

            # If original was .png/.webp etc, remove the old file
            if output_path != image_path:
                image_path.unlink()

            result["bytes_after"] = output_path.stat().st_size
            result["status"] = "resized"

    except (OSError, SyntaxError, ValueError, Image.DecompressionBombError) as e:
        result["status"] = "corrupt"
        result["error"] = str(e)
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)

    return result


def collect_images(dataset_root: Path) -> list[Path]:
    """Collect all image files from the dataset."""
    images = []
    images_dir = dataset_root / "images"
    if not images_dir.exists():
        print(f"ERROR: {images_dir} does not exist!")
        sys.exit(1)

    for path in images_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            images.append(path)

    return sorted(images)


def main():
    parser = argparse.ArgumentParser(
        description="Resize flora dataset images in-place for efficient YOLO training"
    )
    parser.add_argument(
        "--target", type=int, default=640,
        help="Maximum size for the longest side (default: 640)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Preview what would happen without modifying files"
    )
    parser.add_argument(
        "--workers", type=int, default=None,
        help="Number of parallel workers (default: CPU count)"
    )
    args = parser.parse_args()

    target_size = args.target
    dry_run = args.dry_run
    workers = args.workers or os.cpu_count()

    print("=" * 65)
    print("  Flora Dataset In-Place Resizer")
    print("=" * 65)
    print(f"  Dataset root : {DATASET_ROOT}")
    print(f"  Target size  : {target_size}px (longest side)")
    print(f"  JPEG quality : {JPEG_QUALITY}")
    print(f"  Workers      : {workers}")
    print(f"  Mode         : {'DRY RUN (no changes)' if dry_run else 'LIVE (modifying files)'}")
    print("=" * 65)

    # Collect images
    print("\n📁 Scanning for images...")
    images = collect_images(DATASET_ROOT)
    print(f"   Found {len(images):,} images")

    if not images:
        print("No images found. Exiting.")
        return

    # Process with progress
    print(f"\n🔄 Processing images with {workers} workers...")
    stats = Stats(total=len(images))

    try:
        from tqdm import tqdm
        has_tqdm = True
    except ImportError:
        has_tqdm = False
        print("   (install tqdm for a progress bar: pip install tqdm)")

    task_args = [(str(img), target_size, dry_run) for img in images]
    start_time = time.time()

    results = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_single_image, arg): arg for arg in task_args}

        if has_tqdm:
            pbar = tqdm(total=len(futures), unit="img", desc="   Resizing")

        for future in as_completed(futures):
            result = future.result()
            results.append(result)

            if result["status"] in ("resized", "would_resize"):
                stats.resized += 1
                stats.bytes_before += result["bytes_before"]
                stats.bytes_after += result["bytes_after"]
            elif result["status"] == "skipped":
                stats.skipped += 1
                stats.bytes_before += result["bytes_before"]
                stats.bytes_after += result["bytes_after"]
            elif result["status"] == "corrupt":
                stats.corrupt += 1
                stats.corrupt_files.append(result["path"])
            elif result["status"] == "error":
                stats.errors += 1
                stats.error_files.append(
                    f"{result['path']}: {result.get('error', '?')}"
                )

            if has_tqdm:
                pbar.update(1)

        if has_tqdm:
            pbar.close()

    elapsed = time.time() - start_time

    # Handle corrupt files — remove them and their labels
    if stats.corrupt_files:
        print(f"\n🗑️  Cleaning up {len(stats.corrupt_files)} corrupt images...")
        for corrupt_path_str in stats.corrupt_files:
            corrupt_path = Path(corrupt_path_str)
            label_path = get_label_path(corrupt_path)

            if dry_run:
                print(f"   Would delete: {corrupt_path.name}")
                if label_path.exists():
                    print(f"   Would delete label: {label_path.name}")
            else:
                try:
                    corrupt_path.unlink(missing_ok=True)
                    print(f"   Deleted: {corrupt_path.name}")
                except Exception as e:
                    print(f"   Failed to delete {corrupt_path.name}: {e}")

                try:
                    if label_path.exists():
                        label_path.unlink()
                        print(f"   Deleted label: {label_path.name}")
                except Exception as e:
                    print(f"   Failed to delete label {label_path.name}: {e}")

    # Summary
    action = "Would resize" if dry_run else "Resized"
    before_gb = stats.bytes_before / (1024**3)
    after_gb = stats.bytes_after / (1024**3)
    saved_gb = before_gb - after_gb

    print("\n" + "=" * 65)
    print("  SUMMARY")
    print("=" * 65)
    print(f"  Total images     : {stats.total:,}")
    print(f"  {action:16s} : {stats.resized:,}")
    print(f"  Already ≤{target_size}px  : {stats.skipped:,}")
    print(f"  Corrupt (removed): {stats.corrupt}")
    print(f"  Errors           : {stats.errors}")
    print(f"  ─────────────────────────────────────")
    print(f"  Size before      : {before_gb:.2f} GB")
    print(f"  Size after       : {after_gb:.2f} GB")
    print(f"  Space saved      : {saved_gb:.2f} GB ({saved_gb/before_gb*100:.0f}%)" if before_gb > 0 else "")
    print(f"  Time elapsed     : {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"  Speed            : {stats.total/elapsed:.0f} images/sec")
    print("=" * 65)

    if stats.error_files:
        print("\n⚠️  Errors encountered:")
        for err in stats.error_files[:20]:
            print(f"   {err}")
        if len(stats.error_files) > 20:
            print(f"   ... and {len(stats.error_files) - 20} more")

    if dry_run:
        print("\n💡 This was a dry run. To actually resize, run without --dry-run")
    else:
        print("\n✅ Done! Your dataset is now optimized for YOLO training.")
        print("   Next steps:")
        print(f"   1. Delete the old zip:  rm yolo_flora.zip")
        print(f"   2. Re-zip:  cd datasets && zip -r ../yolo_flora_640.zip yolo_flora/")
        print(f"   3. Upload yolo_flora_640.zip to Kaggle")
        print(f"   4. Train with imgsz=640 (much better for plant features)")


if __name__ == "__main__":
    main()
