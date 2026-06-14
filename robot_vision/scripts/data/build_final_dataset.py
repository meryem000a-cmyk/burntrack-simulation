#!/usr/bin/env python3
"""
Build consolidated 17-species dataset for YOLO classification.

- Source: datasets/yolo_flora_cls/{train,val}/  (32,829 images total)
- Output: datasets/final_17species/{train,val,test}/{species_name}/
- Uses symlinks (no copies — saves 30 GB+)
- Stratified 80/10/10 split per species
- eval_images/ is NOT touched — remains independent held-out test set

Usage:
    ./flora_env/bin/python scripts/data/build_final_dataset.py
    # or with the yolos venv:
    ~/Documents/Vision/yolos/.venv/bin/python scripts/data/build_final_dataset.py
"""

import random
from pathlib import Path
from collections import defaultdict

SEED = 42
random.seed(SEED)

SRC_TRAIN = Path("datasets/yolo_flora_cls/train")
SRC_VAL = Path("datasets/yolo_flora_cls/val")
OUT = Path("datasets/final_17species")

TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
# TEST_RATIO = 0.10 (remaining)


def extract_species(folder_name: str) -> str:
    """Convert 'acacia_dry' or 'acacia_not_dry' → 'acacia'."""
    return folder_name.rsplit("_", 2)[0]


def collect_images(src: Path) -> dict[str, list[Path]]:
    """Collect all image paths grouped by species from a directory tree.

    Expects structure: src/{species_state}/*.jpg
    """
    species_map: dict[str, list[Path]] = defaultdict(list)
    for cls_dir in sorted(src.iterdir()):
        if not cls_dir.is_dir():
            continue
        species = extract_species(cls_dir.name)
        for img_path in cls_dir.glob("*.jpg"):
            species_map[species].append(img_path)
    return dict(species_map)


def split_species_images(
    images: list[Path],
) -> tuple[list[Path], list[Path], list[Path]]:
    """Stratified 80/10/10 split for one species."""
    random.shuffle(images)
    n = len(images)
    n_train = max(1, int(n * TRAIN_RATIO))
    n_val = max(1, int(n * VAL_RATIO))
    train = images[:n_train]
    val = images[n_train : n_train + n_val]
    test = images[n_train + n_val :]
    return train, val, test


def make_symlinks(src_paths: list[Path], dst_dir: Path):
    """Create numbered symlinks in dst_dir pointing to src_paths."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    for i, src in enumerate(src_paths):
        link = dst_dir / f"{i:05d}.jpg"
        if not link.exists():
            link.symlink_to(src.resolve())


def main():
    print("=" * 60)
    print("  Building Final 17-Species Dataset")
    print("=" * 60)

    # Collect all images
    print("\n  Collecting images...")
    train_map = collect_images(SRC_TRAIN)
    val_map = collect_images(SRC_VAL)
    all_species = sorted(set(train_map.keys()) | set(val_map.keys()))

    print(f"  Found {len(all_species)} species from {len(train_map)} train classes")
    print(f"  Species: {', '.join(all_species)}")

    # Merge train + val images per species
    merged: dict[str, list[Path]] = {}
    for sp in all_species:
        merged[sp] = train_map.get(sp, []) + val_map.get(sp, [])

    total = sum(len(v) for v in merged.values())
    print(f"\n  Total images available: {total:,}")
    print()

    # Split per species
    train_all: list[Path] = []
    val_all: list[Path] = []
    test_all: list[Path] = []

    print(f"  {'Species':<20} {'Train':>7} {'Val':>6} {'Test':>6} {'Total':>7}")
    print(f"  {'-'*20} {'-'*7} {'-'*6} {'-'*6} {'-'*7}")

    for species in all_species:
        images = merged[species]
        train_imgs, val_imgs, test_imgs = split_species_images(images)
        train_all.extend(train_imgs)
        val_all.extend(val_imgs)
        test_all.extend(test_imgs)
        print(
            f"  {species:<20} {len(train_imgs):>7} {len(val_imgs):>6}"
            f" {len(test_imgs):>6} {len(images):>7}"
        )

    print(f"  {'-'*20} {'-'*7} {'-'*6} {'-'*6} {'-'*7}")
    print(
        f"  {'TOTAL':<20} {len(train_all):>7} {len(val_all):>6}"
        f" {len(test_all):>6} {len(train_all)+len(val_all)+len(test_all):>7}"
    )

    # Create symlinks
    print("\n  Creating dataset structure...")
    for split_name, split_images in [
        ("train", train_all),
        ("val", val_all),
        ("test", test_all),
    ]:
        # Group by species for symlink creation
        by_species: dict[str, list[Path]] = defaultdict(list)
        for img_path in split_images:
            # Determine species from the parent dir name
            species = extract_species(img_path.parent.name)
            by_species[species].append(img_path)

        species_list = sorted(by_species.keys())
        for species in species_list:
            dst = OUT / split_name / species
            make_symlinks(by_species[species], dst)

        n_species = len(species_list)
        n_images = sum(len(v) for v in by_species.values())
        print(f"    {split_name}: {n_images:,} images across {n_species} species")

    # Summary
    print(f"\n{'='*60}")
    print(f"  Dataset ready at: {OUT.resolve()}")
    print(f"  Structure:")
    print(f"    {OUT}/train/{{species}}/  — {len(train_all):,} images")
    print(f"    {OUT}/val/{{species}}/    — {len(val_all):,} images")
    print(f"    {OUT}/test/{{species}}/   — {len(test_all):,} images")
    print(f"\n  Held-out test set (eval_images/): {sum(1 for _ in Path('eval_images').rglob('*.jpg')):,} images")
    print(f"  (Not included — use for final evaluation only)")
    print(f"\n  To train: point YOLO data= argument to {OUT.resolve()}")


if __name__ == "__main__":
    main()
