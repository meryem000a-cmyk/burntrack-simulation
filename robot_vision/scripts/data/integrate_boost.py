#!/usr/bin/env python3
"""
Integrates boosted images from datasets/boost_staging/ into yolo_flora/
by copying images and creating matching YOLO label files.

Maps each species to its class_id (species_id * 2 = not_dry variant).
Uses a full-image dummy bbox since these are classification images
and the training script uses full images anyway (class_id // 2 mapping).
"""

import os
import shutil
from pathlib import Path

STAGING = Path("datasets/boost_staging")
YOLO_IMAGES = Path("datasets/yolo_flora/images/train")
YOLO_LABELS = Path("datasets/yolo_flora/labels/train")

# Species → base class_id (not_dry variant = species_index * 2)
SPECIES_TO_CLASS = {
    "adansonia": 0, "acacia": 2, "vachellia": 4, "senegalia": 6,
    "combretum": 8, "brachystegia": 10, "colophospermum": 12,
    "ficus": 14, "khaya": 16, "macaranga": 18, "euphorbia": 20,
    "aloe": 22, "protea": 24, "erica": 26, "themeda": 28,
    "andropogon": 30, "tamarix": 32,
}

total_added = 0

for species_dir in sorted(STAGING.iterdir()):
    if not species_dir.is_dir():
        continue
    species = species_dir.name
    class_id = SPECIES_TO_CLASS.get(species)
    if class_id is None:
        print(f"  ⚠️  Unknown species: {species}, skipping")
        continue

    images = list(species_dir.glob("*.jpg")) + list(species_dir.glob("*.jpeg")) + list(species_dir.glob("*.png"))
    added = 0

    for img_path in images:
        # Prefix filename to avoid collisions with existing images
        new_name = f"boost_{species}_{img_path.name}"
        dst_img = YOLO_IMAGES / new_name
        dst_lbl = YOLO_LABELS / (Path(new_name).stem + ".txt")

        if dst_img.exists():
            continue

        shutil.copy2(img_path, dst_img)
        dst_lbl.write_text(f"{class_id} 0.5 0.5 1.0 1.0\n")
        added += 1

    print(f"  {species:<20} +{added} images (class_id={class_id})")
    total_added += added

print(f"\n  ✅ Integrated {total_added} new images into yolo_flora/")
print(f"  Total train images now: {len(list(YOLO_IMAGES.glob('*')))}")
