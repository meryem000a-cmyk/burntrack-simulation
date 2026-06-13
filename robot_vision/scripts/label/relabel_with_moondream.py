#!/usr/bin/env python3
"""
Relabel Boosted Images with Moondream (Local, Free)
=====================================================
Runs Moondream 2B locally on boosted images to fix their dryness labels.
The boosted images were all defaulted to "not_dry" by integrate_boost.py,
but some may actually be dry specimens.

This runs on CPU — expect ~15-30s per image on Ryzen 5 4500U.
For 4,527 boosted images ≈ 19-38 hours (run overnight x2).

Saves progress after every 25 images — safe to Ctrl+C and resume.

Usage:
    ./flora_env/bin/python relabel_with_moondream.py

To limit to a subset (e.g. 200 images for a quick test):
    ./flora_env/bin/python relabel_with_moondream.py --limit 200
"""

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import torch

from PIL import Image

# ── Config ──
YOLO_IMAGES = Path("datasets/yolo_flora/images/train")
YOLO_LABELS = Path("datasets/yolo_flora/labels/train")
PAIRS_FILE = Path("datasets/vlm_distill/pairs.jsonl")
PROGRESS_FILE = Path("datasets/vlm_distill/moondream_progress.json")

SPECIES = [
    "adansonia", "acacia", "vachellia", "senegalia", "combretum",
    "brachystegia", "colophospermum", "ficus", "khaya", "macaranga",
    "euphorbia", "aloe", "protea", "erica", "themeda", "andropogon", "tamarix",
]

SPECIES_TO_CLASS = {sp: i * 2 for i, sp in enumerate(SPECIES)}


def load_moondream():
    """Load Moondream 2B via transformers for local CPU inference."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = "vikhyatk/moondream2"
    revision = "2025-01-09"

    print(f"  📦 Loading {model_id} (downloads ~3.5GB on first run)...")
    t0 = time.time()
    hf_token = "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"
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        device_map="cpu",
        torch_dtype=torch.float32,
        token=hf_token,
    )
    model.eval()
    print(f"  ✅ Moondream loaded in {time.time() - t0:.0f}s")
    return model, tokenizer


def classify_dryness(model_tuple, image_path: str) -> str:
    """Ask Moondream if a plant is dry or alive. Returns 'yes' or 'no'."""
    model, tokenizer = model_tuple
    img = Image.open(image_path).convert("RGB")

    answer = model.query(img, "Is this plant dry, dead, or brown? Answer with one word: YES or NO.", tokenizer)["answer"]
    answer = answer.strip().upper()

    if "YES" in answer:
        return "yes"
    return "no"


def get_boosted_images() -> list[dict]:
    """Find all boosted images and their current labels."""
    boosted = []
    for img_path in sorted(YOLO_IMAGES.glob("boost_*.jpg")):
        lbl_path = YOLO_LABELS / (img_path.stem + ".txt")
        if not lbl_path.exists():
            continue

        parts = lbl_path.read_text().strip().split()
        class_id = int(parts[0])
        species_id = class_id // 2
        if species_id >= len(SPECIES):
            continue

        # Extract species from filename: boost_erica_12345.jpg → erica
        name_parts = img_path.stem.split("_")
        # Remove "boost" prefix, find species
        species = None
        for sp in SPECIES:
            if sp in img_path.stem:
                species = sp
                break
        if not species:
            species = SPECIES[species_id]

        boosted.append({
            "image_path": str(img_path),
            "label_path": str(lbl_path),
            "species": species,
            "current_class_id": class_id,
            "current_dry": "yes" if class_id % 2 == 1 else "no",
        })

    return boosted


def load_progress() -> set:
    """Load set of already-processed image paths."""
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE) as f:
            data = json.load(f)
            return set(data.get("processed", []))
    return set()


def save_progress(processed: set, stats: dict):
    """Save progress to resume later."""
    with open(PROGRESS_FILE, "w") as f:
        json.dump({"processed": list(processed), "stats": stats}, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="Max images to process (0=all)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without loading model")
    args = parser.parse_args()

    print("=" * 60)
    print("  Relabel Boosted Images with Moondream 2B (local)")
    print("=" * 60)

    # Find boosted images
    boosted = get_boosted_images()
    print(f"\n  Found {len(boosted)} boosted images in YOLO dataset")

    # Filter already processed
    processed = load_progress()
    todo = [b for b in boosted if b["image_path"] not in processed]
    print(f"  Already processed: {len(processed)}")
    print(f"  Remaining: {len(todo)}")

    if args.limit > 0:
        todo = todo[:args.limit]
        print(f"  Limited to: {len(todo)}")

    if not todo:
        print("  ✅ All boosted images already relabeled!")
        return

    # Estimate time
    est_per_img = 15  # seconds on Ryzen 5
    est_total = len(todo) * est_per_img
    print(f"\n  ⏱  Estimated time: {est_total // 3600}h {(est_total % 3600) // 60}m")
    print(f"  (Safe to Ctrl+C — progress is saved after each image)\n")

    if args.dry_run:
        print("  [DRY RUN] Would process the above images. Exiting.")
        return

    # Load model
    model = load_moondream()

    # Process
    stats = {"total": 0, "changed_to_dry": 0, "stayed_not_dry": 0, "errors": 0}
    start_time = time.time()

    for i, item in enumerate(todo):
        try:
            dryness = classify_dryness(model, item["image_path"])
            old_dry = item["current_dry"]

            # Update YOLO label if dryness changed
            if dryness != old_dry:
                species = item["species"]
                base_class = SPECIES_TO_CLASS[species]
                new_class_id = base_class + (1 if dryness == "yes" else 0)

                # Rewrite label file with corrected class ID
                lbl_path = Path(item["label_path"])
                parts = lbl_path.read_text().strip().split()
                parts[0] = str(new_class_id)
                lbl_path.write_text(" ".join(parts) + "\n")

                stats["changed_to_dry"] += 1
                marker = "🔄 DRY"
            else:
                stats["stayed_not_dry"] += 1
                marker = "  ok"

            stats["total"] += 1

        except Exception as e:
            stats["errors"] += 1
            marker = f"❌ {e}"

        processed.add(item["image_path"])

        # Progress
        elapsed = time.time() - start_time
        rate = (i + 1) / elapsed if elapsed > 0 else 0
        eta = (len(todo) - i - 1) / rate / 60 if rate > 0 else 0
        species_short = item["species"][:12]

        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{i+1:>5}/{len(todo)}] {marker:>8} | {species_short:<12} | "
                  f"{rate:.1f} img/s | ETA: {eta:.0f}min | "
                  f"changed: {stats['changed_to_dry']}")

        # Save progress every 25 images
        if (i + 1) % 25 == 0:
            save_progress(processed, stats)

    # Final save
    save_progress(processed, stats)

    # Summary
    print(f"\n  {'='*50}")
    print(f"  RELABELING COMPLETE")
    print(f"  {'='*50}")
    print(f"  Processed:      {stats['total']}")
    print(f"  Changed to dry: {stats['changed_to_dry']}")
    print(f"  Stayed not_dry: {stats['stayed_not_dry']}")
    print(f"  Errors:         {stats['errors']}")
    print(f"\n  Now re-run generate_teacher_labels.py to update training pairs.")


if __name__ == "__main__":
    main()
