#!/usr/bin/env python3
"""
Generate Teacher Labels for NanoFloraVLM Distillation
======================================================
Uses Gemini 2.5 Flash API to create image→text training pairs
from the existing flora dataset.

Output: datasets/vlm_distill/pairs.jsonl
Each line: {"image_path": "...", "prompt": "...", "answer": "species: X, dry: Y"}

Usage:
    ./flora_env/bin/python generate_teacher_labels.py
"""

import json
import os
import random
import time
from collections import defaultdict
from pathlib import Path

from PIL import Image

# ── Config ──
API_KEY = "AIzaSyBYbAm1603zmMl6tu7WIJjORuFKrzAiLqo"
OUTPUT_DIR = Path("datasets/vlm_distill")
PAIRS_FILE = OUTPUT_DIR / "pairs.jsonl"

# Sources: cropped images from the species classifier pipeline
CROP_DIR = Path("datasets/species_cls/cropped")
# Also try the raw annotated dataset
RAW_IMAGES = Path("datasets/yolo_flora/images")
RAW_LABELS = Path("datasets/yolo_flora/labels")

SPECIES = [
    "adansonia", "acacia", "vachellia", "senegalia", "combretum",
    "brachystegia", "colophospermum", "ficus", "khaya", "macaranga",
    "euphorbia", "aloe", "protea", "erica", "themeda", "andropogon", "tamarix",
]

# Question templates for diversity
PROMPTS = [
    "What species of plant is this?",
    "What plant is this?",
    "Identify this plant.",
    "What type of plant do you see?",
    "Is this plant dry or alive?",
    "What is the condition of this plant?",
    "Identify this plant and its condition.",
    "What species is this and is it dry?",
    "Name this plant species.",
    "Describe this plant.",
]

# Rate limiting
REQUESTS_PER_MINUTE = 50
DELAY_BETWEEN_REQUESTS = 60.0 / REQUESTS_PER_MINUTE

# Max images per species per split (to keep training manageable)
MAX_PER_SPECIES = 200
SEED = 42
random.seed(SEED)


def get_gemini_client():
    """Initialize Gemini API client."""
    try:
        from google import genai
        return genai.Client(api_key=API_KEY)
    except ImportError:
        print("❌ Install google-genai: pip install google-genai")
        raise


def classify_with_gemini(client, image_path: str, prompt: str) -> str | None:
    """Send image + prompt to Gemini, get structured response."""
    from google.genai import types

    with open(image_path, "rb") as f:
        image_data = f.read()

    system_prompt = (
        "You are a botanical expert analyzing African flora. "
        "Always respond in this exact format: species: <name>, dry: <yes/no>\n"
        "The species must be one of: " + ", ".join(SPECIES) + "\n"
        "Example: species: acacia, dry: no\n"
        "If unsure of species, make your best guess from the list. "
        "If unsure of dryness, default to 'no'."
    )

    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[
                    types.Part.from_bytes(data=image_data, mime_type="image/jpeg"),
                    f"{system_prompt}\n\nUser question: {prompt}",
                ],
            )
            return response.text.strip()
        except Exception as e:
            if attempt < 2:
                wait = 5 * (attempt + 1)
                print(f"    ⚠️  Retry {attempt+1}/3 ({e}), waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"    ❌ Failed after 3 attempts: {e}")
                return None
    return None


def normalize_answer(raw: str, known_species: str = None) -> str | None:
    """Parse Gemini response into normalized structured format."""
    raw = raw.lower().strip()

    # Try to extract species
    species = None
    dry = "no"

    # Look for "species: X" pattern
    if "species:" in raw:
        parts = raw.split("species:")
        if len(parts) > 1:
            sp_part = parts[1].split(",")[0].strip()
            for s in SPECIES:
                if s in sp_part:
                    species = s
                    break

    # Fallback: match any species name in the response
    if species is None:
        for s in SPECIES:
            if s in raw:
                species = s
                break

    # Final fallback: use known species from directory name
    if species is None and known_species:
        species = known_species

    if species is None:
        return None

    # Extract dryness
    if "dry: yes" in raw or "dry:yes" in raw:
        dry = "yes"
    elif "dry" in raw and ("dead" in raw or "brown" in raw or "cured" in raw or "wilted" in raw):
        dry = "yes"

    return f"species: {species}, dry: {dry}"


def collect_cropped_images() -> list[dict]:
    """Collect images from the cropped species_cls dataset."""
    samples = []

    for split in ["train", "val"]:
        split_dir = CROP_DIR / split
        if not split_dir.exists():
            continue
        for species_dir in sorted(split_dir.iterdir()):
            if not species_dir.is_dir():
                continue
            species = species_dir.name
            if species not in SPECIES:
                continue
            images = list(species_dir.glob("*.jpg"))
            random.shuffle(images)
            for img_path in images[:MAX_PER_SPECIES]:
                samples.append({
                    "image_path": str(img_path),
                    "known_species": species,
                    "split": split,
                })

    return samples


def collect_raw_images() -> list[dict]:
    """Collect images from the raw YOLO dataset with known labels."""
    samples = []

    for split in ["train", "val"]:
        img_dir = RAW_IMAGES / split
        lbl_dir = RAW_LABELS / split
        if not img_dir.exists():
            continue

        for img_path in sorted(img_dir.glob("*.jpg"))[:500]:
            lbl_path = lbl_dir / (img_path.stem + ".txt")
            if not lbl_path.exists():
                continue
            try:
                parts = lbl_path.read_text().strip().split()
                class_id = int(parts[0])
                species_id = class_id // 2
                if species_id >= len(SPECIES):
                    continue
                is_dry = class_id % 2 == 1
                samples.append({
                    "image_path": str(img_path),
                    "known_species": SPECIES[species_id],
                    "known_dry": "yes" if is_dry else "no",
                    "split": split,
                })
            except Exception:
                continue

    return samples


def generate_labels_gemini(samples: list[dict], use_api: bool = True) -> int:
    """Generate training labels using Gemini API or known labels."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load existing pairs to skip
    existing = set()
    if PAIRS_FILE.exists():
        with open(PAIRS_FILE) as f:
            for line in f:
                d = json.loads(line)
                existing.add(d["image_path"])
        print(f"  Found {len(existing)} existing pairs, skipping those.")

    # Filter out already-processed
    todo = [s for s in samples if s["image_path"] not in existing]
    print(f"  {len(todo)} images to process ({len(existing)} already done)")

    if not todo:
        print("  ✅ All images already labeled!")
        return 0

    client = get_gemini_client() if use_api else None
    written = 0
    start_time = time.time()

    with open(PAIRS_FILE, "a") as f:
        for i, sample in enumerate(todo):
            img_path = sample["image_path"]
            known_species = sample.get("known_species")
            known_dry = sample.get("known_dry")

            # Pick 1-2 random prompts per image
            prompts = random.sample(PROMPTS, min(2, len(PROMPTS)))

            for prompt in prompts:
                if use_api and client:
                    raw = classify_with_gemini(client, img_path, prompt)
                    answer = normalize_answer(raw, known_species) if raw else None
                    time.sleep(DELAY_BETWEEN_REQUESTS)
                else:
                    # Use known labels directly (no API needed)
                    dry = known_dry if known_dry else "no"
                    answer = f"species: {known_species}, dry: {dry}"

                if answer:
                    pair = {
                        "image_path": img_path,
                        "prompt": prompt,
                        "answer": answer,
                    }
                    f.write(json.dumps(pair) + "\n")
                    written += 1

            # Progress
            if (i + 1) % 50 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed * 60
                eta = (len(todo) - i - 1) / rate if rate > 0 else 0
                print(f"  [{i+1}/{len(todo)}] {rate:.0f} img/min, ETA: {eta:.0f} min")

    return written


def generate_from_known_labels(samples: list[dict]) -> int:
    """
    Fast path: generate training pairs from known YOLO labels
    without calling any API. This is free and instant.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    existing = set()
    if PAIRS_FILE.exists():
        with open(PAIRS_FILE) as f:
            for line in f:
                d = json.loads(line)
                existing.add((d["image_path"], d["prompt"]))

    written = 0
    with open(PAIRS_FILE, "a") as f:
        for sample in samples:
            species = sample.get("known_species")
            dry = sample.get("known_dry", "no")
            if not species:
                continue

            answer = f"species: {species}, dry: {dry}"

            # Generate multiple prompt-answer pairs per image
            for prompt in random.sample(PROMPTS, min(3, len(PROMPTS))):
                key = (sample["image_path"], prompt)
                if key in existing:
                    continue
                pair = {
                    "image_path": sample["image_path"],
                    "prompt": prompt,
                    "answer": answer,
                }
                f.write(json.dumps(pair) + "\n")
                written += 1

    return written


def main():
    print("=" * 60)
    print("  Teacher Label Generation for NanoFloraVLM")
    print("=" * 60)

    # Collect all available images
    cropped = collect_cropped_images()
    raw = collect_raw_images()
    print(f"\n  Found {len(cropped)} cropped images, {len(raw)} raw images")

    # Phase 1: Generate from known labels (free, instant)
    print(f"\n  Phase 1: Labels from known YOLO annotations (free)...")
    known_samples = [s for s in cropped + raw if s.get("known_species")]
    n1 = generate_from_known_labels(known_samples)
    print(f"  ✅ Generated {n1} pairs from known labels")

    # Phase 2: Optionally enhance with Gemini API
    # Only use API for images where we want richer/verified labels
    use_api = os.environ.get("USE_GEMINI_API", "0") == "1"
    if use_api:
        print(f"\n  Phase 2: Enriching with Gemini API...")
        api_samples = random.sample(cropped, min(500, len(cropped)))
        n2 = generate_labels_gemini(api_samples, use_api=True)
        print(f"  ✅ Generated {n2} API-enriched pairs")
    else:
        print(f"\n  Phase 2: Skipping Gemini API (set USE_GEMINI_API=1 to enable)")

    # Stats
    total_pairs = sum(1 for _ in open(PAIRS_FILE)) if PAIRS_FILE.exists() else 0
    species_dist = defaultdict(int)
    if PAIRS_FILE.exists():
        with open(PAIRS_FILE) as f:
            for line in f:
                d = json.loads(line)
                ans = d["answer"]
                for sp in SPECIES:
                    if sp in ans:
                        species_dist[sp] += 1
                        break

    print(f"\n  {'='*50}")
    print(f"  Total training pairs: {total_pairs}")
    print(f"\n  {'Species':<20} {'Pairs':>8}")
    print(f"  {'-'*20} {'-'*8}")
    for sp in SPECIES:
        print(f"  {sp:<20} {species_dist.get(sp, 0):>8}")

    print(f"\n  📁 Output: {PAIRS_FILE}")
    print(f"  🏁 Ready for training!")


if __name__ == "__main__":
    main()
