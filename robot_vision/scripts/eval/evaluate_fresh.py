#!/usr/bin/env python3
"""
Fresh Evaluation Pipeline for Flora Detection Model
=====================================================
Downloads fresh images from the web, annotates them with:
  - Gemini 2.5 Flash → dry/alive classification
  - YOLO-World → real bounding boxes
Then evaluates best(1).pt against this independent test set.

Usage:
    ./flora_env/bin/python evaluate_fresh.py
"""

import gc
import io
import json
import os
import re
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

import requests
from PIL import Image

# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────

MODEL_PATH = "best(1).pt"
EVAL_DIR = Path("datasets/fresh_eval")
IMAGES_DIR = EVAL_DIR / "images"
LABELS_DIR = EVAL_DIR / "labels"
RESULTS_DIR = EVAL_DIR / "results"

API_KEY = "AIzaSyBYbAm1603zmMl6tu7WIJjORuFKrzAiLqo"

# Number of images to download per species
IMAGES_PER_SPECIES = 5

# The 17 species in the model
SPECIES = [
    "adansonia",      # baobab
    "acacia",
    "vachellia",
    "senegalia",
    "combretum",
    "brachystegia",
    "colophospermum",  # mopane
    "ficus",
    "khaya",           # African mahogany
    "macaranga",
    "euphorbia",
    "aloe",
    "protea",
    "erica",
    "themeda",         # red grass
    "andropogon",
    "tamarix",
]

# Class mapping (species + dryness → class ID), matches data.yaml
CLASS_MAP = {}
CLASS_NAMES = {}
for i, sp in enumerate(SPECIES):
    CLASS_MAP[f"{sp}_not_dry"] = i * 2
    CLASS_MAP[f"{sp}_dry"] = i * 2 + 1
    CLASS_NAMES[i * 2] = f"{sp}_not_dry"
    CLASS_NAMES[i * 2 + 1] = f"{sp}_dry"

# Human-readable search terms per species
SEARCH_TERMS = {
    "adansonia": "Adansonia baobab tree Africa",
    "acacia": "Acacia tree Africa savanna",
    "vachellia": "Vachellia tortilis tree Africa",
    "senegalia": "Senegalia tree Africa",
    "combretum": "Combretum tree Africa bushveld",
    "brachystegia": "Brachystegia miombo woodland tree",
    "colophospermum": "Colophospermum mopane tree Africa",
    "ficus": "Ficus tree Africa wild fig",
    "khaya": "Khaya African mahogany tree",
    "macaranga": "Macaranga tree tropical Africa",
    "euphorbia": "Euphorbia plant Africa succulent tree",
    "aloe": "Aloe plant Africa wild",
    "protea": "Protea flower plant South Africa",
    "erica": "Erica heather plant South Africa fynbos",
    "themeda": "Themeda triandra red grass Africa",
    "andropogon": "Andropogon grass Africa savanna",
    "tamarix": "Tamarix tree salt cedar Africa",
}

# Also add dry variants for search
DRY_SEARCH_TERMS = {
    "adansonia": "baobab tree dry season leafless",
    "acacia": "acacia tree dry season brown",
    "vachellia": "vachellia tree dry dead brown",
    "senegalia": "senegalia tree dry brown Africa",
    "combretum": "combretum tree dry season deciduous",
    "brachystegia": "brachystegia tree dry season miombo",
    "colophospermum": "mopane tree dry season brown leaves",
    "ficus": "ficus tree dry brown dead",
    "khaya": "mahogany tree dry season deciduous",
    "macaranga": "macaranga tree dry brown leaves",
    "euphorbia": "euphorbia dry brown dead plant",
    "aloe": "aloe plant dry brown dead",
    "protea": "protea flower dry brown seed head",
    "erica": "erica heather dry brown dead",
    "themeda": "themeda grass dry brown cured",
    "andropogon": "andropogon grass dry brown straw",
    "tamarix": "tamarix tree dry brown dead",
}


def download_images_duckduckgo(query: str, max_images: int = 5) -> list[bytes]:
    """Download images using DuckDuckGo image search."""
    print(f"    🔍 Searching: '{query}'...")

    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                       "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    # Step 1: Get the vqd token
    try:
        token_resp = requests.get(
            "https://duckduckgo.com/",
            params={"q": query},
            headers=headers,
            timeout=10,
        )
        # Extract vqd token from response
        vqd_match = re.search(r'vqd=([\d-]+)', token_resp.text)
        if not vqd_match:
            # Try alternate pattern
            vqd_match = re.search(r'vqd="([\d-]+)"', token_resp.text)
        if not vqd_match:
            print(f"    ⚠️  Could not get search token for '{query}'")
            return []
        vqd = vqd_match.group(1)
    except Exception as e:
        print(f"    ⚠️  Search token error: {e}")
        return []

    # Step 2: Fetch image results
    try:
        img_resp = requests.get(
            "https://duckduckgo.com/i.js",
            params={
                "l": "us-en",
                "o": "json",
                "q": query,
                "vqd": vqd,
                "f": ",,,,,",
                "p": "1",
            },
            headers=headers,
            timeout=10,
        )
        results = img_resp.json().get("results", [])
    except Exception as e:
        print(f"    ⚠️  Image search error: {e}")
        return []

    # Step 3: Download actual images
    downloaded = []
    for result in results[:max_images * 3]:  # Try 3x to get enough
        if len(downloaded) >= max_images:
            break
        img_url = result.get("image", "")
        if not img_url:
            continue
        try:
            r = requests.get(img_url, timeout=8, headers=headers)
            if r.status_code == 200 and len(r.content) > 5000:
                # Verify it's a valid image
                img = Image.open(io.BytesIO(r.content))
                img.verify()
                downloaded.append(r.content)
        except Exception:
            continue

    print(f"    ✅ Downloaded {len(downloaded)} images")
    return downloaded


def download_images_gemini(query: str, species: str, max_images: int = 5) -> list[tuple[bytes, str]]:
    """
    Use Gemini to generate image search URLs, then download them.
    Returns list of (image_bytes, suggested_url).
    """
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=API_KEY)

    prompt = f"""I need {max_images} real, royalty-free photograph URLs of {query}.
Return ONLY a JSON array of direct image URLs. Example:
["https://example.com/photo1.jpg", "https://example.com/photo2.jpg"]
Focus on clear, real photographs showing the whole plant/tree. No diagrams, illustrations, or drawings.
Return ONLY the JSON array, no other text."""

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[prompt],
        )
        text = response.text.strip()
        # Try to extract JSON array
        match = re.search(r'\[.*?\]', text, re.DOTALL)
        if match:
            urls = json.loads(match.group())
            downloaded = []
            headers = {
                "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"
            }
            for url in urls[:max_images * 2]:
                if len(downloaded) >= max_images:
                    break
                try:
                    r = requests.get(url, timeout=8, headers=headers)
                    if r.status_code == 200 and len(r.content) > 5000:
                        img = Image.open(io.BytesIO(r.content))
                        img.verify()
                        downloaded.append((r.content, url))
                except Exception:
                    continue
            return downloaded
    except Exception as e:
        print(f"    ⚠️  Gemini URL gen failed: {e}")
    return []


def annotate_dryness_gemini(img_path: str) -> str:
    """Use Gemini 2.5 Flash to classify a plant image as dry or not_dry."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=API_KEY)

    with open(img_path, "rb") as f:
        image_data = f.read()

    prompt = (
        "Look at this plant image carefully. "
        "Is the majority of the foliage dry, dead, brown, or cured? "
        "Or is it green, alive, and healthy? "
        "Answer with EXACTLY one word: 'DRY' or 'LIVE'."
    )

    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[
                    types.Part.from_bytes(data=image_data, mime_type="image/jpeg"),
                    prompt,
                ],
            )
            ans = response.text.strip().upper()
            if "DRY" in ans:
                return "dry"
            return "not_dry"
        except Exception as e:
            if attempt < 2:
                print(f"      ⚠️  Gemini retry ({attempt+1}/3): {e}")
                time.sleep(5 * (attempt + 1))
            else:
                print(f"      ❌ Gemini failed after 3 attempts, defaulting to not_dry")
                return "not_dry"

    return "not_dry"


def annotate_bbox_yoloworld(model, img_path: str) -> str:
    """Use YOLO-World to find the plant bounding box. Returns 'cx cy w h' string."""
    PLANT_PROMPTS = [
        "plant", "tree", "shrub", "flower", "grass",
        "succulent", "cactus", "bush", "leaf", "vegetation",
    ]

    results = model.predict(
        img_path,
        conf=0.15,
        device="cpu",
        verbose=False,
        imgsz=640,
    )

    result = results[0]
    img_h, img_w = result.orig_shape
    boxes = result.boxes

    if boxes is not None and len(boxes) > 0:
        best_idx, best_area = -1, 0
        for j in range(len(boxes)):
            xyxy = boxes.xyxy[j].cpu().numpy()
            area = (xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1])
            if area / (img_w * img_h) >= 0.02 and area > best_area:
                best_area = area
                best_idx = j

        if best_idx >= 0:
            x1, y1, x2, y2 = boxes.xyxy[best_idx].cpu().numpy()
            cx = max(0, min(1, ((x1 + x2) / 2) / img_w))
            cy = max(0, min(1, ((y1 + y2) / 2) / img_h))
            w = max(0.01, min(1, (x2 - x1) / img_w))
            h = max(0.01, min(1, (y2 - y1) / img_h))
            return f"{cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"

    return "0.5 0.5 0.85 0.85"  # fallback


def evaluate_model(model_path: str, eval_dir: Path):
    """Run the trained model on fresh test images and compare against ground truth."""
    from ultralytics import YOLO

    model = YOLO(model_path)
    images_dir = eval_dir / "images"
    labels_dir = eval_dir / "labels"
    results_dir = eval_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    image_files = sorted(images_dir.glob("*.jpg"))
    if not image_files:
        print("❌ No images to evaluate!")
        return

    print(f"\n{'='*65}")
    print(f"  Evaluating {model_path} on {len(image_files)} fresh images")
    print(f"{'='*65}")

    # Track metrics
    correct_species = 0
    correct_dryness = 0
    correct_full = 0
    total = 0
    iou_scores = []

    per_class_tp = defaultdict(int)
    per_class_fn = defaultdict(int)
    per_class_fp = defaultdict(int)
    per_species_correct = defaultdict(int)
    per_species_total = defaultdict(int)

    detailed_results = []

    for img_path in image_files:
        label_path = labels_dir / (img_path.stem + ".txt")
        if not label_path.exists():
            continue

        # Read ground truth
        gt_text = label_path.read_text().strip()
        if not gt_text:
            continue
        gt_parts = gt_text.split()
        gt_class_id = int(gt_parts[0])
        gt_bbox = list(map(float, gt_parts[1:5]))
        gt_name = CLASS_NAMES.get(gt_class_id, f"unknown_{gt_class_id}")
        gt_species = gt_name.rsplit("_", 1)[0] if "_dry" in gt_name or "_not_dry" in gt_name else gt_name
        gt_dryness = "dry" if gt_name.endswith("_dry") else "not_dry"

        # Run inference
        results = model.predict(
            str(img_path),
            conf=0.1,
            device="cpu",
            verbose=False,
            imgsz=640,
            save=True,
            project=str(results_dir),
            name="predictions",
            exist_ok=True,
        )

        result = results[0]
        boxes = result.boxes

        pred_class_id = -1
        pred_conf = 0
        pred_bbox = None

        if boxes is not None and len(boxes) > 0:
            # Pick highest confidence detection
            confs = boxes.conf.cpu().numpy()
            best_idx = confs.argmax()
            pred_class_id = int(boxes.cls[best_idx].cpu().numpy())
            pred_conf = float(confs[best_idx])
            pred_xyxy = boxes.xyxy[best_idx].cpu().numpy()
            img_h, img_w = result.orig_shape
            pred_bbox = [
                ((pred_xyxy[0] + pred_xyxy[2]) / 2) / img_w,
                ((pred_xyxy[1] + pred_xyxy[3]) / 2) / img_h,
                (pred_xyxy[2] - pred_xyxy[0]) / img_w,
                (pred_xyxy[3] - pred_xyxy[1]) / img_h,
            ]

        pred_name = CLASS_NAMES.get(pred_class_id, "no_detection")
        pred_species = pred_name.rsplit("_", 1)[0] if "_dry" in pred_name or "_not_dry" in pred_name else pred_name
        pred_dryness = "dry" if pred_name.endswith("_dry") else "not_dry"

        # Calculate IoU if we have both boxes
        iou = 0
        if pred_bbox is not None:
            # Convert xywh to xyxy for IoU
            gt_x1 = gt_bbox[0] - gt_bbox[2] / 2
            gt_y1 = gt_bbox[1] - gt_bbox[3] / 2
            gt_x2 = gt_bbox[0] + gt_bbox[2] / 2
            gt_y2 = gt_bbox[1] + gt_bbox[3] / 2

            pd_x1 = pred_bbox[0] - pred_bbox[2] / 2
            pd_y1 = pred_bbox[1] - pred_bbox[3] / 2
            pd_x2 = pred_bbox[0] + pred_bbox[2] / 2
            pd_y2 = pred_bbox[1] + pred_bbox[3] / 2

            inter_x1 = max(gt_x1, pd_x1)
            inter_y1 = max(gt_y1, pd_y1)
            inter_x2 = min(gt_x2, pd_x2)
            inter_y2 = min(gt_y2, pd_y2)

            if inter_x2 > inter_x1 and inter_y2 > inter_y1:
                inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                gt_area = (gt_x2 - gt_x1) * (gt_y2 - gt_y1)
                pd_area = (pd_x2 - pd_x1) * (pd_y2 - pd_y1)
                iou = inter_area / (gt_area + pd_area - inter_area + 1e-6)

            iou_scores.append(iou)

        # Track accuracy
        total += 1
        species_match = (pred_species == gt_species)
        dryness_match = (pred_dryness == gt_dryness)
        full_match = (pred_class_id == gt_class_id)

        if species_match:
            correct_species += 1
        if dryness_match:
            correct_dryness += 1
        if full_match:
            correct_full += 1
            per_class_tp[gt_class_id] += 1
        else:
            per_class_fn[gt_class_id] += 1
            if pred_class_id >= 0:
                per_class_fp[pred_class_id] += 1

        per_species_total[gt_species] += 1
        if species_match:
            per_species_correct[gt_species] += 1

        # Log detail
        status = "✅" if full_match else ("🟡" if species_match else "❌")
        detail = {
            "image": img_path.name,
            "gt_class": gt_name,
            "pred_class": pred_name,
            "confidence": pred_conf,
            "iou": iou,
            "species_match": species_match,
            "dryness_match": dryness_match,
            "full_match": full_match,
        }
        detailed_results.append(detail)

        print(f"  {status} {img_path.name}")
        print(f"     GT:   {gt_name}")
        print(f"     Pred: {pred_name} ({pred_conf:.2f})")
        if pred_bbox:
            print(f"     IoU:  {iou:.3f}")
        print()

    # ── Summary Report ──
    print("\n" + "=" * 65)
    print("  EVALUATION SUMMARY")
    print("=" * 65)

    if total > 0:
        print(f"\n  Total images evaluated: {total}")
        print(f"  Full class accuracy:   {correct_full}/{total} ({100*correct_full/total:.1f}%)")
        print(f"  Species accuracy:      {correct_species}/{total} ({100*correct_species/total:.1f}%)")
        print(f"  Dryness accuracy:      {correct_dryness}/{total} ({100*correct_dryness/total:.1f}%)")

        if iou_scores:
            avg_iou = sum(iou_scores) / len(iou_scores)
            print(f"  Mean IoU:              {avg_iou:.3f}")

        # Per-species breakdown
        print(f"\n  {'Species':<20} {'Correct':>8} {'Total':>6} {'Accuracy':>10}")
        print(f"  {'-'*20} {'-'*8} {'-'*6} {'-'*10}")
        for sp in SPECIES:
            t = per_species_total.get(sp, 0)
            c = per_species_correct.get(sp, 0)
            if t > 0:
                print(f"  {sp:<20} {c:>8} {t:>6} {100*c/t:>9.1f}%")

    # Save detailed results — convert numpy types to native Python for JSON
    for d in detailed_results:
        for k, v in d.items():
            if hasattr(v, 'item'):  # numpy scalar
                d[k] = v.item()
            elif isinstance(v, float):
                d[k] = float(v)

    # Convert iou_scores from potential numpy floats
    iou_scores = [float(x) for x in iou_scores]

    report_path = results_dir / "evaluation_report.json"
    with open(report_path, "w") as f:
        json.dump({
            "model": model_path,
            "total": total,
            "full_class_accuracy": float(correct_full / total) if total else 0,
            "species_accuracy": float(correct_species / total) if total else 0,
            "dryness_accuracy": float(correct_dryness / total) if total else 0,
            "mean_iou": float(sum(iou_scores) / len(iou_scores)) if iou_scores else 0,
            "per_species": {
                sp: {
                    "correct": per_species_correct.get(sp, 0),
                    "total": per_species_total.get(sp, 0),
                }
                for sp in SPECIES
            },
            "detailed": detailed_results,
        }, f, indent=2)
    print(f"\n  📄 Detailed report saved to: {report_path}")

    return detailed_results


# ──────────────────────────────────────────────────────────────────────
# Main Pipeline
# ──────────────────────────────────────────────────────────────────────

def main():
    from ultralytics import YOLO

    # Setup directories
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    LABELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Check how many images we already have
    existing_images = list(IMAGES_DIR.glob("*.jpg"))
    if len(existing_images) >= 20:
        print(f"✅ Found {len(existing_images)} existing fresh images, skipping download")
        print("   (Delete datasets/fresh_eval/images/ to re-download)")
    else:
        # ── Phase 1: Download Fresh Images ──
        print("=" * 65)
        print("  Phase 1: Downloading Fresh Test Images")
        print("=" * 65)

        downloaded_count = 0

        for species in SPECIES:
            print(f"\n🌿 {species}:")

            # Download GREEN/ALIVE images
            query = SEARCH_TERMS[species]
            images = download_images_duckduckgo(query, max_images=3)

            for idx, img_data in enumerate(images):
                fname = f"{species}_alive_{idx:02d}.jpg"
                fpath = IMAGES_DIR / fname
                try:
                    # Convert to standard JPEG
                    img = Image.open(io.BytesIO(img_data))
                    img = img.convert("RGB")
                    img = img.resize((640, 640), Image.LANCZOS)
                    img.save(str(fpath), "JPEG", quality=90)
                    downloaded_count += 1
                except Exception as e:
                    print(f"    ⚠️  Failed to save {fname}: {e}")

            # Download DRY images
            query = DRY_SEARCH_TERMS[species]
            images = download_images_duckduckgo(query, max_images=2)

            for idx, img_data in enumerate(images):
                fname = f"{species}_dry_{idx:02d}.jpg"
                fpath = IMAGES_DIR / fname
                try:
                    img = Image.open(io.BytesIO(img_data))
                    img = img.convert("RGB")
                    img = img.resize((640, 640), Image.LANCZOS)
                    img.save(str(fpath), "JPEG", quality=90)
                    downloaded_count += 1
                except Exception as e:
                    print(f"    ⚠️  Failed to save {fname}: {e}")

            time.sleep(1)  # Be polite to DDG

        print(f"\n✅ Downloaded {downloaded_count} fresh images total")

    # ── Phase 2: Annotate with Gemini + YOLO-World ──
    # Check which images still need annotation
    image_files = sorted(IMAGES_DIR.glob("*.jpg"))
    unannotated = [
        f for f in image_files
        if not (LABELS_DIR / (f.stem + ".txt")).exists()
    ]

    if unannotated:
        print(f"\n{'='*65}")
        print(f"  Phase 2: Annotating {len(unannotated)} images")
        print(f"{'='*65}")

        # Load YOLO-World for bbox detection
        print("\n📦 Loading YOLO-World (yolov8s-worldv2)...")
        world_model = YOLO("yolov8s-worldv2.pt")
        world_model.set_classes([
            "plant", "tree", "shrub", "flower", "grass",
            "succulent", "cactus", "bush", "leaf", "vegetation",
        ])

        for img_path in unannotated:
            fname = img_path.stem
            print(f"\n  📋 Annotating {img_path.name}...")

            # Extract species from filename (e.g., "acacia_alive_01" → "acacia")
            # Handle multi-word species names
            parts = fname.split("_")
            if "alive" in parts:
                species = "_".join(parts[:parts.index("alive")])
                filename_hint = "not_dry"
            elif "dry" in parts:
                species = "_".join(parts[:parts.index("dry")])
                filename_hint = "dry"
            else:
                species = parts[0]
                filename_hint = "not_dry"

            if species not in SPECIES:
                print(f"    ⚠️  Unknown species '{species}', skipping")
                continue

            # Step 1: Classify dryness with Gemini 2.5 Flash
            print(f"    🧠 Gemini 2.5 Flash classifying dryness...")
            dryness = annotate_dryness_gemini(str(img_path))
            print(f"    → Gemini says: {dryness} (filename hint: {filename_hint})")

            # Use filename hint as tiebreaker if Gemini disagrees
            # but trust Gemini for the actual label since it sees the image
            class_name = f"{species}_{dryness}"
            class_id = CLASS_MAP.get(class_name)
            if class_id is None:
                print(f"    ⚠️  Unknown class '{class_name}', skipping")
                continue

            # Step 2: Get bounding box with YOLO-World
            print(f"    📦 YOLO-World detecting bbox...")
            bbox_str = annotate_bbox_yoloworld(world_model, str(img_path))
            print(f"    → BBox: {bbox_str}")

            # Write label
            label_path = LABELS_DIR / (img_path.stem + ".txt")
            label_path.write_text(f"{class_id} {bbox_str}\n")
            print(f"    ✅ Label: {class_id} ({class_name}) {bbox_str}")

            time.sleep(0.5)  # Rate limiting for Gemini

        # Cleanup
        del world_model
        gc.collect()
    else:
        print(f"\n✅ All {len(image_files)} images already annotated")

    # ── Phase 3: Evaluate ──
    print(f"\n{'='*65}")
    print(f"  Phase 3: Running Model Evaluation")
    print(f"{'='*65}")

    evaluate_model(MODEL_PATH, EVAL_DIR)

    print(f"\n🏁 Evaluation pipeline complete!")
    print(f"   Images:  {IMAGES_DIR}")
    print(f"   Labels:  {LABELS_DIR}")
    print(f"   Results: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
