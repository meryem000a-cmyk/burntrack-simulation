#!/usr/bin/env python3
"""
Download 30 images per species from GBIF for evaluation.

Usage:
    python3 download_eval_data.py
    # Images saved to  eval_images/{species}/img_0001.jpg  (224x224 JPEG)
"""

import io
import json
import random
import time
import urllib.request
import urllib.error
from pathlib import Path

from PIL import Image

random.seed(random.randint(0, 99999))

# Species list MUST match the YOLO model's class names.
# Read from model file to stay in sync.
from ultralytics import YOLO  # noqa: E402
_MODEL_PATH = Path("yolos/best(3).pt")
if _MODEL_PATH.exists():
    _m = YOLO(str(_MODEL_PATH))
    SPECIES = [v for k, v in sorted(_m.names.items())]
else:
    SPECIES = [
        "acacia", "adansonia", "aloe", "andropogon", "baobab", "brachystegia",
        "colophospermum", "combretum", "erica", "euphorbia", "ficus",
        "khaya", "macaranga", "protea", "senegalia", "tamarix", "themeda",
    ]

OUT_DIR = Path("eval_images")
TARGET = 100
SLEEP = 0.1
TIMEOUT = 8
MAX_CONSECUTIVE_FAIL = 10

USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64)"

OUT_DIR.mkdir(exist_ok=True)


def gbif_occurrence_images(query, limit=120):
    try:
        offset = random.randint(0, 500)
        url = (
            f"https://api.gbif.org/v1/occurrence/search"
            f"?q={query}&mediaType=StillImage&limit={limit}&offset={offset}"
        )
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())

        urls = []
        for result in data.get("results", []):
            for media in result.get("media", []):
                url = media.get("identifier")
                if url and any(url.endswith(ext) for ext in (".jpg", ".jpeg", ".png")):
                    urls.append(url)
        return urls
    except Exception as e:
        print(f"    GBIF search failed: {e}")
        return []


def download_and_save(urls, species_dir, target=TARGET):
    """Download up to target images, resize to 224x224, save as JPEG."""
    saved = len(list(species_dir.glob("*.jpg")))
    consec_fail = 0
    for url in urls:
        if saved >= target:
            break
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
                data = resp.read()
            img = Image.open(io.BytesIO(data)).convert("RGB")
            img = img.resize((224, 224), Image.LANCZOS)
            path = species_dir / f"img_{saved:04d}.jpg"
            img.save(path, "JPEG", quality=85)
            saved += 1
            consec_fail = 0
            if saved % 10 == 0 or saved == target:
                print(f"    [{saved}/{target}] ...")
        except urllib.error.HTTPError as e:
            if e.code == 403:
                continue
            consec_fail += 1
        except Exception:
            consec_fail += 1
            if consec_fail >= MAX_CONSECUTIVE_FAIL:
                print(f"    {consec_fail} consecutive failures, skipping rest")
                break
        time.sleep(SLEEP)
    return saved


def main():
    print("=" * 60)
    print("  Download Evaluation Data — 30 images/species")
    print("  Source: GBIF (occurrence search)")
    print("=" * 60)
    total_downloaded = 0
    for species in SPECIES:
        species_dir = OUT_DIR / species
        species_dir.mkdir(exist_ok=True)
        existing = len(list(species_dir.glob("*.jpg")))
        needed = TARGET - existing
        if needed <= 0:
            print(f"  {species}: already {existing} images, skipping")
            continue
        print(f"\n  {species}: need {needed} more")
        urls = gbif_occurrence_images(species)
        print(f"    Found {len(urls)} image URLs")
        saved = download_and_save(urls, species_dir, TARGET)
        total_downloaded += saved
        if saved < needed:
            print(f"    Got {saved}, still missing {needed - saved}")
        else:
            print(f"    Done ({saved})")
    print(f"\n  Downloaded {total_downloaded} new images to {OUT_DIR}/")


if __name__ == "__main__":
    main()
