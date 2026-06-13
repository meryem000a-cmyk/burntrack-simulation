#!/usr/bin/env python3
"""
boost_underrepresented.py — Downloads extra images ONLY for species
that are underrepresented in the current dataset.

Current distribution (17 species, mapped from 34 classes):
  erica:           325  ← CRITICAL (needs ~2000 more)
  khaya:          1024  ← LOW
  ficus:          1286  ← LOW
  macaranga:      1302  ← LOW
  brachystegia:   1330  ← LOW
  colophospermum: 1394  ← LOW
  ---------------------
  protea:         1551  (acceptable)
  themeda:        1546
  adansonia:      1634
  tamarix:        1835
  ... rest are 2500+

This script targets the 6 weakest species and downloads extra
images from GBIF + iNaturalist to bring them closer to ~2500.

⏳ LONG RUNNING: ~30-60 minutes (network-bound, rate-limited by iNaturalist API)
   Run it yourself:  ./flora_env/bin/python boost_underrepresented.py
"""

import os
import sys

# Add parent to path so we can import the existing acquire functions
sys.path.insert(0, os.path.dirname(__file__))
from acquire_data import acquire_gbif_african_flora, acquire_inaturalist_flora

OUTPUT_BASE = os.path.join(os.path.dirname(__file__), "datasets", "yolo_flora", "images", "train")

# Species that need more data, with how many to download
# Target: bring each up to ~2500 total
BOOST_TARGETS = {
    "erica":           {"need": 2200, "gbif_key": 2874415, "inat_id": 55776},
    "khaya":           {"need": 1500, "gbif_key": 3190507, "inat_id": 126187},
    "ficus":           {"need": 1200, "gbif_key": 3097368, "inat_id": 50999},
    "macaranga":       {"need": 1200, "gbif_key": 3073879, "inat_id": 133556},
    "brachystegia":    {"need": 1200, "gbif_key": 2952646, "inat_id": 139468},
    "colophospermum":  {"need": 1100, "gbif_key": 2974566, "inat_id": 428750},
}

# Africa place_id for iNaturalist
AFRICA_PLACE_ID = 97394

def main():
    print("=" * 60)
    print("  BOOSTING UNDERREPRESENTED SPECIES")
    print("=" * 60)

    for species, info in BOOST_TARGETS.items():
        need = info["need"]
        gbif_share = need // 2
        inat_share = need - gbif_share
        out_dir = os.path.join(OUTPUT_BASE)

        # Download to a temp staging dir, then we'll create proper labels
        staging = os.path.join(os.path.dirname(__file__), "datasets", "boost_staging", species)
        os.makedirs(staging, exist_ok=True)

        existing = len([f for f in os.listdir(staging) if f.endswith((".jpg", ".jpeg", ".png"))]) if os.path.exists(staging) else 0
        if existing >= need:
            print(f"\n  ✅ {species}: already have {existing} staged images, skipping")
            continue

        remaining = need - existing
        print(f"\n  📥 {species}: need {need}, have {existing}, downloading {remaining} more...")

        # GBIF first (faster, no rate limit)
        print(f"    [GBIF] Downloading up to {gbif_share} images...")
        acquire_gbif_african_flora(
            taxon_key=info["gbif_key"],
            output_dir=staging,
            max_records=gbif_share,
        )

        # iNaturalist (slower, 1.2s rate limit per image)
        current = len([f for f in os.listdir(staging) if f.endswith((".jpg", ".jpeg", ".png"))])
        still_need = need - current
        if still_need > 0:
            print(f"    [iNaturalist] Downloading up to {still_need} more...")
            acquire_inaturalist_flora(
                taxon_id=info["inat_id"],
                place_id=AFRICA_PLACE_ID,
                output_dir=staging,
                max_records=still_need,
            )

        final_count = len([f for f in os.listdir(staging) if f.endswith((".jpg", ".jpeg", ".png"))])
        print(f"    ✅ {species}: {final_count} images staged")

    print(f"\n{'='*60}")
    print(f"  DONE — Staged images are in datasets/boost_staging/")
    print(f"  Next step: integrate into yolo_flora dataset with proper labels")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
