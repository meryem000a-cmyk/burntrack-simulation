#!/usr/bin/env python3
"""
download_srtm.py — Download SRTM 30m elevation tiles
=====================================================
Downloads all 1°×1° SRTM HGT tiles needed to cover the
South Africa fire region (lat -35 to -22, lon 18 to 33).

Source: AWS Terrain Tiles (public, no login required)

Features:
  - Resume support (skips already-downloaded files)
  - Retry with exponential backoff
  - Auto-decompresses .hgt.gz → .hgt
  - Parallel downloads (configurable)
"""

import os
import sys
import time
import gzip
import shutil
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── Configuration ──
OUTPUT_DIR = "/home/anwar/Documents/burntrack-simulation/south africa data/srtm"
BASE_URL = "https://elevation-tiles-prod.s3.amazonaws.com/skadi"

# Fire bounding box (with 1° buffer)
LAT_MIN, LAT_MAX = -36, -22
LON_MIN, LON_MAX = 17, 33

MAX_RETRIES = 5
RETRY_BACKOFF = 2
PARALLEL_DOWNLOADS = 4
CHUNK_SIZE = 1024 * 64


def generate_tile_list():
    """Generate all SRTM tile names covering the fire region."""
    tiles = []
    for lat in range(LAT_MIN, LAT_MAX):
        for lon in range(LON_MIN, LON_MAX):
            lat_str = f"S{abs(lat):02d}" if lat < 0 else f"N{lat:02d}"
            lon_str = f"E{lon:03d}" if lon >= 0 else f"W{abs(lon):03d}"
            tile_name = f"{lat_str}{lon_str}"
            tiles.append(tile_name)
    return tiles


def download_tile(tile_name):
    """Download and decompress a single SRTM tile."""
    lat_dir = tile_name[:3]  # e.g. "S34"
    url = f"{BASE_URL}/{lat_dir}/{tile_name}.hgt.gz"
    gz_path = os.path.join(OUTPUT_DIR, f"{tile_name}.hgt.gz")
    hgt_path = os.path.join(OUTPUT_DIR, f"{tile_name}.hgt")

    # Already decompressed?
    if os.path.exists(hgt_path) and os.path.getsize(hgt_path) > 0:
        size_mb = os.path.getsize(hgt_path) / 1e6
        return tile_name, "DONE", f"Already have {size_mb:.1f} MB"

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, stream=True, timeout=30)

            if resp.status_code == 404 or resp.status_code == 403:
                return tile_name, "SKIP", "No data (ocean/void)"

            if resp.status_code != 200:
                raise Exception(f"HTTP {resp.status_code}")

            # Download compressed file
            with open(gz_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                    if chunk:
                        f.write(chunk)

            # Decompress
            with gzip.open(gz_path, "rb") as f_in:
                with open(hgt_path, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)

            # Clean up .gz
            os.remove(gz_path)

            size_mb = os.path.getsize(hgt_path) / 1e6
            return tile_name, "OK", f"Downloaded & extracted ({size_mb:.1f} MB)"

        except Exception as e:
            wait = RETRY_BACKOFF * (2 ** (attempt - 1))
            if attempt < MAX_RETRIES:
                print(f"  ⚠️  {tile_name} attempt {attempt}/{MAX_RETRIES} failed: {e}. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                # Clean up partial files
                for p in [gz_path, hgt_path]:
                    if os.path.exists(p):
                        os.remove(p)
                return tile_name, "FAIL", f"Failed after {MAX_RETRIES} attempts: {e}"

    return tile_name, "FAIL", "Unexpected exit"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    tiles = generate_tile_list()
    print("=" * 60)
    print("⛰️  SRTM 30m Elevation — Tile Downloader")
    print("=" * 60)
    print(f"  Region: lat [{LAT_MIN}, {LAT_MAX}], lon [{LON_MIN}, {LON_MAX}]")
    print(f"  Tiles to check: {len(tiles)}")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"  Parallel downloads: {PARALLEL_DOWNLOADS}")
    print(f"  Resume support: ✅")
    print("=" * 60)

    results = {"OK": [], "DONE": [], "SKIP": [], "FAIL": []}

    with ThreadPoolExecutor(max_workers=PARALLEL_DOWNLOADS) as executor:
        futures = {executor.submit(download_tile, t): t for t in tiles}
        for i, future in enumerate(as_completed(futures), 1):
            tile_name, status, msg = future.result()
            icon = {"OK": "✅", "DONE": "⏭️", "SKIP": "🌊", "FAIL": "❌"}[status]
            print(f"  [{i:3d}/{len(tiles)}] {icon} {tile_name}: {msg}")
            results[status].append(tile_name)

    # Summary
    print("\n" + "=" * 60)
    print("📊 SUMMARY")
    print("=" * 60)
    print(f"  ✅ Downloaded:      {len(results['OK'])}")
    print(f"  ⏭️  Already had:     {len(results['DONE'])}")
    print(f"  🌊 Skipped (ocean): {len(results['SKIP'])}")
    print(f"  ❌ Failed:          {len(results['FAIL'])}")

    if results["FAIL"]:
        print("\n  Failed tiles (re-run to retry):")
        for t in results["FAIL"]:
            print(f"    - {t}")
        print("\n  💡 Just re-run — it will resume where it left off!")

    total_size = sum(
        os.path.getsize(os.path.join(OUTPUT_DIR, f))
        for f in os.listdir(OUTPUT_DIR)
        if f.endswith(".hgt")
    )
    print(f"\n  📦 Total on disk: {total_size / 1e9:.2f} GB")
    print("=" * 60)


if __name__ == "__main__":
    main()
