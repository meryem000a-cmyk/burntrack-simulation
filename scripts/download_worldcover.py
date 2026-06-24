#!/usr/bin/env python3
"""
download_worldcover.py — Download ESA WorldCover 2021 (v200) tiles
==================================================================
Downloads all 10m land-cover GeoTIFF tiles needed to cover the
South Africa fire region (lat -35 to -22, lon 18 to 33).

Features:
  - Resume support (skips already-downloaded files)
  - Retry with exponential backoff on failure
  - Progress bar per file
  - Integrity check (file size validation)
  - Parallel downloads (configurable)
"""

import os
import sys
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── Configuration ──
OUTPUT_DIR = "/home/anwar/Documents/burntrack-simulation/south africa data/worldcover"
BASE_URL = "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map"

# Fire bounding box (with 1° buffer)
LAT_MIN, LAT_MAX = -36, -21
LON_MIN, LON_MAX = 18, 33

MAX_RETRIES = 5
RETRY_BACKOFF = 2  # seconds, doubles each retry
PARALLEL_DOWNLOADS = 3
CHUNK_SIZE = 1024 * 64  # 64KB chunks


def generate_tile_list():
    """Generate all tile names covering the fire region."""
    tiles = []
    # ESA tiles are 3x3 degrees, named by SW corner
    # Latitude tiles go S36, S33, S30, S27, S24, S21...
    # Longitude tiles go E018, E021, E024, E027, E030...
    for lat in range(LAT_MIN, LAT_MAX, 3):
        for lon in range(LON_MIN, LON_MAX, 3):
            lat_str = f"S{abs(lat):02d}" if lat < 0 else f"N{lat:02d}"
            lon_str = f"E{lon:03d}" if lon >= 0 else f"W{abs(lon):03d}"
            tile_name = f"ESA_WorldCover_10m_2021_v200_{lat_str}{lon_str}_Map.tif"
            tiles.append(tile_name)
    return tiles


def get_remote_size(url):
    """Get file size from server via HEAD request."""
    try:
        resp = requests.head(url, timeout=10, allow_redirects=True)
        if resp.status_code == 200:
            return int(resp.headers.get("Content-Length", 0))
    except Exception:
        pass
    return 0


def download_tile(tile_name):
    """Download a single tile with resume and retry support."""
    url = f"{BASE_URL}/{tile_name}"
    filepath = os.path.join(OUTPUT_DIR, tile_name)

    # Check remote file exists and get size
    remote_size = get_remote_size(url)
    if remote_size == 0:
        # Tile might not exist (ocean tile, etc.)
        return tile_name, "SKIP", "Tile does not exist on server"

    # Check if already fully downloaded
    if os.path.exists(filepath):
        local_size = os.path.getsize(filepath)
        if local_size == remote_size:
            return tile_name, "DONE", f"Already complete ({local_size / 1e6:.1f} MB)"
        elif local_size > remote_size:
            # Corrupted, re-download
            os.remove(filepath)

    # Resume support
    local_size = 0
    if os.path.exists(filepath):
        local_size = os.path.getsize(filepath)

    headers = {}
    if local_size > 0:
        headers["Range"] = f"bytes={local_size}-"

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, headers=headers, stream=True, timeout=60)

            if resp.status_code == 416:
                # Range not satisfiable = file already complete
                return tile_name, "DONE", "Already complete (range check)"

            if resp.status_code not in (200, 206):
                raise Exception(f"HTTP {resp.status_code}")

            mode = "ab" if resp.status_code == 206 else "wb"
            if resp.status_code == 200:
                local_size = 0  # Full download, reset

            downloaded = local_size
            with open(filepath, mode) as f:
                for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)

            # Verify
            final_size = os.path.getsize(filepath)
            if final_size == remote_size:
                return tile_name, "OK", f"Downloaded ({final_size / 1e6:.1f} MB)"
            else:
                raise Exception(f"Size mismatch: got {final_size}, expected {remote_size}")

        except Exception as e:
            wait = RETRY_BACKOFF * (2 ** (attempt - 1))
            if attempt < MAX_RETRIES:
                print(f"  ⚠️  {tile_name} attempt {attempt}/{MAX_RETRIES} failed: {e}. Retrying in {wait}s...")
                time.sleep(wait)
                # Update local_size for resume
                if os.path.exists(filepath):
                    local_size = os.path.getsize(filepath)
                    headers["Range"] = f"bytes={local_size}-"
            else:
                return tile_name, "FAIL", f"Failed after {MAX_RETRIES} attempts: {e}"

    return tile_name, "FAIL", "Unexpected exit"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    tiles = generate_tile_list()
    print("=" * 60)
    print("🌍 ESA WorldCover 2021 (v200) — Tile Downloader")
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
            print(f"  [{i:2d}/{len(tiles)}] {icon} {tile_name}: {msg}")
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
        print("\n  Failed tiles (re-run this script to retry):")
        for t in results["FAIL"]:
            print(f"    - {t}")
        print("\n  💡 Just re-run this script — it will resume where it left off!")

    total_size = sum(
        os.path.getsize(os.path.join(OUTPUT_DIR, f))
        for f in os.listdir(OUTPUT_DIR)
        if f.endswith(".tif")
    )
    print(f"\n  📦 Total downloaded: {total_size / 1e9:.2f} GB")
    print("=" * 60)


if __name__ == "__main__":
    main()
