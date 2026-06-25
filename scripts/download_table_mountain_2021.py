#!/usr/bin/env python3
"""
download_table_mountain_2021.py
================================
Downloads everything needed to validate BurnTrack on the Table Mountain
fire of April 18-30, 2021 (Cape Town, South Africa), against a fused
30 m Landsat-derived burned-area product.

Already in the repo (gitignored, in 'south africa data/'):
    firms/fire_archive_SV-C2_765831.shp   VIIRS 375 m hotspots (Apr 2021)
    firms/fire_archive_M-C61_765830.shp   MODIS 1 km hotspots (Apr 2021)
    data.grib                             ERA5-Land hourly (Mar-Apr 2021)
    srtm/S33E018.hgt, S34E018.hgt, ...    SRTM 30 m DEM tiles
    worldcover/ESA_WorldCover_*.tif       ESA WorldCover 10 m land cover

This script downloads:
    south africa data/table_mountain_2021/landsat/pre/*.tif   Landsat 8/9 pre-fire
    south africa data/table_mountain_2021/landsat/post/*.tif  Landsat 8/9 post-fire

Required env vars (already in .env):
    NASA_FIRMS_API_KEY
    CDSAPI

Bbox: lon [18.418, 18.470], lat [-33.962, -33.933]
Fire window: 2021-04-18 to 2021-04-30
Pre-fire composite window:  2021-03-01 -> 2021-04-17
Post-fire composite window: 2021-04-19 -> 2021-07-31
"""

import argparse
import os
import sys
from pathlib import Path

import planetary_computer
import requests
from dotenv import load_dotenv
from pystac_client import Client

load_dotenv()

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
FIRE_NAME = "table_mountain_2021"
BBOX = [18.418, -33.962, 18.470, -33.933]  # min_lon, min_lat, max_lon, max_lat
PRE_WINDOW = ("2021-03-01", "2021-04-17")
POST_WINDOW = ("2021-04-19", "2021-07-31")
CLOUD_TOL = 50
OUT_DIR = Path("south africa data") / FIRE_NAME

# Landsat Collection 2 Level-2 bands needed for dNBR
LANDSAT_BANDS = ["swir22", "nir08", "qa_pixel"]


def query_landsat(window, max_cloud=CLOUD_TOL):
    """Query STAC for Landsat 8/9 scenes in the bbox + date window."""
    catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
    search = catalog.search(
        collections=["landsat-c2-l2"],
        bbox=BBOX,
        datetime=f"{window[0]}T00:00:00Z/{window[1]}T23:59:59Z",
        query={
            "eo:cloud_cover": {"lt": max_cloud},
            "platform": {"in": ["landsat-8", "landsat-9"]},
        },
    )
    return list(search.items())


def download_asset(item, band, out_path):
    """Sign a STAC asset and stream it to disk."""
    signed = planetary_computer.sign(item)
    href = signed.assets[band].href
    r = requests.get(href, stream=True, timeout=120)
    r.raise_for_status()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        for chunk in r.iter_content(1 << 16):
            f.write(chunk)


def download_window(window, label, max_cloud=CLOUD_TOL):
    """Download all bands for all Landsat scenes in the window."""
    print(f"\n=== {label} window: {window[0]} -> {window[1]} (cloud<{max_cloud}%) ===")
    items = query_landsat(window, max_cloud=max_cloud)
    print(f"  Found {len(items)} scenes")
    if not items:
        # Auto-relax
        relaxed = max_cloud + 20
        print(f"  No scenes under {max_cloud}% — retrying with cloud<{relaxed}%")
        items = query_landsat(window, max_cloud=relaxed)
        print(f"  Found {len(items)} scenes after relaxation")
    if not items:
        print(f"  !! No Landsat scenes for {label}; aborting that window")
        return 0

    n = 0
    for item in items:
        item_id = item.id
        cloud = item.properties.get("eo:cloud_cover", 99)
        date = item.datetime.strftime("%Y%m%d")
        out_dir = OUT_DIR / "landsat" / label
        # Only download bands we need (saves bandwidth)
        for band in LANDSAT_BANDS:
            if band not in item.assets:
                continue
            out_path = out_dir / f"{item_id}_{band}.tif"
            if out_path.exists():
                print(f"  ✓ {out_path.name} (cached)")
                n += 1
                continue
            try:
                download_asset(item, band, out_path)
                print(f"  ✓ {out_path.name} (cloud={cloud:.1f}%)")
                n += 1
            except Exception as e:
                print(f"  ✗ {item_id} {band}: {e}")
    return n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-cloud", type=int, default=CLOUD_TOL)
    parser.add_argument("--skip-existing", action="store_true", default=True)
    args = parser.parse_args()

    print(f"=== Landsat download for {FIRE_NAME} ===")
    print(f"  Bbox: {BBOX}")
    print(f"  Output: {OUT_DIR}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    n_pre = download_window(PRE_WINDOW, "pre", max_cloud=args.max_cloud)
    n_post = download_window(POST_WINDOW, "post", max_cloud=args.max_cloud)

    print(f"\n=== Done ===")
    print(f"  Pre-fire scenes downloaded: {n_pre // len(LANDSAT_BANDS)}")
    print(f"  Post-fire scenes downloaded: {n_post // len(LANDSAT_BANDS)}")
    print(f"  Output: {OUT_DIR / 'landsat'}")


if __name__ == "__main__":
    main()
