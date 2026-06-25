#!/usr/bin/env python3
"""
download_knysna_2017.py
=======================
Downloads everything needed to validate BurnTrack on the Knysna fires
of June 6-12, 2017 (Western Cape, South Africa), against a fused 30 m
Landsat-derived burned-area product.

The Knysna fires of June 2017 were one of the most destructive in the
region's history, burning ~15,000 ha and destroying >800 structures.
Well-documented in:
    - Smith et al. (2018) South African Journal of Science
    - Phaduli et al. (2019) South African Geographical Journal
    - Forsyth et al. (2019) Chapter 5, Western Cape Government report

Required env vars (already in .env):
    NASA_FIRMS_API_KEY
    CDSAPI

This script downloads:
    south africa data/knysna_2017/firms_viirs.csv     NASA FIRMS VIIRS 375 m
    south africa data/knysna_2017/firms_modis.csv     NASA FIRMS MODIS 1 km
    south africa data/knysna_2017/era5_knysna_2017.grib  ERA5-Land hourly
    south africa data/knysna_2017/srtm/               SRTM 30 m DEM tiles
    south africa data/knysna_2017/worldcover/         ESA WorldCover 10 m
    south africa data/knysna_2017/landsat/pre/*.tif   Landsat 8 pre-fire
    south africa data/knysna_2017/landsat/post/*.tif  Landsat 8 post-fire

Bbox: lon [22.85, 23.20], lat [-34.15, -33.92]
Fire window: 2017-06-06 to 2017-06-12
Pre-fire composite window:  2017-03-01 -> 2017-06-05
Post-fire composite window: 2017-06-20 -> 2017-09-30
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import planetary_computer
import requests
from dotenv import load_dotenv
from pystac_client import Client

load_dotenv()

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
FIRE_NAME = "knysna_2017"
BBOX = [22.85, -34.15, 23.20, -33.92]  # min_lon, min_lat, max_lon, max_lat
PRE_WINDOW = ("2017-03-01", "2017-06-05")
POST_WINDOW = ("2017-06-20", "2017-09-30")
CLOUD_TOL = 50
OUT_DIR = Path("south africa data") / FIRE_NAME

LANDSAT_BANDS = ["swir22", "nir08", "qa_pixel"]


# ----------------------------------------------------------------------
# NASA FIRMS (active fire)
# ----------------------------------------------------------------------
def download_firms():
    from burntrack.data.firms import download_firms_region
    key = os.environ["NASA_FIRMS_API_KEY"]
    if not key:
        print("  !! NASA_FIRMS_API_KEY missing from .env; skipping FIRMS")
        return
    bbox_str = f"{BBOX[0]},{BBOX[1]},{BBOX[2]},{BBOX[3]}"
    print("\n=== NASA FIRMS VIIRS ===")
    df_v = download_firms_region(
        api_key=key,
        region_name=FIRE_NAME,
        bbox=bbox_str,
        date_start=datetime(2017, 5, 25),
        days_range=20,
        sensor_source="VIIRS_NOAA20_NRT",
    )
    df_v.to_csv(OUT_DIR / "firms_viirs.csv", index=False)
    print(f"  ✓ firms_viirs.csv: {len(df_v):,} detections")

    print("\n=== NASA FIRMS MODIS ===")
    df_m = download_firms_region(
        api_key=key,
        region_name=FIRE_NAME,
        bbox=bbox_str,
        date_start=datetime(2017, 5, 25),
        days_range=20,
        sensor_source="MODIS_NRT",
    )
    df_m.to_csv(OUT_DIR / "firms_modis.csv", index=False)
    print(f"  ✓ firms_modis.csv: {len(df_m):,} detections")


# ----------------------------------------------------------------------
# ERA5 (Copernicus CDS)
# ----------------------------------------------------------------------
def download_era5():
    cds_key = os.environ.get("CDSAPI")
    if not cds_key:
        print("  !! CDSAPI missing from .env; skipping ERA5")
        return
    try:
        import cdsapi
    except ImportError:
        print("  !! cdsapi package not installed; skipping ERA5")
        return
    c = cdsapi.Client()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    target = OUT_DIR / "era5_knysna_2017.grib"
    if target.exists():
        print(f"  ✓ {target.name} (cached)")
        return
    print("\n=== ERA5-Land hourly ===")
    c.retrieve(
        "reanalysis-era5-land",
        {
            "variable": [
                "2m_temperature", "2m_dewpoint_temperature",
                "10m_u_component_of_wind", "10m_v_component_of_wind",
                "total_precipitation",
            ],
            "year": "2017",
            "month": ["05", "06", "07"],
            "day": [f"{d:02d}" for d in range(1, 32)],
            "time": [f"{h:02d}:00" for h in range(24)],
            "area": [BBOX[3], BBOX[0], BBOX[1], BBOX[2]],  # N, W, S, E
            "format": "grib",
        },
        str(target),
    )
    print(f"  ✓ {target.name}")


# ----------------------------------------------------------------------
# SRTM (Planetary Computer)
# ----------------------------------------------------------------------
def download_srtm():
    print("\n=== SRTM 30 m DEM ===")
    catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
    search = catalog.search(collections=["nasadem"], bbox=BBOX)
    items = list(search.items())
    if not items:
        print("  !! No NASADEM tiles in bbox")
        return
    out_dir = OUT_DIR / "srtm"
    out_dir.mkdir(parents=True, exist_ok=True)
    for item in items:
        signed = planetary_computer.sign(item)
        href = signed.assets["elevation"].href
        fname = Path(href).name
        out = out_dir / fname
        if out.exists():
            print(f"  ✓ {fname} (cached)")
            continue
        r = requests.get(href, stream=True, timeout=120)
        r.raise_for_status()
        with open(out, "wb") as f:
            for chunk in r.iter_content(1 << 16):
                f.write(chunk)
        print(f"  ✓ {fname}")


# ----------------------------------------------------------------------
# ESA WorldCover (Planetary Computer)
# ----------------------------------------------------------------------
def download_worldcover():
    print("\n=== ESA WorldCover 10 m ===")
    catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
    search = catalog.search(collections=["esa-worldcover"], bbox=BBOX)
    items = list(search.items())
    if not items:
        print("  !! No WorldCover tiles in bbox")
        return
    out_dir = OUT_DIR / "worldcover"
    out_dir.mkdir(parents=True, exist_ok=True)
    for item in items:
        signed = planetary_computer.sign(item)
        href = signed.assets["map"].href
        fname = Path(href).name
        out = out_dir / fname
        if out.exists():
            print(f"  ✓ {fname} (cached)")
            continue
        r = requests.get(href, stream=True, timeout=120)
        r.raise_for_status()
        with open(out, "wb") as f:
            for chunk in r.iter_content(1 << 16):
                f.write(chunk)
        print(f"  ✓ {fname}")


# ----------------------------------------------------------------------
# Landsat 8/9 (Planetary Computer)
# ----------------------------------------------------------------------
def query_landsat(window, max_cloud=CLOUD_TOL):
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
    signed = planetary_computer.sign(item)
    href = signed.assets[band].href
    r = requests.get(href, stream=True, timeout=120)
    r.raise_for_status()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        for chunk in r.iter_content(1 << 16):
            f.write(chunk)


def download_window(window, label, max_cloud=CLOUD_TOL):
    print(f"\n=== Landsat {label}: {window[0]} -> {window[1]} (cloud<{max_cloud}%) ===")
    items = query_landsat(window, max_cloud=max_cloud)
    print(f"  Found {len(items)} scenes")
    if not items:
        relaxed = max_cloud + 20
        print(f"  No scenes under {max_cloud}% — retrying with cloud<{relaxed}%")
        items = query_landsat(window, max_cloud=relaxed)
        print(f"  Found {len(items)} scenes after relaxation")
    if not items:
        return 0
    n = 0
    for item in items:
        item_id = item.id
        cloud = item.properties.get("eo:cloud_cover", 99)
        out_dir = OUT_DIR / "landsat" / label
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


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-cloud", type=int, default=CLOUD_TOL)
    parser.add_argument("--skip-firms", action="store_true")
    parser.add_argument("--skip-era5", action="store_true")
    parser.add_argument("--skip-srtm", action="store_true")
    parser.add_argument("--skip-worldcover", action="store_true")
    parser.add_argument("--skip-landsat", action="store_true")
    args = parser.parse_args()

    print(f"=== Download all inputs for {FIRE_NAME} ===")
    print(f"  Bbox: {BBOX}")
    print(f"  Output: {OUT_DIR}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not args.skip_firms:
        download_firms()
    if not args.skip_era5:
        download_era5()
    if not args.skip_srtm:
        download_srtm()
    if not args.skip_worldcover:
        download_worldcover()
    if not args.skip_landsat:
        n_pre = download_window(PRE_WINDOW, "pre", max_cloud=args.max_cloud)
        n_post = download_window(POST_WINDOW, "post", max_cloud=args.max_cloud)
        print(f"\n  Pre-fire scenes: {n_pre // len(LANDSAT_BANDS)}")
        print(f"  Post-fire scenes: {n_post // len(LANDSAT_BANDS)}")

    print(f"\n=== All inputs ready in {OUT_DIR} ===")


if __name__ == "__main__":
    main()
