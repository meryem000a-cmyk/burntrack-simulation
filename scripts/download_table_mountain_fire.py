#!/usr/bin/env python3
"""
download_table_mountain_fire.py

Downloads the real FIRMS and Open-Meteo data for the famous 
April 2021 Table Mountain / Cape Town Wildfire.

This fire occurred in the Fynbos biome (Protea and Erica), which 
perfectly matches the vision model classes found in infer.py!
"""

import os
import sys
from datetime import datetime
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from burntrack.data.firms import download_firms_region, reconstruct_propagation
from burntrack.data.weather import fetch_weather_for_points
from burntrack.data.real_dataset import compute_rothermel_baseline

def main():
    api_key = os.environ.get("NASA_FIRMS_API_KEY", "bb880143fa048ebb2da6d6d0057ae5ef")
    
    print("=====================================================")
    print("🏔️  Table Mountain Fire (April 2021) Dataset Builder")
    print("=====================================================")
    
    # 1. South Africa Fynbos (Cape Town) Bounding Box
    cape_town_bbox = "18.0,-35.0,21.0,-33.0"  # lon_min, lat_min, lon_max, lat_max
    
    # Fire dates: April 18 to April 21, 2021
    date_start = datetime(2021, 4, 18)
    
    print("🛰️  Downloading NASA FIRMS data for Cape Town...")
    try:
        raw_df = download_firms_region(
            api_key=api_key,
            region_name="south_africa_fynbos",
            bbox=cape_town_bbox,
            date_start=date_start,
            days_range=4,  # April 18, 19, 20, 21
            chunk_days=4,
            sensor_source="VIIRS_SNPP_SP"
        )
        print(f"📊 Downloaded {len(raw_df)} fire detections.")
    except Exception as e:
        print(f"❌ FIRMS download failed: {e}")
        sys.exit(1)

    if raw_df.empty:
        print("❌ No fires found in that range. Exiting.")
        sys.exit(1)

    # 2. Reconstruct
    print("\n🔍 Clustering fire vectors...")
    propagation_df = reconstruct_propagation(raw_df)
    print(f"✅ Extracted {len(propagation_df)} valid fire vectors.")
    
    if propagation_df.empty:
        print("❌ Not enough vectors to train on. Exiting.")
        sys.exit(1)

    # 3. Weather
    print("\n🌤️ Fetching Open-Meteo Historical Weather (April 2021)...")
    weather_df = fetch_weather_for_points(propagation_df)
    
    # 4. Rothermel
    print("\n🧮 Computing Rothermel Baseline...")
    dataset_df = compute_rothermel_baseline(weather_df)
    
    # 5. Save
    output_path = os.path.join(PROJECT_ROOT, "data", "processed", "table_mountain_2021.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    dataset_df.to_csv(output_path, index=False)
    
    print(f"🎉 Success! Real Table Mountain fire dataset saved to: {output_path}")

if __name__ == "__main__":
    main()
