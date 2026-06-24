#!/usr/bin/env python3
"""
build_from_local_firms.py — Build a real dataset using pre-downloaded FIRMS CSVs
================================================================================

This script bypasses the NASA FIRMS API completely by reading the 61,000+ row
dataset you already have in PLBD_robot.

It will:
1. Reconstruct fire propagation (DBSCAN) to find the True Rate of Spread.
2. Fetch Open-Meteo historical weather for those exact locations.
3. Compute the Rothermel baseline.
4. Save the final dataset.
"""

import os
import sys
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from burntrack.data.firms import reconstruct_propagation
from burntrack.data.weather import fetch_weather_for_points
from burntrack.data.real_dataset import compute_rothermel_baseline

def main():
    print("=====================================================")
    print("🔥 BurnTrack - Local FIRMS Data Processor")
    print("=====================================================")

    # 1. Load the local raw FIRMS data
    local_firms_path = "/home/anwar/Documents/PLBD_robot/datasets/fire_ros/raw/firms/firms_viirs_morocco.csv"
    if not os.path.exists(local_firms_path):
        print(f"❌ Error: Could not find local FIRMS data at {local_firms_path}")
        sys.exit(1)

    print(f"📥 Loading local FIRMS data from {local_firms_path}...")
    df = pd.read_csv(local_firms_path)
    df = df[df["acq_date"] != "acq_date"].copy()  # Remove accidental header rows
    df["region"] = "morocco"
    df["latitude"] = pd.to_numeric(df["latitude"])
    df["longitude"] = pd.to_numeric(df["longitude"])
    print(f"📊 Loaded {len(df):,} raw fire detections.")

    # 2. Reconstruct propagation
    print("\n🔍 Reconstructing fire fronts (DBSCAN clustering)...")
    propagation_df = reconstruct_propagation(df)
    if propagation_df.empty:
        print("❌ No propagation vectors extracted.")
        sys.exit(1)
    
    print(f"✅ Extracted {len(propagation_df):,} valid fire vectors with observed ROS.")

    # 3. Fetch weather
    print("\n🌤️ Fetching real historical weather from Open-Meteo...")
    print("   (Note: If this freezes or fails, you have hit the Open-Meteo 10,000 requests/day limit.")
    print("    Connect to a VPN or your phone's hotspot to instantly bypass it!)")
    
    try:
        weather_df = fetch_weather_for_points(propagation_df)
    except Exception as e:
        print(f"\n❌ Weather fetch failed: {e}")
        print("💡 TIP: Change your IP address (VPN/Hotspot) and try again.")
        sys.exit(1)

    if weather_df.empty:
        print("❌ Unable to associate weather data.")
        sys.exit(1)

    # 4. Compute Rothermel baseline
    print("\n🧮 Computing Rothermel baseline for all points...")
    dataset_df = compute_rothermel_baseline(weather_df)

    # 5. Save output
    output_path = os.path.join(PROJECT_ROOT, "data", "processed", "local_african_dataset.csv")
    dataset_df.to_csv(output_path, index=False)
    
    print("=====================================================")
    print("🎉 PIPELINE COMPLETE")
    print("=====================================================")
    print(f"📁 Output file          : {output_path}")
    print(f"📈 Total Training Rows  : {len(dataset_df):,}")
    print(f"🔥 ROS observed (mean)  : {dataset_df['ros_observed'].mean():.3f} m/min")
    print(f"📐 Delta ROS (mean)     : {dataset_df['delta_ros'].mean():.3f} m/min")
    print("=====================================================")

if __name__ == "__main__":
    main()
