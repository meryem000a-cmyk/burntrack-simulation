"""
firms.py — NASA FIRMS active fire data download and propagation reconstruction.
================================================================================

Extracted from data_pipeline/build_real_dataset.py.

Provides:
    - Multi-region FIRMS download with parallel requests
    - DBSCAN-based propagation reconstruction (fire front tracking)
    - Retry logic and rate limiting for HTTP requests

Africa bounding boxes cover 4 major fire regions:
    west_sahel, central_savanna, east_africa, madagascar
"""

import os
import time
import random
import warnings
import concurrent.futures
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
from sklearn.cluster import DBSCAN

from .weather import request_with_retry

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION
# =============================================================================

AFRICA_REGIONS = {
    "west_sahel": "-18.0,4.0,15.0,16.0",
    "central_savanna": "12.0,-18.0,35.0,-5.0",
    "east_africa": "30.0,-12.0,42.0,5.0",
    "madagascar": "43.0,-26.0,51.0,-11.0",
}

MAX_DISTANCE_M = 1500
MAX_TIME_DIFF_MIN = 1440


# =============================================================================
# FIRMS DOWNLOAD FUNCTIONS
# =============================================================================

def download_firms_region(
    api_key: str,
    region_name: str,
    bbox: str = None,
    date_start: datetime = None,
    days_range: int = 4,
    sensor_source: str = "VIIRS_NOAA20_NRT",
) -> pd.DataFrame:
    """Download FIRMS active fire data for a single region.

    Args:
        api_key: NASA FIRMS API key.
        region_name: Label for the region (used as metadata).
        bbox: Bounding box string "lon1,lat1,lon2,lat2". Uses AFRICA_REGIONS
              if not provided.
        date_start: Start date. Defaults to 10 June 2026.
        days_range: Number of days of data to fetch.
        sensor_source: Sensor identifier (default VIIRS_NOAA20_NRT, 375 m).

    Returns:
        DataFrame with fire detection points.
    """
    if bbox is None:
        bbox = AFRICA_REGIONS.get(region_name)
        if bbox is None:
            raise ValueError(f"Unknown region '{region_name}'. Available: {list(AFRICA_REGIONS.keys())}")

    if date_start is None:
        date_start = datetime(2026, 6, 10)

    date_str = date_start.strftime("%Y-%m-%d")
    url = (
        f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/"
        f"{api_key}/{sensor_source}/{bbox}/{days_range}/{date_str}"
    )

    try:
        response = request_with_retry(url)
        df = pd.read_csv(response.url)
        print(f"  [{region_name.upper()}] Loaded: {len(df)} detections.")
        df["region"] = region_name
        return df
    except Exception as e:
        print(f"  [{region_name.upper()}] Download error: {e}")
        return pd.DataFrame()


def download_all_africa_fires(
    api_key: str,
    regions_dict: dict = None,
    date_start: datetime = None,
    days_range: int = 4,
    sensor_source: str = "VIIRS_NOAA20_NRT",
) -> pd.DataFrame:
    """Download active fire data for all African regions in parallel.

    Args:
        api_key: NASA FIRMS API key.
        regions_dict: Dict of {name: bbox} per region. Defaults to AFRICA_REGIONS.
        date_start: Start date. Defaults to 10 June 2026.
        days_range: Number of days of data.
        sensor_source: Sensor identifier.

    Returns:
        Combined DataFrame with deduplicated fire points.
    """
    if regions_dict is None:
        regions_dict = AFRICA_REGIONS

    if date_start is None:
        date_start = datetime(2026, 6, 10)

    print(
        f"  Launching parallel download for Africa "
        f"({len(regions_dict)} regions)..."
    )

    all_dfs = []
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=len(regions_dict)
    ) as executor:
        futures = {
            executor.submit(
                download_firms_region,
                api_key,
                name,
                bbox,
                date_start,
                days_range,
                sensor_source,
            ): name
            for name, bbox in regions_dict.items()
        }
        for future in concurrent.futures.as_completed(futures):
            df = future.result()
            if not df.empty:
                all_dfs.append(df)

    if not all_dfs:
        return pd.DataFrame()

    combined = pd.concat(all_dfs, ignore_index=True)
    if "latitude" in combined.columns:
        combined = combined.drop_duplicates(
            subset=["latitude", "longitude", "acq_date", "acq_time"]
        )
    print(f"  Total unique points across Africa: {len(combined)}")
    return combined


# =============================================================================
# PROPAGATION RECONSTRUCTION VIA DBSCAN (from build_real_dataset.py lines 166-250)
# =============================================================================

def reconstruct_propagation(df: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct rate of spread (ROS_observed) via spatio-temporal tracking.

    Uses DBSCAN spatial clustering per region, then matches each fire
    detection to its nearest predecessor within MAX_TIME_DIFF_MIN to
    compute effective ROS and spread direction.

    Args:
        df: DataFrame with columns:
            latitude, longitude, acq_date, acq_time, confidence,
            bright_ti4, region.

    Returns:
        DataFrame with columns:
            latitude, longitude, datetime, ros_observed, spread_direction,
            confidence, brightness_k, region.
    """
    print("  Reconstructing spatio-temporal propagation...")

    if len(df) < 2:
        return pd.DataFrame()

    df = df.copy()

    # Convert acq_date and acq_time to datetime
    df["acq_time_str"] = df["acq_time"].astype(str).str.zfill(4)
    df["datetime_str"] = (
        df["acq_date"]
        + " "
        + df["acq_time_str"].str[:2]
        + ":"
        + df["acq_time_str"].str[2:]
    )
    df["datetime"] = pd.to_datetime(df["datetime_str"], format="%Y-%m-%d %H:%M")

    df = df.sort_values("datetime").reset_index(drop=True)

    # Simple cartographic projection
    mean_lat = df["latitude"].mean()
    lat_to_m = 111000.0
    lon_to_m = 111000.0 * np.cos(np.radians(mean_lat))
    df["x"] = df["longitude"] * lon_to_m
    df["y"] = df["latitude"] * lat_to_m

    propagations = []

    for region_name, r_group in df.groupby("region"):
        if len(r_group) < 2:
            continue

        coords = r_group[["x", "y"]].values
        db = DBSCAN(eps=3000, min_samples=2).fit(coords)
        r_group = r_group.copy()
        r_group["fire_id"] = db.labels_

        r_group = r_group[r_group["fire_id"] != -1].reset_index(drop=True)

        for fire_id, f_group in r_group.groupby("fire_id"):
            f_group = f_group.sort_values("datetime")

            for i in range(1, len(f_group)):
                target = f_group.iloc[i]
                predecessors = f_group.iloc[:i]

                time_diffs = (
                    (target["datetime"] - predecessors["datetime"]).dt.total_seconds()
                    / 60.0
                )
                valid_idx = (time_diffs > 0) & (time_diffs <= MAX_TIME_DIFF_MIN)

                if not valid_idx.any():
                    continue

                valid_predecessors = predecessors[valid_idx]
                time_diffs_valid = time_diffs[valid_idx]

                dists = np.sqrt(
                    (target["x"] - valid_predecessors["x"]) ** 2
                    + (target["y"] - valid_predecessors["y"]) ** 2
                )

                min_dist_idx = dists.idxmin()
                dist_m = dists[min_dist_idx]
                time_min = time_diffs_valid[min_dist_idx]

                if dist_m <= MAX_DISTANCE_M and time_min > 0:
                    ros_obs = dist_m / time_min

                    if 0.1 <= ros_obs <= 40.0:
                        pred_point = valid_predecessors.loc[min_dist_idx]
                        dx = target["x"] - pred_point["x"]
                        dy = target["y"] - pred_point["y"]
                        direction_deg = np.degrees(np.arctan2(dx, dy)) % 360.0

                        propagations.append(
                            {
                                "latitude": target["latitude"],
                                "longitude": target["longitude"],
                                "datetime": target["datetime"],
                                "ros_observed": ros_obs,
                                "spread_direction": direction_deg,
                                "confidence": target["confidence"],
                                "brightness_k": target["bright_ti4"],
                                "region": region_name,
                            }
                        )

    res_df = pd.DataFrame(propagations)
    print(f"  Propagation reconstructed: {len(res_df)} valid velocity vectors.")
    return res_df
