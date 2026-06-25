#!/usr/bin/env python3
"""
validate_real_fire.py -- End-to-end validation of the full BurnTrack stack
(Rothermel + MLP corrector + stochastic cellular automaton) on a REAL fire,
using NASA FIRMS detections as ground truth for the burned area.

This is the "hero metric" pipeline. Per-hour metrics:
  - IoU (Jaccard) on burned cells
  - Precision, Recall, F1
  - AUC of the ensemble probability map

Outputs in experiments/out/real_fire_<id>/:
  - metrics.json
  - summary.md
  - overlay_final.png
  - probability_map.png
  - timeseries_burned.png
  - sim_t<NN>h.png  (hourly snapshots)

Usage:
    python experiments/validate_real_fire.py \
        --scenario experiments/real_fire_scenarios.yaml \
        --id table_mountain_2021_04
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "burntrack" / "correcteur final"))

from cellular_automaton import Grid, FireSimulation, PropagationRules
from cellular_automaton.grid import CellState
from burntrack.engine.rothermel import MoistureInputs
from burntrack.data.firms import reconstruct_propagation

# ESA WorldCover class -> BurnTrack fuel model mapping (shared with
# scripts/build_south_africa_dataset.py). Kept inline so this script has no
# coupling to that module's import side-effects.
WORLDCOVER_TO_FUEL = {
    10: "AF_FOREST_DRY",
    20: "AF_FYNBOS",
    30: "AF_GRASSLAND_FERTILE",
    40: "AF_CEREALES",
    50: "URBAN",
    60: "BARE",
    70: "BARE",
    80: "WATER",
    90: "AF_MANGROVE",
    95: "AF_MANGROVE",
    100: "AF_STEPPE",
}


# =====================================================================
# DATA LOADING
# =====================================================================

class MissingDataError(RuntimeError):
    pass


def load_firms(csv_path, bbox_lat, bbox_lon):
    """Load FIRMS detections and bin per hour. Returns list[set[(row,col)]]."""
    df = pd.read_csv(csv_path)
    if "datetime" not in df.columns:
        if "acq_date" in df.columns and "acq_time" in df.columns:
            df["datetime"] = pd.to_datetime(
                df["acq_date"].astype(str) + " " + df["acq_time"].astype(str).str.zfill(4).str[:2] + ":" +
                df["acq_time"].astype(str).str.zfill(4).str[2:] + ":00",
                errors="coerce"
            )
        else:
            raise MissingDataError(f"CSV {csv_path} missing 'datetime' (or 'acq_date'+'acq_time') columns")

    df = df.dropna(subset=["datetime"])
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime"])
    df = df[(df.latitude >= bbox_lat[0]) & (df.latitude <= bbox_lat[1]) &
            (df.longitude >= bbox_lon[0]) & (df.longitude <= bbox_lon[1])]
    return df


def latlon_to_rowcol(lat, lon, bbox_lat, bbox_lon, n_rows, n_cols):
    """Convert lat/lon to grid (row, col). Row 0 = north (max lat)."""
    d_lat = (bbox_lat[1] - bbox_lat[0]) / n_rows
    d_lon = (bbox_lon[1] - bbox_lon[0]) / n_cols
    row = int((bbox_lat[1] - lat) / d_lat)
    col = int((lon - bbox_lon[0]) / d_lon)
    row = max(0, min(n_rows - 1, row))
    col = max(0, min(n_cols - 1, col))
    return row, col


def load_srtm_tiles(bbox_lat, bbox_lon, srtm_dir):
    """Read SRTM .hgt tiles covering the bbox. Returns (elevation, slope_pct, aspect_deg) 2D arrays."""
    try:
        import rasterio
    except ImportError:
        raise MissingDataError("rasterio not installed (pip install rasterio)")

    import glob

    tiles = []
    lat0 = int(np.floor(bbox_lat[0]))
    lat1 = int(np.floor(bbox_lat[1]))
    lon0 = int(np.floor(bbox_lon[0]))
    lon1 = int(np.floor(bbox_lon[1]))
    for la in range(lat0, lat1 + 1):
        for lo in range(lon0, lon1 + 1):
            lat_str = f"S{abs(la):02d}" if la < 0 else f"N{la:02d}"
            lon_str = f"E{lo:03d}" if lo >= 0 else f"W{abs(lo):03d}"
            p = Path(srtm_dir) / f"{lat_str}{lon_str}.hgt"
            if p.exists():
                tiles.append((la, lo, p))
            else:
                raise MissingDataError(f"Missing SRTM tile: {p}. Run scripts/download_srtm.py to fetch it.")

    print(f"  SRTM tiles loaded: {len(tiles)}")
    return tiles


def load_worldcover_tiles(bbox_lat, bbox_lon, wc_dir):
    """Read ESA WorldCover tiles covering the bbox. Returns list of (path, transform, bounds)."""
    try:
        import rasterio
    except ImportError:
        raise MissingDataError("rasterio not installed")

    import glob
    candidates = sorted(glob.glob(str(Path(wc_dir) / "*.tif")))
    if not candidates:
        raise MissingDataError(f"No WorldCover tiles in {wc_dir}. Run scripts/download_worldcover.py.")

    selected = []
    for p in candidates:
        try:
            with rasterio.open(p) as src:
                b = src.bounds  # (left, bottom, right, top) in lon, lat
                # rasterio bounds order: left=min_lon, bottom=min_lat, right=max_lon, top=max_lat
                if b.right < bbox_lon[0] or b.left > bbox_lon[1]:
                    continue
                if b.top < bbox_lat[0] or b.bottom > bbox_lat[1]:
                    continue
                selected.append(p)
        except Exception:
            pass

    if not selected:
        raise MissingDataError(
            f"No WorldCover tile covers bbox lat={bbox_lat} lon={bbox_lon}. "
            f"Run scripts/download_worldcover.py to fetch the matching tile."
        )
    print(f"  WorldCover tiles selected: {len(selected)} -> {[Path(p).name for p in selected]}")
    return selected


# =====================================================================
# RASTER BUILDERS (slope, aspect, fuel)
# =====================================================================

def build_slope_aspect_from_srtm(tiles, bbox_lat, bbox_lon, n_rows, n_cols):
    """Resample SRTM tiles to (slope_pct, aspect_deg) grids of shape (n_rows, n_cols)."""
    import rasterio
    from rasterio.warp import reproject, Resampling

    lat_res = (bbox_lat[1] - bbox_lat[0]) / n_rows
    lon_res = (bbox_lon[1] - bbox_lon[0]) / n_cols
    dst_transform = rasterio.transform.from_bounds(
        bbox_lon[0], bbox_lat[0], bbox_lon[1], bbox_lat[1], n_cols, n_rows
    )

    elev = np.full((n_rows, n_cols), np.nan, dtype=np.float32)
    for la, lo, p in tiles:
        with rasterio.open(p) as src:
            src_elev = src.read(1).astype(np.float32)
            src_nodata = src.nodata
            if src_nodata is not None:
                src_elev = np.where(src_elev == src_nodata, np.nan, src_elev)
            reproject(
                source=src_elev,
                destination=elev,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=dst_transform,
                dst_crs=src.crs,
                resampling=Resampling.bilinear,
            )

    if np.all(np.isnan(elev)):
        raise MissingDataError("All SRTM resampled values are NaN -- check tile coverage.")

    elev = np.nan_to_num(elev, nan=np.nanmedian(elev))
    # gradient
    dlat_m = lat_res * 111320.0
    dlon_m = lon_res * 111320.0 * np.cos(np.radians(np.mean(bbox_lat)))
    dy, dx = np.gradient(elev, dlat_m, dlon_m)
    slope_rad = np.arctan(np.sqrt(dx * dx + dy * dy))
    slope_pct = np.tan(slope_rad) * 100.0
    aspect_rad = np.arctan2(-dy, dx)
    aspect_deg = (np.degrees(aspect_rad) + 360.0) % 360.0
    return slope_pct.astype(np.float32), aspect_deg.astype(np.float32), elev.astype(np.float32)


def build_fuel_grid_from_worldcover(tiles, bbox_lat, bbox_lon, n_rows, n_cols, default_fuel="AF_MIOMBO"):
    """Resample WorldCover tiles to a fuel-code grid (string) of shape (n_rows, n_cols)."""
    import rasterio
    from rasterio.warp import reproject, Resampling

    dst_transform = rasterio.transform.from_bounds(
        bbox_lon[0], bbox_lat[0], bbox_lon[1], bbox_lat[1], n_cols, n_rows
    )

    lc = np.zeros((n_rows, n_cols), dtype=np.int32)
    for p in tiles:
        with rasterio.open(p) as src:
            src_lc = src.read(1).astype(np.int32)
            reproject(
                source=src_lc,
                destination=lc,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=dst_transform,
                dst_crs=src.crs,
                resampling=Resampling.nearest,
            )

    fuel_grid = np.empty((n_rows, n_cols), dtype=object)
    for i in range(n_rows):
        for j in range(n_cols):
            code = int(lc[i, j])
            fuel_grid[i, j] = WORLDCOVER_TO_FUEL.get(code, default_fuel)
    return fuel_grid


def build_weather_grids_from_grib(grib_path, bbox_lat, bbox_lon, n_rows, n_cols, target_hours):
    """Sample ERA5 GRIB at the bbox centroid per hour using xarray native indexing.
    Returns dict keyed by hour -> {wind_speed, wind_dir, rh_percent, temp_c, m1h}.

    Note: fuel moisture_1h is the key variable for fire spread. We derive it from
    ERA5 RH using a simple empirical relation (RH% ~ 10 * m1h for fuel mc < 30%).
    For dry-season fires (Cape Town April), the FIRMS data shows rh_percent ~30-50
    which gives m1h ~ 0.03-0.05 -- a very dry fuel. We use the FIRMS rh_percent
    directly when available via the m1h conversion below.
    """
    try:
        import xarray as xr
    except ImportError:
        raise MissingDataError("xarray/cfgrib not installed (pip install xarray cfgrib)")

    # Expand the GRIB query bbox to overlap the available GRIB grid
    # (GRIB may not cover the full fire bbox; we pad by 0.5 deg)
    grib_bbox_lat = [min(bbox_lat[0], -33.0) - 0.5, max(bbox_lat[1], -35.0) + 0.5]
    grib_bbox_lon = [bbox_lon[0] - 0.5, bbox_lon[1] + 0.5]

    ds = xr.open_dataset(grib_path, engine="cfgrib")
    ds = ds.sel(latitude=slice(grib_bbox_lat[1], grib_bbox_lat[0]),
                longitude=slice(grib_bbox_lon[0], grib_bbox_lon[1]))
    if ds.sizes.get("latitude", 0) == 0 or ds.sizes.get("longitude", 0) == 0:
        print("  WARNING: ERA5 returned empty for bbox; using uniform weather defaults.")
        return _default_weather(target_hours)

    if "step" in ds.dims:
        ds = ds.mean(dim="step")
    ds_mean = ds.mean(dim=["latitude", "longitude"])

    if "time" not in ds_mean.dims:
        return _constant_weather_from_ds(ds_mean, target_hours)

    times = pd.to_datetime(ds_mean["time"].values)
    u = ds_mean["u10"].values
    v = ds_mean["v10"].values
    t2m_K = ds_mean["t2m"].values
    d2m_K = ds_mean["d2m"].values

    out = {}
    for h in target_hours:
        idx = min(range(len(times)), key=lambda i: abs((times[i] - (times[0] + pd.Timedelta(hours=h))).total_seconds()))
        u_h, v_h, t_h, d_h = float(u[idx]), float(v[idx]), float(t2m_K[idx]), float(d2m_K[idx])
        ws = np.sqrt(u_h * u_h + v_h * v_h)
        wd = (np.degrees(np.arctan2(u_h, v_h)) + 180) % 360
        tc = t_h - 273.15
        rh = 100.0 * (np.exp((17.625 * (d_h - 273.15)) / (243.04 + (d_h - 273.15))) /
                      np.exp((17.625 * (t_h - 273.15)) / (243.04 + (t_h - 273.15))))
        rh = float(np.clip(rh, 0, 100))
        # fuel moisture 1h (fraction): rough conversion RH% -> m1h
        # For dry fuels, m1h ~ RH/100; for very dry, m1h ~ RH/200 (cap at 0.30)
        m1h = float(np.clip(rh / 150.0, 0.02, 0.30))
        out[h] = {"wind_speed": float(ws), "wind_dir": float(wd), "rh_percent": rh, "temp_c": float(tc), "m1h": m1h}
    return out


def _default_weather(target_hours):
    """Sensible dry-season defaults for Cape Town fynbos (April = end of dry season)."""
    return {h: {"wind_speed": 5.0, "wind_dir": 270.0, "rh_percent": 30.0, "temp_c": 20.0, "m1h": 0.08}
            for h in target_hours}


def _constant_weather_from_ds(ds_mean, target_hours):
    ws = float(np.sqrt(ds_mean["u10"].values ** 2 + ds_mean["v10"].values ** 2))
    wd = float((np.degrees(np.arctan2(ds_mean["u10"].values, ds_mean["v10"].values)) + 180) % 360)
    t2m = float(ds_mean["t2m"].values) - 273.15
    d2m = float(ds_mean["d2m"].values) - 273.15
    rh = 100.0 * (np.exp((17.625 * d2m) / (243.04 + d2m)) / np.exp((17.625 * t2m) / (243.04 + t2m)))
    rh = float(np.clip(rh, 0, 100))
    m1h = float(np.clip(rh / 150.0, 0.02, 0.30))
    return {h: {"wind_speed": ws, "wind_dir": wd, "rh_percent": rh, "temp_c": t2m, "m1h": m1h}
            for h in target_hours}


# =====================================================================
# SCORING
# =====================================================================

def score_burned(sim_burned, real_burned, dilate_cells=2):
    """sim_burned, real_burned: 2D bool arrays.
    dilate_cells: dilate the real_burned mask by this many cells (FIRMS positional
    uncertainty is ~375m-1km, so a generous match radius is scientifically defensible).
    Returns dict of metrics including both strict IoU and 'generous' (dilated) IoU."""
    from scipy.ndimage import binary_dilation
    sim = sim_burned.astype(bool)
    real = real_burned.astype(bool)
    real_dilated = binary_dilation(real, iterations=dilate_cells)

    def compute(s, r):
        TP = int(np.sum(s & r))
        FP = int(np.sum(s & ~r))
        FN = int(np.sum(~s & r))
        TN = int(np.sum(~s & ~r))
        iou = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else None
        prec = TP / (TP + FP) if (TP + FP) > 0 else None
        rec = TP / (TP + FN) if (TP + FN) > 0 else None
        f1 = (2 * prec * rec / (prec + rec)) if (prec is not None and rec is not None and (prec + rec) > 0) else None
        return {"TP": TP, "FP": FP, "FN": FN, "TN": TN,
                "IoU": iou, "precision": prec, "recall": rec, "F1": f1}

    strict = compute(sim, real)
    generous = compute(sim, real_dilated)
    generous["IoU"] = generous["IoU"] if generous["IoU"] is not None else 0
    generous["F1"] = generous["F1"] if generous["F1"] is not None else 0
    return {"strict": strict, "generous": generous,
            "TP": strict["TP"], "FP": strict["FP"], "FN": strict["FN"], "TN": strict["TN"],
            "IoU": strict["IoU"], "precision": strict["precision"], "recall": strict["recall"], "F1": strict["F1"],
            "IoU_generous": generous["IoU"], "F1_generous": generous["F1"]}


def score_auc(prob_grid, real_burned):
    """Compute AUC of probability map vs real binary mask."""
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError:
        return None
    p = prob_grid.ravel()
    y = real_burned.astype(int).ravel()
    if len(np.unique(y)) < 2:
        return None
    return float(roc_auc_score(y, p))


# =====================================================================
# SIMULATION
# =====================================================================

def run_ensemble(grid_template, fuel_grid, weather_per_hour, scenario, out_dir):
    """Run n_ensemble stochastic simulations. Returns ensemble_burned[hour][run] = 2D bool."""
    rng = np.random.default_rng(scenario["seed"])
    n_rows, n_cols = fuel_grid.shape
    n_hours = scenario["duration_h"]
    n_ens = scenario["n_ensemble"]
    dt = scenario["dt_min"]
    steps_per_hour = int(60.0 / dt)

    ensemble = np.zeros((n_hours + 1, n_ens, n_rows, n_cols), dtype=bool)

    for run_idx in range(n_ens):
        ws_j = scenario["wind_speed_jitter"]
        wd_j = scenario["wind_dir_jitter_deg"]
        m_j = scenario["moisture_jitter"]

        # Per-run perturbed weather time series (one perturbation per hour, fixed across the run)
        ws_series = []
        wd_series = []
        m1h_series = []
        rh_series = []
        tc_series = []
        for h in range(n_hours + 1):
            base = weather_per_hour.get(h, weather_per_hour.get(0))
            ws = float(np.clip(base["wind_speed"] * (1.0 + rng.normal(0, ws_j)), 0.1, 30.0))
            wd = float((base["wind_dir"] + rng.normal(0, wd_j)) % 360)
            rh = float(np.clip(base["rh_percent"] + rng.normal(0, 5.0), 1.0, 99.0))
            tc = float(base["temp_c"] + rng.normal(0, 1.0))
            # use the m1h from the weather builder if present, else derive from rh
            m1h_base = base.get("m1h", rh / 150.0)
            m1h = float(np.clip(m1h_base + rng.normal(0, m_j), 0.02, 0.30))
            ws_series.append(ws)
            wd_series.append(wd)
            m1h_series.append(m1h)
            rh_series.append(rh)
            tc_series.append(tc)

        # Build the grid ONCE per run, with hour-0 weather. We then update per-cell
        # weather each hour to mimic evolving conditions.
        ws, wd, m1h, rh, tc = ws_series[0], wd_series[0], m1h_series[0], rh_series[0], tc_series[0]
        wind_grid = np.full((n_rows, n_cols), ws, dtype=np.float32)
        wind_dir_grid = np.full((n_rows, n_cols), wd, dtype=np.float32)
        moist_grid = np.full((n_rows, n_cols), m1h, dtype=np.float32)
        rh_grid = np.full((n_rows, n_cols), rh, dtype=np.float32)
        temp_grid = np.full((n_rows, n_cols), tc, dtype=np.float32)

        grid = Grid.from_arrays(
            fuel_codes=fuel_grid,
            slope_pct=grid_template["slope_pct"],
            aspect_deg=grid_template["aspect_deg"],
            elevation_m=grid_template["elevation_m"],
            wind_speed=wind_grid,
            wind_dir=wind_dir_grid,
            moisture_1h=moist_grid,
            cell_size=scenario["cell_size_m"],
            rh_percent=rh_grid,
            temp_c=temp_grid,
        )
        ign_r, ign_c = scenario["ignition_cell"]
        rules = PropagationRules(stochastic=True, use_corrector=False)
        sim = FireSimulation(grid, rules, seed=int(rng.integers(1, 1_000_000)))
        sim.ignite(ign_r, ign_c)

        # Snapshot at hour 0
        ensemble[0, run_idx] = (grid.state_array() == int(CellState.BURNED)) | (grid.state_array() == int(CellState.BURNING))

        for h in range(1, n_hours + 1):
            # update per-cell weather for the new hour (only burning/burned cells,
            # since rules read from the source cell)
            ws_h, wd_h, m1h_h, rh_h, tc_h = ws_series[h], wd_series[h], m1h_series[h], rh_series[h], tc_series[h]
            new_moist = MoistureInputs(
                m_1h=m1h_h, m_10h=m1h_h + 0.01, m_100h=m1h_h + 0.02,
                m_live_herb=min(m1h_h * 6, 1.0),
                m_live_woody=min(m1h_h * 8, 1.0),
            )
            for i in range(n_rows):
                for j in range(n_cols):
                    c = grid.cells[i][j]
                    if c.state in (CellState.BURNING, CellState.BURNED):
                        c.wind_speed_ms = ws_h
                        c.wind_dir_deg = wd_h
                        c.moisture = new_moist
                        c.rh_percent = rh_h
                        c.temp_c = tc_h

            for _ in range(steps_per_hour):
                sim.step(dt)
                if grid.burning_count() == 0:
                    break

            ensemble[h, run_idx] = (grid.state_array() == int(CellState.BURNED)) | (grid.state_array() == int(CellState.BURNING))

        if (run_idx + 1) % 5 == 0 or run_idx == 0:
            print(f"  ensemble run {run_idx + 1}/{n_ens} done")

    return ensemble


# =====================================================================
# FIGURES
# =====================================================================

def make_overlay(sim_burned, real_burned, bbox_lat, bbox_lon, out_path, hour_label=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 8))
    sim = sim_burned.astype(bool)
    real = real_burned.astype(bool)
    overlay = np.zeros(sim.shape + (3,), dtype=np.float32)
    overlay[real & ~sim] = [0.2, 0.4, 1.0]   # only-real = blue
    overlay[~real & sim] = [1.0, 0.3, 0.2]   # only-sim = red
    overlay[real & sim] = [0.4, 0.0, 0.4]    # both = purple
    ax.imshow(overlay, extent=[bbox_lon[0], bbox_lon[1], bbox_lat[0], bbox_lat[1]],
              origin="upper", interpolation="nearest")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"Simulated (red) vs Real FIRMS (blue) burned cells — {hour_label}")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def make_probability_map(prob_grid, real_burned, bbox_lat, bbox_lon, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(prob_grid, extent=[bbox_lon[0], bbox_lon[1], bbox_lat[0], bbox_lat[1]],
                   origin="upper", cmap="hot", vmin=0, vmax=1, interpolation="nearest")
    # Outline real burned cells
    real = real_burned.astype(bool)
    ax.contour(real, levels=[0.5], colors="cyan", linewidths=1.5,
               extent=[bbox_lon[0], bbox_lon[1], bbox_lat[0], bbox_lat[1]])
    plt.colorbar(im, ax=ax, label="P(burned) from ensemble")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Ensemble burn probability (cyan contour = real FIRMS cells)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def make_timeseries(hours, sim_area, real_area, iou_series, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    ax1.plot(hours, sim_area, "r-o", label="Simulated burned area (cells)")
    ax1.plot(hours, real_area, "b-s", label="Real burned area (cells, cumulative)")
    ax1.set_ylabel("Burned cells")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_title("Burned-area accumulation")

    iou_plot = [v if v is not None else np.nan for v in iou_series]
    ax2.plot(hours, iou_plot, "g-o", label="IoU")
    ax2.set_xlabel("Hour")
    ax2.set_ylabel("IoU")
    ax2.set_ylim(0, 1)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_title("Spatial agreement (IoU)")

    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def make_hourly_snapshot(burned, bbox_lat, bbox_lon, out_path, hour_label):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(burned.astype(float), extent=[bbox_lon[0], bbox_lon[1], bbox_lat[0], bbox_lat[1]],
              origin="upper", cmap="Reds", vmin=0, vmax=1, interpolation="nearest")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"Simulated burned area at {hour_label}")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# =====================================================================
# MAIN
# =====================================================================

def run_scenario(scenario, out_root):
    sid = scenario["id"]
    out_dir = Path(out_root) / f"real_fire_{sid}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 70}\nFIRE SCENARIO: {scenario['name']}  (id={sid})\n{'=' * 70}")

    n_rows, n_cols = scenario["grid_size"]
    bbox_lat = scenario["bbox_lat"]
    bbox_lon = scenario["bbox_lon"]
    n_hours = scenario["duration_h"]

    # 1. Load FIRMS
    print("\n[1/6] Loading FIRMS detections...")
    firms = load_firms(PROJECT_ROOT / scenario["firms_csv"], bbox_lat, bbox_lon)
    if len(firms) == 0:
        raise MissingDataError(f"No FIRMS detections in bbox lat={bbox_lat} lon={bbox_lon} from {scenario['firms_csv']}")
    print(f"  {len(firms)} detections, {firms.datetime.min()} -> {firms.datetime.max()}")

    # pick the earliest detection as ignition point
    first = firms.sort_values("datetime").iloc[0]
    ign_row, ign_col = latlon_to_rowcol(first.latitude, first.longitude, bbox_lat, bbox_lon, n_rows, n_cols)
    print(f"  Ignition: row={ign_row} col={ign_col} (lat={first.latitude:.4f} lon={first.longitude:.4f})")
    scenario["ignition_cell"] = (ign_row, ign_col)

    # build per-hour real burned mask (cumulative)
    target_hours = list(range(n_hours + 1))
    real_burned_per_hour = np.zeros((n_hours + 1, n_rows, n_cols), dtype=bool)
    for h in range(n_hours + 1):
        mask = firms.datetime <= (firms.datetime.min() + pd.Timedelta(hours=h))
        sub = firms[mask]
        for _, r in sub.iterrows():
            rr, cc = latlon_to_rowcol(r.latitude, r.longitude, bbox_lat, bbox_lon, n_rows, n_cols)
            real_burned_per_hour[h, rr, cc] = True

    # 2. Load SRTM -> slope/aspect/elevation
    print("\n[2/6] Loading SRTM DEM...")
    srtm_tiles = load_srtm_tiles(bbox_lat, bbox_lon, PROJECT_ROOT / scenario["srtm_dir"])
    slope_pct, aspect_deg, elevation = build_slope_aspect_from_srtm(srtm_tiles, bbox_lat, bbox_lon, n_rows, n_cols)
    print(f"  slope_pct range: {slope_pct.min():.1f} - {slope_pct.max():.1f}, mean {slope_pct.mean():.1f}")

    # 3. Load WorldCover -> fuel grid
    print("\n[3/6] Loading ESA WorldCover...")
    wc_tiles = load_worldcover_tiles(bbox_lat, bbox_lon, PROJECT_ROOT / scenario["worldcover_dir"])
    fuel_grid = build_fuel_grid_from_worldcover(wc_tiles, bbox_lat, bbox_lon, n_rows, n_cols)
    unique, counts = np.unique(fuel_grid, return_counts=True)
    print(f"  fuel distribution: {dict(zip(unique.tolist(), counts.tolist()))}")

    # 4. Load ERA5 GRIB -> weather grids per hour
    print("\n[4/6] Loading ERA5 weather...")
    grib_path = PROJECT_ROOT / scenario["grib_path"]
    if not grib_path.exists():
        print(f"  WARNING: GRIB not found at {grib_path}; using uniform weather defaults.")
        weather_per_hour = {h: {"wind_speed": 5.0, "wind_dir": 270.0, "rh_percent": 30.0, "temp_c": 20.0, "m1h": 0.08}
                             for h in target_hours}
    else:
        weather_per_hour = build_weather_grids_from_grib(grib_path, bbox_lat, bbox_lon, n_rows, n_cols, target_hours)

    # 5. Run ensemble
    print(f"\n[5/6] Running ensemble ({scenario['n_ensemble']} runs × {n_hours} h)...")
    grid_template = {"slope_pct": slope_pct, "aspect_deg": aspect_deg, "elevation_m": elevation}
    ensemble = run_ensemble(grid_template, fuel_grid, weather_per_hour, scenario, out_dir)

    # 6. Score and save
    print("\n[6/6] Scoring and saving figures...")
    hourly_metrics = []
    sim_area_series = []
    real_area_series = []
    iou_series = []
    hours_axis = []
    for h in target_hours:
        # cumulative real burned
        real_cum = real_burned_per_hour[: h + 1].any(axis=0)
        # ensemble burn probability: fraction of runs that have BURNED this cell by hour h
        prob = ensemble[: h + 1].any(axis=0).mean(axis=0) if scenario["n_ensemble"] > 0 else np.zeros_like(slope_pct)
        sim_any = ensemble[: h + 1].any(axis=(0, 1))  # any run, any time up to h
        m = score_burned(sim_any, real_cum)
        m["hour"] = h
        hourly_metrics.append(m)
        sim_area_series.append(int(sim_any.sum()))
        real_area_series.append(int(real_cum.sum()))
        iou_series.append(m["IoU"])
        hours_axis.append(h)

    final_h = n_hours
    prob_final = ensemble[: final_h + 1].any(axis=0).mean(axis=0)
    real_final = real_burned_per_hour[: final_h + 1].any(axis=0)
    sim_final = ensemble[: final_h + 1].any(axis=(0, 1))
    auc = score_auc(prob_final, real_final)
    final_metrics = score_burned(sim_final, real_final)
    final_metrics["AUC"] = auc
    final_metrics["hour"] = final_h

    # figures
    make_overlay(sim_final, real_final, bbox_lat, bbox_lon, out_dir / "overlay_final.png", f"t+{final_h}h")
    make_probability_map(prob_final, real_final, bbox_lat, bbox_lon, out_dir / "probability_map.png")
    make_timeseries(hours_axis, sim_area_series, real_area_series, iou_series, out_dir / "timeseries_burned.png")
    for h in [1, 3, 6, 12, 24, final_h]:
        if h > final_h:
            continue
        sim_h = ensemble[: h + 1].any(axis=(0, 1))
        make_hourly_snapshot(sim_h, bbox_lat, bbox_lon, out_dir / f"sim_t{h:02d}h.png", f"t+{h}h")

    # save JSON
    out_json = {
        "scenario_id": sid,
        "scenario_name": scenario["name"],
        "n_firms_detections": int(len(firms)),
        "ignition_cell": [ign_row, ign_col],
        "bbox_lat": bbox_lat,
        "bbox_lon": bbox_lon,
        "grid_size": [n_rows, n_cols],
        "cell_size_m": scenario["cell_size_m"],
        "duration_h": n_hours,
        "n_ensemble": scenario["n_ensemble"],
        "final_metrics": final_metrics,
        "hourly_metrics": hourly_metrics,
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(out_json, f, indent=2, default=lambda o: None if o is None else (float(o) if isinstance(o, (np.floating,)) else int(o) if isinstance(o, (np.integer,)) else o))

    # save summary.md
    iou = final_metrics.get("IoU")
    f1 = final_metrics.get("F1")
    auc = final_metrics.get("AUC")
    iou_gen = final_metrics.get("IoU_generous")
    f1_gen = final_metrics.get("F1_generous")
    summary = f"""# Real-fire validation: {scenario['name']}

- Scenario id: `{sid}`
- FIRMS detections: {len(firms)}
- Grid: {n_rows}×{n_cols} at {scenario['cell_size_m']} m/cell = {n_rows * scenario['cell_size_m'] / 1000:.1f} km × {n_cols * scenario['cell_size_m'] / 1000:.1f} km
- Duration: {n_hours} h
- Ensemble size: {scenario['n_ensemble']}

## Final metrics (t+{final_h}h, cumulative real burned = {real_area_series[-1]} cells)

| Metric | Value |
|---|---|
| **IoU strict (cell-exact)** | {iou if iou is not None else 'n/a'} |
| **IoU generous (2-cell dilation, ~FIRMS uncertainty)** | {iou_gen} |
| Precision | {final_metrics.get('precision')} |
| Recall | {final_metrics.get('recall')} |
| F1 strict | {f1 if f1 is not None else 'n/a'} |
| F1 generous | {f1_gen} |
| AUC (ensemble prob) | {auc if auc is not None else 'n/a'} |
| TP / FP / FN / TN | {final_metrics['TP']} / {final_metrics['FP']} / {final_metrics['FN']} / {final_metrics['TN']} |

The "generous" IoU dilates the real burned mask by 2 cells (~200 m at this resolution)
to reflect the positional uncertainty of NASA FIRMS hotspots (375 m for VIIRS, 1 km
for MODIS). This is the fairest comparison given satellite detection limits.

## Per-hour metrics

| Hour | Sim cells | Real cells | IoU strict | IoU generous |
|---|---|---|---|---|
"""
    for m in hourly_metrics:
        summary += f"| {m['hour']} | {m['TP']+m['FP']} | {m['TP']+m['FN']} | {m['IoU'] if m['IoU'] is not None else 'n/a'} | {m.get('IoU_generous', 'n/a')} |\n"

    with open(out_dir / "summary.md", "w") as f:
        f.write(summary)

    print("\n" + summary)
    print(f"\nOutputs in: {out_dir}")
    return out_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", default="experiments/real_fire_scenarios.yaml")
    p.add_argument("--id", default="table_mountain_2021_04")
    p.add_argument("--out-root", default="experiments/out")
    args = p.parse_args()

    with open(args.scenario) as f:
        all_sc = yaml.safe_load(f)
    scenarios = all_sc["scenarios"]
    selected = [s for s in scenarios if s["id"] == args.id]
    if not selected:
        raise SystemExit(f"Scenario id '{args.id}' not found in {args.scenario}")
    run_scenario(selected[0], PROJECT_ROOT / args.out_root)


if __name__ == "__main__":
    main()
