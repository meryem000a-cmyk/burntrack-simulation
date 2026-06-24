# ReadIfAgent

> **If you are an AI agent picking up this project, read this first.**

## Project: BurnTrack Simulation

A wildfire spread simulation that combines the **Rothermel physical model** with an **MLP neural network corrector**. The MLP learns the systematic biases of the physics engine and corrects them using real satellite fire data and weather reanalysis.

**Architecture:** `ROS_burntrack = ROS_rothermel + Δ_MLP`

---

## Current State (June 2026)

### Corrector — what to know before touching it

The corrector was hit by a **target-leakage bug** in late June 2026 and was retrained clean on 2026-06-24. Do not re-introduce the leaked `thermal_proxy` feature (was: `delta_ros * 0.95 + noise`).

- `burntrack/correcteur final/source/model.py` exposes three classes: `BurnTrackMLPMinimal` (the one we now ship), `BurnTrackAdvancedCorrector` (PINN, deprecated — was trained on the leaky feature), `BurnTrackFTGatedCorrector` (deprecated for the same reason). `bridge.py` only loads the minimal MLP.
- `burntrack/correcteur final/bridge.py` → `BurnTrackPredictor.predict(fuel_id, wind_speed, moisture_1h, moisture_live, slope_pct, ...)` returns a dict with `ros_rothermel`, `delta_mlp`, `ros_burntrack`. No more `target_real_ros` parameter.
- `burntrack/correcteur final/train_correcteur_final.py` → `FEATURE_COLS` has 31 features (no `thermal_proxy`). Output checkpoint is `burntrack/correcteur final/checkpoints/burntrack_mlp_minimal.pt`.

### Honest metric split (do not mix them up)

| Dataset | n | R² (MLP) | MAE | Notes |
|---|---|---|---|---|
| Literature (7 published studies) | 1,840 | 0.987 | 0.735 m/min | Curated, controlled conditions, small — what the report abstract still quotes |
| South Africa FIRMS (real) | 2,866 | 0.12 | 5.38 m/min | Honest, post-leakage-fix. ROS reconstructed from satellite hotspots is structurally noisy |

The big drop on FIRMS is **expected and not a bug**: satellites give ignition points, not continuous fronts, and ERA5-Land reanalysis is coarse. A field-grade front-line measurement campaign would be the proper fix.

### Data Sources (all in `south africa data/`, gitignored — 7.4 GB)
| File | Source | What it is |
|---|---|---|
| `d8b9f698fa6debeaac2b10e21c97ba5f.zip` | Copernicus CDS | ERA5-Land hourly weather (GRIB), April 2021, Western Cape |
| `data.grib` | Extracted from above | t2m, d2m, u10, v10, tp |
| `DL_FIRE_M-C61_765830.zip` | NASA FIRMS | MODIS Collection 6.1 active fire detections |
| `DL_FIRE_SV-C2_765831.zip` | NASA FIRMS | VIIRS SNPP Collection 2 active fire detections (375m) |
| `srtm/` | SRTM 30m DEM (`.hgt`) | Real slope/aspect per fire point |
| `worldcover/` | ESA WorldCover v200 | 10m land-cover GeoTIFFs |

### What's Been Built
| File | Purpose |
|---|---|
| `scripts/build_south_africa_dataset.py` | Merges FIRMS + ERA5 + SRTM + WorldCover → training CSV (real slope, real fuel) |
| `scripts/download_worldcover.py` | Downloads ESA WorldCover tiles with resume/retry |
| `scripts/download_srtm.py` | Downloads SRTM 30m tiles |
| `scripts/download_table_mountain_fire.py` | Downloads Table Mountain 2021 fire data |
| `scripts/build_from_local_firms.py` | Builds dataset from locally cached FIRMS |
| `scripts/make_real_figures.py` | Stale — references `ros_surrogate.pt` from a different pipeline; do not use |
| `burntrack/correcteur final/train_correcteur_final.py` | Training script (clean, no synthetic, no leakage) |
| `data/processed/south_africa_manual_dataset.csv` | Current real-data training CSV: 2,866 fire vectors |

### σ₁h unit caveat (do not "fix" without retraining)
The BEHAVE fuel models in `burntrack/engine/fuel_models.py` keep their `sigma_*` values in **English units (ft⁻¹)** as in Anderson (1982) / Scott & Burgan (2005). The dataclass comment says SI but the numbers are ft⁻¹. The report's table at `rapport/rapport_part2.tex` caption now states this explicitly. Do **not** bulk-convert without a full re-calibration of Rothermel outputs.

### Report writing rules
- Logo background must be white and larger. "Simplify not delete" — over-explain everything.
- When stating R², always specify which dataset. Literature 0.987 and FIRMS 0.12 are both real; never quote one without the other.
- The 50 fuel models = 28 African + 22 BEHAVE. Of the 40 Scott & Burgan models, BurnTrack implements 22.

### Key Constraints
- **NO synthetic data.** Real observations only.
- **NO API-downloaded data.** Only use files manually dropped into `south africa data/`.
- **Privacy first.** Local Postgres, local embeddings. No cloud.
- **User is Anwar**, engineering student presenting to teachers.
- **Branch policy:** all work on `integration`, never push to `main` (per `pm.md`).

---

## Codebase Map
```
burntrack-simulation/
├── burntrack/
│   ├── correcteur final/     # MLP training + checkpoints + results
│   ├── data/                 # firms.py, weather.py, real_dataset.py
│   ├── engine/               # rothermel.py, fuel_models.py
│   └── simulation/
├── cellular_automaton/       # Grid, rules, simulation
├── data/processed/           # Shared CSVs (SA manual dataset, table_mountain_2021)
├── scripts/                  # Build + download + figure scripts
├── south africa data/        # 7.4 GB raw data — GITIGNORED, local only
├── experiments/              # CA validation, calibration (out/ gitignored)
├── rapport/                  # LaTeX source + figures
├── risk_map/                 # Risk-map generator
├── robot_nav/                # D* Lite planner
├── robot_vision/             # YOLO + CNN vision
└── visualization/            # WebSocket server + 3D UI
```
