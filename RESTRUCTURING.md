# BurnTrack Project — Restructuring

**Date:** June 25, 2026

## What changed

All old, deprecated, or unused files have been moved to `archive/`. Nothing was deleted — everything is preserved and still accessible.

## Archive layout

```
archive/
├── corrector_old/       burntrack/corrector/ — pre-leakage-fix MLP corrector (deprecated Jun 24)
│                        Replaced by burntrack/correcteur final/ (31-feature MLP, clean)
├── models_old/          Old model checkpoints: RF, XGBoost, CatBoost, ensemble, atlas_v2, v3
│                        Current model: burntrack/correcteur final/checkpoints/burntrack_mlp_minimal.pt
├── configs_old/         configs/ — old YAML/JSON config files
├── scripts_old/         Old training, debug, synthetic generation, and validation scripts
│                        Current scripts: scripts/download_*.py, scripts/build_*.py
├── rapport_old/         Page-padding scripts (inflate, simplify, stretch) + old report PDF
│                        Current report: rapport/rapport_plbd_groupe7.pdf
├── docs_old/            Old session logs, verification report, pm.md handoff docs
├── data_old/            Old synthetic dataset generator, FIRMS zip, service key
│                        Current data: data/processed/south_africa_manual_dataset.csv,
│                        data/processed/table_mountain_2021.csv, data/processed/local_firms_ros_observed.csv
├── figures_old/         Old figures (metrics.json, scatter plots)
└── misc/                Prototype photos, Fritzing schematic, swap file, _local/
```

## Current active layout

```
burntrack-simulation/
├── burntrack/
│   ├── correcteur final/   ← Current MLP corrector (31 features, clean, no leakage)
│   ├── data/               ← firms.py, weather.py, real_dataset.py, burned_area.py
│   ├── engine/             ← rothermel.py, fuel_models.py
│   └── utils/              ← logging.py, simulation.py
├── cellular_automaton/     ← Grid, rules, simulation, mlp_corrector
├── experiments/            ← validate_real_fire.py, calibrate_ca.py, real_fire_scenarios.yaml
├── rapport/                ← LaTeX report + figures + PDF
├── risk_map/               ← Bouskoura risk map generator
├── robot_nav/              ← D* Lite planner
├── robot_vision/           ← YOLO + CNN vision
├── scripts/                ← Download + build scripts (active only)
├── tests/                  ← Unit tests
├── visualization/          ← WebSocket server + 3D UI
├── ReadIfAgent.md          ← Agent handoff doc (current state)
├── pm.md                   ← Agent handoff prompt
├── pyproject.toml          ← Package config
├── requirements.txt        ← Dependencies
└── .env                    ← API keys (gitignored)
```

## Key files for the report

| File | Purpose |
|------|---------|
| `rapport/rapport_plbd_groupe7.pdf` | Final report PDF |
| `rapport/rapport_plbd_groupe7.tex` | Main LaTeX source |
| `rapport/rapport_part2.tex` | Part 2 (validation results, conclusion) |
| `rapport/figures/` | All report figures |
| `experiments/out/` | Validation outputs (JSON, figures, summary) |
| `south africa data/` | ERA5 GRIB, FIRMS shapefiles, SRTM, WorldCover, Landsat scenes |

## Validation metrics (final)

### Knysna 2017 — Landsat-fused (dNBR threshold 0.05)

| Metric | Value |
|--------|-------|
| IoU strict | 0.516 |
| IoU generous | 0.620 |
| F1 strict | 0.681 |
| F1 generous | 0.765 |
| Precision | 0.793 |
| Recall | 0.597 |

### Table Mountain 2021 — Landsat-fused (dNBR threshold 0.40)

| Metric | Value |
|--------|-------|
| IoU strict | 0.070 |
| IoU generous | 0.067 |
| F1 strict | 0.131 |
| Precision | 0.933 |
| Recall | 0.071 |
| AUC | 0.532 |

### Table Mountain 2021 — FIRMS-only (original)

| Metric | Value |
|--------|-------|
| IoU strict | 0.02 |
| IoU generous | 0.05 |
| F1 generous | 0.10 |
| AUC | 0.55 |

## How to run validation

```bash
# Knysna (fastest — ~54s with 5 workers)
venv/bin/python -u experiments/validate_real_fire.py --id knysna_2017_06

# Table Mountain (~2 min)
venv/bin/python -u experiments/validate_real_fire.py --id table_mountain_2021_04
```

## How to rebuild the report

```bash
cd rapport && pdflatex -shell-escape rapport_plbd_groupe7.tex
```
