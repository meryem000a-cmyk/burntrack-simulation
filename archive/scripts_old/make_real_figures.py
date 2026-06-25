"""Generate 4 poster-quality figures from real FIRMS data.

Usage:
    python scripts/make_real_figures.py

Outputs to figures/:
    01_rothermel_vs_correction_vs_realite.png  (hero: 3-panel pred vs obs)
    02_metriques_par_intensite.png               (metrics by ROS bin)
    03_couverture_geographique_firms.png          (geographic scatter + density)
    04_courbes_entrainement_reelles.png           (training log)
    metrics.json                                  (all quantitative results)
"""

import json
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BURNTTRACK_ROOT = Path(__file__).parents[1]
PLBD_ROOT = Path("/home/anwar/Documents/PLBD_robot")
DATA_DIR = PLBD_ROOT / "datasets" / "fire_ros"
MODELS_DIR = PLBD_ROOT / "models"
FIGS_DIR = BURNTTRACK_ROOT / "figures"
FIGS_DIR.mkdir(exist_ok=True)

# Import burntrack Rothermel engine
sys.path.insert(0, str(BURNTTRACK_ROOT))
from burntrack.engine.rothermel import (
    BurnTrackRothermel,
    EnvironmentalConditions,
    FuelModel,
    MoistureInputs,
    RothermelEngine,
)

# Import PLBD_robot feature_schema
sys.path.insert(0, str(PLBD_ROOT))
from training.feature_schema import FEATURE_NAMES, NORMALIZATION, NUM_FEATURES

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

POSTER_DPI = 300
MAX_ROS = 16.0  # m/min clip for plots

PALETTE = {
    "roth": "#d62728",
    "corr": "#1f77b4",
    "obs":  "#2ca02c",
    "bg":   "#f8f9fa",
    "grid": "#e0e0e0",
}

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 13,
    "axes.titlesize": 15,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.facecolor": "white",
    "axes.facecolor": PALETTE["bg"],
    "axes.grid": True,
    "grid.color": PALETTE["grid"],
    "grid.alpha": 0.6,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.25,
})


# ---------------------------------------------------------------------------
# Surrogate model
# ---------------------------------------------------------------------------
class RossSurrogate(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(NUM_FEATURES, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def load_surrogate(path: Path) -> RossSurrogate:
    model = RossSurrogate()
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def normalize_features(arr: np.ndarray) -> np.ndarray:
    out = np.empty_like(arr, dtype=np.float32)
    for i, name in enumerate(FEATURE_NAMES):
        lo, hi = NORMALIZATION[name]
        out[..., i] = (arr[..., i] - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0)


def denormalize_ros(ros_norm: np.ndarray) -> np.ndarray:
    return ros_norm * 120.0


def predict_surrogate(model: RossSurrogate, features_norm: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        pred = model(torch.from_numpy(features_norm)).numpy()
    return denormalize_ros(pred).flatten()


# ---------------------------------------------------------------------------
# Rothermel baseline
# ---------------------------------------------------------------------------
# GR4 reference values (standard fuel model for dry climate grass)
_GR4_W1H = 4.15
_GR4_DELTA = 0.6096
_GR4_W_LIVE_HERB = 4.18
_GR4_W_LIVE_WOODY = 0.0


def build_fuel_model(row: dict, use_gr4_fixed: bool = True) -> FuelModel:
    if use_gr4_fixed:
        w1h = _GR4_W1H
        w_live_herb = _GR4_W_LIVE_HERB
        w_live_woody = _GR4_W_LIVE_WOODY
        delta = _GR4_DELTA
    else:
        w1h = float(row["fuel_load"])
        w_live_herb = 0.0
        w_live_woody = 0.0
        delta = float(row["fuel_depth"])
    fuel = FuelModel(
        name="GR4" if use_gr4_fixed else "GR4_dynamic",
        w_1h=max(w1h, 0.01),
        w_10h=0.0,
        w_100h=0.0,
        w_live_herb=max(w_live_herb, 0.0),
        w_live_woody=max(w_live_woody, 0.0),
        sigma_1h=10499.0,
        sigma_10h=0.0,
        sigma_100h=0.0,
        sigma_live_herb=0.0 if w_live_herb <= 0 else 9449.0,
        sigma_live_woody=0.0 if w_live_woody <= 0 else 8399.0,
        delta=max(delta, 0.01),
        mx=15.0,
        h_dead=18608.0,
        h_live=18608.0,
        st=0.0555,
        se=0.01,
    )
    return fuel


def compute_rothermel_ros(engine: RothermelEngine, row: dict) -> float:
    fuel = build_fuel_model(row)
    moisture = MoistureInputs(
        m_1h=float(row["moisture_1h"]),
        m_10h=float(row["moisture_10h"]),
        m_100h=float(row["moisture_100h"]),
        m_live_herb=float(row["moisture_live_herb"]),
        m_live_woody=float(row["moisture_live_woody"]),
    )

    slope_frac = float(row["slope"])
    wind_rad = float(row["wind_dir"])  # radians
    aspect_rad = float(row["aspect"])  # radians

    angle_diff = abs(wind_rad - aspect_rad)
    angle_diff = min(angle_diff, 2 * np.pi - angle_diff)
    angle_diff_deg = np.degrees(angle_diff)

    conditions = EnvironmentalConditions(
        wind_speed=float(row["wind_speed"]),
        slope_pct=slope_frac * 100.0,
        angle_wind_slope=angle_diff_deg,
    )

    output = engine.compute(fuel, moisture, conditions)
    return output.ros


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_metrics(obs: np.ndarray, pred: np.ndarray, label: str) -> dict:
    mae = float(np.mean(np.abs(pred - obs)))
    rmse = float(np.sqrt(np.mean((pred - obs) ** 2)))
    bias = float(np.mean(pred - obs))
    numer = np.sum((obs - pred) ** 2)
    denom = max(np.sum((obs - np.mean(obs)) ** 2), 1e-10)
    r2 = float(1 - numer / denom)
    return {"label": label, "r2": r2, "mae": mae, "rmse": rmse, "bias": bias, "n": int(len(obs))}


# ---------------------------------------------------------------------------
# Figure 1: Hero — 3-panel pred vs obs
# ---------------------------------------------------------------------------
def fig_hero(obs, roth, corr, metrics_roth, metrics_corr):
    limits = (0, min(MAX_ROS, float(np.percentile(obs, 98)) * 1.2))
    panels = [
        ("Rothermel seul (GR4 standard)", roth, PALETTE["roth"], metrics_roth),
        ("Rothermel + Correction ML", corr, PALETTE["corr"], metrics_corr),
        ("Réalité (obs vs obs)", obs, PALETTE["obs"], None),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
    for ax, (title, pred, color, met) in zip(axes, panels):
        mask = (obs > 0) & (pred >= 0)
        x = np.clip(obs[mask], 0.01, limits[1])
        y = np.clip(pred[mask], 0.01, limits[1])

        hb = ax.hexbin(x, y, gridsize=35, bins="log", cmap="Blues", mincnt=1, alpha=0.7)
        ax.plot(limits, limits, "k-", lw=1.5, alpha=0.5, label="y=x")
        ax.set_xlim(*limits)
        ax.set_ylim(*limits)
        ax.set_xlabel("Observé (m/min)")
        if ax == axes[0]:
            ax.set_ylabel("Prédit (m/min)")
        ax.set_title(title, fontweight="bold", color=color)

        if met:
            text = (
                f"R²={met['r2']:.3f}\n"
                f"MAE={met['mae']:.2f}\n"
                f"RMSE={met['rmse']:.2f}\n"
                f"Biais={met['bias']:.3f}"
            )
            ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=11,
                    verticalalignment="top", fontfamily="monospace",
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85))

        cb = plt.colorbar(hb, ax=ax, shrink=0.7)
        cb.set_label("Densité (log)", fontsize=11)
        ax.set_aspect("equal", adjustable="box")

    fig.suptitle(
        "Rothermel vs Correction ML vs Réalité — observations FIRMS (Maroc)",
        fontsize=16, fontweight="bold", y=0.98
    )
    plt.tight_layout()
    fig.savefig(FIGS_DIR / "01_rothermel_vs_correction_vs_realite.png", dpi=POSTER_DPI)
    plt.close(fig)
    print("  Figure 1: 01_rothermel_vs_correction_vs_realite.png")


# ---------------------------------------------------------------------------
# Figure 2: Metrics by ROS bin
# ---------------------------------------------------------------------------
def fig_metrics_by_bin(obs, roth, corr, bins):
    bin_labels = []
    r2_roth, r2_corr = [], []
    mae_roth, mae_corr = [], []
    bias_roth, bias_corr = [], []
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        mask = (obs >= lo) & (obs < hi)
        if mask.sum() < 10:
            continue
        bin_labels.append(f"{lo:.0f}–{hi:.0f}")
        r2_roth.append(compute_metrics(obs[mask], roth[mask], "")["r2"])
        r2_corr.append(compute_metrics(obs[mask], corr[mask], "")["r2"])
        mae_roth.append(compute_metrics(obs[mask], roth[mask], "")["mae"])
        mae_corr.append(compute_metrics(obs[mask], corr[mask], "")["mae"])
        bias_roth.append(compute_metrics(obs[mask], roth[mask], "")["bias"])
        bias_corr.append(compute_metrics(obs[mask], corr[mask], "")["bias"])

    x = np.arange(len(bin_labels))
    width = 0.35

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
    titles = ["R² par classe", "MAE par classe (m/min)", "Biais par classe (m/min)"]
    data_roth = [r2_roth, mae_roth, bias_roth]
    data_corr = [r2_corr, mae_corr, bias_corr]

    for ax, title, dr, dc in zip(axes, titles, data_roth, data_corr):
        ax.bar(x - width / 2, dr, width, label="Rothermel", color=PALETTE["roth"], alpha=0.75)
        ax.bar(x + width / 2, dc, width, label="Corrigé", color=PALETTE["corr"], alpha=0.75)
        ax.set_xticks(x)
        ax.set_xticklabels(bin_labels, rotation=30, ha="right")
        ax.set_xlabel("ROS observé (m/min)")
        ax.set_title(title, fontweight="bold")
        ax.legend()

    fig.suptitle("Performance par classe d'intensité", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    fig.savefig(FIGS_DIR / "02_metriques_par_intensite.png", dpi=POSTER_DPI)
    plt.close(fig)
    print("  Figure 2: 02_metriques_par_intensite.png")


# ---------------------------------------------------------------------------
# Figure 3: Geographic coverage
# ---------------------------------------------------------------------------
def fig_geographic(features, labels):
    has_latlon = {"lat", "lon"}.issubset(set(features.columns))
    if not has_latlon:
        print("  WARNING: No lat/lon in features. Trying firms_observed_ros.csv...")
        csv_path = DATA_DIR / "firms_observed_ros.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            fig, ax = plt.subplots(figsize=(16, 10))
            sc = ax.scatter(df["lon"], df["lat"], c=df["ros_m_min"], s=30, cmap="hot_r",
                          edgecolors="k", linewidths=0.3, alpha=0.8, vmin=0, vmax=df["ros_m_min"].quantile(0.95))
            cbar = plt.colorbar(sc, ax=ax, shrink=0.7)
            cbar.set_label("ROS (m/min)")
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            ax.set_title("Points d'observation FIRMS — Maroc (N={})".format(len(df)), fontweight="bold")
            ax.grid(True, alpha=0.3)
            fig.savefig(FIGS_DIR / "03_couverture_geographique_firms.png", dpi=POSTER_DPI)
            plt.close(fig)
            print("  Figure 3 (fallback): 03_couverture_geographique_firms.png ({} points)".format(len(df)))
            return

        print("  WARNING: No geographic data available. Skipping Figure 3.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    ax = axes[0]
    sc = ax.scatter(features["lon"], features["lat"], c=labels, s=8, cmap="hot_r",
                    alpha=0.6, vmin=0, vmax=np.percentile(labels, 95))
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.set_title("ROS observé par localisation", fontweight="bold")
    cbar = plt.colorbar(sc, ax=ax, shrink=0.8)
    cbar.set_label("ROS (m/min)")

    ax = axes[1]
    hb = ax.hexbin(features["lon"], features["lat"], gridsize=35, cmap="Blues", bins="log", mincnt=1, alpha=0.7)
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.set_title("Densité d'observations (log)", fontweight="bold")
    cbar = plt.colorbar(hb, ax=ax, shrink=0.8)
    cbar.set_label("N (log)")

    fig.suptitle("Couverture géographique des observations FIRMS réelles (N={}, Maroc)".format(len(features)),
                 fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    fig.savefig(FIGS_DIR / "03_couverture_geographique_firms.png", dpi=POSTER_DPI)
    plt.close(fig)
    print("  Figure 3: 03_couverture_geographique_firms.png")


# ---------------------------------------------------------------------------
# Figure 4: Training curves
# ---------------------------------------------------------------------------
def fig_training_curves(log_path):
    with open(log_path) as f:
        data = json.load(f)

    log = data["log"]
    epochs = [e["epoch"] for e in log]
    train_loss = [e["train_loss"] for e in log]
    val_loss = [e["val_loss"] for e in log]
    lrs = [e["lr"] for e in log]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    ax = axes[0]
    ax.plot(epochs, train_loss, label="Entraînement", color=PALETTE["corr"], lw=2)
    ax.plot(epochs, val_loss, label="Validation", color=PALETTE["roth"], lw=2)
    ax.set_xlabel("Époque"); ax.set_ylabel("Perte (Huber)")
    ax.set_title("Courbes de perte", fontweight="bold")
    ax.legend()

    if "test_metrics" in data and data["test_metrics"]:
        tm = data["test_metrics"]
        text = f"MAE={tm['mae']:.2f}\nRMSE={tm['rmse']:.2f}\nR²={tm['r2']:.3f}"
        ax.text(0.97, 0.95, text, transform=ax.transAxes, fontsize=10,
                verticalalignment="top", horizontalalignment="right", fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85))

    ax = axes[1]
    ax.plot(epochs, lrs, color="green", lw=2)
    ax.set_xlabel("Époque"); ax.set_ylabel("Taux d'apprentissage")
    ax.set_title("Learning rate schedule", fontweight="bold")

    fig.suptitle("Courbes d'entraînement réelles du correcteur MLP",
                 fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    fig.savefig(FIGS_DIR / "04_courbes_entrainement_reelles.png", dpi=POSTER_DPI)
    plt.close(fig)
    print("  Figure 4: 04_courbes_entrainement_reelles.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("  BurnTrack Poster Figures — Real FIRMS Data")
    print("=" * 60)

    # 1. Load data
    print("\n[1] Loading test data...")
    train_feats = pd.read_parquet(DATA_DIR / "train_features.parquet")
    train_labels = pd.read_parquet(DATA_DIR / "train_labels.parquet")["ros_m_min"].values
    val_feats = pd.read_parquet(DATA_DIR / "val_features.parquet")
    val_labels = pd.read_parquet(DATA_DIR / "val_labels.parquet")["ros_m_min"].values
    test_feats = pd.read_parquet(DATA_DIR / "test_features.parquet")
    test_labels = pd.read_parquet(DATA_DIR / "test_labels.parquet")["ros_m_min"].values
    full_feats = pd.read_parquet(DATA_DIR / "features.parquet")
    full_labels = pd.read_parquet(DATA_DIR / "labels.parquet")["ros_m_min"].values

    print(f"  Train: {len(train_labels)}, Val: {len(val_labels)}, Test: {len(test_labels)}, Full: {len(full_labels)}")

    # Use test set for evaluation
    obs = test_labels.astype(np.float32)
    feat_test = test_feats.copy()

    # 2. Load surrogate model
    print("\n[2] Loading surrogate model...")
    model_path = MODELS_DIR / "ros_surrogate.pt"
    if not model_path.exists():
        print(f"  ERROR: Model not found at {model_path}")
        sys.exit(1)
    model = load_surrogate(model_path)
    print("  Model loaded (14→128→128→64→1 MLP)")

    feat_norm = normalize_features(feat_test.values.astype(np.float32))
    pred_corr = predict_surrogate(model, feat_norm)
    pred_corr = np.clip(pred_corr, 0.0, None)
    print(f"  Surrogate predictions: min={pred_corr.min():.3f}, max={pred_corr.max():.3f}, "
          f"mean={pred_corr.mean():.3f}")

    # 3. Run Rothermel baseline (using standard GR4 fuel model)
    print("\n[3] Running Rothermel baseline (standard GR4 fuel model)...")
    engine = RothermelEngine()
    pred_roth = []
    n_skip = 0
    for idx in range(len(feat_test)):
        row = feat_test.iloc[idx]
        try:
            ros = compute_rothermel_ros(engine, row)
            pred_roth.append(ros)
        except Exception as e:
            pred_roth.append(0.0)
            n_skip += 1
    pred_roth = np.array(pred_roth, dtype=np.float32)
    pred_roth = np.clip(pred_roth, 0.0, None)

    print(f"  Rothermel predictions: min={pred_roth.min():.3f}, max={pred_roth.max():.3f}, "
          f"mean={pred_roth.mean():.3f}, median={np.median(pred_roth):.3f}, skipped={n_skip}")

    # 4. Compute metrics
    print("\n[4] Computing metrics...")
    metrics_roth = compute_metrics(obs, pred_roth, "Rothermel")
    metrics_corr = compute_metrics(obs, pred_corr, "Corrigé")

    print(f"  Rothermel: R²={metrics_roth['r2']:.4f}, MAE={metrics_roth['mae']:.3f}, RMSE={metrics_roth['rmse']:.3f}, Bias={metrics_roth['bias']:.3f}")
    print(f"  Corrigé:   R²={metrics_corr['r2']:.4f}, MAE={metrics_corr['mae']:.3f}, RMSE={metrics_corr['rmse']:.3f}, Bias={metrics_corr['bias']:.3f}")
    print(f"  Improvement: ΔR²={metrics_corr['r2'] - metrics_roth['r2']:.4f}, "
          f"ΔMAE={metrics_corr['mae'] - metrics_roth['mae']:.3f}, ΔRMSE={metrics_corr['rmse'] - metrics_roth['rmse']:.3f}")

    # 5. Per-bin metrics
    bins = [0, 2, 5, 10, 20, 50]
    bin_metrics = []
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        mask = (obs >= lo) & (obs < hi)
        if mask.sum() < 10:
            continue
        bin_metrics.append({
            "bin": f"{lo}-{hi}",
            "n": int(mask.sum()),
            "Rothermel": compute_metrics(obs[mask], pred_roth[mask], ""),
            "Corrigé": compute_metrics(obs[mask], pred_corr[mask], ""),
        })

    # 6. Save metrics.json
    metrics_all = {
        "overall": {"Rothermel": metrics_roth, "Corrigé": metrics_corr},
        "per_bin": bin_metrics,
        "improvement": {
            "delta_r2": metrics_corr["r2"] - metrics_roth["r2"],
            "delta_mae": metrics_corr["mae"] - metrics_roth["mae"],
            "delta_rmse": metrics_corr["rmse"] - metrics_roth["rmse"],
        },
    }
    with open(FIGS_DIR / "metrics.json", "w") as f:
        json.dump(metrics_all, f, indent=2)
    print(f"\n  Saved metrics to figures/metrics.json")

    # 7. Generate figures
    print("\n[5] Generating figures...")
    fig_hero(obs, pred_roth, pred_corr, metrics_roth, metrics_corr)
    fig_metrics_by_bin(obs, pred_roth, pred_corr, np.array([0, 2, 5, 10, 20, 50]))

    # Figure 3: try to find lat/lon in features
    fig_geographic(full_feats, full_labels)

    # Figure 4: training curves
    log_path = MODELS_DIR / "training_log.json"
    if log_path.exists():
        fig_training_curves(log_path)
    else:
        print("  WARNING: training_log.json not found. Skipping Figure 4.")

    print("\n" + "=" * 60)
    print("  Done! All figures in figures/")
    print("=" * 60)


if __name__ == "__main__":
    main()
