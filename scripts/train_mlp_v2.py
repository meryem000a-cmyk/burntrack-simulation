"""
train_mlp_v2.py — Train BurnTrack MLP Corrector v2

Architecture: 8->64->32->1, GELU, LayerNorm, Dropout(0.2)
Features: wind, humidity, slope, ros_rothermel, h_dead, sigma, m_live, mx
Loss: Weighted MSE (real=10x, synth=1x)
Early stopping: loss on real data only
"""
import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from burntrack.corrector.mlp_v2 import BurnTrackMLP

FEATURES = ["wind_speed_ms", "rh_percent", "slope_pct", "ros_rothermel",
            "h_dead_kj_kg", "sigma_m2_m3", "m_live_woody", "mx_percent"]
REAL_WEIGHT = 10.0
SYNTH_WEIGHT = 1.0
BATCH_SIZE = 32
LR = 3e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 300
PATIENCE = 30
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def build_X(df):
    return np.stack([
        df["wind_speed_ms"].values,
        df["rh_percent"].values,
        df["slope_pct"].values,
        df["ros_rothermel"].values,
        df["h_dead_kj_kg"].values,
        df["sigma_m2_m3"].values,
        df["m_live_woody"].values,
        df["mx_percent"].values,
    ], axis=1)


def load_data():
    real = pd.read_csv("data/processed/african_ground_truth.csv")
    synth = pd.read_csv("data/processed/train.csv")

    X_real = build_X(real)
    y_real = real["delta_ros"].values

    X_synth = build_X(synth)
    y_synth = synth["delta_ros"].values

    return X_real, y_real, X_synth, y_synth


def train():
    X_real, y_real, X_synth, y_synth = load_data()

    X_all = np.vstack([X_real, X_synth])
    y_all = np.concatenate([y_real, y_synth])
    n_real = len(y_real)
    n_synth = len(y_synth)

    weights = np.concatenate([
        np.full(n_real, REAL_WEIGHT),
        np.full(n_synth, SYNTH_WEIGHT),
    ])

    scaler = StandardScaler()
    X_all_s = scaler.fit_transform(X_all)
    X_real_s = scaler.transform(X_real)

    X_all_t = torch.tensor(X_all_s, dtype=torch.float32)
    y_all_t = torch.tensor(y_all, dtype=torch.float32)
    w_all_t = torch.tensor(weights, dtype=torch.float32)

    X_real_t = torch.tensor(X_real_s, dtype=torch.float32)
    y_real_t = torch.tensor(y_real, dtype=torch.float32)

    dataset = TensorDataset(X_all_t, y_all_t, w_all_t)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = BurnTrackMLP().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_mae_real = float("inf")
    patience_counter = 0

    print(f"Training MLP v2 on {DEVICE}")
    print(f"  Real:   {n_real} samples (weight={REAL_WEIGHT}x)")
    print(f"  Synth:  {n_synth} samples (weight={SYNTH_WEIGHT}x)")
    print(f"  Total:  {n_real + n_synth} samples")
    print(f"  Features: wind, humidity, slope, ros_rothermel, h_dead, sigma, m_live, mx")
    print()

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        n_batch = 0

        for bx, by, bw in loader:
            bx, by, bw = bx.to(DEVICE), by.to(DEVICE), bw.to(DEVICE)
            optimizer.zero_grad()
            pred = model(bx)
            loss = torch.mean(bw * (pred - by) ** 2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item() * len(bx)
            n_batch += len(bx)

        scheduler.step()
        avg_loss = total_loss / n_batch

        model.eval()
        with torch.no_grad():
            pred_real = model(X_real_t.to(DEVICE))
            mae_real = torch.mean(torch.abs(pred_real.cpu() - y_real_t)).item()

        if epoch % 10 == 0 or epoch == EPOCHS - 1:
            print(f"  Epoch {epoch:3d}/{EPOCHS} | loss={avg_loss:.4f} | MAE_real={mae_real:.3f}")

        if mae_real < best_mae_real:
            best_mae_real = mae_real
            patience_counter = 0
            torch.save(model.state_dict(), "models/mlp_v2_best.pt")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch}")
                break

    model.load_state_dict(torch.load("models/mlp_v2_best.pt", map_location=DEVICE))

    import joblib
    joblib.dump(scaler, "models/mlp_v2_scaler.joblib")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n=== FINAL MODEL ===")
    print(f"  Model params: {n_params}")
    print(f"  Best MAE real: {best_mae_real:.3f}")
    print(f"  Saved: models/mlp_v2_best.pt + models/mlp_v2_scaler.joblib")


if __name__ == "__main__":
    train()
