"""
train_mlp_v2_full.py — Train MLP with 31 features (fair comparison with RF/XGBoost)
"""
import json
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

FEATURES_31 = json.load(open("models/rf_feature_names.json"))
REAL_WEIGHT = 10.0
BATCH_SIZE = 32
LR = 3e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 300
PATIENCE = 30
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class MLP31(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(31, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train():
    rt = pd.read_csv("data/processed/real_train.csv")
    rv = pd.read_csv("data/processed/real_val.csv")
    st = pd.read_csv("data/processed/train.csv")

    X_rt = rt[FEATURES_31].fillna(0).values
    y_rt = rt["delta_ros"].values
    X_rv = rv[FEATURES_31].fillna(0).values
    y_rv = rv["delta_ros"].values
    X_st = st[FEATURES_31].fillna(0).values
    y_st = st["delta_ros"].values

    X_train = np.vstack([X_rt, X_st])
    y_train = np.concatenate([y_rt, y_st])
    weights = np.concatenate([np.full(len(y_rt), REAL_WEIGHT), np.full(len(y_st), 1.0)])

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_rv_s = scaler.transform(X_rv)

    X_train_t = torch.tensor(X_train_s, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    w_train_t = torch.tensor(weights, dtype=torch.float32)
    X_rv_t = torch.tensor(X_rv_s, dtype=torch.float32)
    y_rv_t = torch.tensor(y_rv, dtype=torch.float32)

    dataset = TensorDataset(X_train_t, y_train_t, w_train_t)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = MLP31().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_mae_real = float("inf")
    patience_counter = 0

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Training MLP-31 on {DEVICE} ({n_params} params)")
    print(f"  Real: {len(y_rt)} (weight={REAL_WEIGHT}x), Synth: {len(y_st)}")
    print()

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        n_samples = 0
        for bx, by, bw in loader:
            bx, by, bw = bx.to(DEVICE), by.to(DEVICE), bw.to(DEVICE)
            optimizer.zero_grad()
            pred = model(bx)
            loss = torch.mean(bw * (pred - by) ** 2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item() * len(bx)
            n_samples += len(bx)
        scheduler.step()

        model.eval()
        with torch.no_grad():
            pred_rv = model(X_rv_t.to(DEVICE)).cpu().numpy()
            y_pred_rv = pred_rv
            ss_res = np.sum((y_rv - y_pred_rv) ** 2)
            ss_tot = np.sum((y_rv - np.mean(y_rv)) ** 2)
            r2 = 1 - ss_res / ss_tot
            rmse = np.sqrt(np.mean((y_rv - y_pred_rv) ** 2))
            mae = np.mean(np.abs(y_rv - y_pred_rv))
            pred_rt = model(torch.tensor(scaler.transform(X_rt), dtype=torch.float32).to(DEVICE)).cpu().numpy()
            mae_real = np.mean(np.abs(y_rt - pred_rt))

        if epoch % 20 == 0 or epoch == EPOCHS - 1:
            print(f"  Epoch {epoch:3d}/{EPOCHS} | loss={total_loss/n_samples:.4f} | MAE_real={mae_real:.3f} | R2={r2:.4f} | RMSE={rmse:.3f}")

        if mae_real < best_mae_real:
            best_mae_real = mae_real
            best_r2 = r2
            best_rmse = rmse
            best_mae = mae
            patience_counter = 0
            torch.save(model.state_dict(), "models/mlp_v2_31_best.pt")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch}")
                break

    model.load_state_dict(torch.load("models/mlp_v2_31_best.pt", map_location=DEVICE))
    import joblib
    joblib.dump(scaler, "models/mlp_v2_31_scaler.joblib")

    print(f"\n=== FINAL (MLP-31 features) ===")
    print(f"  R2 val:   {best_r2:.4f}")
    print(f"  RMSE val: {best_rmse:.3f}")
    print(f"  MAE val:  {best_mae:.3f}")
    print(f"  MAE real: {best_mae_real:.3f}")
    print(f"\n  Comparison:")
    print(f"    RF:   R2=0.6013  RMSE=2.960")
    print(f"    XGB:  R2=0.6456  RMSE=2.791")
    print(f"    MLP5: R2=0.3904  RMSE=3.660  (5 features)")
    print(f"    MLP31: R2={best_r2:.4f}  RMSE={best_rmse:.3f}  (31 features)")


if __name__ == "__main__":
    train()
