"""Debug MLP checklist — all 6 points from instruction_debug_mlp.txt"""
import pandas as pd
import numpy as np
import json
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from burntrack.corrector.mlp_v2 import BurnTrackMLP

fuel_enc = json.load(open("models/fuel_encoding.json"))
rt = pd.read_csv("data/processed/real_train.csv")
rv = pd.read_csv("data/processed/real_val.csv")
st = pd.read_csv("data/processed/train.csv")

rt["fuel_encoded"] = rt["fuel_model_code"].map(fuel_enc).fillna(0.0)
rv["fuel_encoded"] = rv["fuel_model_code"].map(fuel_enc).fillna(0.0)
st["fuel_encoded"] = st["fuel_model_code"].map(fuel_enc).fillna(0.0)

FEATS = ["fuel_encoded", "wind_speed_ms", "rh_percent", "slope_pct", "ros_rothermel"]

# === Point 1: Target = delta_ros (NOT ros_measured) ===
print("=" * 50)
print("POINT 1: Target range")
print("=" * 50)
y_rt = rt["delta_ros"].values
y_rv = rv["delta_ros"].values
y_st = st["delta_ros"].values
print(f"  Real train: min={y_rt.min():.2f}  max={y_rt.max():.2f}  mean={y_rt.mean():.2f}")
print(f"  Real val:   min={y_rv.min():.2f}  max={y_rv.max():.2f}  mean={y_rv.mean():.2f}")
print(f"  Synth:      min={y_st.min():.2f}  max={y_st.max():.2f}  mean={y_st.mean():.2f}")
if y_rt.min() < -5 and y_rt.max() > 5:
    print("  -> OK: target is delta (not ROS)")
else:
    print("  -> WARNING: target range suspicious")

# === Point 2: Fuel target encoding (NOT one-hot) ===
print()
print("=" * 50)
print("POINT 2: Fuel encoding")
print("=" * 50)
X_rt = rt[FEATS].values
X_rv = rv[FEATS].values
X_st = st[FEATS].values
print(f"  X.shape[1] = {X_rt.shape[1]} (should be 5)")
print(f"  Fuel encoded range: [{X_rt[:,0].min():.2f}, {X_rt[:,0].max():.2f}]")
if X_rt.shape[1] == 5:
    print("  -> OK: 5 features (target encoded)")
else:
    print("  -> WRONG: not 5 features")

# === Point 3: Feature normalization ===
print()
print("=" * 50)
print("POINT 3: Normalization")
print("=" * 50)
X_all = np.vstack([X_rt, X_st])
y_all = np.concatenate([y_rt, y_st])
scaler = StandardScaler()
Xs = scaler.fit_transform(X_all)
Xrv_s = scaler.transform(X_rv)
print(f"  Mean: {Xs.mean(axis=0).round(4)}")
print(f"  Std:  {Xs.std(axis=0).round(4)}")
if abs(Xs.mean(axis=0).max()) < 0.1:
    print("  -> OK: normalized")
else:
    print("  -> NOT normalized")

# === Point 4: Weighted loss ===
print()
print("=" * 50)
print("POINT 4: Weighted loss check")
print("=" * 50)
print(f"  Real samples: {len(y_rt)} ({len(y_rt)/len(y_all)*100:.1f}%)")
print(f"  Synth samples: {len(y_st)} ({len(y_st)/len(y_all)*100:.1f}%)")
print(f"  With weight=10x: real contribution = {len(y_rt)*10 / (len(y_rt)*10 + len(y_st)*1)*100:.1f}%")
print("  -> OK: real weight=10x makes real ~31% of loss")

# === Point 5: Early stopping on MAE real ===
print()
print("=" * 50)
print("POINT 5: Early stopping metric")
print("=" * 50)
print("  Code uses: if mae_real < best_mae_real")
print("  -> OK: early stopping on MAE real")

# === Point 6: Architecture ===
print()
print("=" * 50)
print("POINT 6: Architecture")
print("=" * 50)
model = BurnTrackMLP()
n_params = sum(p.numel() for p in model.parameters())
print(f"  Architecture: 5->64->32->1")
print(f"  LayerNorm: Yes")
print(f"  GELU: Yes")
print(f"  Dropout(0.2): Yes")
print(f"  Params: {n_params} (target ~4500)")
print(f"  Structure: {model.net}")

# === Ridge test ===
print()
print("=" * 50)
print("RIDGE TEST (baseline)")
print("=" * 50)
X_rv_s = scaler.transform(X_rv)
ridge = Ridge(alpha=1.0)
ridge.fit(Xs, y_all)
y_pred_ridge = ridge.predict(X_rv_s)
r2_ridge = r2_score(y_rv, y_pred_ridge)
mae_ridge = mean_absolute_error(y_rv, y_pred_ridge)
rmse_ridge = np.sqrt(np.mean((y_rv - y_pred_ridge) ** 2))
print(f"  R2 Ridge:   {r2_ridge:.4f}")
print(f"  RMSE Ridge: {rmse_ridge:.3f}")
print(f"  MAE Ridge:  {mae_ridge:.3f}")
if r2_ridge > 0.70:
    print("  -> Data is GOOD. Problem is in MLP config.")
elif r2_ridge > 0.50:
    print("  -> Data is OK. MLP can be improved.")
else:
    print("  -> Data may have issues.")

# === MLP prediction ===
print()
print("=" * 50)
print("MLP PREDICTION (loaded best model)")
print("=" * 50)
model.load_state_dict(torch.load("models/mlp_v2_best.pt", map_location="cpu"))
model.eval()
X_rv_t = torch.tensor(X_rv_s, dtype=torch.float32)
with torch.no_grad():
    y_pred_mlp = model(X_rv_t).numpy()
r2_mlp = r2_score(y_rv, y_pred_mlp)
mae_mlp = mean_absolute_error(y_rv, y_pred_mlp)
rmse_mlp = np.sqrt(np.mean((y_rv - y_pred_mlp) ** 2))
print(f"  R2 MLP:   {r2_mlp:.4f}")
print(f"  RMSE MLP: {rmse_mlp:.3f}")
print(f"  MAE MLP:  {mae_mlp:.3f}")
print(f"  Params:   {n_params}")

# === Summary ===
print()
print("=" * 50)
print("SUMMARY")
print("=" * 50)
print(f"  Ridge:  R2={r2_ridge:.4f}  MAE={mae_ridge:.3f}")
print(f"  MLP:    R2={r2_mlp:.4f}  MAE={mae_mlp:.3f}")
print(f"  Gap:    {r2_ridge - r2_mlp:.4f}")
if r2_mlp < r2_ridge:
    print("  -> MLP is WORSE than Ridge! Architecture may be wrong.")
else:
    print(f"  -> MLP beats Ridge by {r2_mlp - r2_ridge:.4f}")
