"""Deep debug: Why is Ridge R2=-0.31?"""
import pandas as pd
import json
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error

fuel_enc = json.load(open("models/fuel_encoding.json"))
rt = pd.read_csv("data/processed/real_train.csv")
rv = pd.read_csv("data/processed/real_val.csv")
st = pd.read_csv("data/processed/train.csv")
sv = pd.read_csv("data/processed/val.csv")

FEATS = ["fuel_encoded", "wind_speed_ms", "rh_percent", "slope_pct", "ros_rothermel"]

for df in [rt, rv, st, sv]:
    df["fuel_encoded"] = df["fuel_model_code"].map(fuel_enc).fillna(0.0)

print("=== FUEL ENCODING ANALYSIS ===")
print(f"AF_MIOMBO encoding: {fuel_enc.get('AF_MIOMBO', 'NOT FOUND')}")
print(f"Real train fuels: {rt['fuel_model_code'].unique()}")
print()

print("=== DELTA ROS DISTRIBUTION ===")
print(f"Real train:  mean={rt['delta_ros'].mean():.3f}  std={rt['delta_ros'].std():.3f}")
print(f"Real val:    mean={rv['delta_ros'].mean():.3f}  std={rv['delta_ros'].std():.3f}")
miombo = st[st['fuel_model_code'] == 'AF_MIOMBO']
print(f"Synth MIOMBO: mean={miombo['delta_ros'].mean():.3f}  std={miombo['delta_ros'].std():.3f}")
print(f"Synth ALL:    mean={st['delta_ros'].mean():.3f}  std={st['delta_ros'].std():.3f}")
print()

print("=== FEATURE DISTRIBUTIONS (real vs synth) ===")
for f in FEATS[1:]:
    print(f"  {f:20s} Real: [{rt[f].min():.2f}, {rt[f].max():.2f}] mean={rt[f].mean():.2f}  |  Synth: [{st[f].min():.2f}, {st[f].max():.2f}] mean={st[f].mean():.2f}")
print()

print("=== RIDGE ON REAL ONLY (no synth contamination) ===")
from sklearn.model_selection import cross_val_score
X_rt = rt[FEATS].values
y_rt = rt['delta_ros'].values
scaler = StandardScaler()
X_rt_s = scaler.fit_transform(X_rt)
ridge = Ridge(alpha=1.0)
scores = cross_val_score(ridge, X_rt_s, y_rt, cv=5, scoring='r2')
print(f"  5-fold CV R2 on real only: {scores.mean():.4f} (+/- {scores.std():.4f})")
print()

print("=== RIDGE TRAINED ON SYNTH, TESTED ON REAL ===")
X_st = st[FEATS].values
y_st = st['delta_ros'].values
X_all = np.vstack([X_rt, X_st])
y_all = np.concatenate([y_rt, y_st])
scaler2 = StandardScaler()
X_all_s = scaler2.fit_transform(X_all)
X_rv = rv[FEATS].values
y_rv = rv['delta_ros'].values
X_rv_s = scaler2.transform(X_rv)
ridge2 = Ridge(alpha=1.0)
ridge2.fit(X_all_s, y_all)
y_pred = ridge2.predict(X_rv_s)
print(f"  R2 (synth+real train -> real val): {r2_score(y_rv, y_pred):.4f}")
print(f"  MAE: {mean_absolute_error(y_rv, y_pred):.3f}")
print()

print("=== WHAT IF WE REMOVE FUEL ENCODING? ===")
FEATS_NO_FUEL = ["wind_speed_ms", "rh_percent", "slope_pct", "ros_rothermel"]
X_nofuel = st[FEATS_NO_FUEL].values
y_synth = st['delta_ros'].values
X_nofuel_rt = rt[FEATS_NO_FUEL].values
y_real = rt['delta_ros'].values
X_nofuel_all = np.vstack([X_nofuel_rt, X_nofuel])
y_nofuel_all = np.concatenate([y_real, y_synth])
scaler3 = StandardScaler()
X_nofuel_s = scaler3.fit_transform(X_nofuel_all)
X_rv_nofuel = scaler3.transform(rv[FEATS_NO_FUEL].values)
ridge3 = Ridge(alpha=1.0)
ridge3.fit(X_nofuel_s, y_nofuel_all)
y_pred3 = ridge3.predict(X_rv_nofuel)
print(f"  R2 without fuel encoding: {r2_score(y_rv, y_pred3):.4f}")
