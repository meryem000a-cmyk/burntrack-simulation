"""Train XGBoost corrector on real+synthetic data with real weight=50x."""
import json
import numpy as np
import pandas as pd
import joblib
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler

FEATURES = json.load(open("models/rf_feature_names.json"))

rt = pd.read_csv("data/processed/real_train.csv")
rv = pd.read_csv("data/processed/real_val.csv")
st = pd.read_csv("data/processed/train.csv")

X_rt = rt[FEATURES].fillna(0).values
y_rt = rt["delta_ros"].values
X_rv = rv[FEATURES].fillna(0).values
y_rv = rv["delta_ros"].values
X_st = st[FEATURES].fillna(0).values
y_st = st["delta_ros"].values

X_all = np.vstack([X_rt, X_st])
y_all = np.concatenate([y_rt, y_st])
w = np.concatenate([np.full(len(y_rt), 50), np.full(len(y_st), 1)])

sc = StandardScaler()
Xs = sc.fit_transform(X_all)
Xv = sc.transform(X_rv)

configs = [
    {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.1, "subsample": 0.8, "colsample_bytree": 0.8},
    {"n_estimators": 500, "max_depth": 10, "learning_rate": 0.03, "subsample": 0.7, "colsample_bytree": 0.7},
    {"n_estimators": 800, "max_depth": 5, "learning_rate": 0.05, "subsample": 0.9, "colsample_bytree": 0.9, "min_child_weight": 5},
    {"n_estimators": 500, "max_depth": 8, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 0.1, "reg_lambda": 1.0},
]

print("=== XGBoost Hyperparameter Search ===\n")
best_r2 = -999
best_cfg = None

for c in configs:
    xgb = XGBRegressor(**c, random_state=42, n_jobs=-1)
    xgb.fit(Xs, y_all, sample_weight=w)
    yp = xgb.predict(Xv)
    ss_res = np.sum((y_rv - yp) ** 2)
    ss_tot = np.sum((y_rv - np.mean(y_rv)) ** 2)
    r2 = 1 - ss_res / ss_tot
    rmse = np.sqrt(np.mean((y_rv - yp) ** 2))
    mae = np.mean(np.abs(y_rv - yp))
    print(f"  depth={c['max_depth']:2d} lr={c['learning_rate']:.3f} n={c['n_estimators']:4d} -> R2={r2:.4f}  RMSE={rmse:.3f}  MAE={mae:.3f}")
    if r2 > best_r2:
        best_r2 = r2
        best_cfg = c

print(f"\nBest: depth={best_cfg['max_depth']} lr={best_cfg['learning_rate']} n={best_cfg['n_estimators']} -> R2={best_r2:.4f}")

print("\n=== Training final model with best config ===")
xgb_final = XGBRegressor(**best_cfg, random_state=42, n_jobs=-1)
xgb_final.fit(Xs, y_all, sample_weight=w)

yp_real = xgb_final.predict(Xv)
ss_res = np.sum((y_rv - yp_real) ** 2)
ss_tot = np.sum((y_rv - np.mean(y_rv)) ** 2)
r2_real = 1 - ss_res / ss_tot
rmse_real = np.sqrt(np.mean((y_rv - yp_real) ** 2))
mae_real = np.mean(np.abs(y_rv - yp_real))

print(f"\nREAL VAL:  R2={r2_real:.4f}  RMSE={rmse_real:.3f}  MAE={mae_real:.3f}")

# Also eval on RF for comparison
rf = joblib.load("models/rf_corrector.joblib")
rf_scaler = joblib.load("models/rf_scaler.joblib")
yp_rf = rf.predict(rf_scaler.transform(X_rv))
ss_res_rf = np.sum((y_rv - yp_rf) ** 2)
r2_rf = 1 - ss_res_rf / ss_tot
rmse_rf = np.sqrt(np.mean((y_rv - yp_rf) ** 2))
mae_rf = np.mean(np.abs(y_rv - yp_rf))
print(f"RF  VAL:  R2={r2_rf:.4f}  RMSE={rmse_rf:.3f}  MAE={mae_rf:.3f}")

print(f"\nComparison: XGBoost R2={r2_real:.4f} vs RF R2={r2_rf:.4f} -> {'XGBoost WINS' if r2_real > r2_rf else 'RF WINS'}")

print("\nFeature importance:")
imp = xgb_final.feature_importances_
for i in np.argsort(imp)[::-1][:15]:
    print(f"  {FEATURES[i]:25s} {imp[i]:.4f}")

joblib.dump(xgb_final, "models/xgb_corrector.joblib")
joblib.dump(sc, "models/xgb_scaler.joblib")
print("\nSaved: models/xgb_corrector.joblib + xgb_scaler.joblib")
