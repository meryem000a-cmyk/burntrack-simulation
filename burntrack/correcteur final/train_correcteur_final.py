#!/usr/bin/env python3
"""
BurnTrack — Entraînement du Correcteur Final MLP
Script adapté pour fonctionner avec les données existantes du projet.

Adapte les colonnes des datasets existants (african_ground_truth.csv,
synthetic_dataset.csv) au format attendu par le correcteur final.

Usage:
    python train_correcteur_final.py
"""

import os
import sys
import json
import pickle

# Fix encodage Windows pour les emojis
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if sys.stderr.encoding != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy import stats

# ─── Ajouter le dossier courant au path pour les imports ───
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from source.model import BurnTrackMLPMinimal, BurnTrackAdvancedCorrector, BurnTrackFTGatedCorrector
from source.loss import WeightedMSELoss

# =====================================================================
# CONFIGURATION
# =====================================================================

# Chemins données (relatifs au projet)
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "..", "data", "processed")
TRAIN_DATA_PATH = os.path.join(DATA_DIR, "south_africa_manual_dataset.csv")
TEST_DATA_PATH = None

# Sorties
CHECKPOINT_DIR = os.path.join(SCRIPT_DIR, "checkpoints")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")

# Hyperparamètres (calibrés pour le MLP minimal, sans fuite de cible)
EPOCHS = 250
BATCH_SIZE = 64
LR = 1e-3
WEIGHT_DECAY = 1e-3
PATIENCE = 30
HIDDEN1 = 64
HIDDEN2 = 32
DROPOUT = 0.2
SEED = 42

# Features attendues par le correcteur (31 features physiques — SANS fuite de cible)
# NOTE: 'thermal_proxy' a été supprimé car il était construit à partir de la
# cible (delta_ros), ce qui constituait une fuite d'information (target leakage).
FEATURE_COLS = [
    # 1. Combustible & Végétation
    'fuel_encoded', 'w_total_kg_m2', 'w_dead_kg_m2', 'w_live_kg_m2', 'delta_m', 'sigma_m2_m3', 'mx_percent',
    # 2. Topographie
    'slope', 'aspect_deg',
    # 3. Météo & Atmosphère
    'wind_speed', 'humidity', 'temp_c', 'vpd_kpa', 'dfmc_percent',
    # 4. Variables internes Rothermel (apprentissage résiduel physique)
    'ros_rothermel', 'phi_w', 'phi_s', 'phi_eff', 'beta_ratio', 'I_R_kW_m2', 'xi', 'tau_min',
    # 5. Interprétations Non-linéaires Avancées
    'wind_sq', 'slope_sq', 'wind_slope_inter', 'wind_hum_ratio', 'energy_flux', 'roth_sq', 'roth_wind', 'temp_vpd', 'brightness_k'
]


def set_seed(seed=42):
    """Reproductibilité."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =====================================================================
# ADAPTATION DES COLONNES
# =====================================================================

def adapt_real_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adapte le format african_ground_truth.csv au format attendu.

    Colonnes source → cible :
        fuel_model_code → fuel_model
        wind_speed_ms   → wind_speed
        rh_percent      → humidity (pas de conversion, c'est déjà en %)
        slope_pct       → slope
        ros_measured    → ros_measured (OK)
        ros_rothermel   → ros_rothermel (OK)
    """
    df = df.copy()

    # Renommage des colonnes
    rename_map = {}
    if 'fuel_model_code' in df.columns and 'fuel_model' not in df.columns:
        rename_map['fuel_model_code'] = 'fuel_model'
    if 'wind_speed_ms' in df.columns and 'wind_speed' not in df.columns:
        rename_map['wind_speed_ms'] = 'wind_speed'
    if 'rh_percent' in df.columns and 'humidity' not in df.columns:
        rename_map['rh_percent'] = 'humidity'
    if 'slope_pct' in df.columns and 'slope' not in df.columns:
        rename_map['slope_pct'] = 'slope'
    if 'ros_observed' in df.columns and 'ros_measured' not in df.columns:
        rename_map['ros_observed'] = 'ros_measured'

    df = df.rename(columns=rename_map)

    # Vérification
    required = ['fuel_model', 'wind_speed', 'humidity', 'slope', 'ros_measured', 'ros_rothermel']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes dans données réelles après adaptation: {missing}")

    return df





# =====================================================================
# PIPELINE DE DONNÉES
# =====================================================================

def analyze_bias(df_real: pd.DataFrame) -> pd.DataFrame:
    """Analyse les biais de Rothermel par fuel model."""
    print("\n" + "=" * 60)
    print("📊 ANALYSE DES BIAIS PAR FUEL")
    print("=" * 60)

    # La colonne 'delta_ros' est déjà présente, plus besoin de la recalculer
    bias_analysis = []
    for fuel in sorted(df_real['fuel_model'].unique()):
        subset = df_real[df_real['fuel_model'] == fuel]['delta_ros']
        if len(subset) < 2:
            continue
        t_stat, p_value = stats.ttest_1samp(subset, 0)
        bias_analysis.append({
            'fuel_model': fuel,
            'n': len(subset),
            'mean_bias': subset.mean(),
            'std_bias': subset.std(),
            'p_value': p_value,
            'significant': p_value < 0.05
        })

    bias_df = pd.DataFrame(bias_analysis).sort_values('mean_bias')
    print(f"\n{len(bias_df)} fuels analysés:")
    print(bias_df.to_string(index=False, float_format='%.3f'))

    herbes = bias_df[bias_df['mean_bias'] < -3]['fuel_model'].tolist()
    boises = bias_df[bias_df['mean_bias'] > 1]['fuel_model'].tolist()
    print(f"\n  Herbes (sur-estimés par Rothermel): {herbes}")
    print(f"  Boisés (sous-estimés par Rothermel): {boises}")

    return bias_df


def compute_target_encoding(bias_df: pd.DataFrame, save_dir: str) -> dict:
    """Target encoding: fuel → biais moyen."""
    fuel_encoding = bias_df.set_index('fuel_model')['mean_bias'].to_dict()

    print(f"\n{'='*60}")
    print("🏷️  TARGET ENCODING")
    print(f"{'='*60}")
    for fuel, bias in sorted(fuel_encoding.items(), key=lambda x: x[1]):
        print(f"  {fuel:25s} → {bias:+.3f} m/min")

    path = os.path.join(save_dir, 'fuel_encoding.json')
    with open(path, 'w') as f:
        json.dump(fuel_encoding, f, indent=2)
    print(f"\n  💾 Sauvegardé: {path}")

    return fuel_encoding


def prepare_data(df_train, df_test_ext, save_dir):
    """Préparation des données, normalisation, target encoding (sans data leakage)."""
    print(f"\n{'='*60}")
    print("🔧 PRÉPARATION DES DONNÉES (RÉELLES UNIQUEMENT)")
    print(f"{'='*60}")

    # Delta ROS (target)
    df_train['delta_ros'] = df_train['ros_measured'] - df_train['ros_rothermel']
    if df_test_ext is not None:
        df_test_ext['delta_ros'] = df_test_ext['ros_measured'] - df_test_ext['ros_rothermel']

    # Interprétations Non-linéaires Avancées
    df_train['wind_sq'] = df_train['wind_speed'] ** 2
    df_train['slope_sq'] = df_train['slope'] ** 2
    df_train['wind_slope_inter'] = df_train['wind_speed'] * df_train['slope']
    df_train['wind_hum_ratio'] = df_train['wind_speed'] / (df_train['humidity'] + 1e-5)
    df_train['energy_flux'] = df_train['w_total_kg_m2'] * df_train['wind_speed']
    df_train['roth_sq'] = df_train['ros_rothermel'] ** 2
    df_train['roth_wind'] = df_train['ros_rothermel'] * df_train['wind_speed']
    df_train['temp_vpd'] = df_train['temp_c'] * df_train['vpd_kpa']
    df_train['lst_c'] = df_train.get('lst_c', df_train['temp_c'] + 10.0)
    df_train['brightness_k'] = pd.to_numeric(df_train.get('brightness_k', 305.0), errors='coerce').fillna(305.0)

    if df_test_ext is not None:
        df_test_ext['wind_sq'] = df_test_ext['wind_speed'] ** 2
        df_test_ext['slope_sq'] = df_test_ext['slope'] ** 2
        df_test_ext['wind_slope_inter'] = df_test_ext['wind_speed'] * df_test_ext['slope']
        df_test_ext['wind_hum_ratio'] = df_test_ext['wind_speed'] / (df_test_ext['humidity'] + 1e-5)
        df_test_ext['energy_flux'] = df_test_ext['w_total_kg_m2'] * df_test_ext['wind_speed']
        df_test_ext['roth_sq'] = df_test_ext['ros_rothermel'] ** 2
        df_test_ext['roth_wind'] = df_test_ext['ros_rothermel'] * df_test_ext['wind_speed']
        df_test_ext['temp_vpd'] = df_test_ext['temp_c'] * df_test_ext['vpd_kpa']
        df_test_ext['lst_c'] = df_test_ext.get('lst_c', df_test_ext['temp_c'] + 10.0)
        df_test_ext['brightness_k'] = pd.to_numeric(df_test_ext.get('brightness_k', 305.0), errors='coerce').fillna(305.0)

    # Séparation train/val (stratifiée par fuel_model)
    raw_features = [c for c in FEATURE_COLS if c != 'fuel_encoded']
    df_train = df_train.dropna(subset=['delta_ros'] + raw_features)
    if df_test_ext is not None:
        df_test_ext = df_test_ext.dropna(subset=['delta_ros'] + raw_features)

    df_train_m, df_val_m = train_test_split(df_train, test_size=0.2, random_state=SEED)
    
    if df_test_ext is not None:
        df_test_m = df_test_ext.copy()
    else:
        # Si pas de test externe, on prend un bout du val
        df_val_m, df_test_m = train_test_split(df_val_m, test_size=0.5, random_state=SEED)

    print(f"  Train : {len(df_train_m)} échantillons")
    print(f"  Val   : {len(df_val_m)} échantillons")
    print(f"  Test  : {len(df_test_m)} échantillons")

    # Target Encoding sur le Train UNIQUEMENT
    bias_df = analyze_bias(df_train_m)
    fuel_encoding = compute_target_encoding(bias_df, save_dir)

    def apply_encoding(df):
        df_enc = df.copy()
        df_enc['fuel_encoded'] = df_enc['fuel_model'].map(fuel_encoding).fillna(0.0)
        return df_enc

    df_train_m = apply_encoding(df_train_m)
    df_val_m = apply_encoding(df_val_m)
    df_test_m = apply_encoding(df_test_m)

    # Normalisation
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(df_train_m[FEATURE_COLS].values)
    X_val_scaled = scaler.transform(df_val_m[FEATURE_COLS].values)
    X_test_scaled = scaler.transform(df_test_m[FEATURE_COLS].values)

    y_train = df_train_m['delta_ros'].values.astype(np.float32)
    y_val = df_val_m['delta_ros'].values.astype(np.float32)
    y_test = df_test_m['delta_ros'].values.astype(np.float32)

    # Masks pour l'évaluation (tous réels)
    is_r_train = np.ones_like(y_train)
    is_r_val = np.ones_like(y_val)
    is_r_test = np.ones_like(y_test)

    # Sauvegarder le scaler
    scaler_path = os.path.join(save_dir, 'scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"\n  💾 Scaler sauvegardé: {scaler_path}")
    print(f"  ✅ Données prêtes. Shape features: {X_train_scaled.shape}")

    return (X_train_scaled, X_val_scaled, X_test_scaled,
            y_train, y_val, y_test,
            FEATURE_COLS, df_test_m, fuel_encoding)


# =====================================================================
# ENTRAÎNEMENT
# =====================================================================

def train_model(model, train_loader, val_loader, loss_fn, device,
                epochs=EPOCHS, patience=PATIENCE, lr=LR, weight_decay=WEIGHT_DECAY):
    """Boucle d'entraînement avec early stopping."""

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )

    history = {
        'train_loss': [], 'val_loss': [],
        'val_mae': [], 'val_mae_real': [], 'lr': []
    }

    best_val_mae = float('inf')
    patience_counter = 0
    best_state = None

    print(f"\n{'='*60}")
    print(f"🚀 ENTRAÎNEMENT")
    print(f"{'='*60}")
    print(f"  Device: {device} | Epochs max: {epochs} | Patience: {patience}")
    print(f"  LR: {lr} | Weight decay: {weight_decay}")
    print(f"  Paramètres: {model.count_parameters():,}")
    print()

    for epoch in range(epochs):
        # ── TRAIN ──
        model.train()
        train_losses = []

        for X_batch, y_batch, is_real_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            is_real_batch = is_real_batch.to(device)

            optimizer.zero_grad()
            pred = model(X_batch)
            loss = loss_fn(pred, y_batch)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        # ── VALIDATION ──
        model.eval()
        val_losses, val_maes, val_maes_real = [], [], []

        with torch.no_grad():
            for X_batch, y_batch, is_real_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                is_real_batch = is_real_batch.to(device)

                pred = model(X_batch)
                val_losses.append(loss_fn(pred, y_batch).item())

                mae = torch.abs(pred - y_batch).mean().item()
                val_maes.append(mae)

                real_mask = is_real_batch.bool()
                if real_mask.sum() > 0:
                    mae_real = torch.abs(pred[real_mask] - y_batch[real_mask]).mean().item()
                    val_maes_real.append(mae_real)

        avg_train = np.mean(train_losses)
        avg_val = np.mean(val_losses)
        avg_mae = np.mean(val_maes)
        avg_mae_real = np.mean(val_maes_real) if val_maes_real else float('nan')

        history['train_loss'].append(avg_train)
        history['val_loss'].append(avg_val)
        history['val_mae'].append(avg_mae)
        history['val_mae_real'].append(avg_mae_real)
        history['lr'].append(optimizer.param_groups[0]['lr'])

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs} | "
                  f"Train: {avg_train:.4f} | "
                  f"Val MAE: {avg_mae:.4f} | "
                  f"Val MAE(réel): {avg_mae_real:.4f} | "
                  f"LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Early stopping sur MAE réel
        target_mae = avg_mae_real if not np.isnan(avg_mae_real) else avg_mae
        if target_mae < best_val_mae:
            best_val_mae = target_mae
            patience_counter = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n  ✋ Early stopping à epoch {epoch+1} "
                      f"(best MAE réel: {best_val_mae:.4f})")
                break

        scheduler.step(target_mae)

    if best_state is not None:
        model.load_state_dict(best_state)

    print(f"\n  ✅ Entraînement terminé. Best MAE réel: {best_val_mae:.4f}")
    return model, history


# =====================================================================
# ÉVALUATION
# =====================================================================

def evaluate_model(model, X_test, y_test, is_real_test, df_test,
                   fuel_encoding, feature_cols, device, save_dir):
    """Évalue et génère les visualisations."""
    from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

    os.makedirs(save_dir, exist_ok=True)

    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X_test, dtype=torch.float32).to(device)
        delta_pred = model(X_t).cpu().numpy()

    real_mask = is_real_test.astype(bool)
    y_real = y_test[real_mask]
    pred_real = delta_pred[real_mask]

    # Baselines
    delta_baseline = np.zeros_like(y_real)  # Rothermel seul (delta=0)

    # Target encoding baseline
    fuel_models = df_test[real_mask].get('fuel_model')
    if fuel_models is not None:
        global_mean = np.mean(list(fuel_encoding.values()))
        delta_target_enc = np.array([fuel_encoding.get(f, global_mean) for f in fuel_models.values])
    else:
        delta_target_enc = np.zeros_like(y_real)

    # Métriques
    results = []
    for name, pred in [('Rothermel seul', delta_baseline),
                        ('Target Encoding', delta_target_enc),
                        ('Modèle Avancé (PINN)', pred_real)]:
        results.append({
            'Modèle': name,
            'MAE': mean_absolute_error(y_real, pred),
            'RMSE': np.sqrt(mean_squared_error(y_real, pred)),
            'R²': r2_score(y_real, pred)
        })

    results_df = pd.DataFrame(results)
    print(f"\n{'='*60}")
    print("📊 ÉVALUATION TEST SET (RÉEL UNIQUEMENT)")
    print(f"{'='*60}")
    print(results_df.round(4).to_string(index=False))

    # Amélioration
    mae_base = results[0]['MAE']
    mae_mlp = results[2]['MAE']
    improvement = (1 - mae_mlp / mae_base) * 100
    print(f"\n  📈 Amélioration MAE vs Rothermel seul: {improvement:+.1f}%")

    # ── Visualisations ──
    try:
        import matplotlib
        matplotlib.use('Agg')  # Backend non-interactif
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        # 1. Comparaison métriques
        x = np.arange(len(results_df))
        width = 0.25
        for i, m in enumerate(['MAE', 'RMSE', 'R²']):
            axes[0, 0].bar(x + i * width, results_df[m], width, label=m)
        axes[0, 0].set_xticks(x + width)
        axes[0, 0].set_xticklabels(results_df['Modèle'], rotation=15)
        axes[0, 0].set_title('Comparaison des métriques', fontweight='bold')
        axes[0, 0].legend()

        # 2. Scatter pred vs réel
        axes[0, 1].scatter(y_real, pred_real, alpha=0.5, s=20, c='steelblue', edgecolors='none')
        max_d = max(np.abs(y_real).max(), np.abs(pred_real).max())
        axes[0, 1].plot([-max_d, max_d], [-max_d, max_d], 'r--', lw=2, label='Parfait')
        axes[0, 1].set_xlabel('Delta réel (m/min)')
        axes[0, 1].set_ylabel('Delta prédit (m/min)')
        r2 = results_df[results_df['Modèle'] == 'Modèle Avancé (PINN)']['R²'].values[0]
        axes[0, 1].set_title(f'Modèle Avancé (PINN) | R²={r2:.3f}', fontweight='bold')
        axes[0, 1].legend()

        # 3. Distribution erreurs
        e_base = np.abs(y_real)
        e_mlp = np.abs(y_real - pred_real)
        axes[1, 0].hist(e_base, bins=30, alpha=0.5, label='Rothermel', color='coral', edgecolor='white')
        axes[1, 0].hist(e_mlp, bins=30, alpha=0.5, label='MLP', color='steelblue', edgecolor='white')
        axes[1, 0].axvline(e_base.mean(), color='coral', linestyle='--', lw=2, label=f'MAE Roth={e_base.mean():.2f}')
        axes[1, 0].axvline(e_mlp.mean(), color='steelblue', linestyle='--', lw=2, label=f'MAE MLP={e_mlp.mean():.2f}')
        axes[1, 0].set_xlabel('Erreur absolue (m/min)')
        axes[1, 0].set_title('Distribution des erreurs', fontweight='bold')
        axes[1, 0].legend()

        # 4. Résiduel par fuel
        if fuel_models is not None:
            residual = y_real - pred_real
            res_df = pd.DataFrame({'fuel': fuel_models.values, 'residual': residual})
            unique_fuels = sorted(res_df['fuel'].unique())
            if len(unique_fuels) <= 20:
                sns.boxplot(data=res_df, x='fuel', y='residual',
                            order=unique_fuels, ax=axes[1, 1], palette='RdBu_r')
            else:
                top_fuels = res_df['fuel'].value_counts().head(15).index.tolist()
                sns.boxplot(data=res_df[res_df['fuel'].isin(top_fuels)],
                            x='fuel', y='residual', ax=axes[1, 1], palette='RdBu_r')
            axes[1, 1].axhline(y=0, color='red', linestyle='--', lw=2)
            axes[1, 1].set_title('Résiduel par fuel (MLP)', fontweight='bold')
            axes[1, 1].tick_params(axis='x', rotation=45)

        plt.suptitle('🔥 BurnTrack – Évaluation Correcteur Final MLP', fontsize=14, fontweight='bold')
        plt.tight_layout()
        eval_path = os.path.join(save_dir, 'evaluation_correcteur_final.png')
        plt.savefig(eval_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\n  📊 Visualisation sauvegardée: {eval_path}")

        # 5. Feature importance
        if hasattr(model, 'input_proj'):
            weights_l1 = model.input_proj[0].weight.detach().cpu().numpy()
        elif hasattr(model, 'input_layer'):
            weights_l1 = model.input_layer[0].weight.detach().cpu().numpy()
        else:
            weights_l1 = model.layer1.weight.detach().cpu().numpy()
        importance = np.abs(weights_l1).mean(axis=0)

        fig2, ax2 = plt.subplots(figsize=(12, 10))
        colors = ['#2ca02c' if i > np.mean(importance) else '#ff7f0e' for i in importance]
        bars = ax2.barh(feature_cols, importance, color=colors)
        ax2.set_xlabel('Importance moyenne (|poids|)')
        ax2.set_title('Importance des features (couche 1)', fontweight='bold')
        for bar, imp in zip(bars, importance):
            ax2.text(imp + 0.005, bar.get_y() + bar.get_height() / 2, f'{imp:.3f}',
                     va='center', fontsize=9)
        plt.tight_layout()
        imp_path = os.path.join(save_dir, 'feature_importance.png')
        plt.savefig(imp_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  📊 Feature importance sauvegardée: {imp_path}")

        # 6. Training history
    except ImportError as e:
        print(f"  ⚠️  Matplotlib non disponible, pas de visualisations: {e}")

    return results_df


# =====================================================================
# PIPELINE PRINCIPAL
# =====================================================================

def main():
    """Pipeline complet d'entraînement du correcteur final."""
    set_seed(SEED)

    print("=" * 60)
    print("🔥 BURNTRACK — ENTRAÎNEMENT CORRECTEUR FINAL MLP")
    print("=" * 60)

    # ── 1. Chargement des données ──
    print(f"\n📂 Chargement des données...")
    print(f"  Train: {TRAIN_DATA_PATH}")
    
    if not os.path.exists(TRAIN_DATA_PATH):
        print(f"❌ Fichier d'entraînement non trouvé: {TRAIN_DATA_PATH}")
        print("  Générez-le d'abord avec scripts/build_from_local_firms.py")
        return

    df_train_raw = pd.read_csv(TRAIN_DATA_PATH)
    print(f"  Train: {len(df_train_raw):,} lignes")

    df_test_raw = None
    if TEST_DATA_PATH and os.path.exists(TEST_DATA_PATH):
        print(f"  Test: {TEST_DATA_PATH}")
        df_test_raw = pd.read_csv(TEST_DATA_PATH)
        print(f"  Test: {len(df_test_raw):,} lignes")

    # ── 2. Adaptation des colonnes ──
    print(f"\n🔄 Adaptation des colonnes...")
    df_train = adapt_real_data(df_train_raw)
    df_test = adapt_real_data(df_test_raw) if df_test_raw is not None else None
    print(f"  ✅ Colonnes adaptées")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # ── 4. Préparation des données ──
    (X_train, X_val, X_test,
     y_train, y_val, y_test,
     feature_cols, df_test_full, fuel_encoding) = prepare_data(df_train, df_test, SCRIPT_DIR)

    # ── 6. DataLoaders ──
    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32),
            torch.ones(len(y_train), dtype=torch.float32)
        ),
        batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.float32),
            torch.ones(len(y_val), dtype=torch.float32)
        ),
        batch_size=BATCH_SIZE, shuffle=False
    )

    # ── 7. Modèle ──
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_features = X_train.shape[1]
    model = BurnTrackMLPMinimal(
        n_features=n_features,
        hidden1=HIDDEN1, hidden2=HIDDEN2, dropout=DROPOUT
    ).to(device)

    print(f"\n🧠 MLP Minimal (sans fuite de cible): {n_features} → {HIDDEN1} → {HIDDEN2} → 1")
    print(f"  Paramètres: {model.count_parameters():,}")
    print(f"  Device: {device}")

    # ── 8. Loss ──
    # On n'utilise plus de poids synthétiques vs réels
    loss_fn = nn.MSELoss()

    # ── 9. Entraînement ──
    model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        device=device,
        epochs=EPOCHS,
        patience=PATIENCE,
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    # ── 10. Évaluation ──
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results_df = evaluate_model(
        model=model,
        X_test=X_test,
        y_test=y_test,
        is_real_test=np.ones(len(y_test)),
        df_test=df_test_full,
        fuel_encoding=fuel_encoding,
        feature_cols=feature_cols,
        device=device,
        save_dir=RESULTS_DIR
    )

    # ── 11. Sauvegarde checkpoint ──
    checkpoint_path = os.path.join(CHECKPOINT_DIR, "burntrack_mlp_minimal.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'n_features': n_features,
        'feature_cols': feature_cols,
        'fuel_encoding': fuel_encoding,
        'history': history,
        'best_val_mae': min(
            [x for x in history['val_mae_real'] if not np.isnan(x)]
        ) if any(not np.isnan(x) for x in history['val_mae_real']) else min(history['val_mae']),
        'test_metrics': results_df.to_dict(),
        'config': {
            'hidden1': HIDDEN1, 'hidden2': HIDDEN2, 'dropout': DROPOUT,
            'lr': LR, 'weight_decay': WEIGHT_DECAY, 'batch_size': BATCH_SIZE,
            'seed': SEED,
        }
    }, checkpoint_path)

    print(f"\n{'='*60}")
    print(f"💾 SAUVEGARDE")
    print(f"{'='*60}")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Taille: ~{os.path.getsize(checkpoint_path)/1024:.1f} KB")
    print(f"  Scaler: {os.path.join(SCRIPT_DIR, 'scaler.pkl')}")
    print(f"  Fuel encoding: {os.path.join(SCRIPT_DIR, 'fuel_encoding.json')}")
    print(f"  Résultats: {RESULTS_DIR}/")

    print(f"\n{'='*60}")
    print(f"✅ ENTRAÎNEMENT CORRECTEUR FINAL TERMINÉ")
    print(f"{'='*60}")
    print(f"\nStructure additive: ROS_burntrack = ROS_rothermel + Δ_MLP")
    print(f"Le MLP corrige les biais systématiques du moteur physique.")


if __name__ == '__main__':
    main()
