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

from source.model import BurnTrackMLPMinimal
from source.loss import WeightedMSELoss

# =====================================================================
# CONFIGURATION
# =====================================================================

# Chemins données (relatifs au projet)
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "..", "data", "processed")
REAL_DATA_PATH = os.path.join(DATA_DIR, "african_ground_truth.csv")
SYNTH_DATA_PATH = os.path.join(DATA_DIR, "synthetic_dataset.csv")

# Sorties
CHECKPOINT_DIR = os.path.join(SCRIPT_DIR, "checkpoints")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")

# Hyperparamètres
EPOCHS = 200
BATCH_SIZE = 64
LR = 1e-3
WEIGHT_DECAY = 1e-3
PATIENCE = 20
WEIGHT_REAL = 3.0
WEIGHT_SYNTH = 1.0
HIDDEN1 = 64
HIDDEN2 = 32
DROPOUT = 0.2
SEED = 42

# Features attendues par le MLP (5 features)
FEATURE_COLS = ['fuel_encoded', 'wind_speed', 'humidity', 'slope', 'ros_rothermel']


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

    df = df.rename(columns=rename_map)

    # Vérification
    required = ['fuel_model', 'wind_speed', 'humidity', 'slope', 'ros_measured', 'ros_rothermel']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes dans données réelles après adaptation: {missing}")

    return df


def adapt_synth_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adapte le format synthetic_dataset.csv au format attendu.

    Colonnes source → cible :
        fuel_model      → fuel_model (OK)
        wind_speed      → wind_speed (OK)
        rh_percent      → humidity
        slope_deg       → slope (conversion deg→%)
        ros_observed    → ros_measured
        ros_rothermel   → ros_rothermel (OK)
    """
    df = df.copy()

    rename_map = {}
    if 'rh_percent' in df.columns and 'humidity' not in df.columns:
        rename_map['rh_percent'] = 'humidity'
    if 'ros_observed' in df.columns and 'ros_measured' not in df.columns:
        rename_map['ros_observed'] = 'ros_measured'

    df = df.rename(columns=rename_map)

    # Conversion slope_deg → slope (en %)
    if 'slope_deg' in df.columns and 'slope' not in df.columns:
        df['slope'] = np.tan(np.radians(df['slope_deg'])) * 100.0

    # Vérification
    required = ['fuel_model', 'wind_speed', 'humidity', 'slope', 'ros_measured', 'ros_rothermel']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes dans données synth après adaptation: {missing}")

    return df


# =====================================================================
# PIPELINE DE DONNÉES
# =====================================================================

def analyze_bias(df_real: pd.DataFrame) -> pd.DataFrame:
    """Analyse les biais de Rothermel par fuel model."""
    print("\n" + "=" * 60)
    print("📊 ANALYSE DES BIAIS PAR FUEL")
    print("=" * 60)

    df_real['delta_ros'] = df_real['ros_measured'] - df_real['ros_rothermel']

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


def prepare_data(df_real, df_synth, fuel_encoding, save_dir):
    """Merge, split stratifié, normalisation."""
    print(f"\n{'='*60}")
    print("🔧 PRÉPARATION DES DONNÉES")
    print(f"{'='*60}")

    # Delta ROS (target)
    df_real['delta_ros'] = df_real['ros_measured'] - df_real['ros_rothermel']
    df_synth['delta_ros'] = df_synth['ros_measured'] - df_synth['ros_rothermel']

    # Target encoding
    global_mean = np.mean(list(fuel_encoding.values()))
    df_real['fuel_encoded'] = df_real['fuel_model'].map(fuel_encoding).fillna(global_mean)
    df_synth['fuel_encoded'] = df_synth['fuel_model'].map(fuel_encoding).fillna(global_mean)

    # Source tag
    df_real['source'] = 'real'
    df_synth['source'] = 'synthetic'

    # Colonnes communes
    needed_cols = FEATURE_COLS + ['delta_ros', 'source', 'ros_rothermel', 'fuel_model']
    real_cols = [c for c in needed_cols if c in df_real.columns]
    synth_cols = [c for c in needed_cols if c in df_synth.columns]
    common = list(set(real_cols) & set(synth_cols))

    df_merged = pd.concat([df_real[common], df_synth[common]], ignore_index=True)
    print(f"\n  Dataset fusionné: {len(df_merged)} éch. "
          f"(réels: {len(df_real)}, synth: {len(df_synth)})")

    # Vérifier que toutes les features sont là
    available_features = [c for c in FEATURE_COLS if c in df_merged.columns]
    print(f"  Features: {available_features}")

    if len(available_features) != len(FEATURE_COLS):
        missing = set(FEATURE_COLS) - set(available_features)
        print(f"  ⚠️  Features manquantes: {missing}")

    # Target
    df_merged['target'] = df_merged['delta_ros']

    # Split stratifié
    df_real_m = df_merged[df_merged['source'] == 'real']
    df_synth_m = df_merged[df_merged['source'] == 'synthetic']

    real_train, real_temp = train_test_split(df_real_m, test_size=0.3, random_state=SEED)
    real_val, real_test = train_test_split(real_temp, test_size=0.5, random_state=SEED)

    synth_train, synth_temp = train_test_split(df_synth_m, test_size=0.2, random_state=SEED)
    synth_val, synth_test = train_test_split(synth_temp, test_size=0.5, random_state=SEED)

    df_train = pd.concat([real_train, synth_train])
    df_val = pd.concat([real_val, synth_val])
    df_test = pd.concat([real_test, synth_test])

    print(f"\n  Split:")
    print(f"    Train: {len(df_train):,} (réels: {len(real_train):,}, synth: {len(synth_train):,})")
    print(f"    Val:   {len(df_val):,} (réels: {len(real_val):,}, synth: {len(synth_val):,})")
    print(f"    Test:  {len(df_test):,} (réels: {len(real_test):,}, synth: {len(synth_test):,})")

    # Normalisation
    scaler = StandardScaler()
    X_train = scaler.fit_transform(df_train[available_features].values)
    X_val = scaler.transform(df_val[available_features].values)
    X_test = scaler.transform(df_test[available_features].values)

    y_train = df_train['target'].values.astype(np.float32)
    y_val = df_val['target'].values.astype(np.float32)
    y_test = df_test['target'].values.astype(np.float32)

    is_real_train = (df_train['source'] == 'real').values.astype(np.float32)
    is_real_val = (df_val['source'] == 'real').values.astype(np.float32)
    is_real_test = (df_test['source'] == 'real').values.astype(np.float32)

    # Sauvegarder le scaler
    scaler_path = os.path.join(save_dir, 'scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"\n  💾 Scaler sauvegardé: {scaler_path}")
    print(f"  ✅ Données prêtes. Shape features: {X_train.shape}")

    return (X_train, X_val, X_test,
            y_train, y_val, y_test,
            is_real_train, is_real_val, is_real_test,
            available_features, df_test)


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
            loss = loss_fn(pred, y_batch, is_real_batch)
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
                val_losses.append(loss_fn(pred, y_batch, is_real_batch).item())

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
                        ('MLP Minimal', pred_real)]:
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
        r2 = results_df[results_df['Modèle'] == 'MLP Minimal']['R²'].values[0]
        axes[0, 1].set_title(f'MLP Minimal | R²={r2:.3f}', fontweight='bold')
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
        weights_l1 = model.layer1.weight.detach().cpu().numpy()
        importance = np.abs(weights_l1).mean(axis=0)

        fig2, ax2 = plt.subplots(figsize=(10, 6))
        colors = ['#2ca02c' if i > np.mean(importance) else '#ff7f0e' for i in importance]
        bars = ax2.barh(feature_cols, importance, color=colors)
        ax2.set_xlabel('Importance moyenne (|poids|)')
        ax2.set_title('Importance des features (couche 1)', fontweight='bold')
        for bar, imp in zip(bars, importance):
            ax2.text(imp + 0.005, bar.get_y() + bar.get_height() / 2, f'{imp:.3f}',
                     va='center', fontsize=10)
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
    print(f"  Réelles:      {REAL_DATA_PATH}")
    print(f"  Synthétiques: {SYNTH_DATA_PATH}")

    if not os.path.exists(REAL_DATA_PATH):
        print(f"❌ Fichier non trouvé: {REAL_DATA_PATH}")
        return
    if not os.path.exists(SYNTH_DATA_PATH):
        print(f"❌ Fichier non trouvé: {SYNTH_DATA_PATH}")
        return

    df_real_raw = pd.read_csv(REAL_DATA_PATH)
    df_synth_raw = pd.read_csv(SYNTH_DATA_PATH)
    print(f"  Réelles: {len(df_real_raw):,} lignes, {len(df_real_raw.columns)} colonnes")
    print(f"  Synthétiques: {len(df_synth_raw):,} lignes, {len(df_synth_raw.columns)} colonnes")

    # ── 2. Adaptation des colonnes ──
    print(f"\n🔄 Adaptation des colonnes...")
    df_real = adapt_real_data(df_real_raw)
    df_synth = adapt_synth_data(df_synth_raw)
    print(f"  ✅ Colonnes adaptées")

    # ── 3. Analyse des biais ──
    bias_df = analyze_bias(df_real)

    # ── 4. Target encoding ──
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    fuel_encoding = compute_target_encoding(bias_df, SCRIPT_DIR)

    # ── 5. Préparation des données ──
    (X_train, X_val, X_test,
     y_train, y_val, y_test,
     is_r_train, is_r_val, is_r_test,
     feature_cols, df_test_full) = prepare_data(df_real, df_synth, fuel_encoding, SCRIPT_DIR)

    # ── 6. DataLoaders ──
    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32),
            torch.tensor(is_r_train, dtype=torch.float32)
        ),
        batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.float32),
            torch.tensor(is_r_val, dtype=torch.float32)
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

    print(f"\n🧠 Modèle: {n_features} → {HIDDEN1} → {HIDDEN2} → 1")
    print(f"  Paramètres: {model.count_parameters():,}")
    print(f"  Device: {device}")

    # ── 8. Loss ──
    loss_fn = WeightedMSELoss(weight_real=WEIGHT_REAL, weight_synth=WEIGHT_SYNTH)

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
        is_real_test=is_r_test,
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
            'weight_real': WEIGHT_REAL, 'weight_synth': WEIGHT_SYNTH,
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
