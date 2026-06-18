"""
Évaluation du modèle BurnTrack
Métriques: MAE, RMSE, R² + Visualisations
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from typing import Dict


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, name: str) -> Dict:
    """Calcule MAE, RMSE, R²."""
    return {
        'Modèle': name,
        'MAE': mean_absolute_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'R²': r2_score(y_true, y_pred)
    }


def evaluate_model(model, X_test, y_test, is_real_test, df_test, 
                   fuel_encoded_test, mae_baseline, device, feature_cols, 
                   save_dir='results'):
    """Évalue le modèle et compare avec baseline."""
    import os
    os.makedirs(save_dir, exist_ok=True)

    model.eval()
    with torch.no_grad():
        X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
        delta_pred = model(X_test_t).cpu().numpy()

    real_mask = is_real_test.astype(bool)
    y_test_r = y_test[real_mask]

    delta_baseline = np.zeros_like(y_test_r)
    delta_target_enc = fuel_encoded_test[real_mask]
    delta_mlp = delta_pred[real_mask]

    results = [
        compute_metrics(y_test_r, delta_baseline, 'Rothermel seul'),
        compute_metrics(y_test_r, delta_target_enc, 'Target Encoding'),
        compute_metrics(y_test_r, delta_mlp, 'MLP Minimal')
    ]

    results_df = pd.DataFrame(results)
    print("=== ÉVALUATION TEST SET (RÉEL UNIQUEMENT) ===")
    print(results_df.round(3).to_string(index=False))

    # Visualisations
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    metrics = ['MAE', 'RMSE', 'R²']
    x = np.arange(len(results_df))
    for i, m in enumerate(metrics):
        axes[0, 0].bar(x + i*0.25, results_df[m], 0.25, label=m)
    axes[0, 0].set_xticks(x + 0.25)
    axes[0, 0].set_xticklabels(results_df['Modèle'], rotation=15)
    axes[0, 0].set_title('Comparaison des métriques', fontweight='bold')
    axes[0, 0].legend()

    axes[0, 1].scatter(y_test_r, delta_mlp, alpha=0.6, s=30, c='steelblue', edgecolors='none')
    max_d = max(np.abs(y_test_r).max(), np.abs(delta_mlp).max())
    axes[0, 1].plot([-max_d, max_d], [-max_d, max_d], 'r--', lw=2, label='Parfait')
    axes[0, 1].set_xlabel('Delta réel (m/min)')
    axes[0, 1].set_ylabel('Delta prédit (m/min)')
    r2_mlp = results_df[results_df['Modèle']=='MLP Minimal']['R²'].values[0]
    axes[0, 1].set_title(f'MLP Minimal | R²={r2_mlp:.3f}', fontweight='bold')
    axes[0, 1].legend()

    e_base = np.abs(y_test_r)
    e_mlp = np.abs(y_test_r - delta_mlp)
    axes[1, 0].hist(e_base, bins=30, alpha=0.5, label='Rothermel', color='coral', edgecolor='white')
    axes[1, 0].hist(e_mlp, bins=30, alpha=0.5, label='MLP', color='steelblue', edgecolor='white')
    axes[1, 0].axvline(e_base.mean(), color='coral', linestyle='--', lw=2, label=f'MAE={e_base.mean():.2f}')
    axes[1, 0].axvline(e_mlp.mean(), color='steelblue', linestyle='--', lw=2, label=f'MAE={e_mlp.mean():.2f}')
    axes[1, 0].set_xlabel('Erreur absolue (m/min)')
    axes[1, 0].set_title('Distribution des erreurs', fontweight='bold')
    axes[1, 0].legend()

    residual = y_test_r - delta_mlp
    test_fuels = df_test[real_mask]['fuel_model'].values
    res_df = pd.DataFrame({'fuel': test_fuels, 'residual': residual})
    sns.boxplot(data=res_df, x='fuel', y='residual', 
                order=sorted(np.unique(test_fuels)), ax=axes[1, 1], palette='RdBu_r')
    axes[1, 1].axhline(y=0, color='red', linestyle='--', lw=2)
    axes[1, 1].set_title('Résiduel par fuel (MLP)', fontweight='bold')
    axes[1, 1].tick_params(axis='x', rotation=45)

    plt.suptitle('Évaluation Comparative BurnTrack', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/evaluation.png', dpi=150, bbox_inches='tight')
    plt.show()

    return results_df


def plot_feature_importance(model, feature_cols, save_dir='results'):
    """Visualise l'importance des features (poids couche 1)."""
    import os
    os.makedirs(save_dir, exist_ok=True)

    weights_l1 = model.layer1.weight.detach().cpu().numpy()
    importance = np.abs(weights_l1).mean(axis=0)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#2ca02c' if i > 0.1 else '#ff7f0e' if i > 0.05 else '#d62728' for i in importance]
    bars = ax.barh(feature_cols, importance, color=colors)
    ax.set_xlabel('Importance moyenne (|poids|)')
    ax.set_title('Importance des features (couche 1)', fontweight='bold')

    for bar, imp in zip(bars, importance):
        ax.text(imp + 0.005, bar.get_y() + bar.get_height()/2, f'{imp:.3f}', 
               va='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(f'{save_dir}/feature_importance.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("\nInterprétation:")
    print(f"  fuel_encoded: {importance[0]:.3f} -> Fuel dominant (R²_fuel=0.79)")
    for i, col in enumerate(feature_cols[1:], 1):
        print(f"  {col:15s}: {importance[i]:.3f}")
