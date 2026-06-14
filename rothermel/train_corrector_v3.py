"""
train_corrector_v3.py
=====================
Script d'entraînement du corrector v3.

Usage :
    python train_corrector_v3.py --dataset train.csv --epochs 200 --batch_size 64

Le script :
1. Charge le dataset (CSV)
2. Normalise les features continues avec StandardScaler
3. Entraîne le modèle AtlasCorrectorV3
4. Sauvegarde le modèle et le scaler
5. Évalue sur le jeu de validation
"""

import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import sys
import json
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ia_corrector_v3 import (
    AtlasCorrectorV3, CorrectorDatasetLoader, corrector_loss,
    REQUIRED_FEATURES, FUEL_MODEL_ENCODING, N_FUEL_MODELS,
    encode_fuel_model
)


# =============================================================================
# DATASET PYTORCH
# =============================================================================

class CorrectorDataset(Dataset):
    """Dataset PyTorch pour le corrector v3."""

    def __init__(self, df: pd.DataFrame, scaler=None, fit_scaler=False):
        """
        Args:
            df : DataFrame avec toutes les colonnes requises
            scaler : CorrectorDatasetLoader (optionnel)
            fit_scaler : si True, fit le scaler sur ces données
        """
        self.df = df.reset_index(drop=True)

        # Extraire features continues
        continuous_features = [f for f in REQUIRED_FEATURES if f != 'fuel_model_code']

        # Vérifier que toutes les colonnes existent
        missing = [f for f in continuous_features if f not in df.columns]
        if missing:
            raise ValueError(f"Colonnes manquantes dans le dataset : {missing}")

        self.X_continuous = df[continuous_features].values.astype(np.float32)

        # Fuel model indices
        self.fuel_indices = df['fuel_model_code'].apply(encode_fuel_model).values.astype(np.int64)

        # Target : delta_ros
        if 'delta_ros' not in df.columns:
            raise ValueError("Colonne 'delta_ros' manquante (target d'entraînement)")
        self.y = df['delta_ros'].values.astype(np.float32)

        # Normalisation
        if scaler is not None:
            if fit_scaler:
                self.X_continuous = scaler.fit_transform(self.X_continuous)
            else:
                self.X_continuous = scaler.transform(self.X_continuous)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.X_continuous[idx]),
            torch.tensor(self.fuel_indices[idx]),
            torch.tensor(self.y[idx])
        )


# =============================================================================
# ENTRAÎNEMENT
# =============================================================================

def train_epoch(model, dataloader, optimizer, device):
    """Entraîne le modèle sur une epoch."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for x_cont, fuel_idx, y in dataloader:
        x_cont = x_cont.to(device)
        fuel_idx = fuel_idx.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        out = model(x_cont, fuel_idx)
        loss = corrector_loss(out, y, l2_lambda=1e-4, model=model)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


def validate(model, dataloader, device):
    """Évalue le modèle sur le jeu de validation."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    n_batches = 0

    with torch.no_grad():
        for x_cont, fuel_idx, y in dataloader:
            x_cont = x_cont.to(device)
            fuel_idx = fuel_idx.to(device)
            y = y.to(device)

            out = model(x_cont, fuel_idx)
            loss = corrector_loss(out, y, l2_lambda=0.0, model=None)  # Pas de L2 en val

            total_loss += loss.item()
            all_preds.extend(out[:, 0].cpu().numpy())
            all_targets.extend(y.cpu().numpy())
            n_batches += 1

    avg_loss = total_loss / n_batches
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # Métriques
    mae = np.mean(np.abs(all_preds - all_targets))
    rmse = np.sqrt(np.mean((all_preds - all_targets) ** 2))
    r2 = 1 - np.sum((all_targets - all_preds) ** 2) / np.sum((all_targets - np.mean(all_targets)) ** 2)

    return avg_loss, mae, rmse, r2, all_preds, all_targets


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Entraîne le corrector v3')
    parser.add_argument('--dataset', type=str, default='train.csv', help='Fichier CSV d"entraînement')
    parser.add_argument('--val', type=str, default='val.csv', help='Fichier CSV de validation')
    parser.add_argument('--epochs', type=int, default=200, help='Nombre d"epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Taille du batch')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--patience', type=int, default=20, help='Early stopping patience')
    parser.add_argument('--output_dir', type=str, default='models', help='Répertoire de sortie')
    args = parser.parse_args()

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device : {device}")

    # Créer le répertoire de sortie
    os.makedirs(args.output_dir, exist_ok=True)

    # Charger les données
    print(f"\nChargement des données...")
    train_df = pd.read_csv(args.dataset)
    val_df = pd.read_csv(args.val)

    print(f"Train : {len(train_df):,} échantillons")
    print(f"Val   : {len(val_df):,} échantillons")

    # Scaler
    scaler = CorrectorDatasetLoader()

    # Datasets
    train_dataset = CorrectorDataset(train_df, scaler=scaler, fit_scaler=True)
    val_dataset = CorrectorDataset(val_df, scaler=scaler, fit_scaler=False)

    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Modèle
    n_continuous = len(REQUIRED_FEATURES) - 1
    model = AtlasCorrectorV3(
        n_features=n_continuous,
        n_fuel_models=N_FUEL_MODELS,
        embedding_dim=16,
        hidden_dims=[256, 128, 64],
        dropout_rate=0.3,
        correction_scale=5.0,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModèle : {n_params:,} paramètres")
    print(f"Features continues : {n_continuous}")
    print(f"Architecture : [256, 128, 64]")

    # Optimiseur et scheduler
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    # Entraînement
    print(f"\n{'='*60}")
    print("DÉBUT DE L'ENTRAÎNEMENT")
    print(f"{'='*60}")

    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'val_mae': [], 'val_rmse': [], 'val_r2': []}

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss, val_mae, val_rmse, val_r2, _, _ = validate(model, val_loader, device)

        scheduler.step(val_loss)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_mae'].append(val_mae)
        history['val_rmse'].append(val_rmse)
        history['val_r2'].append(val_r2)

        # Affichage
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{args.epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"MAE: {val_mae:.3f} m/min | "
                  f"RMSE: {val_rmse:.3f} m/min | "
                  f"R²: {val_r2:.3f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0

            # Sauvegarder le meilleur modèle
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_mae': val_mae,
                'val_rmse': val_rmse,
                'val_r2': val_r2,
            }, f'{args.output_dir}/corrector_v3_best.pt')
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping à l'epoch {epoch} (best: {best_epoch})")
                break

    # Sauvegarder le scaler
    scaler.save(f'{args.output_dir}/scaler.joblib')

    # Sauvegarder l'historique (convertir numpy types en Python natifs)
    history_serializable = {
        k: [float(v) for v in vals] if isinstance(vals, list) else vals
        for k, vals in history.items()
    }
    with open(f'{args.output_dir}/history.json', 'w') as f:
        json.dump(history_serializable, f, indent=2)

    # Évaluation finale
    print(f"\n{'='*60}")
    print("RÉSULTATS FINAUX")
    print(f"{'='*60}")
    print(f"Meilleure epoch : {best_epoch}")
    print(f"Meilleure val loss : {best_val_loss:.4f}")
    print(f"Meilleur MAE : {min(history['val_mae']):.3f} m/min")
    print(f"Meilleur RMSE : {min(history['val_rmse']):.3f} m/min")
    print(f"Meilleur R² : {max(history['val_r2']):.3f}")
    print()
    print(f"Modèle sauvegardé : {args.output_dir}/corrector_v3_best.pt")
    print(f"Scaler sauvegardé : {args.output_dir}/scaler.joblib")
    print(f"Historique sauvegardé : {args.output_dir}/history.json")

    # Test rapide
    print(f"\n{'='*60}")
    print("TEST RAPIDE")
    print(f"{'='*60}")

    model.eval()
    with torch.no_grad():
        x_cont = torch.tensor(val_dataset.X_continuous[:5], dtype=torch.float32).to(device)
        fuel_idx = torch.tensor(val_dataset.fuel_indices[:5], dtype=torch.int64).to(device)
        out = model(x_cont, fuel_idx)

        for i in range(5):
            pred = out[i, 0].item()
            target = val_dataset.y[i]
            ros_roth = val_df.iloc[i]['ros_rothermel']
            ros_corr = ros_roth + pred
            print(f"  Sample {i+1}: ROS_Roth={ros_roth:.2f} | "
                  f"delta_pred={pred:+.3f} | ROS_corr={ros_corr:.2f} | "
                  f"target_delta={target:+.3f}")


if __name__ == "__main__":
    main()