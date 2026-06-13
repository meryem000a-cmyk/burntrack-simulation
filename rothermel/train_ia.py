"""
train_ia.py
===========
Script d'entraînement complet de l'IA correctrice BurnTrack (AtlasCorrectorV2).

Fonctionnalités :
- Chargement dataset avec StandardScaler obligatoire
- Architecture MLP [128→64→32] avec BatchNorm + Dropout
- Perte NLL hétéroscédastique (incertitude)
- MC Dropout pour l'inférence
- Feature importance (permutation)
- Early stopping + sauvegarde model.pt + scaler.pkl + config.json
- Traçabilité : version Rothermel utilisée

Usage :
    python train_ia.py --dataset synthetic_dataset.csv --epochs 200
"""

import os
import json
import time
import warnings
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# Import du modèle IA corrigé
from ia_corrector_v2 import AtlasCorrectorV2, build_ia_vector, DatasetLoader, compute_feature_importance


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class TrainingConfig:
    """Configuration de l'entraînement."""
    # Données
    dataset_path: str = 'synthetic_dataset.csv'
    test_size: float = 0.15
    val_size: float = 0.15

    # Modèle
    hidden_dims: List[int] = None
    dropout_rate: float = 0.2

    # Entraînement
    epochs: int = 200
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    patience: int = 20  # Early stopping

    # MC Dropout
    n_mcdropout_samples: int = 50

    # Sorties
    output_dir: str = 'models'
    model_name: str = 'atlas_corrector_v2'

    # Traçabilité
    rothermel_version: str = 'v2_corrected'

    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [128, 64, 32]


# =============================================================================
# DATASET PYTORCH
# =============================================================================

class BurnTrackDataset(Dataset):
    """Dataset PyTorch pour l'IA correctrice."""

    FEATURE_COLS = [
        'ros_rothermel', 'temp_c', 'rh_percent', 'wind_speed',
        'vpd_kpa', 'slope_deg', 'fuel_model_encoded', 'fuel_moisture'
    ]
    TARGET_COL = 'correction_factor'

    def __init__(self, df: pd.DataFrame, scaler: Optional[StandardScaler] = None, fit_scaler: bool = False):
        self.df = df.reset_index(drop=True)

        # Features
        X = self.df[self.FEATURE_COLS].values.astype(np.float32)

        # Normalisation
        if fit_scaler:
            self.scaler = StandardScaler()
            self.X = self.scaler.fit_transform(X)
        elif scaler is not None:
            self.scaler = scaler
            self.X = scaler.transform(X)
        else:
            raise ValueError(" scaler requis si fit_scaler=False")

        # Target
        self.y = self.df[self.TARGET_COL].values.astype(np.float32).reshape(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.float32)


# =============================================================================
# FONCTION DE PERTE
# =============================================================================

def heteroscedastic_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Negative Log-Likelihood avec hétéroscédasticité.

    prediction : [correction_factor, log_var]
    target : correction_factor_vrai
    """
    correction = prediction[:, 0:1]
    log_var = prediction[:, 1:2]

    precision = torch.exp(-log_var)
    loss = 0.5 * (log_var + precision * (target - correction) ** 2)

    return loss.mean()


# =============================================================================
# ENTRAÎNEMENT
# =============================================================================

class Trainer:
    """Gestionnaire d'entraînement."""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Device : {self.device}")

        os.makedirs(config.output_dir, exist_ok=True)

        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_mae': [],
            'val_r2': [],
            'best_epoch': 0,
            'best_val_loss': float('inf'),
        }

    def load_data(self) -> Tuple[DataLoader, DataLoader, DataLoader, StandardScaler]:
        """Charge et split le dataset."""
        print(f"\nChargement dataset : {self.config.dataset_path}")
        df = pd.read_csv(self.config.dataset_path)
        print(f"   Échantillons : {len(df)}")

        # Split
        n_total = len(df)
        n_test = int(n_total * self.config.test_size)
        n_val = int(n_total * self.config.val_size)
        n_train = n_total - n_test - n_val

        # Dataset complet pour fit scaler
        full_dataset = BurnTrackDataset(df, fit_scaler=True)
        scaler = full_dataset.scaler

        # Split indices
        indices = list(range(n_total))
        np.random.seed(42)
        np.random.shuffle(indices)

        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train+n_val]
        test_idx = indices[n_train+n_val:]

        # Subsets
        train_dataset = torch.utils.data.Subset(full_dataset, train_idx)
        val_dataset = torch.utils.data.Subset(full_dataset, val_idx)
        test_dataset = torch.utils.data.Subset(full_dataset, test_idx)

        # Dataloaders
        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=self.config.batch_size, shuffle=False)

        print(f"   Train : {n_train} | Val : {n_val} | Test : {n_test}")

        return train_loader, val_loader, test_loader, scaler

    def create_model(self) -> AtlasCorrectorV2:
        """Crée le modèle."""
        model = AtlasCorrectorV2(
            n_features=8,
            hidden_dims=self.config.hidden_dims,
            dropout_rate=self.config.dropout_rate,
        )
        return model.to(self.device)

    def train_epoch(self, model: AtlasCorrectorV2, loader: DataLoader, optimizer: torch.optim.Optimizer) -> float:
        """Entraîne une epoch."""
        model.train()
        total_loss = 0.0

        for X_batch, y_batch in loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            optimizer.zero_grad()
            pred = model(X_batch)
            loss = heteroscedastic_loss(pred, y_batch)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            total_loss += loss.item() * len(X_batch)

        return total_loss / len(loader.dataset)

    def validate(self, model: AtlasCorrectorV2, loader: DataLoader) -> Dict:
        """Validation."""
        model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                pred = model(X_batch)
                loss = heteroscedastic_loss(pred, y_batch)
                total_loss += loss.item() * len(X_batch)

                all_preds.extend(pred[:, 0].cpu().numpy())
                all_targets.extend(y_batch.cpu().numpy().flatten())

        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)

        return {
            'loss': total_loss / len(loader.dataset),
            'mae': mean_absolute_error(all_targets, all_preds),
            'r2': r2_score(all_targets, all_preds),
        }

    def train(self) -> AtlasCorrectorV2:
        """Boucle d'entraînement complète."""
        print("\n" + "=" * 60)
        print("ENTRAÎNEMENT ATLAS CORRECTOR V2")
        print("=" * 60)

        # Chargement données
        train_loader, val_loader, test_loader, scaler = self.load_data()
        self.scaler = scaler

        # Modèle
        model = self.create_model()
        n_params = sum(p.numel() for p in model.parameters())
        print(f"\nParamètres du modèle : {n_params:,}")
        print(f"Architecture : {self.config.hidden_dims}")
        print(f"Dropout : {self.config.dropout_rate}")

        # Optimiseur
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=10 
        )


        # Entraînement
        best_val_loss = float('inf')
        patience_counter = 0
        start_time = time.time()

        for epoch in range(1, self.config.epochs + 1):
            train_loss = self.train_epoch(model, train_loader, optimizer)
            val_metrics = self.validate(model, val_loader)

            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_mae'].append(val_metrics['mae'])
            self.history['val_r2'].append(val_metrics['r2'])

            scheduler.step(val_metrics['loss'])

            # Early stopping
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                self.history['best_epoch'] = epoch
                self.history['best_val_loss'] = best_val_loss
                patience_counter = 0

                # Sauvegarde meilleur modèle
                self.best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1

            # Log
            if epoch % 10 == 0 or epoch == 1:
                print(f"Epoch {epoch:3d}/{self.config.epochs} | "
                      f"Train Loss: {train_loss:.5f} | "
                      f"Val Loss: {val_metrics['loss']:.5f} | "
                      f"Val MAE: {val_metrics['mae']:.4f} | "
                      f"Val R²: {val_metrics['r2']:.4f}")

            if patience_counter >= self.config.patience:
                print(f"\n⏹️ Early stopping à l'epoch {epoch}")
                break

        elapsed = time.time() - start_time
        print(f"\n⏱️ Temps d'entraînement : {elapsed:.1f}s ({elapsed/60:.1f} min)")

        # Restaure meilleur modèle
        model.load_state_dict(self.best_model_state)

        # Évaluation finale
        print("\n" + "=" * 60)
        print("ÉVALUATION FINALE")
        print("=" * 60)

        test_metrics = self.validate(model, test_loader)
        print(f"Test Loss : {test_metrics['loss']:.5f}")
        print(f"Test MAE  : {test_metrics['mae']:.4f}")
        print(f"Test R²   : {test_metrics['r2']:.4f}")

        self.test_metrics = test_metrics

        return model

    def save(self, model: AtlasCorrectorV2):
        """Sauvegarde le modèle, scaler et config."""
        print("\n" + "=" * 60)
        print("SAUVEGARDE")
        print("=" * 60)

        base_path = os.path.join(self.config.output_dir, self.config.model_name)

        # Modèle
        model_path = f"{base_path}.pt"
        torch.save(model.state_dict(), model_path)
        print(f"✅ Modèle : {model_path}")

        # Scaler
        scaler_path = f"{base_path}_scaler.pkl"
        joblib.dump(self.scaler, scaler_path)
        print(f"✅ Scaler : {scaler_path}")

        # Config
        config_dict = asdict(self.config)
        config_dict['history'] = self.history
        config_dict['test_metrics'] = self.test_metrics
        config_dict['n_parameters'] = sum(p.numel() for p in model.parameters())

        config_path = f"{base_path}_config.json"
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        print(f"✅ Config : {config_path}")

        # Graphes
        self._plot_training(base_path)
        self._plot_predictions(model, base_path)

    def _plot_training(self, base_path: str):
        """Graphe de l'historique d'entraînement."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        epochs = range(1, len(self.history['train_loss']) + 1)

        axes[0].plot(epochs, self.history['train_loss'], label='Train')
        axes[0].plot(epochs, self.history['val_loss'], label='Val')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss (NLL)')
        axes[0].set_title('Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(epochs, self.history['val_mae'])
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('MAE')
        axes[1].set_title('Validation MAE')
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(epochs, self.history['val_r2'])
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('R²')
        axes[2].set_title('Validation R²')
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = f"{base_path}_training.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"✅ Graphe entraînement : {plot_path}")

    def _plot_predictions(self, model: AtlasCorrectorV2, base_path: str):
        """Graphe des prédictions vs réel."""
        df = pd.read_csv(self.config.dataset_path)
        dataset = BurnTrackDataset(df, scaler=self.scaler, fit_scaler=False)
        loader = DataLoader(dataset, batch_size=256, shuffle=False)

        model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(self.device)
                pred = model(X_batch)
                all_preds.extend(pred[:, 0].cpu().numpy())
                all_targets.extend(y_batch.numpy().flatten())

        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(all_targets, all_preds, alpha=0.3, s=10)
        ax.plot([all_targets.min(), all_targets.max()], 
                [all_targets.min(), all_targets.max()], 
                'r--', lw=2, label='y=x')
        ax.set_xlabel('Correction Factor (Réel)')
        ax.set_ylabel('Correction Factor (Prédit)')
        ax.set_title(f'Prédictions vs Réel (R² = {r2_score(all_targets, all_preds):.3f})')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plot_path = f"{base_path}_predictions.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"✅ Graphe prédictions : {plot_path}")


# =============================================================================
# POINT D'ENTRÉE
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Entraînement IA correctrice BurnTrack")
    parser.add_argument("--dataset", type=str, default="synthetic_dataset.csv", help="Chemin dataset")
    parser.add_argument("--epochs", type=int, default=200, help="Nombre d'epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Taille batch")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--patience", type=int, default=20, help="Patience early stopping")
    parser.add_argument("--output-dir", type=str, default="models", help="Dossier sortie")
    parser.add_argument("--model-name", type=str, default="atlas_corrector_v2", help="Nom modèle")

    args = parser.parse_args()

    config = TrainingConfig(
        dataset_path=args.dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        patience=args.patience,
        output_dir=args.output_dir,
        model_name=args.model_name,
    )

    trainer = Trainer(config)
    model = trainer.train()
    trainer.save(model)

    print("\n" + "=" * 60)
    print("ENTRAÎNEMENT TERMINÉ")
    print("=" * 60)