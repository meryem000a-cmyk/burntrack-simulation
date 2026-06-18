"""
Boucle d'entraînement BurnTrack
AdamW + ReduceLROnPlateau + Early stopping + Gradient clipping
"""

import numpy as np
import torch
from torch.utils.data import DataLoader
from typing import Tuple, Dict


def train_model(model, train_loader, val_loader, loss_fn, device,
                epochs: int = 200, patience: int = 20, 
                lr: float = 1e-3, weight_decay: float = 1e-3) -> Tuple:
    """
    Entraîne le modèle avec early stopping et suivi des métriques.

    Args:
        model: Instance de BurnTrackMLPMinimal
        train_loader: DataLoader d'entraînement
        val_loader: DataLoader de validation
        loss_fn: Fonction de loss (WeightedMSELoss)
        device: torch.device
        epochs: Nombre max d'epochs
        patience: Patience pour early stopping
        lr: Learning rate initial
        weight_decay: Weight decay pour AdamW

    Returns:
        (model, history) où history est un dict de listes
    """

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True
    )

    history = {
        'train_loss': [], 'val_loss': [], 
        'val_mae': [], 'val_mae_real': [], 'lr': []
    }

    best_val_mae = float('inf')
    patience_counter = 0
    best_state = None

    print(f"\n=== ENTRAÎNEMENT ===")
    print(f"Device: {device} | Epochs max: {epochs} | Patience: {patience}")
    print(f"LR: {lr} | Weight decay: {weight_decay}")

    for epoch in range(epochs):
        # ---- TRAIN ----
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

        # ---- VALIDATION ----
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
            print(f"Epoch {epoch+1:3d}/{epochs} | "
                  f"Train: {avg_train:.4f} | "
                  f"Val MAE: {avg_mae:.4f} | "
                  f"Val MAE(r): {avg_mae_real:.4f} | "
                  f"LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Early stopping sur MAE réel
        if avg_mae_real < best_val_mae:
            best_val_mae = avg_mae_real
            patience_counter = 0
            best_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n✋ Early stopping à epoch {epoch+1} "
                      f"(best MAE réel: {best_val_mae:.4f})")
                break

        scheduler.step(avg_mae_real)

    if best_state is not None:
        model.load_state_dict(best_state)

    print(f"\n✅ Entraînement terminé. Best MAE réel: {best_val_mae:.4f}")

    return model, history
