#!/usr/bin/env python3
"""
BurnTrack - Script principal
Projet étudiant ingénieurs: Correction des biais de Rothermel par MLP minimal

Usage:
    python main.py --train    # Entraînement
    python main.py --eval     # Évaluation
    python main.py --predict data/nouvelles_donnees.csv  # Inférence
"""

import argparse
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.model import BurnTrackMLPMinimal
from src.loss import WeightedMSELoss
from src.dataset import BurnTrackDataset
from src.train import train_model
from src.evaluate import evaluate_model, plot_feature_importance


def main_train():
    """Pipeline complet d'entraînement."""
    print("=" * 60)
    print("🔥 BURNTRACK - ENTRAÎNEMENT")
    print("=" * 60)

    # 1. Chargement et analyse
    dataset = BurnTrackDataset(
        real_path="data/african_ground_truth.csv",
        synth_path="data/synthetic_dataset_balanced_v2.csv"
    )

    bias_df = dataset.analyze_bias()
    dataset.compute_target_encoding(bias_df)

    # 2. Préparation
    X_train, X_val, X_test, y_train, y_val, y_test, is_r_train, is_r_val, is_r_test = dataset.prepare()

    # 3. DataLoaders
    batch_size = 64
    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32),
            torch.tensor(is_r_train, dtype=torch.float32)
        ),
        batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.float32),
            torch.tensor(is_r_val, dtype=torch.float32)
        ),
        batch_size=batch_size, shuffle=False
    )

    # 4. Modèle
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BurnTrackMLPMinimal(n_features=X_train.shape[1]).to(device)

    print(f"\nModèle: {X_train.shape[1]} -> 64 -> 32 -> 1")
    print(f"Paramètres: {model.count_parameters():,}")
    print(f"Device: {device}")

    # 5. Loss
    loss_fn = WeightedMSELoss(weight_real=3.0, weight_synth=1.0)

    # 6. Entraînement
    model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        device=device,
        epochs=200,
        patience=20,
        lr=1e-3,
        weight_decay=1e-3
    )

    # 7. Évaluation
    print("\n" + "=" * 60)
    print("📊 ÉVALUATION")
    print("=" * 60)

    mae_baseline = np.mean(np.abs(y_test[is_r_test.astype(bool)]))

    # Récupération fuel_encoded pour le test set
    test_df = dataset.df_merged.iloc[-len(y_test):].copy()
    fuel_encoded_test = test_df['fuel_encoded'].values

    results_df = evaluate_model(
        model=model,
        X_test=X_test,
        y_test=y_test,
        is_real_test=is_r_test,
        df_test=test_df,
        fuel_encoded_test=fuel_encoded_test,
        mae_baseline=mae_baseline,
        device=device,
        feature_cols=dataset.feature_cols
    )

    # Feature importance
    plot_feature_importance(model, dataset.feature_cols)

    # 8. Sauvegarde
    os.makedirs("checkpoints", exist_ok=True)
    checkpoint_path = "checkpoints/burntrack_mlp_minimal.pt"

    torch.save({
        'model_state_dict': model.state_dict(),
        'n_features': X_train.shape[1],
        'feature_cols': dataset.feature_cols,
        'fuel_encoding': dataset.fuel_encoding,
        'history': history,
        'best_val_mae': min(history['val_mae_real']),
        'test_metrics': results_df.to_dict(),
    }, checkpoint_path)

    print(f"\n💾 Modèle sauvegardé: {checkpoint_path}")
    print(f"   Taille: ~{os.path.getsize(checkpoint_path)/1024:.1f} KB")

    print("\n" + "=" * 60)
    print("✅ ENTRAÎNEMENT TERMINÉ")
    print("=" * 60)


def main_predict(input_path: str):
    """Inférence sur de nouvelles données."""
    from src.predict import BurnTrackPredictor

    predictor = BurnTrackPredictor(
        model_path="checkpoints/burntrack_mlp_minimal.pt",
        scaler_path="scaler.pkl",
        fuel_encoding_path="fuel_encoding.json"
    )

    df = pd.read_csv(input_path)
    ros_corrected = predictor.predict(df)

    df['ros_corrected'] = ros_corrected
    df['delta_pred'] = ros_corrected - df['ros_rothermel']

    output_path = input_path.replace('.csv', '_predicted.csv')
    df.to_csv(output_path, index=False)
    print(f"✅ Prédictions sauvegardées: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='BurnTrack - Correction biais Rothermel')
    parser.add_argument('--train', action='store_true', help='Entraînement')
    parser.add_argument('--predict', type=str, help='Chemin vers CSV à prédire')

    args = parser.parse_args()

    if args.train:
        main_train()
    elif args.predict:
        main_predict(args.predict)
    else:
        parser.print_help()
