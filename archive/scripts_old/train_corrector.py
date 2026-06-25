"""
Unified corrector trainer for BurnTrack.

Usage:
    python scripts/train_corrector.py --model mlp --dataset data/processed/train.csv --val data/processed/val.csv
    python scripts/train_corrector.py --model rf --dataset data/processed/real_train.csv --val data/processed/real_val.csv
    python scripts/train_corrector.py --model xgb --dataset data/processed/train.csv --val data/processed/val.csv
"""
import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)


def train_mlp(args):
    try:
        from burntrack.corrector.training import train_mlp as _train_mlp
    except ImportError:
        print("ERROR: burntrack.corrector.training.train_mlp not found. Is PyTorch installed?")
        sys.exit(1)

    return _train_mlp(
        train_path=args.dataset,
        val_path=args.val,
        hidden_dims=args.hidden_dims,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        output_dir=args.output_dir,
        seed=args.seed,
    )


def train_rf(args):
    try:
        from burntrack.corrector.training import train_rf as _train_rf
    except ImportError:
        print("ERROR: burntrack.corrector.training.train_rf not found. Is scikit-learn installed?")
        sys.exit(1)

    return _train_rf(
        train_path=args.dataset,
        val_path=args.val,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        output_dir=args.output_dir,
        seed=args.seed,
    )


def train_xgboost(args):
    try:
        from burntrack.corrector.training import train_xgboost as _train_xgboost
    except ImportError:
        print("ERROR: burntrack.corrector.training.train_xgboost not found. Is xgboost installed?")
        sys.exit(1)

    return _train_xgboost(
        train_path=args.dataset,
        val_path=args.val,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        output_dir=args.output_dir,
        seed=args.seed,
    )


def main():
    parser = argparse.ArgumentParser(description="Train a Burntrack corrector model")
    parser.add_argument("--model", type=str, required=True, choices=["mlp", "rf", "xgb"])
    parser.add_argument("--dataset", type=str, required=True, help="Path to training CSV")
    parser.add_argument("--val", type=str, required=True, help="Path to validation CSV")
    parser.add_argument("--output-dir", type=str, default="models/", help="Output directory for model artifacts")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[128, 64, 32])
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)

    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--max-depth", type=int, default=12)

    parser.add_argument("--learning-rate", type=float, default=0.05)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    train_fn = {"mlp": train_mlp, "rf": train_rf, "xgb": train_xgboost}[args.model]
    metrics = train_fn(args)

    print()
    print(f"{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    row = f"  train_r2  | {metrics.get('train_r2', 'N/A'):>10}"
    if isinstance(metrics.get("train_r2"), float):
        row = f"  train_r2  | {metrics['train_r2']:10.4f}"
    else:
        row = f"  train_r2  | {str(metrics.get('train_r2', 'N/A')):>10}"
    print(row)

    row = f"  val_r2    | {metrics.get('val_r2', 'N/A'):>10}"
    if isinstance(metrics.get("val_r2"), float):
        row = f"  val_r2    | {metrics['val_r2']:10.4f}"
    else:
        row = f"  val_r2    | {str(metrics.get('val_r2', 'N/A')):>10}"
    print(row)

    row = f"  val_mae   | {metrics.get('val_mae', 'N/A'):>10}"
    if isinstance(metrics.get("val_mae"), float):
        row = f"  val_mae   | {metrics['val_mae']:10.4f}"
    else:
        row = f"  val_mae   | {str(metrics.get('val_mae', 'N/A')):>10}"
    print(row)

    row = f"  val_rmse  | {metrics.get('val_rmse', 'N/A'):>10}"
    if isinstance(metrics.get("val_rmse"), float):
        row = f"  val_rmse  | {metrics['val_rmse']:10.4f}"
    else:
        row = f"  val_rmse  | {str(metrics.get('val_rmse', 'N/A')):>10}"
    print(row)
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
