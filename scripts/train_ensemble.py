"""
train_ensemble.py
=================
Pipeline script to build the dataset, validate it, and train the full ensemble corrector.

Usage:
    python scripts/train_ensemble.py --output-dir models/ensemble --use-mlp
"""

import os
import sys
import argparse
import logging
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from burntrack.data.build_ground_truth import build_ground_truth_dataset
from burntrack.corrector.validation import split_by_fire_event
from burntrack.corrector.training import train_ensemble

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Train the Burntrack ensemble corrector")
    parser.add_argument("--output-dir", type=str, default="models/ensemble", help="Output directory for model artifacts")
    parser.add_argument("--data-dir", type=str, default="data/processed", help="Directory for processed datasets")
    parser.add_argument("--use-mlp", action="store_true", help="Also train the MLP ensemble model")
    
    args = parser.parse_args()
    
    # 1. Build Ground Truth
    logger.info("Step 1: Building Ground Truth Dataset")
    african_df, global_df = build_ground_truth_dataset(output_dir=args.data_dir)
    
    if african_df.empty:
        logger.error("Failed to build dataset. Exiting.")
        sys.exit(1)
        
    # 2. Split Dataset by Fire Event
    logger.info("Step 2: Splitting dataset by fire event to prevent leakage")
    # Using the primary African dataset for training
    train_df, val_df, test_df = split_by_fire_event(
        african_df, 
        fire_id_col='fire_id', 
        val_size=0.15, 
        test_size=0.15, 
        random_state=42
    )
    
    logger.info(f"Train size: {len(train_df)}, Val size: {len(val_df)}, Test size: {len(test_df)}")
    
    # 3. Train Ensemble
    logger.info("Step 3: Training Ensemble Corrector")
    report = train_ensemble(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        output_dir=args.output_dir,
        use_mlp=args.use_mlp,
        global_df=global_df,
    )
    
    print("\nTraining Complete! Report:\n")
    print(report)

if __name__ == "__main__":
    main()
