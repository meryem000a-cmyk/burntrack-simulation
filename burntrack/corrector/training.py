"""
training.py
===========
Module for training the StackedCorrectEnsemble on the ground-truth dataset.
Supports two-stage training: pre-train on global data, fine-tune on African data.
"""

import os
import logging
import pandas as pd
import numpy as np

from burntrack.corrector.ensemble import StackedCorrectEnsemble
from burntrack.corrector.mlp import MLPEnsembleCorrector
from burntrack.corrector.features import CorrectorFeatureExtractor, _encode_fuel_model
from burntrack.corrector.validation import evaluate_corrector, save_metrics, format_metrics_report

logger = logging.getLogger(__name__)

def train_ensemble(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: str = 'models/ensemble',
    use_mlp: bool = False,
    global_df: pd.DataFrame = None,
):
    """
    Train the corrector ensemble with optional global pre-training.

    Args:
        train_df: African training DataFrame
        val_df: African validation DataFrame
        test_df: African test DataFrame
        output_dir: Directory to save the models and metrics
        use_mlp: Whether to also train the MLPEnsembleCorrector
        global_df: Optional global transfer dataset for pre-training

    Returns:
        metrics_report (str)
    """
    os.makedirs(output_dir, exist_ok=True)
    extractor = CorrectorFeatureExtractor()

    logger.info("Extracting features...")
    X_train, fuel_train = extractor.extract_dataframe(train_df, include_fuel_idx=True)
    y_train = train_df['delta_ros'].values
    X_val, fuel_val = extractor.extract_dataframe(val_df, include_fuel_idx=True)
    y_val = val_df['delta_ros'].values
    X_test, fuel_test = extractor.extract_dataframe(test_df, include_fuel_idx=True)
    y_test = test_df['delta_ros'].values

    # Stage 1: Pre-train on global data if available
    ensemble = StackedCorrectEnsemble()

    if global_df is not None and len(global_df) > 0:
        logger.info("Stage 1: Pre-training base learners on global transfer data...")
        X_global, fuel_global = extractor.extract_dataframe(global_df, include_fuel_idx=True)
        y_global = global_df['delta_ros'].values
        ensemble.fit(X_global, y_global, X_val, y_val)

    # Stage 2: Fine-tune on African data
    logger.info("Stage 2: Fine-tuning on African ground-truth data...")
    ensemble.fit(X_train, y_train, X_val, y_val)

    logger.info("Saving ensemble...")
    ensemble.save(os.path.join(output_dir, "stacking_ensemble"))

    # Train MLP
    if use_mlp:
        mlp_ensemble = MLPEnsembleCorrector(n_continuous_features=X_train.shape[1])
        if global_df is not None and len(global_df) > 0:
            logger.info("Pre-training MLP on global data...")
            mlp_ensemble.fit(X_global, y_global, fuel_global,
                             X_val, y_val, fuel_val,
                             epochs=100, batch_size=64, lr=1e-3)
        logger.info("Fine-tuning MLP on African data...")
        mlp_ensemble.fit(X_train, y_train, fuel_train,
                         X_val, y_val, fuel_val,
                         epochs=200, batch_size=64, lr=1e-3)
        mlp_ensemble.save(os.path.join(output_dir, "mlp_ensemble"))

    # Evaluate
    logger.info("Evaluating on TEST set...")
    preds, uncs, cis_lower, cis_upper = [], [], [], []
    for _, row in test_df.iterrows():
        row_dict = row.to_dict()
        res = ensemble.predict_with_uncertainty(row_dict)
        preds.append(res['delta_ros'])
        uncs.append(res['uncertainty'])
        cis_lower.append(res['ci_lower'])
        cis_upper.append(res['ci_upper'])

    test_df_eval = test_df.copy()
    test_df_eval['delta_ros_pred'] = preds
    test_df_eval['uncertainty'] = uncs
    test_df_eval['ci_lower'] = cis_lower
    test_df_eval['ci_upper'] = cis_upper

    metrics = evaluate_corrector(
        y_true=test_df_eval['delta_ros'].values,
        y_pred=test_df_eval['delta_ros_pred'].values,
        ros_rothermel=test_df_eval['ros_rothermel'].values
        if 'ros_rothermel' in test_df_eval else None,
        ci_lower=test_df_eval['ci_lower'].values,
        ci_upper=test_df_eval['ci_upper'].values,
        uncertainty=test_df_eval['uncertainty'].values,
        fire_ids=test_df_eval['fire_id'].values
        if 'fire_id' in test_df_eval else None,
        fuel_models=test_df_eval['fuel_model_code'].values
        if 'fuel_model_code' in test_df_eval else None,
    )
    save_metrics(metrics, os.path.join(output_dir, "test_metrics.json"))

    report = format_metrics_report(metrics)
    with open(os.path.join(output_dir, "test_report.md"), "w") as f:
        f.write(report)
    logger.info("\n" + report)
    return report
