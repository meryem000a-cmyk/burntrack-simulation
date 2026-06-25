"""
Validation strategy for BurnTrack fire behavior correction models.

Key principle: NEVER use random row-level splits. Always split by fire event
to prevent information leakage between observations from the same fire.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable, Any
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
import logging
import json
import os

logger = logging.getLogger(__name__)


def split_by_fire_event(
    df: pd.DataFrame,
    fire_id_col: str = 'fire_id',
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split dataset by unique fire events, NOT random rows.
    All observations from a single fire event go exclusively to one split.
    
    This is the ONLY valid way to evaluate fire behavior model generalization.
    Random row splits leak fire-event-level information, causing overfitting.
    
    Args:
        df: Input DataFrame with fire observations
        fire_id_col: Column containing unique fire event identifiers
        test_size: Fraction of fires for test set
        val_size: Fraction of fires for validation set
        random_state: Random seed for reproducibility
        
    Returns:
        train_df, val_df, test_df: DataFrames split by fire event
    """
    unique_fires = df[fire_id_col].unique()
    rng = np.random.RandomState(random_state)
    rng.shuffle(unique_fires)
    
    n_test = max(1, int(len(unique_fires) * test_size))
    n_val = max(1, int(len(unique_fires) * val_size))
    
    test_fires = set(unique_fires[:n_test])
    val_fires = set(unique_fires[n_test:n_test + n_val])
    train_fires = set(unique_fires[n_test + n_val:])
    
    train_df = df[df[fire_id_col].isin(train_fires)].copy()
    val_df = df[df[fire_id_col].isin(val_fires)].copy()
    test_df = df[df[fire_id_col].isin(test_fires)].copy()
    
    logger.info(
        f"Fire-event split: {len(train_fires)} train fires ({len(train_df)} rows), "
        f"{len(val_fires)} val fires ({len(val_df)} rows), "
        f"{len(test_fires)} test fires ({len(test_df)} rows)"
    )
    
    return train_df, val_df, test_df


def stratified_split_by_fuel(
    df: pd.DataFrame,
    fire_id_col: str = 'fire_id',
    fuel_col: str = 'fuel_model_code',
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split by fire events while preserving fuel model distribution across folds.
    
    Each fire event gets a dominant fuel type. Splitting is stratified by fuel type
    so that all fuel models appear in all splits (when possible).
    """
    fire_fuel = df.groupby(fire_id_col)[fuel_col].first()
    fires = fire_fuel.index.values
    fuels = fire_fuel.values
    
    # First split: train+val vs test
    try:
        trainval_fires, test_fires = train_test_split(
            fires, test_size=test_size, stratify=fuels, random_state=random_state
        )
    except ValueError:
        # Not enough samples for stratification, fall back to random
        logger.warning("Not enough samples for stratified split, using random")
        trainval_fires, test_fires = train_test_split(
            fires, test_size=test_size, random_state=random_state
        )
    
    # Second split: train vs val
    trainval_fuels = fire_fuel[trainval_fires].values
    adjusted_val = val_size / (1.0 - test_size)
    try:
        train_fires, val_fires = train_test_split(
            trainval_fires, test_size=adjusted_val, stratify=trainval_fuels,
            random_state=random_state
        )
    except ValueError:
        train_fires, val_fires = train_test_split(
            trainval_fires, test_size=adjusted_val, random_state=random_state
        )
    
    train_df = df[df[fire_id_col].isin(set(train_fires))].copy()
    val_df = df[df[fire_id_col].isin(set(val_fires))].copy()
    test_df = df[df[fire_id_col].isin(set(test_fires))].copy()
    
    # Log fuel distribution across splits
    for name, split_df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
        dist = split_df[fuel_col].value_counts().to_dict()
        logger.info(f"{name} split fuel distribution: {dist}")
    
    return train_df, val_df, test_df


def evaluate_corrector(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    ros_rothermel: Optional[np.ndarray] = None,
    ci_lower: Optional[np.ndarray] = None,
    ci_upper: Optional[np.ndarray] = None,
    uncertainty: Optional[np.ndarray] = None,
    fire_ids: Optional[np.ndarray] = None,
    fuel_models: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Comprehensive evaluation with aggregate, per-fire, per-fuel, and calibration metrics.
    
    Args:
        y_true: Ground-truth delta_ros values
        y_pred: Predicted delta_ros values
        ros_rothermel: Rothermel baseline ROS (for computing corrected ROS metrics)
        ci_lower: Lower bound of 95% confidence interval
        ci_upper: Upper bound of 95% confidence interval
        uncertainty: Predicted standard deviation / uncertainty
        fire_ids: Fire event IDs for per-fire evaluation
        fuel_models: Fuel model codes for per-fuel evaluation
    
    Returns:
        Dict with sections: 'aggregate', 'per_fire', 'by_fuel', 'calibration', 'residuals'
    """
    results: Dict[str, Any] = {
        'aggregate': {},
        'per_fire': [],
        'by_fuel': [],
        'calibration': {},
        'residuals': {},
    }
    
    # --- Aggregate metrics ---
    results['aggregate']['r2'] = float(r2_score(y_true, y_pred))
    results['aggregate']['mae'] = float(mean_absolute_error(y_true, y_pred))
    results['aggregate']['rmse'] = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    results['aggregate']['bias'] = float(np.mean(y_pred - y_true))
    results['aggregate']['mape'] = float(np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 0.01))) * 100)
    results['aggregate']['n_samples'] = len(y_true)
    
    # Corrected ROS metrics (if baseline available)
    if ros_rothermel is not None:
        ros_corrected = ros_rothermel + y_pred
        ros_true = ros_rothermel + y_true
        results['aggregate']['ros_corrected_r2'] = float(r2_score(ros_true, ros_corrected))
        results['aggregate']['ros_corrected_mae'] = float(mean_absolute_error(ros_true, ros_corrected))
    
    # --- Per-fire metrics ---
    if fire_ids is not None:
        unique_fires = np.unique(fire_ids)
        for fid in unique_fires:
            mask = fire_ids == fid
            if np.sum(mask) < 2:
                continue
            yt = y_true[mask]
            yp = y_pred[mask]
            fire_metrics: Dict[str, Any] = {
                'fire_id': str(fid),
                'n_observations': int(np.sum(mask)),
                'r2': float(r2_score(yt, yp)) if np.var(yt) > 0 else float('nan'),
                'mae': float(mean_absolute_error(yt, yp)),
                'rmse': float(np.sqrt(mean_squared_error(yt, yp))),
                'bias': float(np.mean(yp - yt)),
            }
            if fuel_models is not None:
                fire_metrics['fuel_model'] = str(fuel_models[mask][0])
            results['per_fire'].append(fire_metrics)
    
    # --- By fuel model ---
    if fuel_models is not None:
        unique_fuels = np.unique(fuel_models)
        for fm in unique_fuels:
            mask = fuel_models == fm
            if np.sum(mask) < 2:
                continue
            yt = y_true[mask]
            yp = y_pred[mask]
            results['by_fuel'].append({
                'fuel_model': str(fm),
                'n_observations': int(np.sum(mask)),
                'r2': float(r2_score(yt, yp)) if np.var(yt) > 0 else float('nan'),
                'mae': float(mean_absolute_error(yt, yp)),
                'rmse': float(np.sqrt(mean_squared_error(yt, yp))),
                'bias': float(np.mean(yp - yt)),
            })
    
    # --- Uncertainty calibration ---
    if ci_lower is not None and ci_upper is not None:
        coverage_95 = float(np.mean((ci_lower <= y_true) & (y_true <= ci_upper)))
        results['calibration']['95ci_coverage'] = coverage_95
        results['calibration']['ece'] = float(np.abs(coverage_95 - 0.95))
        results['calibration']['sharpness'] = float(np.mean(ci_upper - ci_lower))
        results['calibration']['mean_ci_width'] = float(np.mean(ci_upper - ci_lower))
    
    if uncertainty is not None:
        # Calibration across quantile levels
        residuals = np.abs(y_true - y_pred)
        # Check multiple coverage levels
        for level in [0.50, 0.80, 0.90, 0.95, 0.99]:
            z = {0.50: 0.674, 0.80: 1.282, 0.90: 1.645, 0.95: 1.960, 0.99: 2.576}[level]
            actual_coverage = float(np.mean(residuals <= z * uncertainty))
            results['calibration'][f'{int(level*100)}ci_coverage'] = actual_coverage
    
    # --- Residual analysis ---
    residuals_arr = y_true - y_pred
    results['residuals']['mean'] = float(np.mean(residuals_arr))
    results['residuals']['std'] = float(np.std(residuals_arr))
    results['residuals']['skewness'] = float(_skewness(residuals_arr))
    results['residuals']['kurtosis'] = float(_kurtosis(residuals_arr))
    results['residuals']['q25'] = float(np.percentile(residuals_arr, 25))
    results['residuals']['q50'] = float(np.percentile(residuals_arr, 50))
    results['residuals']['q75'] = float(np.percentile(residuals_arr, 75))
    
    return results


def _skewness(x: np.ndarray) -> float:
    """Compute Fisher skewness."""
    n = len(x)
    if n < 3:
        return 0.0
    m = np.mean(x)
    s = np.std(x, ddof=1)
    if s == 0:
        return 0.0
    return float((n / ((n-1) * (n-2))) * np.sum(((x - m) / s)**3))


def _kurtosis(x: np.ndarray) -> float:
    """Compute excess kurtosis."""
    n = len(x)
    if n < 4:
        return 0.0
    m = np.mean(x)
    s = np.std(x, ddof=1)
    if s == 0:
        return 0.0
    k4 = np.mean(((x - m) / s)**4)
    return float(k4 - 3.0)


def calibrate_uncertainty(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    uncertainty: np.ndarray,
) -> 'IsotonicRegression':
    """
    Calibrate uncertainty estimates using isotonic regression.
    
    Maps raw predicted uncertainties to calibrated uncertainties such that
    the 95% CI contains the true value ~95% of the time.
    
    Returns:
        Fitted IsotonicRegression model: raw_uncertainty -> calibrated_uncertainty
    """
    from sklearn.isotonic import IsotonicRegression
    
    residuals = np.abs(y_true - y_pred)
    
    iso_reg = IsotonicRegression(out_of_bounds='clip', increasing=True)
    iso_reg.fit(uncertainty, residuals)
    
    # Validate calibration
    calibrated_unc = iso_reg.predict(uncertainty)
    coverage_95 = np.mean(residuals <= 1.96 * calibrated_unc)
    logger.info(f"Calibrated 95% CI coverage: {coverage_95:.3f} (target: 0.95)")
    
    return iso_reg


def apply_calibrated_uncertainty(
    predictions: Dict[str, np.ndarray],
    calibrator: 'IsotonicRegression',
) -> Dict[str, np.ndarray]:
    """
    Apply calibrated uncertainty to model predictions.
    
    Args:
        predictions: Dict with 'delta_ros' and 'uncertainty' keys
        calibrator: Fitted isotonic regression model
    
    Returns:
        Updated predictions dict with calibrated uncertainty and CI bounds
    """
    raw_unc = predictions['uncertainty']
    calibrated_unc = calibrator.predict(raw_unc)
    
    return {
        **predictions,
        'uncertainty': calibrated_unc,
        'uncertainty_raw': raw_unc,
        'ci_lower': predictions['delta_ros'] - 1.96 * calibrated_unc,
        'ci_upper': predictions['delta_ros'] + 1.96 * calibrated_unc,
    }


def save_metrics(metrics: Dict, output_dir: str, filename: str = 'test_metrics.json'):
    """Save evaluation metrics to JSON file."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    with open(path, 'w') as f:
        json.dump(metrics, f, indent=2, default=str)
    logger.info(f"Metrics saved to {path}")


def format_metrics_report(metrics: Dict) -> str:
    """Format evaluation metrics as a readable string report."""
    lines = []
    lines.append('=' * 60)
    lines.append('EVALUATION RESULTS')
    lines.append('=' * 60)
    
    agg = metrics.get('aggregate', {})
    lines.append(f"R²:      {agg.get('r2', 'N/A'):.4f}" if isinstance(agg.get('r2'), (int, float)) else f"R²:      {agg.get('r2', 'N/A')}")
    lines.append(f"MAE:     {agg.get('mae', 'N/A'):.4f} m/min" if isinstance(agg.get('mae'), (int, float)) else f"MAE:     {agg.get('mae', 'N/A')}")
    lines.append(f"RMSE:    {agg.get('rmse', 'N/A'):.4f} m/min" if isinstance(agg.get('rmse'), (int, float)) else f"RMSE:    {agg.get('rmse', 'N/A')}")
    lines.append(f"Bias:    {agg.get('bias', 'N/A'):.4f} m/min" if isinstance(agg.get('bias'), (int, float)) else f"Bias:    {agg.get('bias', 'N/A')}")
    lines.append(f"Samples: {agg.get('n_samples', 'N/A')}")
    
    # Calibration
    cal = metrics.get('calibration', {})
    if cal:
        lines.append('')
        lines.append('--- Calibration ---')
        for k, v in cal.items():
            lines.append(f"  {k}: {v:.3f}" if isinstance(v, (int, float)) else f"  {k}: {v}")
    
    # Per-fuel
    by_fuel = metrics.get('by_fuel', [])
    if by_fuel:
        lines.append('')
        lines.append('--- Per Fuel Model ---')
        for fm in sorted(by_fuel, key=lambda x: x.get('r2', 0), reverse=True):
            lines.append(f"  {fm['fuel_model']:25s}: R²={fm['r2']:.3f}, MAE={fm['mae']:.3f}, n={fm['n_observations']}")
    
    return '\n'.join(lines)
