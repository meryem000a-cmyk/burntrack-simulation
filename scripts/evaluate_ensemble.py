"""
evaluate_ensemble.py
====================
Standalone evaluation script for the trained BurnTrack ensemble.
"""

import os
import sys
import argparse
import pandas as pd
import json

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from burntrack.corrector.ensemble import StackedCorrectEnsemble
from burntrack.corrector.validation import evaluate_corrector, format_metrics_report

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, required=True, help="Path to trained model directory")
    parser.add_argument("--test-data", type=str, required=True, help="Path to test CSV dataset")
    args = parser.parse_args()
    
    print(f"Loading ensemble from {args.model_dir}...")
    ensemble = StackedCorrectEnsemble.load(args.model_dir)
    
    print(f"Loading test data from {args.test_data}...")
    test_df = pd.read_csv(args.test_data)
    
    print("Generating predictions...")
    preds = []
    uncs = []
    cis_lower = []
    cis_upper = []
    
    for _, row in test_df.iterrows():
        res = ensemble.predict_with_uncertainty(row.to_dict())
        preds.append(res['delta_ros'])
        uncs.append(res['uncertainty'])
        cis_lower.append(res['ci_lower'])
        cis_upper.append(res['ci_upper'])
        
    test_df['delta_ros_pred'] = preds
    test_df['uncertainty'] = uncs
    test_df['ci_lower'] = cis_lower
    test_df['ci_upper'] = cis_upper
    
    print("Evaluating metrics...")
    metrics = evaluate_corrector(
        y_true=test_df['delta_ros'].values,
        y_pred=test_df['delta_ros_pred'].values,
        ros_rothermel=test_df['ros'].values if 'ros' in test_df else None,
        ci_lower=test_df['ci_lower'].values,
        ci_upper=test_df['ci_upper'].values,
        uncertainty=test_df['uncertainty'].values,
        fire_ids=test_df['fire_id'].values if 'fire_id' in test_df else None,
        fuel_models=test_df['fuel_model_code'].values if 'fuel_model_code' in test_df else None
    )
    report = format_metrics_report(metrics)
    
    print("\n" + "="*80)
    print("EVALUATION REPORT")
    print("="*80)
    print(report)
    
    report_path = os.path.join(args.model_dir, "standalone_eval_report.md")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\nReport saved to {report_path}")

if __name__ == "__main__":
    main()
