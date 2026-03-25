"""
Automated SHAP-based Feature Pruning and Re-evaluation.

This script reads the global SHAP importance from the full model run,
drops features contributing less than a specified threshold (e.g., bottom 50%),
and executes a pruned forward-chaining validation to verify if removing
structural noise improves AUPRC.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import importlib.util

root_path = Path(__file__).resolve().parent.parent
sys.path.append(str(root_path))

from src.config import OUTPUT_FEATURES_FILE, DATA_PATH

# Dynamically import script 04
module_name = "04_train_model_forward_chaining"
file_path = Path(__file__).resolve().parent / f"{module_name}.py"
if not file_path.exists():
    raise FileNotFoundError(f"Missing {file_path}. Ensure script 04 is in the same directory.")

spec = importlib.util.spec_from_file_location(module_name, file_path)
train_module = importlib.util.module_from_spec(spec)
sys.modules[module_name] = train_module
spec.loader.exec_module(train_module)

forward_chaining_validation = train_module.forward_chaining_validation

# ============================================================================
# CONFIGURATION
# ============================================================================
# Retain only the top X% of features based on global mean absolute SHAP
PRUNE_PERCENTILE = 50 

def main():
    print("=" * 80)
    print("AUTOMATED FEATURE PRUNING & RE-EVALUATION")
    print("=" * 80)

    results_dir = DATA_PATH / "results"
    shap_file = results_dir / "shap_feature_importance.csv"
    
    if not shap_file.exists():
        raise FileNotFoundError(f"Missing SHAP data at {shap_file}. Run script 05 first.")

    # 1. Load and Prune Features
    shap_df = pd.read_csv(shap_file)
    
    # Calculate the threshold value (e.g., median if PRUNE_PERCENTILE is 50)
    threshold_val = np.percentile(shap_df['mean_abs_shap'], PRUNE_PERCENTILE)
    
    pruned_df = shap_df[shap_df['mean_abs_shap'] >= threshold_val]
    pruned_features = pruned_df['feature'].tolist()
    
    print(f"\n[PRUNING] Threshold set at {PRUNE_PERCENTILE}th percentile (|SHAP| >= {threshold_val:.6f})")
    print(f"[PRUNING] Dropping {len(shap_df) - len(pruned_features)} features.")
    print(f"[PRUNING] Retaining {len(pruned_features)} features.")
    
    # 2. Load Dataset
    features_path = DATA_PATH / OUTPUT_FEATURES_FILE
    if not features_path.exists():
        raise FileNotFoundError(f"Missing {features_path}.")
        
    df = pd.read_parquet(features_path)
    window_col = 'window_id' if 'window_id' in df.columns else 'date'

    # 3. Execute Pruned Validation Loop
    print("\n" + "-" * 80)
    print(f"EVALUATING PRUNED MODEL ({len(pruned_features)} features)")
    print("-" * 80)
    
    results_df, summary, _, _ = forward_chaining_validation(
        df=df,
        feature_cols=pruned_features,
        window_col=window_col,
        target_col="is_fraud",
        min_train_windows=2,
        n_trials=20, # Set to 50 for final run
        verbose=True
    )
    
    print("\n" + "=" * 80)
    print("PRUNED MODEL AGGREGATE RESULTS")
    print("=" * 80)
    print(f"Test Windows:       {summary['n_test_windows']}")
    print(f"Total test samples: {summary['total_test_samples']:,}")
    print(f"Overall AUPRC:      {summary['overall_auprc']:.4f}")
    print(f"Mean Window AUPRC:  {summary['mean_window_auprc']:.4f} ± {summary['std_window_auprc']:.4f}")
    print(f"Overall ROC-AUC:    {summary['overall_roc_auc']:.4f}")

    # 4. Save Pruned Results
    results_df.to_csv(results_dir / "pruned_model_results.csv", index=False)
    
    # Output the exact retained list for documentation
    print("\n[RETAINED FEATURE SET]")
    print(pruned_features)

if __name__ == "__main__":
    main()
