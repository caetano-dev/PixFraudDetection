"""
Iterative, Temporally-Aware SHAP Recursive Feature Elimination (RFE) Pipeline.

This script implements a memory-efficient feature selection pipeline that:
1. Removes collinear features based on Spearman correlation
2. Protects temporally important features from elimination
3. Iteratively eliminates low-performing features while tracking AUPRC
4. Operates within strict 8GB RAM constraints

Hardware Constraint: Execution environment limited to 8GB RAM
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import importlib.util
import json
import gc
from typing import List, Dict, Tuple, Set
from scipy.stats import spearmanr

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
CORRELATION_THRESHOLD = 0.80  # Drop one feature from pairs with correlation > this
DROP_PERCENTAGE = 5  # Drop bottom 5% of features per iteration
DELTA_AUPRC_THRESHOLD = -0.01  # Stop if AUPRC drops more than this
TEMPORAL_PROTECTION_PERCENTILE = 50  # Protect features in top 50% in any window
N_TRIALS = 20  # Optuna trials for hyperparameter tuning
MAX_ITERATIONS = 20  # Safety limit to prevent runaway loops 


# ============================================================================
# MEMORY-EFFICIENT UTILITIES
# ============================================================================
# Note: cleanup_memory function removed - del must be called in the scope
# where variables are created to properly release memory


# ============================================================================
# STEP 1: COLLINEARITY FILTER
# ============================================================================

def compute_collinearity_filter(
    df: pd.DataFrame,
    feature_cols: List[str],
    shap_importance: Dict[str, float],
    threshold: float = 0.80
) -> Tuple[List[str], List[Tuple[str, str, float]]]:
    """
    Remove collinear features based on Spearman correlation.
    Between correlated pairs, drop the feature with lower mean absolute SHAP.
    
    Returns:
        - List of features after collinearity filtering
        - List of dropped feature pairs (feat1, feat2, correlation)
    """
    print("\n" + "=" * 80)
    print("STEP 1: COLLINEARITY FILTER")
    print("=" * 80)
    
    # Compute Spearman correlation on available features only
    available_features = [f for f in feature_cols if f in df.columns]
    print(f"Computing Spearman correlation for {len(available_features)} features...")
    
    # Use only a sample of data to save memory
    sample_size = min(10000, len(df))
    df_sample = df[available_features].sample(n=sample_size, random_state=42)
    
    # Compute correlation matrix
    corr_matrix = df_sample.corr(method='spearman').abs()
    del df_sample
    gc.collect()
    
    # Find highly correlated pairs
    dropped_features = set()
    dropped_pairs = []
    
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            feat1 = corr_matrix.columns[i]
            feat2 = corr_matrix.columns[j]
            correlation = corr_matrix.iloc[i, j]
            
            if correlation > threshold:
                # Drop the feature with lower SHAP importance
                shap1 = shap_importance.get(feat1, 0)
                shap2 = shap_importance.get(feat2, 0)
                
                if shap1 < shap2:
                    to_drop = feat1
                else:
                    to_drop = feat2
                
                if to_drop not in dropped_features:
                    dropped_features.add(to_drop)
                    dropped_pairs.append((feat1, feat2, correlation))
                    print(f"  Dropping '{to_drop}' (SHAP={shap_importance.get(to_drop, 0):.4f}) - "
                          f"correlated with '{feat1 if to_drop == feat2 else feat2}' (r={correlation:.3f})")
    
    del corr_matrix
    gc.collect()
    
    retained_features = [f for f in available_features if f not in dropped_features]
    
    print(f"\n[COLLINEARITY FILTER] Dropped {len(dropped_features)} features due to correlation > {threshold}")
    print(f"[COLLINEARITY FILTER] Retained {len(retained_features)} features")
    
    return retained_features, dropped_pairs


# ============================================================================
# STEP 2: TEMPORAL STABILITY FILTER
# ============================================================================

def compute_temporal_protection(
    df: pd.DataFrame,
    feature_cols: List[str],
    window_col: str,
    target_col: str,
    percentile: float = 50
) -> Set[str]:
    """
    Identify features that rank in the top percentile in at least one window.
    These features are protected from immediate elimination.
    
    Returns:
        Set of protected feature names
    """
    print("\n" + "=" * 80)
    print("STEP 2: TEMPORAL STABILITY ANALYSIS")
    print("=" * 80)
    print("Running forward-chaining validation to extract per-window SHAP values...")
    
    # Run forward-chaining to get window-specific SHAP
    results_df, summary, global_shap_values, global_x_test_df = forward_chaining_validation(
        df=df,
        feature_cols=feature_cols,
        window_col=window_col,
        target_col=target_col,
        min_train_windows=2,
        n_trials=N_TRIALS,
        verbose=False
    )
    
    # Compute per-window mean absolute SHAP
    print(f"\nAnalyzing temporal importance across {len(global_shap_values)} windows...")
    
    # Stack all SHAP values with window labels
    window_shap_importance = {}
    
    for window_idx, shap_values in enumerate(global_shap_values):
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        
        for feat_idx, feat_name in enumerate(feature_cols):
            if feat_name not in window_shap_importance:
                window_shap_importance[feat_name] = []
            window_shap_importance[feat_name].append(mean_abs_shap[feat_idx])
    
    # Determine protected features (top percentile in any window)
    protected_features = set()
    
    for feat_name, shap_values_across_windows in window_shap_importance.items():
        # Check if feature ranks in top percentile in any window
        for window_shap in shap_values_across_windows:
            all_window_shaps = [window_shap_importance[f][0] for f in feature_cols]
            threshold = np.percentile(all_window_shaps, percentile)
            
            if window_shap >= threshold:
                protected_features.add(feat_name)
                break
    
    del results_df, global_shap_values, global_x_test_df, window_shap_importance
    gc.collect()
    
    print(f"\n[TEMPORAL PROTECTION] {len(protected_features)} features protected (top {percentile}% in ≥1 window)")
    print(f"Protected features: {sorted(protected_features)[:10]}{'...' if len(protected_features) > 10 else ''}")
    
    return protected_features


# ============================================================================
# STEP 3: ITERATIVE SHAP-RFE LOOP
# ============================================================================

def run_shap_rfe_loop(
    df: pd.DataFrame,
    initial_features: List[str],
    protected_features: Set[str],
    shap_importance: Dict[str, float],
    window_col: str,
    target_col: str
) -> Tuple[List[str], List[Dict]]:
    """
    Iteratively eliminate lowest-performing features until AUPRC degrades.
    
    Returns:
        - Final retained feature list
        - Iteration history
    """
    print("\n" + "=" * 80)
    print("STEP 3: ITERATIVE SHAP-RFE LOOP")
    print("=" * 80)
    
    current_features = initial_features.copy()
    iteration_history = []
    
    # Baseline evaluation
    print("\n[ITERATION 0] Baseline evaluation with all features...")
    results_df, summary, temp_shap, temp_x_test = forward_chaining_validation(
        df=df,
        feature_cols=current_features,
        window_col=window_col,
        target_col=target_col,
        min_train_windows=2,
        n_trials=N_TRIALS,
        verbose=False
    )
    
    baseline_auprc = summary['overall_auprc']
    
    iteration_history.append({
        'iteration': 0,
        'n_features': len(current_features),
        'features_dropped': 0,
        'overall_auprc': baseline_auprc,
        'delta_from_baseline': 0.0,
        'features': current_features.copy()
    })
    
    print(f"  Features: {len(current_features)} | AUPRC: {baseline_auprc:.4f}")
    del results_df, temp_shap, temp_x_test
    gc.collect()
    
    # Iterative elimination
    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n[ITERATION {iteration}] Identifying features to drop...")
        
        # Identify droppable features (not protected)
        droppable_features = [f for f in current_features if f not in protected_features]
        
        if len(droppable_features) == 0:
            print("  No droppable features remaining (all protected). Stopping.")
            break
        
        # Calculate number of features to drop (5% or minimum 1)
        n_to_drop = max(1, int(len(droppable_features) * DROP_PERCENTAGE / 100))
        
        # Sort droppable features by SHAP importance (ascending)
        droppable_sorted = sorted(droppable_features, key=lambda f: shap_importance.get(f, 0))
        features_to_drop = droppable_sorted[:n_to_drop]
        
        print(f"  Dropping {n_to_drop} lowest SHAP features: {features_to_drop}")
        
        # Create new feature set
        candidate_features = [f for f in current_features if f not in features_to_drop]
        
        # Evaluate with reduced feature set
        print(f"  Retraining with {len(candidate_features)} features...")
        results_df, summary, temp_shap, temp_x_test = forward_chaining_validation(
            df=df,
            feature_cols=candidate_features,
            window_col=window_col,
            target_col=target_col,
            min_train_windows=2,
            n_trials=N_TRIALS,
            verbose=False
        )
        
        new_auprc = summary['overall_auprc']
        delta_from_baseline = new_auprc - baseline_auprc
        
        print(f"  AUPRC: {new_auprc:.4f} (Δ from baseline={delta_from_baseline:+.4f})")
        
        iteration_history.append({
            'iteration': iteration,
            'n_features': len(candidate_features),
            'features_dropped': n_to_drop,
            'overall_auprc': new_auprc,
            'delta_from_baseline': delta_from_baseline,
            'dropped_features': features_to_drop,
            'features': candidate_features.copy()
        })
        
        del results_df, temp_shap, temp_x_test
        gc.collect()
        
        # Check stopping criterion: absolute degradation from baseline
        if delta_from_baseline < DELTA_AUPRC_THRESHOLD:
            print(f"\n[STOPPING] ΔAUPRC from baseline ({delta_from_baseline:+.4f}) < threshold ({DELTA_AUPRC_THRESHOLD})")
            print(f"[STOPPING] Reverting to iteration {iteration - 1} feature set")
            final_features = current_features
            break
        else:
            # Accept the feature reduction
            current_features = candidate_features
            
            if len(droppable_features) <= n_to_drop:
                print("\n[STOPPING] All droppable features have been eliminated")
                final_features = current_features
                break
    else:
        # Max iterations reached
        print(f"\n[STOPPING] Maximum iterations ({MAX_ITERATIONS}) reached")
        final_features = current_features
    
    return final_features, iteration_history


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("=" * 80)
    print("ITERATIVE, TEMPORALLY-AWARE SHAP RECURSIVE FEATURE ELIMINATION")
    print("=" * 80)
    print(f"Hardware constraint: 8GB RAM")
    print(f"Configuration: Correlation threshold={CORRELATION_THRESHOLD}, "
          f"Drop%={DROP_PERCENTAGE}, ΔAUPRC threshold={DELTA_AUPRC_THRESHOLD}")

    results_dir = DATA_PATH / "results"
    results_dir.mkdir(exist_ok=True, parents=True)
    shap_file = results_dir / "shap_feature_importance.csv"
    
    if not shap_file.exists():
        raise FileNotFoundError(f"Missing SHAP data at {shap_file}. Run script 04 first.")

    # Load global SHAP importance
    shap_df = pd.read_csv(shap_file)
    shap_importance = dict(zip(shap_df['feature'], shap_df['mean_abs_shap']))
    initial_features = shap_df['feature'].tolist()
    
    print(f"\n[INITIAL] Starting with {len(initial_features)} features")
    
    # Load dataset
    features_path = DATA_PATH / OUTPUT_FEATURES_FILE
    if not features_path.exists():
        raise FileNotFoundError(f"Missing {features_path}.")
        
    df = pd.read_parquet(features_path)
    window_col = 'window_id' if 'window_id' in df.columns else 'date'
    target_col = 'is_fraud'
    
    # STEP 1: Collinearity Filter
    features_after_collinearity, dropped_pairs = compute_collinearity_filter(
        df=df,
        feature_cols=initial_features,
        shap_importance=shap_importance,
        threshold=CORRELATION_THRESHOLD
    )
    
    # STEP 2: Temporal Stability Filter
    protected_features = compute_temporal_protection(
        df=df,
        feature_cols=features_after_collinearity,
        window_col=window_col,
        target_col=target_col,
        percentile=TEMPORAL_PROTECTION_PERCENTILE
    )
    
    # STEP 3: Iterative SHAP-RFE
    final_features, iteration_history = run_shap_rfe_loop(
        df=df,
        initial_features=features_after_collinearity,
        protected_features=protected_features,
        shap_importance=shap_importance,
        window_col=window_col,
        target_col=target_col
    )
    
    # STEP 4: Save Results
    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    
    discarded_features = [f for f in initial_features if f not in final_features]
    print(f"\nFeatures: {len(initial_features)} → {len(final_features)} (dropped {len(discarded_features)})")
    print(f"Final AUPRC: {iteration_history[-1]['overall_auprc']:.4f}")
    print(f"Baseline AUPRC: {iteration_history[0]['overall_auprc']:.4f}")
    print(f"Overall ΔAUPRC: {iteration_history[-1]['overall_auprc'] - iteration_history[0]['overall_auprc']:+.4f}")
    
    # Save retained features
    output_file = results_dir / "rfe_retained_features.json"
    with open(output_file, 'w') as f:
        json.dump({
            'n_features': len(final_features),
            'features': final_features,
            'baseline_auprc': iteration_history[0]['overall_auprc'],
            'final_auprc': iteration_history[-1]['overall_auprc']
        }, f, indent=2)
    print(f"\n✓ Retained features saved to: {output_file}")
    
    # Save discarded features
    discarded_file = results_dir / "rfe_discarded_features.json"
    with open(discarded_file, 'w') as f:
        json.dump({
            'n_discarded': len(discarded_features),
            'discarded_features': discarded_features,
            'collinearity_pairs': [(f1, f2, float(corr)) for f1, f2, corr in dropped_pairs]
        }, f, indent=2)
    print(f"✓ Discarded features saved to: {discarded_file}")
    
    # Save iteration history
    history_file = results_dir / "rfe_iteration_history.csv"
    history_df = pd.DataFrame([
        {
            'iteration': h['iteration'],
            'n_features': h['n_features'],
            'features_dropped': h['features_dropped'],
            'overall_auprc': h['overall_auprc'],
            'delta_from_baseline': h['delta_from_baseline']
        }
        for h in iteration_history
    ])
    history_df.to_csv(history_file, index=False)
    print(f"✓ Iteration history saved to: {history_file}")
    
    # Print summary
    print("\n" + "-" * 80)
    print("DISCARDED FEATURES SUMMARY")
    print("-" * 80)
    print(f"Total discarded: {len(discarded_features)}")
    if discarded_features:
        print("Features removed:")
        for feat in sorted(discarded_features):
            print(f"  - {feat} (SHAP: {shap_importance.get(feat, 0):.4f})")
    
    print("\n" + "-" * 80)
    print("RETAINED FEATURES")
    print("-" * 80)
    for feat in sorted(final_features):
        print(f"  - {feat} (SHAP: {shap_importance.get(feat, 0):.4f})")


if __name__ == "__main__":
    main()
