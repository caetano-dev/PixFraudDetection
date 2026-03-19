"""
Forward-Chaining Time-Series Cross-Validation for Money Laundering Detection.

This script implements strict temporal validation using forward-chaining:
- For each window t in the test set:
  1. Train on all windows < t
  2. Predict on window t
  3. Record performance metrics

This approach ensures NO target leakage from future windows and provides
a realistic evaluation of model performance in production deployment.
"""
import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, precision_recall_curve
from pathlib import Path
import sys
from typing import Dict, List, Tuple

root_path = Path(__file__).resolve().parent.parent
sys.path.append(str(root_path))

from src.config import OUTPUT_FEATURES_FILE, DATA_PATH


def compute_precision_at_k(y_true: np.ndarray, y_probs: np.ndarray, k_values: list) -> dict:
    """Computes Precision@K to measure real-world investigator efficiency."""
    results = {}
    sorted_indices = np.argsort(y_probs)[::-1]
    
    for k in k_values:
        if k > len(y_true):
            continue
        top_k_indices = sorted_indices[:k]
        top_k_labels = y_true[top_k_indices]
        precision = np.sum(top_k_labels) / k
        results[k] = precision
    return results


def forward_chaining_validation(
    df: pd.DataFrame,
    window_col: str = "window_id",
    target_col: str = "is_fraud",
    min_train_windows: int = 2,
    model_type: str = "xgboost",
    n_estimators: int = 100,
    max_depth: int = 3,
    learning_rate: float = 0.1,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Perform forward-chaining cross-validation on sliding window data.
    
    For each window t:
        - Train on all data from windows [0, t-1]
        - Predict on window t
        - Record metrics for window t
    
    Args:
        df: Feature dataframe with window_id, features, and is_fraud
        window_col: Column name for window identifier
        target_col: Column name for target variable
        min_train_windows: Minimum number of windows needed for training
        model_type: "xgboost" or "lightgbm"
        n_estimators: Number of boosting rounds
        max_depth: Maximum tree depth
        learning_rate: Learning rate
    
    Returns:
        results_df: DataFrame with per-window performance metrics
        summary: Dictionary with aggregate metrics across all test windows
    """
    # Sort by window to ensure temporal order
    df = df.sort_values(by=window_col).reset_index(drop=True)
    
    # Get unique windows in sorted order
    unique_windows = sorted(df[window_col].unique())
    print(f"\nTotal windows available: {len(unique_windows)}")
    print(f"Window range: {unique_windows[0]} to {unique_windows[-1]}")
    
    # Define columns to drop (non-predictive metadata)
    cols_to_drop = [
        window_col, "window_start", "window_end", "entity_id", target_col,
        "leiden_macro_id", "leiden_micro_id", "is_rank_anomaly", "credit_card_count_sent", "tx_count", "cash_count_sent", "cash_count_recv", "degree", "weirdnodes_magnitude", "wire_count_recv", "k_core", "weirdnodes_residual", "leiden_micro_size" 
    ]
    drop_cols = [col for col in cols_to_drop if col in df.columns]
    
    # Storage for per-window results
    window_results = []
    all_y_true = []
    all_y_probs = []
    
    # Forward-chaining loop
    print(f"\nStarting forward-chaining validation (min_train_windows={min_train_windows})...\n")
    
    for test_window_idx in range(min_train_windows, len(unique_windows)):
        test_window_id = unique_windows[test_window_idx]
        train_window_ids = unique_windows[:test_window_idx]
        
        # Temporal boundary enforcement
        train_df = df[df[window_col].isin(train_window_ids)]
        test_df = df[df[window_col] == test_window_id]
        
        # Sanity check: ensure no overlap
        assert len(set(train_df[window_col]) & {test_window_id}) == 0, \
            f"Data leakage detected: test window {test_window_id} appears in training set!"
        
        # Extract features and labels
        X_train = train_df.drop(columns=drop_cols)
        y_train = train_df[target_col].values
        
        X_test = test_df.drop(columns=drop_cols)
        y_test = test_df[target_col].values
        
        # Handle edge case: no fraud in test window
        if len(np.unique(y_test)) < 2:
            print(f"  Window {test_window_id:3d}: Skipped (no variance in labels)")
            continue
        
        # Handle edge case: no fraud in training set
        if len(np.unique(y_train)) < 2:
            print(f"  Window {test_window_id:3d}: Skipped (no fraud in training history)")
            continue
        
        # Compute class weights
        num_neg = np.sum(y_train == 0)
        num_pos = np.sum(y_train == 1)
        scale_pos_weight = num_neg / num_pos if num_pos > 0 else 1.0
        
        # Train model
        if model_type == "xgboost":
            model = xgb.XGBClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                scale_pos_weight=scale_pos_weight,
                objective='binary:logistic',
                eval_metric='aucpr',
                random_state=42,
                tree_method='hist',
                verbosity=0
            )
        elif model_type == "lightgbm":
            model = lgb.LGBMClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                scale_pos_weight=scale_pos_weight,
                objective='binary',
                metric='auc',
                random_state=42,
                n_jobs=-1,
                verbosity=-1
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
        
        model.fit(X_train, y_train)
        
        # Predict on test window
        y_probs = model.predict_proba(X_test)[:, 1]
        
        # Store for aggregate metrics
        all_y_true.extend(y_test)
        all_y_probs.extend(y_probs)
        
        # Compute window-specific metrics
        try:
            auprc = average_precision_score(y_test, y_probs)
            roc_auc = roc_auc_score(y_test, y_probs)
        except ValueError as e:
            print(f"  Window {test_window_id:3d}: Metric computation failed - {e}")
            continue
        
        # Find optimal F1 threshold
        precisions, recalls, thresholds = precision_recall_curve(y_test, y_probs)
        f1_scores = np.divide(
            2 * precisions * recalls,
            precisions + recalls,
            out=np.zeros_like(precisions),
            where=(precisions + recalls) != 0
        )
        best_threshold_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[best_threshold_idx] if best_threshold_idx < len(thresholds) else 0.5
        
        y_pred = (y_probs >= optimal_threshold).astype(int)
        f1 = f1_score(y_test, y_pred)
        
        # Precision@K
        k_vals = [10, 50, 100, 500]
        prec_at_k = compute_precision_at_k(y_test, y_probs, k_vals)
        
        # Baseline fraud rate in test window
        fraud_rate = np.mean(y_test)
        
        # Store results
        result = {
            "test_window_id": test_window_id,
            "train_windows": f"[{train_window_ids[0]}, {train_window_ids[-1]}]",
            "n_train_samples": len(X_train),
            "n_test_samples": len(X_test),
            "test_fraud_rate": fraud_rate,
            "auprc": auprc,
            "roc_auc": roc_auc,
            "f1_score": f1,
            "optimal_threshold": optimal_threshold,
        }
        
        for k, prec in prec_at_k.items():
            result[f"precision_at_{k}"] = prec
            result[f"lift_at_{k}"] = prec / fraud_rate if fraud_rate > 0 else 0
        
        window_results.append(result)
        
        print(f"  Window {test_window_id:3d}: AUPRC={auprc:.4f}, ROC-AUC={roc_auc:.4f}, F1={f1:.4f}, "
              f"Fraud={fraud_rate:.4%}, Train_size={len(X_train):,}")
    
    # Aggregate metrics across all test windows
    results_df = pd.DataFrame(window_results)
    
    # Compute overall metrics on concatenated predictions
    all_y_true = np.array(all_y_true)
    all_y_probs = np.array(all_y_probs)
    
    overall_auprc = average_precision_score(all_y_true, all_y_probs)
    overall_roc_auc = roc_auc_score(all_y_true, all_y_probs)
    
    # Overall optimal threshold
    overall_precisions, overall_recalls, overall_thresholds = precision_recall_curve(all_y_true, all_y_probs)
    overall_f1_scores = np.divide(
        2 * overall_precisions * overall_recalls,
        overall_precisions + overall_recalls,
        out=np.zeros_like(overall_precisions),
        where=(overall_precisions + overall_recalls) != 0
    )
    overall_best_idx = np.argmax(overall_f1_scores)
    overall_threshold = overall_thresholds[overall_best_idx] if overall_best_idx < len(overall_thresholds) else 0.5
    
    overall_y_pred = (all_y_probs >= overall_threshold).astype(int)
    overall_f1 = f1_score(all_y_true, overall_y_pred)
    
    overall_prec_at_k = compute_precision_at_k(all_y_true, all_y_probs, k_vals)
    overall_fraud_rate = np.mean(all_y_true)
    
    summary = {
        "n_test_windows": len(window_results),
        "total_test_samples": len(all_y_true),
        "overall_fraud_rate": overall_fraud_rate,
        "overall_auprc": overall_auprc,
        "overall_roc_auc": overall_roc_auc,
        "overall_f1": overall_f1,
        "overall_threshold": overall_threshold,
        "mean_window_auprc": results_df["auprc"].mean(),
        "std_window_auprc": results_df["auprc"].std(),
        "mean_window_roc_auc": results_df["roc_auc"].mean(),
        "std_window_roc_auc": results_df["roc_auc"].std(),
    }
    
    for k in k_vals:
        summary[f"overall_precision_at_{k}"] = overall_prec_at_k.get(k, 0)
        summary[f"overall_lift_at_{k}"] = (overall_prec_at_k.get(k, 0) / overall_fraud_rate 
                                           if overall_fraud_rate > 0 else 0)
    
    return results_df, summary


def main():
    features_path = DATA_PATH / OUTPUT_FEATURES_FILE
    if not features_path.exists():
        raise FileNotFoundError(f"Missing {features_path}. Run 03_extract_features.py first.")
    
    print("=" * 80)
    print("FORWARD-CHAINING TIME-SERIES CROSS-VALIDATION")
    print("=" * 80)
    print(f"Loading feature matrix from {features_path}...")
    df = pd.read_parquet(features_path)
    
    print(f"\nDataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Check if we have window_id (new format) or date (old format)
    if "window_id" in df.columns:
        window_col = "window_id"
        print(f"\nUsing sliding window format with window_id")
    elif "date" in df.columns:
        window_col = "date"
        print(f"\nWARNING: Using legacy date-based format. Consider running with updated SQL.")
    else:
        raise ValueError("Neither 'window_id' nor 'date' column found in features!")
    
    # Validate temporal structure
    print(f"\nTemporal structure validation:")
    print(f"  Total unique windows: {df[window_col].nunique()}")
    print(f"  Total samples: {len(df):,}")
    print(f"  Fraud rate: {df['is_fraud'].mean():.4%}")
    
    # ========================================================================
    # XGBoost Forward-Chaining Validation
    # ========================================================================
    print("\n" + "=" * 80)
    print("XGBOOST FORWARD-CHAINING VALIDATION")
    print("=" * 80)
    
    xgb_results, xgb_summary = forward_chaining_validation(
        df=df,
        window_col=window_col,
        target_col="is_fraud",
        min_train_windows=2,
        model_type="xgboost",
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
    )
    
    print("\n" + "-" * 80)
    print("XGBOOST AGGREGATE RESULTS")
    print("-" * 80)
    print(f"Test windows evaluated:    {xgb_summary['n_test_windows']}")
    print(f"Total test samples:        {xgb_summary['total_test_samples']:,}")
    print(f"Overall fraud rate:        {xgb_summary['overall_fraud_rate']:.4%}")
    print(f"\nOverall Performance:")
    print(f"  AUPRC:                   {xgb_summary['overall_auprc']:.4f}")
    print(f"  ROC-AUC:                 {xgb_summary['overall_roc_auc']:.4f}")
    print(f"  F1-Score:                {xgb_summary['overall_f1']:.4f} (threshold={xgb_summary['overall_threshold']:.4f})")
    print(f"\nPer-Window Statistics:")
    print(f"  Mean AUPRC:              {xgb_summary['mean_window_auprc']:.4f} ± {xgb_summary['std_window_auprc']:.4f}")
    print(f"  Mean ROC-AUC:            {xgb_summary['mean_window_roc_auc']:.4f} ± {xgb_summary['std_window_roc_auc']:.4f}")
    print(f"\nOverall Precision@K:")
    for k in [10, 50, 100, 500]:
        prec_key = f"overall_precision_at_{k}"
        lift_key = f"overall_lift_at_{k}"
        if prec_key in xgb_summary:
            print(f"  @ {k:>3}: {xgb_summary[prec_key]:.4%} (Lift: {xgb_summary[lift_key]:.2f}x)")
    
    # Save per-window results
    output_dir = DATA_PATH / "results"
    output_dir.mkdir(exist_ok=True)
    xgb_results_path = output_dir / "xgboost_forward_chaining_results.csv"
    xgb_results.to_csv(xgb_results_path, index=False)
    print(f"\nPer-window results saved to: {xgb_results_path}")
    
    # ========================================================================
    # LightGBM Forward-Chaining Validation
    # ========================================================================
    print("\n" + "=" * 80)
    print("LIGHTGBM FORWARD-CHAINING VALIDATION")
    print("=" * 80)
    
    lgb_results, lgb_summary = forward_chaining_validation(
        df=df,
        window_col=window_col,
        target_col="is_fraud",
        min_train_windows=2,
        model_type="lightgbm",
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
    )
    
    print("\n" + "-" * 80)
    print("LIGHTGBM AGGREGATE RESULTS")
    print("-" * 80)
    print(f"Test windows evaluated:    {lgb_summary['n_test_windows']}")
    print(f"Total test samples:        {lgb_summary['total_test_samples']:,}")
    print(f"Overall fraud rate:        {lgb_summary['overall_fraud_rate']:.4%}")
    print(f"\nOverall Performance:")
    print(f"  AUPRC:                   {lgb_summary['overall_auprc']:.4f}")
    print(f"  ROC-AUC:                 {lgb_summary['overall_roc_auc']:.4f}")
    print(f"  F1-Score:                {lgb_summary['overall_f1']:.4f} (threshold={lgb_summary['overall_threshold']:.4f})")
    print(f"\nPer-Window Statistics:")
    print(f"  Mean AUPRC:              {lgb_summary['mean_window_auprc']:.4f} ± {lgb_summary['std_window_auprc']:.4f}")
    print(f"  Mean ROC-AUC:            {lgb_summary['mean_window_roc_auc']:.4f} ± {lgb_summary['std_window_roc_auc']:.4f}")
    print(f"\nOverall Precision@K:")
    for k in [10, 50, 100, 500]:
        prec_key = f"overall_precision_at_{k}"
        lift_key = f"overall_lift_at_{k}"
        if prec_key in lgb_summary:
            print(f"  @ {k:>3}: {lgb_summary[prec_key]:.4%} (Lift: {lgb_summary[lift_key]:.2f}x)")
    
    lgb_results_path = output_dir / "lightgbm_forward_chaining_results.csv"
    lgb_results.to_csv(lgb_results_path, index=False)
    print(f"\nPer-window results saved to: {lgb_results_path}")
    
    print("\n" + "=" * 80)
    print("FORWARD-CHAINING VALIDATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
