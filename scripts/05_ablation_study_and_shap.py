"""
Ablation Study and SHAP Interpretability Analysis for Money Laundering Detection.

This script empirically proves that topological graph metrics (PageRank, Betweenness,
HITS, etc.) significantly improve fraud detection over behavioral features alone.

Methodology:
1. Define three feature sets: Behavioral-only, Topological-only, and Full
2. Run forward-chaining validation with Behavioral-only features (baseline)
3. Run forward-chaining validation with Full features (proposed model)
4. Compare metrics (AUPRC, F1, Precision@100) across all test windows
5. Calculate percentage lift from adding topological features
6. Use SHAP to identify which features drive fraud detection
7. Visualize results to prove scientific value of graph-based features
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server environments
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, precision_recall_curve
from pathlib import Path
import sys
from typing import Dict, List, Tuple, Optional

root_path = Path(__file__).resolve().parent.parent
sys.path.append(str(root_path))

from src.config import OUTPUT_FEATURES_FILE, DATA_PATH


# ============================================================================
# FEATURE SET DEFINITIONS
# ============================================================================

# Behavioral features: Simple transaction statistics and payment type counts
BEHAVIORAL_COLS = [
    # Transaction volume and frequency
    'vol_sent', 'vol_recv', 'tx_count', 'time_variance', 'flow_ratio',
    
    # Currency diversity
    'distinct_currencies_sent', 'distinct_currencies_recv',
    
    # Payment format counts (sent)
    'wire_count_sent', 'cash_count_sent', 'bitcoin_count_sent',
    'cheque_count_sent', 'credit_card_count_sent', 'ach_count_sent',
    'reinvestment_count_sent',
    
    # Payment format counts (received)
    'wire_count_recv', 'cash_count_recv', 'bitcoin_count_recv',
    'cheque_count_recv', 'credit_card_count_recv', 'ach_count_recv',
    'reinvestment_count_recv',
]

# Topological features: Graph-based centrality and community metrics
TOPOLOGICAL_COLS = [
    # PageRank variants (volume-weighted with different walk depths)
    'pr_vol_deep', 'pr_vol_shallow', 'pr_count',
    
    # HITS algorithm (hub and authority scores)
    'hits_hub', 'hits_auth',
    
    # Community detection (Leiden algorithm)
    'leiden_macro_size', 'leiden_macro_modularity',
    'leiden_micro_size', 'leiden_micro_modularity',
    
    # Centrality metrics
    'betweenness', 'k_core',
    
    # Degree metrics
    'degree', 'in_degree', 'out_degree',
    
    # Subgraph motifs (structural patterns)
    'fan_out_count', 'fan_in_count',
    'scatter_gather_count', 'gather_scatter_count',
    'cycle_count',
]

# WeirdNodes: Rank stability metrics (if present in data)
WEIRDNODES_COLS = [
    'weirdnodes_magnitude', 'weirdnodes_residual',
    'is_riser', 'is_faller'
]

# Full feature set: Union of all features
FULL_COLS = BEHAVIORAL_COLS + TOPOLOGICAL_COLS + WEIRDNODES_COLS


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

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
    feature_cols: List[str],
    window_col: str = "window_id",
    target_col: str = "is_fraud",
    min_train_windows: int = 2,
    n_estimators: int = 100,
    max_depth: int = 6,
    learning_rate: float = 0.1,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, Dict, Optional[xgb.XGBClassifier], Optional[pd.DataFrame]]:
    """
    Perform forward-chaining cross-validation with specified feature columns.
    
    Modified from 04_train_model_forward_chaining.py to accept feature_cols parameter.
    
    Args:
        df: Feature dataframe with window_id, features, and is_fraud
        feature_cols: List of feature column names to use for training
        window_col: Column name for window identifier
        target_col: Column name for target variable
        min_train_windows: Minimum number of windows needed for training
        n_estimators: Number of boosting rounds
        max_depth: Maximum tree depth
        learning_rate: Learning rate
        verbose: Whether to print per-window results
    
    Returns:
        results_df: DataFrame with per-window performance metrics
        summary: Dictionary with aggregate metrics across all test windows
        final_model: The model trained on the last training fold (for SHAP analysis)
        final_test_df: The last test dataframe (for SHAP analysis)
    """
    # Sort by window to ensure temporal order
    df = df.sort_values(by=window_col).reset_index(drop=True)
    
    # Get unique windows in sorted order
    unique_windows = sorted(df[window_col].unique())
    
    if verbose:
        print(f"\nTotal windows available: {len(unique_windows)}")
        print(f"Window range: {unique_windows[0]} to {unique_windows[-1]}")
        print(f"Features used: {len(feature_cols)}")
    
    # Verify all feature columns exist in dataframe
    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        print(f"WARNING: Missing columns in dataframe: {missing_cols}")
        # Filter to only existing columns
        feature_cols = [col for col in feature_cols if col in df.columns]
        print(f"Proceeding with {len(feature_cols)} available features")
    
    # Storage for per-window results
    window_results = []
    all_y_true = []
    all_y_probs = []
    
    final_model = None
    final_test_df = None
    
    # Forward-chaining loop
    if verbose:
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
        X_train = train_df[feature_cols]
        y_train = train_df[target_col].values
        
        X_test = test_df[feature_cols]
        y_test = test_df[target_col].values
        
        # Handle edge case: no fraud in test window
        if len(np.unique(y_test)) < 2:
            if verbose:
                print(f"  Window {test_window_id}: Skipped (no variance in labels)")
            continue
        
        # Handle edge case: no fraud in training set
        if len(np.unique(y_train)) < 2:
            if verbose:
                print(f"  Window {test_window_id}: Skipped (no fraud in training history)")
            continue
        
        # Compute class weights
        num_neg = np.sum(y_train == 0)
        num_pos = np.sum(y_train == 1)
        scale_pos_weight = num_neg / num_pos if num_pos > 0 else 1.0
        
        # Train model
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
        
        model.fit(X_train, y_train)
        
        # Always store the latest successfully trained model for SHAP analysis
        final_model = model
        final_test_df = test_df
        
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
            if verbose:
                print(f"  Window {test_window_id}: Metric computation failed - {e}")
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
        
        if verbose:
            print(f"  Window {test_window_id}: AUPRC={auprc:.4f}, ROC-AUC={roc_auc:.4f}, F1={f1:.4f}, "
                  f"Fraud={fraud_rate:.4%}, Train_size={len(X_train):,}")
    
    # Aggregate metrics across all test windows
    results_df = pd.DataFrame(window_results)
    
    if len(all_y_true) == 0:
        if verbose:
            print("WARNING: No valid test windows found!")
        return results_df, {}, final_model, final_test_df
    
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
    
    return results_df, summary, final_model, final_test_df


# ============================================================================
# ABLATION STUDY EXECUTION
# ============================================================================

def run_ablation_study(df: pd.DataFrame, window_col: str = "window_id") -> Tuple[pd.DataFrame, pd.DataFrame, Dict, Dict, Optional[xgb.XGBClassifier], Optional[pd.DataFrame]]:
    """
    Run ablation study comparing Behavioral-only vs Full model.
    
    Args:
        df: Feature dataframe
        window_col: Column name for window identifier ('window_id' or 'date')
    
    Returns:
        behavioral_results: Per-window results for behavioral-only model
        full_results: Per-window results for full model
        behavioral_summary: Aggregate metrics for behavioral-only model
        full_summary: Aggregate metrics for full model
        final_model: Final trained model for SHAP analysis
        final_test_df: Final test dataframe for SHAP analysis
    """
    print("=" * 80)
    print("ABLATION STUDY: BEHAVIORAL-ONLY vs FULL MODEL")
    print("=" * 80)
    
    # Run Behavioral-only model
    print("\n" + "-" * 80)
    print("BASELINE: BEHAVIORAL FEATURES ONLY")
    print("-" * 80)
    print(f"Features: {len(BEHAVIORAL_COLS)}")
    print(f"  Transaction stats: vol_sent, vol_recv, tx_count, time_variance, flow_ratio")
    print(f"  Currency diversity: distinct_currencies_sent/recv")
    print(f"  Payment formats: wire, cash, bitcoin, cheque, credit_card, ach, reinvestment (sent/recv)")
    
    behavioral_results, behavioral_summary, _, _ = forward_chaining_validation(
        df=df,
        feature_cols=BEHAVIORAL_COLS,
        window_col=window_col,
        target_col="is_fraud",
        min_train_windows=2,
        verbose=True
    )
    
    # Run Full model
    print("\n" + "-" * 80)
    print("PROPOSED MODEL: FULL FEATURES (BEHAVIORAL + TOPOLOGICAL)")
    print("-" * 80)
    print(f"Features: {len([c for c in FULL_COLS if c in df.columns])}")
    print(f"  Behavioral: {len(BEHAVIORAL_COLS)}")
    print(f"  Topological: {len([c for c in TOPOLOGICAL_COLS if c in df.columns])}")
    print(f"  WeirdNodes: {len([c for c in WEIRDNODES_COLS if c in df.columns])}")
    
    full_results, full_summary, final_model, final_test_df = forward_chaining_validation(
        df=df,
        feature_cols=FULL_COLS,
        window_col=window_col,
        target_col="is_fraud",
        min_train_windows=2,
        verbose=True
    )
    
    return behavioral_results, full_results, behavioral_summary, full_summary, final_model, final_test_df


# ============================================================================
# PERFORMANCE LIFT ANALYSIS
# ============================================================================

def compute_performance_lift(
    behavioral_results: pd.DataFrame,
    full_results: pd.DataFrame,
    behavioral_summary: Dict,
    full_summary: Dict
) -> pd.DataFrame:
    """
    Compute percentage improvement from adding topological features.
    
    Returns:
        comparison_df: DataFrame with side-by-side metrics and lift percentages
    """
    print("\n" + "=" * 80)
    print("PERFORMANCE LIFT FROM TOPOLOGICAL FEATURES")
    print("=" * 80)
    
    # Merge per-window results
    comparison = behavioral_results.merge(
        full_results,
        on='test_window_id',
        suffixes=('_behavioral', '_full')
    )
    
    # Compute lifts
    comparison['auprc_lift_%'] = (
        (comparison['auprc_full'] - comparison['auprc_behavioral']) / 
        comparison['auprc_behavioral'] * 100
    )
    comparison['f1_lift_%'] = (
        (comparison['f1_score_full'] - comparison['f1_score_behavioral']) / 
        comparison['f1_score_behavioral'] * 100
    )
    comparison['precision_at_100_lift_%'] = (
        (comparison['precision_at_100_full'] - comparison['precision_at_100_behavioral']) / 
        comparison['precision_at_100_behavioral'] * 100
    )
    
    # Print per-window comparison
    print("\n" + "-" * 80)
    print("PER-WINDOW METRICS COMPARISON")
    print("-" * 80)
    print(f"{'Window':<15} {'AUPRC (Beh)':>12} {'AUPRC (Full)':>13} {'Lift %':>8} "
          f"{'F1 (Beh)':>10} {'F1 (Full)':>11} {'Lift %':>8} "
          f"{'P@100 (Beh)':>12} {'P@100 (Full)':>13} {'Lift %':>8}")
    print("-" * 80)
    
    for _, row in comparison.iterrows():
        print(f"{str(row['test_window_id']):<15} "
              f"{row['auprc_behavioral']:12.4f} {row['auprc_full']:13.4f} {row['auprc_lift_%']:7.1f}% "
              f"{row['f1_score_behavioral']:10.4f} {row['f1_score_full']:11.4f} {row['f1_lift_%']:7.1f}% "
              f"{row['precision_at_100_behavioral']:12.4f} {row['precision_at_100_full']:13.4f} "
              f"{row['precision_at_100_lift_%']:7.1f}%")
    
    # Print aggregate comparison
    print("\n" + "-" * 80)
    print("AGGREGATE METRICS COMPARISON")
    print("-" * 80)
    
    overall_auprc_lift = (
        (full_summary['overall_auprc'] - behavioral_summary['overall_auprc']) / 
        behavioral_summary['overall_auprc'] * 100
    )
    overall_f1_lift = (
        (full_summary['overall_f1'] - behavioral_summary['overall_f1']) / 
        behavioral_summary['overall_f1'] * 100
    )
    overall_prec100_lift = (
        (full_summary['overall_precision_at_100'] - behavioral_summary['overall_precision_at_100']) / 
        behavioral_summary['overall_precision_at_100'] * 100
    )
    
    print(f"\nOverall AUPRC:")
    print(f"  Behavioral-only: {behavioral_summary['overall_auprc']:.4f}")
    print(f"  Full model:      {full_summary['overall_auprc']:.4f}")
    print(f"  Improvement:     {overall_auprc_lift:+.2f}%")
    
    print(f"\nOverall F1-Score:")
    print(f"  Behavioral-only: {behavioral_summary['overall_f1']:.4f}")
    print(f"  Full model:      {full_summary['overall_f1']:.4f}")
    print(f"  Improvement:     {overall_f1_lift:+.2f}%")
    
    print(f"\nOverall Precision@100:")
    print(f"  Behavioral-only: {behavioral_summary['overall_precision_at_100']:.4f}")
    print(f"  Full model:      {full_summary['overall_precision_at_100']:.4f}")
    print(f"  Improvement:     {overall_prec100_lift:+.2f}%")
    
    print(f"\nMean Window AUPRC:")
    print(f"  Behavioral-only: {behavioral_summary['mean_window_auprc']:.4f} ± {behavioral_summary['std_window_auprc']:.4f}")
    print(f"  Full model:      {full_summary['mean_window_auprc']:.4f} ± {full_summary['std_window_auprc']:.4f}")
    
    # Statistical significance (paired t-test)
    from scipy import stats
    t_stat, p_value = stats.ttest_rel(
        comparison['auprc_behavioral'].values,
        comparison['auprc_full'].values
    )
    print(f"\nStatistical Significance (paired t-test on AUPRC):")
    print(f"  t-statistic: {t_stat:.4f}")
    print(f"  p-value:     {p_value:.4e}")
    if p_value < 0.05:
        print(f"  ✓ Topological features provide STATISTICALLY SIGNIFICANT improvement (p < 0.05)")
    else:
        print(f"  ✗ Difference not statistically significant at α=0.05")
    
    return comparison


# ============================================================================
# SHAP INTERPRETABILITY ANALYSIS
# ============================================================================

def generate_shap_analysis(
    model: xgb.XGBClassifier,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    output_dir: Path,
    sample_size: int = 1000
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Generate SHAP analysis to identify which features drive fraud detection.
    
    Args:
        model: Trained XGBoost model
        test_df: Test dataframe with features
        feature_cols: List of feature columns
        output_dir: Directory to save SHAP plots
        sample_size: Number of samples to use for SHAP (to avoid memory issues)
    
    Returns:
        shap_values: SHAP values array
        feature_importance_df: DataFrame with feature importance rankings
    """
    print("\n" + "=" * 80)
    print("SHAP GLOBAL INTERPRETABILITY ANALYSIS")
    print("=" * 80)
    
    # Filter to available features
    available_features = [col for col in feature_cols if col in test_df.columns]
    X_test = test_df[available_features]
    
    # Sample if too large
    if len(X_test) > sample_size:
        print(f"\nSampling {sample_size} records from {len(X_test)} for SHAP analysis (memory optimization)...")
        X_sample = X_test.sample(n=sample_size, random_state=42)
    else:
        X_sample = X_test
    
    print(f"Computing SHAP values for {len(X_sample)} samples with {len(available_features)} features...")
    
    # Create SHAP explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    
    # Compute mean absolute SHAP values for feature importance
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    feature_importance = pd.DataFrame({
        'feature': available_features,
        'mean_abs_shap': mean_abs_shap,
        'feature_type': [
            'Behavioral' if f in BEHAVIORAL_COLS else
            'WeirdNodes' if f in WEIRDNODES_COLS else
            'Topological'
            for f in available_features
        ]
    }).sort_values('mean_abs_shap', ascending=False)
    
    print("\n" + "-" * 80)
    print("TOP 20 MOST IMPORTANT FEATURES (by mean |SHAP|)")
    print("-" * 80)
    print(f"{'Rank':>4} {'Feature':>30} {'Type':>15} {'Mean |SHAP|':>15}")
    print("-" * 80)
    for idx, row in feature_importance.head(20).iterrows():
        print(f"{row.name+1:4d} {row['feature']:>30} {row['feature_type']:>15} {row['mean_abs_shap']:15.6f}")
    
    # Count feature types in top 20
    top20_counts = feature_importance.head(20)['feature_type'].value_counts()
    print("\n" + "-" * 80)
    print("FEATURE TYPE DISTRIBUTION IN TOP 20")
    print("-" * 80)
    for ftype, count in top20_counts.items():
        print(f"  {ftype:>15}: {count:2d} features ({count/20*100:.1f}%)")
    
    # Generate SHAP summary plot (bar chart showing global importance)
    print(f"\nGenerating SHAP summary plot...")
    plt.figure(figsize=(12, 10))
    shap.summary_plot(
        shap_values,
        X_sample,
        feature_names=available_features,
        max_display=25,
        show=False,
        plot_type="bar"
    )
    plt.title("SHAP Feature Importance: Global Impact on Fraud Detection", fontsize=14, fontweight='bold')
    plt.xlabel("Mean |SHAP value| (average impact on model output)", fontsize=11)
    plt.tight_layout()
    
    shap_plot_path = output_dir / "shap_feature_importance.png"
    plt.savefig(shap_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ SHAP plot saved to: {shap_plot_path}")
    
    # Generate SHAP beeswarm plot (showing feature value distribution)
    print(f"\nGenerating SHAP beeswarm plot...")
    plt.figure(figsize=(12, 10))
    shap.summary_plot(
        shap_values,
        X_sample,
        feature_names=available_features,
        max_display=25,
        show=False
    )
    plt.title("SHAP Feature Impact: Value Distribution and Direction", fontsize=14, fontweight='bold')
    plt.xlabel("SHAP value (impact on model prediction)", fontsize=11)
    plt.tight_layout()
    
    beeswarm_path = output_dir / "shap_beeswarm_plot.png"
    plt.savefig(beeswarm_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ SHAP beeswarm plot saved to: {beeswarm_path}")
    
    return shap_values, feature_importance


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_ablation_comparison(
    comparison_df: pd.DataFrame,
    output_dir: Path
):
    """
    Generate bar chart comparing AUPRC of Behavioral-only vs Full model.
    
    Args:
        comparison_df: DataFrame with per-window metrics for both models
        output_dir: Directory to save plot
    """
    print("\n" + "=" * 80)
    print("GENERATING ABLATION STUDY VISUALIZATIONS")
    print("=" * 80)
    
    # AUPRC comparison plot
    fig, ax = plt.subplots(figsize=(14, 6))
    
    x = comparison_df['test_window_id'].values
    width = 0.35
    x_pos = np.arange(len(x))
    
    behavioral_bars = ax.bar(
        x_pos - width/2,
        comparison_df['auprc_behavioral'].values,
        width,
        label='Behavioral-only (Baseline)',
        color='#e74c3c',
        alpha=0.8
    )
    
    full_bars = ax.bar(
        x_pos + width/2,
        comparison_df['auprc_full'].values,
        width,
        label='Full Model (Behavioral + Topological)',
        color='#27ae60',
        alpha=0.8
    )
    
    # Add value labels on bars
    for bars in [behavioral_bars, full_bars]:
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2.,
                height,
                f'{height:.3f}',
                ha='center',
                va='bottom',
                fontsize=8
            )
    
    # Add lift percentage annotations
    for i, row in comparison_df.iterrows():
        lift = row['auprc_lift_%']
        y_pos = max(row['auprc_behavioral'], row['auprc_full']) + 0.01
        ax.text(
            i,
            y_pos,
            f'+{lift:.1f}%',
            ha='center',
            va='bottom',
            fontsize=9,
            fontweight='bold',
            color='#2c3e50'
        )
    
    ax.set_xlabel('Test Window ID', fontsize=12, fontweight='bold')
    ax.set_ylabel('AUPRC (Area Under Precision-Recall Curve)', fontsize=12, fontweight='bold')
    ax.set_title(
        'Ablation Study: Impact of Topological Features on Fraud Detection\n'
        'Forward-Chaining Validation Across Time Windows',
        fontsize=14,
        fontweight='bold'
    )
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x)
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, max(comparison_df['auprc_full'].max(), comparison_df['auprc_behavioral'].max()) * 1.15)
    
    plt.tight_layout()
    ablation_plot_path = output_dir / "ablation_study_results.png"
    plt.savefig(ablation_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Ablation study plot saved to: {ablation_plot_path}")
    
    # Lift percentage plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.bar(
        x_pos,
        comparison_df['auprc_lift_%'].values,
        color='#3498db',
        alpha=0.8
    )
    
    # Add value labels
    for i, lift in enumerate(comparison_df['auprc_lift_%'].values):
        ax.text(
            i,
            lift + 0.5,
            f'+{lift:.1f}%',
            ha='center',
            va='bottom',
            fontsize=10,
            fontweight='bold'
        )
    
    # Add horizontal line at 0
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
    
    ax.set_xlabel('Test Window ID', fontsize=12, fontweight='bold')
    ax.set_ylabel('AUPRC Improvement (%)', fontsize=12, fontweight='bold')
    ax.set_title(
        'Performance Lift from Adding Topological Graph Metrics\n'
        'Percentage Improvement Over Behavioral-Only Baseline',
        fontsize=14,
        fontweight='bold'
    )
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    lift_plot_path = output_dir / "performance_lift.png"
    plt.savefig(lift_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Performance lift plot saved to: {lift_plot_path}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Execute full ablation study and SHAP analysis pipeline."""
    
    # Load feature data
    features_path = DATA_PATH / OUTPUT_FEATURES_FILE
    if not features_path.exists():
        raise FileNotFoundError(
            f"Missing {features_path}. Run 03_extract_features.py first."
        )
    
    print("=" * 80)
    print("ABLATION STUDY AND SHAP INTERPRETABILITY ANALYSIS")
    print("=" * 80)
    print(f"Loading feature matrix from {features_path}...")
    df = pd.read_parquet(features_path)
    
    print(f"\nDataset shape: {df.shape}")
    print(f"Fraud rate: {df['is_fraud'].mean():.4%}")
    
    # Determine window column (window_id or date)
    if 'window_id' in df.columns:
        window_col = 'window_id'
    elif 'date' in df.columns:
        window_col = 'date'
    else:
        raise ValueError("Neither 'window_id' nor 'date' column found in features!")
    
    print(f"Window column: {window_col}")
    print(f"Total windows: {df[window_col].nunique()}")
    
    # Setup output directories
    output_dir = DATA_PATH / "results"
    output_dir.mkdir(exist_ok=True)
    
    notebooks_dir = root_path / "notebooks"
    notebooks_dir.mkdir(exist_ok=True)
    
    # Run ablation study
    behavioral_results, full_results, behavioral_summary, full_summary, final_model, final_test_df = run_ablation_study(df, window_col)
    
    # Compute performance lift
    comparison_df = compute_performance_lift(
        behavioral_results,
        full_results,
        behavioral_summary,
        full_summary
    )
    
    # Save results
    comparison_path = output_dir / "ablation_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)
    print(f"\n✓ Ablation comparison saved to: {comparison_path}")
    
    behavioral_path = output_dir / "behavioral_only_results.csv"
    behavioral_results.to_csv(behavioral_path, index=False)
    print(f"✓ Behavioral-only results saved to: {behavioral_path}")
    
    full_path = output_dir / "full_model_results.csv"
    full_results.to_csv(full_path, index=False)
    print(f"✓ Full model results saved to: {full_path}")
    
    # Generate visualizations
    plot_ablation_comparison(comparison_df, notebooks_dir)
    
    # Generate SHAP analysis (if we have a final model)
    if final_model is not None and final_test_df is not None:
        shap_values, feature_importance = generate_shap_analysis(
            model=final_model,
            test_df=final_test_df,
            feature_cols=FULL_COLS,
            output_dir=notebooks_dir,
            sample_size=1000
        )
        
        # Save feature importance
        importance_path = output_dir / "shap_feature_importance.csv"
        feature_importance.to_csv(importance_path, index=False)
        print(f"\n✓ SHAP feature importance saved to: {importance_path}")
    else:
        print("\nWARNING: No final model available for SHAP analysis")
    
    print("\n" + "=" * 80)
    print("ABLATION STUDY AND SHAP ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nOutputs:")
    print(f"  - Ablation comparison:      {comparison_path}")
    print(f"  - SHAP feature importance:  {importance_path if final_model else 'N/A'}")
    print(f"  - Ablation plot:            {notebooks_dir / 'ablation_study_results.png'}")
    print(f"  - Performance lift plot:    {notebooks_dir / 'performance_lift.png'}")
    print(f"  - SHAP bar plot:            {notebooks_dir / 'shap_feature_importance.png' if final_model else 'N/A'}")
    print(f"  - SHAP beeswarm plot:       {notebooks_dir / 'shap_beeswarm_plot.png' if final_model else 'N/A'}")
    
    print("\n" + "=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)
    print(f"1. Overall AUPRC improvement: {((full_summary['overall_auprc'] - behavioral_summary['overall_auprc']) / behavioral_summary['overall_auprc'] * 100):+.2f}%")
    print(f"2. Test windows evaluated: {len(comparison_df)}")
    print(f"3. Mean window AUPRC lift: {comparison_df['auprc_lift_%'].mean():+.2f}% ± {comparison_df['auprc_lift_%'].std():.2f}%")
    
    if final_model is not None:
        top_feature = feature_importance.iloc[0]
        print(f"4. Most important feature: {top_feature['feature']} ({top_feature['feature_type']})")
        print(f"5. Top feature type in top 20: {feature_importance.head(20)['feature_type'].value_counts().index[0]}")


if __name__ == "__main__":
    main()
