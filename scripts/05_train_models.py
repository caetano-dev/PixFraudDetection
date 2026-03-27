"""
Phase 2: Model Training
=======================
TCC Pipeline Script 05

Purpose: Generate the unoptimized baseline and full models to isolate the raw
         information gain of the topological features.

Logic: Execute forward-chaining cross-validation. Train and evaluate:
       - Unpruned Baseline (behavioral features only)
       - Unpruned Full Model (behavioral + topological features)

Outputs:
    - data/results/baseline_predictions.parquet
    - data/results/full_predictions.parquet
    - data/results/shap_feature_importance.csv
    - data/results/raw_shap_values.npy
    - data/results/plots/global_shap_beeswarm.png

Hardware Constraint: 8GB RAM - aggressive memory management required.
"""

import gc
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

root_path = Path(__file__).resolve().parent.parent
sys.path.append(str(root_path))

from src.config import OUTPUT_FEATURES_FILE, DATA_PATH


BEHAVIORAL_COLS = [
    'vol_sent', 'vol_recv', 'tx_count', 'time_variance', 'flow_ratio',
    'distinct_currencies_sent', 'distinct_currencies_recv',
    'wire_count_sent', 'cash_count_sent', 'bitcoin_count_sent', 'cheque_count_sent',
    'credit_card_count_sent', 'ach_count_sent', 'reinvestment_count_sent',
    'wire_count_recv', 'cash_count_recv', 'bitcoin_count_recv', 'cheque_count_recv',
    'credit_card_count_recv', 'ach_count_recv', 'reinvestment_count_recv'
]

TOPOLOGICAL_COLS = [
    'pr_vol_deep', 'pr_vol_shallow', 'pr_count', 'hits_hub', 'hits_auth',
    'leiden_macro_size', 'leiden_macro_modularity', 'leiden_micro_size',
    'leiden_micro_modularity', 'betweenness', 'k_core', 'degree', 'in_degree',
    'out_degree', 'fan_out_count', 'fan_in_count', 'scatter_gather_count',
    'gather_scatter_count', 'cycle_count',
    'egonet_node_count', 'egonet_edge_count', 'egonet_density', 'egonet_total_weight',
    'local_clustering_coefficient', 'triangle_count',
    'average_neighbor_degree', 'successor_avg_volume', 'successor_max_volume'
]

FULL_COLS = BEHAVIORAL_COLS + TOPOLOGICAL_COLS

K_VALUES = [10, 50, 100, 500]

XGBOOST_PARAMS = {
    'n_estimators': 200,
    'max_depth': 6,
    'learning_rate': 0.1,
    'min_child_weight': 3,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'gamma': 0.1,
    'objective': 'binary:logistic',
    'eval_metric': 'aucpr',
    'tree_method': 'hist',
    'random_state': 42,
    'n_jobs': -1
}

MIN_TRAIN_WINDOWS = 2
UNDERSAMPLE_RATIO = 20
SHAP_SAMPLE_SIZE = 1000


def compute_precision_at_k(y_true: np.ndarray, y_probs: np.ndarray, k: int) -> float:
    """Compute Precision@K."""
    if k > len(y_true):
        k = len(y_true)
    sorted_indices = np.argsort(y_probs)[::-1]
    top_k_labels = y_true[sorted_indices[:k]]
    return float(np.sum(top_k_labels)) / k


def undersample_data(
    data: pd.DataFrame,
    target_col: str = "is_fraud",
    ratio: int = UNDERSAMPLE_RATIO
) -> pd.DataFrame:
    """Enforce strict 1:N asymmetric undersampling for class balance."""
    fraud = data[data[target_col] == 1]
    normal = data[data[target_col] == 0]
    target_count = len(fraud) * ratio

    if len(normal) > target_count:
        normal = normal.sample(n=target_count, random_state=42)

    result = pd.concat([fraud, normal]).sample(frac=1, random_state=42)
    return result


def calibrate_threshold(
    y_true: np.ndarray,
    y_probs: np.ndarray
) -> float:
    """Find optimal threshold using F1 maximization on validation set."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_probs)
    f1_scores = np.divide(
        2 * precisions * recalls,
        precisions + recalls,
        out=np.zeros_like(precisions),
        where=(precisions + recalls) != 0
    )
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
    return float(optimal_threshold)


def forward_chaining_validation(
    df: pd.DataFrame,
    feature_cols: List[str],
    window_col: str = "window_id",
    target_col: str = "is_fraud",
    model_name: str = "model",
    collect_shap: bool = False,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, Dict, Optional[np.ndarray], Optional[pd.DataFrame]]:
    """
    Execute forward-chaining cross-validation with strict temporal ordering.
    
    This prevents structural leakage by ensuring:
    - Training only uses data from windows strictly BEFORE the test window
    - No future information leaks into the training process
    
    Args:
        df: Features DataFrame with window_col for temporal ordering
        feature_cols: List of feature column names to use
        window_col: Column name for temporal window identifier
        target_col: Binary target column name
        model_name: Name for logging
        collect_shap: Whether to collect SHAP values
        verbose: Print progress
    
    Returns:
        - results_df: Per-window metrics
        - summary: Aggregate metrics
        - stacked_shap: Stacked SHAP values (if collect_shap=True)
        - shap_features_df: Feature values for SHAP (if collect_shap=True)
    """
    df = df.sort_values(by=window_col).reset_index(drop=True)
    unique_windows = sorted(df[window_col].unique())
    feature_cols = [col for col in feature_cols if col in df.columns]

    window_results = []
    all_predictions = []
    
    global_shap_values = []
    global_x_test = []

    if verbose:
        print(f"\n[{model_name}] Forward-chaining CV with {len(feature_cols)} features")
        print(f"[{model_name}] Windows: {len(unique_windows)}, Test windows: {len(unique_windows) - MIN_TRAIN_WINDOWS}")

    for test_window_idx in range(MIN_TRAIN_WINDOWS, len(unique_windows)):
        test_window_id = unique_windows[test_window_idx]
        train_window_ids = unique_windows[:test_window_idx]

        train_df = df[df[window_col].isin(train_window_ids)].copy()
        test_df = df[df[window_col] == test_window_id].copy()

        if len(train_df[target_col].unique()) < 2 or len(test_df[target_col].unique()) < 2:
            if verbose:
                print(f"  Window {test_window_id}: Skipped (insufficient class variance)")
            del train_df, test_df
            gc.collect()
            continue

        split_idx = int(len(train_df) * 0.8)
        inner_train_df = train_df.iloc[:split_idx]
        inner_val_df = train_df.iloc[split_idx:]

        inner_train_df = undersample_data(inner_train_df, target_col)

        X_inner_train = inner_train_df[feature_cols]
        y_inner_train = inner_train_df[target_col].values
        X_inner_val = inner_val_df[feature_cols]
        y_inner_val = inner_val_df[target_col].values
        X_test = test_df[feature_cols]
        y_test = test_df[target_col].values

        calib_model = xgb.XGBClassifier(**XGBOOST_PARAMS)
        calib_model.fit(X_inner_train, y_inner_train)
        val_probs = calib_model.predict_proba(X_inner_val)[:, 1]
        optimal_threshold = calibrate_threshold(y_inner_val, val_probs)

        del calib_model, X_inner_train, y_inner_train, X_inner_val, y_inner_val
        del inner_train_df, inner_val_df, val_probs
        gc.collect()

        final_train_df = undersample_data(train_df, target_col)
        X_train_final = final_train_df[feature_cols]
        y_train_final = final_train_df[target_col].values

        final_model = xgb.XGBClassifier(**XGBOOST_PARAMS)
        final_model.fit(X_train_final, y_train_final)

        del X_train_final, y_train_final, final_train_df, train_df
        gc.collect()

        y_probs = final_model.predict_proba(X_test)[:, 1]
        y_pred = (y_probs >= optimal_threshold).astype(int)

        auprc = average_precision_score(y_test, y_probs)
        roc_auc = roc_auc_score(y_test, y_probs)
        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)

        result = {
            "window_id": test_window_id,
            "n_test_samples": len(y_test),
            "n_fraud": int(y_test.sum()),
            "fraud_rate": float(y_test.mean()),
            "auprc": auprc,
            "roc_auc": roc_auc,
            "f1_score": f1,
            "precision": precision,
            "recall": recall,
            "optimal_threshold": optimal_threshold,
        }

        for k in K_VALUES:
            result[f"P@{k}"] = compute_precision_at_k(y_test, y_probs, k)

        window_results.append(result)

        for i, (entity_id, prob, pred, true_label) in enumerate(
            zip(test_df['entity_id'].values, y_probs, y_pred, y_test)
        ):
            all_predictions.append({
                'window_id': test_window_id,
                'entity_id': entity_id,
                'y_true': int(true_label),
                'y_prob': float(prob),
                'y_pred': int(pred),
                'threshold': optimal_threshold
            })

        if verbose:
            print(f"  Window {test_window_id}: AUPRC={auprc:.4f}, F1={f1:.4f}, "
                  f"P@100={result.get('P@100', 0):.4f}")

        if collect_shap:
            explainer = shap.TreeExplainer(final_model)
            sample_size = min(SHAP_SAMPLE_SIZE, len(X_test))
            X_test_sampled = X_test.sample(sample_size, random_state=42)
            shap_values = explainer.shap_values(X_test_sampled)
            global_shap_values.append(shap_values)
            global_x_test.append(X_test_sampled)
            
            del explainer, shap_values
            gc.collect()

        del final_model, X_test, y_test, y_probs, y_pred, test_df
        gc.collect()

    results_df = pd.DataFrame(window_results)
    predictions_df = pd.DataFrame(all_predictions)

    all_y_true = predictions_df['y_true'].values
    all_y_probs = predictions_df['y_prob'].values
    all_y_pred = predictions_df['y_pred'].values

    summary = {
        "model_name": model_name,
        "n_features": len(feature_cols),
        "n_test_windows": len(window_results),
        "total_test_samples": len(all_y_true),
        "overall_fraud_rate": float(np.mean(all_y_true)),
        "overall_auprc": average_precision_score(all_y_true, all_y_probs),
        "overall_roc_auc": roc_auc_score(all_y_true, all_y_probs),
        "overall_f1": f1_score(all_y_true, all_y_pred),
        "overall_precision": precision_score(all_y_true, all_y_pred, zero_division=0),
        "overall_recall": recall_score(all_y_true, all_y_pred, zero_division=0),
        "mean_window_auprc": results_df["auprc"].mean(),
        "std_window_auprc": results_df["auprc"].std(),
    }

    for k in K_VALUES:
        summary[f"overall_P@{k}"] = compute_precision_at_k(all_y_true, all_y_probs, k)

    stacked_shap = None
    shap_features_df = None
    
    if collect_shap and global_shap_values:
        stacked_shap = np.vstack(global_shap_values)
        shap_features_df = pd.concat(global_x_test, axis=0)
        
        del global_shap_values, global_x_test
        gc.collect()

    return results_df, summary, predictions_df, stacked_shap, shap_features_df


def generate_shap_analysis(
    stacked_shap: np.ndarray,
    shap_features_df: pd.DataFrame,
    output_dir: Path
) -> pd.DataFrame:
    """
    Generate SHAP feature importance analysis and visualization.
    
    Creates:
    - Feature importance CSV ranked by mean absolute SHAP
    - Global beeswarm plot for publication
    """
    print("\n" + "-" * 80)
    print("GENERATING SHAP ANALYSIS")
    print("-" * 80)

    feature_names = shap_features_df.columns.tolist()

    mean_abs_shap = np.abs(stacked_shap).mean(axis=0)
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'mean_abs_shap': mean_abs_shap
    }).sort_values('mean_abs_shap', ascending=False)

    print("\nGlobal SHAP Feature Importance:")
    print(feature_importance.to_string(index=False))

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True, parents=True)

    plt.figure(figsize=(12, 10))
    shap.summary_plot(
        stacked_shap,
        shap_features_df,
        feature_names=feature_names,
        show=False,
        max_display=30
    )
    plt.title("Global SHAP Feature Impact (Full Model, All Windows)",
              fontsize=14, fontweight='bold')
    plt.tight_layout()

    plot_path = plots_dir / "global_shap_beeswarm.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"\n✓ SHAP beeswarm plot saved to: {plot_path}")

    return feature_importance


def print_summary_comparison(baseline_summary: Dict, full_summary: Dict) -> None:
    """Print a formatted comparison of baseline vs full model."""
    print("\n" + "=" * 80)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 80)

    metrics = [
        ("Overall AUPRC", "overall_auprc"),
        ("Overall ROC-AUC", "overall_roc_auc"),
        ("Overall F1", "overall_f1"),
        ("Overall Precision", "overall_precision"),
        ("Overall Recall", "overall_recall"),
        ("Mean Window AUPRC", "mean_window_auprc"),
        ("P@100", "overall_P@100"),
    ]

    print(f"\n{'Metric':<25} {'Baseline':>12} {'Full Model':>12} {'Lift':>12}")
    print("-" * 65)

    for metric_name, metric_key in metrics:
        baseline_val = baseline_summary.get(metric_key, 0)
        full_val = full_summary.get(metric_key, 0)
        lift_pct = ((full_val - baseline_val) / (baseline_val + 1e-9)) * 100

        print(f"{metric_name:<25} {baseline_val:>12.4f} {full_val:>12.4f} {lift_pct:>+11.2f}%")


def main():
    """
    Phase 2 Main Execution: Model Training Pipeline
    """
    print("=" * 80)
    print("TCC PIPELINE - PHASE 2: MODEL TRAINING")
    print("=" * 80)
    print("Purpose: Generate baseline and full models for ablation comparison")
    print("Hardware Constraint: 8GB RAM")

    features_path = DATA_PATH / OUTPUT_FEATURES_FILE
    if not features_path.exists():
        raise FileNotFoundError(
            f"Features file not found at {features_path}. "
            "Run 03_extract_features.py first."
        )

    print(f"\nLoading features from: {features_path}")
    df = pd.read_parquet(features_path)

    window_col = 'window_id' if 'window_id' in df.columns else 'date'
    target_col = 'is_fraud'

    print(f"Dataset shape: {df.shape}")
    print(f"Window column: {window_col}")
    print(f"Fraud rate: {df[target_col].mean():.4%}")

    results_dir = DATA_PATH / "results"
    results_dir.mkdir(exist_ok=True, parents=True)

    print("\n" + "=" * 80)
    print("PHASE 2A: TRAINING BASELINE MODEL (Behavioral Features Only)")
    print(f"Features: {len(BEHAVIORAL_COLS)}")
    print("=" * 80)

    baseline_results, baseline_summary, baseline_preds, _, _ = forward_chaining_validation(
        df=df,
        feature_cols=BEHAVIORAL_COLS,
        window_col=window_col,
        target_col=target_col,
        model_name="Baseline",
        collect_shap=False,
        verbose=True
    )

    baseline_preds_path = results_dir / "baseline_predictions.parquet"
    baseline_preds.to_parquet(baseline_preds_path, index=False)
    print(f"\n✓ Baseline predictions saved to: {baseline_preds_path}")

    baseline_results_path = results_dir / "baseline_window_results.csv"
    baseline_results.to_csv(baseline_results_path, index=False)

    del baseline_results, baseline_preds
    gc.collect()

    print("\n" + "=" * 80)
    print("PHASE 2B: TRAINING FULL MODEL (Behavioral + Topological Features)")
    print(f"Features: {len(FULL_COLS)}")
    print("=" * 80)

    full_results, full_summary, full_preds, stacked_shap, shap_features_df = forward_chaining_validation(
        df=df,
        feature_cols=FULL_COLS,
        window_col=window_col,
        target_col=target_col,
        model_name="Full Model",
        collect_shap=True,
        verbose=True
    )

    full_preds_path = results_dir / "full_predictions.parquet"
    full_preds.to_parquet(full_preds_path, index=False)
    print(f"\n✓ Full model predictions saved to: {full_preds_path}")

    full_results_path = results_dir / "full_window_results.csv"
    full_results.to_csv(full_results_path, index=False)

    del full_results, full_preds, df
    gc.collect()

    print_summary_comparison(baseline_summary, full_summary)

    if stacked_shap is not None and shap_features_df is not None:
        feature_importance = generate_shap_analysis(
            stacked_shap=stacked_shap,
            shap_features_df=shap_features_df,
            output_dir=results_dir
        )

        shap_csv_path = results_dir / "shap_feature_importance.csv"
        feature_importance.to_csv(shap_csv_path, index=False)
        print(f"\n✓ SHAP feature importance saved to: {shap_csv_path}")

        shap_npy_path = results_dir / "raw_shap_values.npy"
        np.save(shap_npy_path, stacked_shap)
        print(f"✓ Raw SHAP values saved to: {shap_npy_path}")

        del stacked_shap, shap_features_df, feature_importance
        gc.collect()

    summary_data = {
        'baseline': baseline_summary,
        'full_model': full_summary
    }
    summary_df = pd.DataFrame([baseline_summary, full_summary])
    summary_df.to_csv(results_dir / "model_summaries.csv", index=False)

    print("\n" + "=" * 80)
    print("PHASE 2 COMPLETE")
    print("=" * 80)
    print(f"\nOutputs saved to: {results_dir}")
    print("  • baseline_predictions.parquet")
    print("  • full_predictions.parquet")
    print("  • shap_feature_importance.csv")
    print("  • raw_shap_values.npy")
    print("  • plots/global_shap_beeswarm.png")

    print("\n→ Proceed to Phase 3: 06_ablation_study.py")


if __name__ == "__main__":
    main()
