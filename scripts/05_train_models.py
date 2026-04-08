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
    accuracy_score,
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


class Tee:
    """Duplicates stdout to both terminal and a file."""
    def __init__(self, file_path):
        self.terminal = sys.stdout
        self.log = open(file_path, 'w', encoding='utf-8')
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    
    def close(self):
        self.log.close()


BEHAVIORAL_COLS = [
    'vol_sent', 'vol_recv', 'tx_count', 'time_variance', 'flow_ratio',
    'distinct_currencies_sent', 'distinct_currencies_recv',
    'wire_count_sent', 'cash_count_sent', 'bitcoin_count_sent', 'cheque_count_sent',
    'credit_card_count_sent', 'ach_count_sent', 'reinvestment_count_sent',
    'wire_count_recv', 'cash_count_recv', 'bitcoin_count_recv', 'cheque_count_recv',
    'credit_card_count_recv', 'ach_count_recv', 'reinvestment_count_recv'
]

TOPOLOGICAL_COLS = [
    'pr_vol_deep', 'pr_vol_shallow', 'pr_count',
    'leiden_macro_modularity', 'leiden_micro_modularity',
    'betweenness', 'k_core',
    'hits_hub', 'hits_auth',
    
    'leiden_macro_size', 'leiden_micro_size',
    
    'degree', 'in_degree', 'out_degree',
    
    'fan_out_count', 'fan_in_count',
    'cycle_count',
    
    'egonet_node_count', 'egonet_edge_count', 'egonet_density', 'egonet_total_weight',
    
    'local_clustering_coefficient', 'triangle_count',
    
    'average_neighbor_degree', 'successor_avg_volume', 'successor_max_volume',
    
    'temporal_fan_out_count', 
    'temporal_fan_in_count',
]

FULL_COLS = BEHAVIORAL_COLS + TOPOLOGICAL_COLS

K_VALUES = [10, 50, 100, 200, 300, 500]

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





def calibrate_threshold(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    target_recall: float = 0.85
) -> float:
    """
    Find optimal threshold using recall-constrained precision maximization.
    
    In AML, recall is the operational priority. This function finds the highest
    possible threshold (maximizing precision/minimizing false alerts) that 
    strictly maintains the minimum target recall.
    
    Args:
        y_true: Ground truth binary labels
        y_probs: Predicted probabilities
        target_recall: Minimum recall to maintain (default 0.85)
        
    Returns:
        Optimal threshold that maintains target_recall with highest precision
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_probs)
    
    # Find all thresholds that meet or exceed the target recall
    # Note: recalls array is in descending order
    valid_indices = np.where(recalls >= target_recall)[0]
    
    if len(valid_indices) == 0:
        # If target recall cannot be achieved, use lowest threshold
        return float(thresholds[0]) if len(thresholds) > 0 else 0.0
    
    # Among valid thresholds, select the one with highest precision
    # (which corresponds to the highest threshold value)
    best_idx = valid_indices[np.argmax(precisions[valid_indices])]
    optimal_threshold = thresholds[best_idx] if best_idx < len(thresholds) else thresholds[-1]
    
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

        X_inner_train = inner_train_df[feature_cols]
        y_inner_train = inner_train_df[target_col].values
        X_inner_val = inner_val_df[feature_cols]
        y_inner_val = inner_val_df[target_col].values
        X_test = test_df[feature_cols]
        y_test = test_df[target_col].values

        # Calculate dynamic scale_pos_weight for imbalanced data (no undersampling)
        n_neg_inner = np.sum(y_inner_train == 0)
        n_pos_inner = np.sum(y_inner_train == 1)
        scale_pos_weight_inner = n_neg_inner / n_pos_inner if n_pos_inner > 0 else 1.0

        # Free memory immediately after extracting arrays
        del inner_train_df
        gc.collect()

        baseline_params = XGBOOST_PARAMS.copy()
        baseline_params['scale_pos_weight'] = scale_pos_weight_inner
        baseline_model = xgb.XGBClassifier(**baseline_params)
        baseline_model.fit(X_inner_train, y_inner_train)
        val_probs = baseline_model.predict_proba(X_inner_val)[:, 1]
        optimal_threshold = calibrate_threshold(y_inner_val, val_probs)

        # ==========================================================
        # FINAL MODEL TRAINING
        # ==========================================================
        X_train_final = train_df[feature_cols]
        y_train_final = train_df[target_col].values

        # Calculate dynamic scale_pos_weight for full training set
        n_neg_final = np.sum(y_train_final == 0)
        n_pos_final = np.sum(y_train_final == 1)
        scale_pos_weight_final = n_neg_final / n_pos_final if n_pos_final > 0 else 1.0

        # Free memory immediately after extracting arrays
        del train_df
        gc.collect()

        final_params = XGBOOST_PARAMS.copy()
        final_params['scale_pos_weight'] = scale_pos_weight_final
        final_model = xgb.XGBClassifier(**final_params)
        final_model.fit(X_train_final, y_train_final)

        # Predict on the test set
        y_probs = final_model.predict_proba(X_test)[:, 1]
        y_pred = (y_probs >= optimal_threshold).astype(int)

        # Safely free memory
        del baseline_model, X_inner_train, y_inner_train, X_inner_val, y_inner_val
        del inner_val_df, val_probs
        gc.collect()

        del X_train_final, y_train_final
        gc.collect()

        auprc = average_precision_score(y_test, y_probs)
        roc_auc = roc_auc_score(y_test, y_probs)
        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        accuracy = accuracy_score(y_test, y_pred)

        result = {
            "window_id": test_window_id,
            "n_test_samples": len(y_test),
            "n_fraud": int(y_test.sum()),
            "fraud_rate": float(y_test.mean()),
            "auprc": auprc,
            "roc_auc": roc_auc,
            "accuracy": accuracy,
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
            p_at_metrics = ", ".join(
                f"P@{k}={result.get(f'P@{k}', 0):.4f}"
                for k in (100, 200, 300, 500)
            )
            print(
                f"  Window {test_window_id}: AUPRC={auprc:.4f}, "
                f"Acc={accuracy:.4f}, F1={f1:.4f}, "
                f"{p_at_metrics}"
            )

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
        "overall_accuracy": accuracy_score(all_y_true, all_y_pred),
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
        ("Overall Accuracy", "overall_accuracy"),
        ("Overall AUPRC", "overall_auprc"),
        ("Overall ROC-AUC", "overall_roc_auc"),
        ("Overall F1", "overall_f1"),
        ("Overall Precision", "overall_precision"),
        ("Overall Recall", "overall_recall"),
        ("Mean Window AUPRC", "mean_window_auprc"),
        ("P@100", "overall_P@100"),
        ("P@200", "overall_P@200"),
        ("P@300", "overall_P@300"),
        ("P@500", "overall_P@500"),
    ]

    print(f"\n{'Metric':<25} {'Baseline':>12} {'Full Model':>12} {'Lift':>12}")
    print("-" * 65)

    for metric_name, metric_key in metrics:
        baseline_val = baseline_summary.get(metric_key, 0)
        full_val = full_summary.get(metric_key, 0)
        lift_pct = ((full_val - baseline_val) / (baseline_val + 1e-9)) * 100

        print(f"{metric_name:<25} {baseline_val:>12.4f} {full_val:>12.4f} {lift_pct:>+11.2f}%")


def plot_metric_comparison(baseline_summary: Dict, full_summary: Dict, output_dir: Path) -> None:
    """Generate an overall metric comparison plot for baseline vs full models."""
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True, parents=True)

    metric_specs = [
        ("Accuracy", "overall_accuracy"),
        ("Precision", "overall_precision"),
        ("Recall", "overall_recall"),
        ("ROC-AUC", "overall_roc_auc"),
        ("F1", "overall_f1"),
    ]

    baseline_values = [baseline_summary.get(key, 0.0) for _, key in metric_specs]
    full_values = [full_summary.get(key, 0.0) for _, key in metric_specs]
    x = np.arange(len(metric_specs))
    width = 0.36

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))

    ax.bar(x - width / 2, baseline_values, width, label='Baseline', color='#e74c3c', alpha=0.85)
    ax.bar(x + width / 2, full_values, width, label='Full Model', color='#27ae60', alpha=0.85)

    for idx, (baseline_val, full_val) in enumerate(zip(baseline_values, full_values)):
        ax.text(idx - width / 2, baseline_val + 0.01, f"{baseline_val:.3f}", ha='center', va='bottom', fontsize=9)
        ax.text(idx + width / 2, full_val + 0.01, f"{full_val:.3f}", ha='center', va='bottom', fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([name for name, _ in metric_specs], rotation=20, ha='right')
    ax.set_ylim(0, min(1.05, max(baseline_values + full_values) * 1.2 if baseline_values or full_values else 1.0))
    ax.set_ylabel('Score', fontsize=11, fontweight='bold')
    ax.set_title('Baseline vs Full Model: Core Classification Metrics', fontsize=13, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    output_path = plots_dir / "model_metric_comparison.png"
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    print(f"\n✓ Metric comparison plot saved to: {output_path}")


def plot_precision_at_k(baseline_summary: Dict, full_summary: Dict, output_dir: Path) -> None:
    """Generate precision@k comparison plot for baseline vs full models."""
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True, parents=True)

    metric_specs = [
        ("P@10", "overall_P@10"),
        ("P@50", "overall_P@50"),
        ("P@100", "overall_P@100"),
        ("P@200", "overall_P@200"),
        ("P@300", "overall_P@300"),
        ("P@500", "overall_P@500"),
    ]

    baseline_values = [baseline_summary.get(key, 0.0) for _, key in metric_specs]
    full_values = [full_summary.get(key, 0.0) for _, key in metric_specs]
    x = np.arange(len(metric_specs))
    width = 0.36

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(13, 7))

    ax.bar(x - width / 2, baseline_values, width, label='Baseline', color='#e74c3c', alpha=0.85)
    ax.bar(x + width / 2, full_values, width, label='Full Model', color='#27ae60', alpha=0.85)

    for idx, (baseline_val, full_val) in enumerate(zip(baseline_values, full_values)):
        ax.text(idx - width / 2, baseline_val + 0.01, f"{baseline_val:.3f}", ha='center', va='bottom', fontsize=9)
        ax.text(idx + width / 2, full_val + 0.01, f"{full_val:.3f}", ha='center', va='bottom', fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([name for name, _ in metric_specs], rotation=15, ha='right')
    ax.set_ylim(0, min(1.05, max(baseline_values + full_values) * 1.2 if baseline_values or full_values else 1.0))
    ax.set_ylabel('Precision@K', fontsize=11, fontweight='bold')
    ax.set_title('Baseline vs Full Model: Precision@K Metrics', fontsize=13, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    output_path = plots_dir / "precision_at_k_comparison.png"
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    print(f"\n✓ Precision@K comparison plot saved to: {output_path}")


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
    
    # Initialize logging to file
    log_file = results_dir / "05_train_models_log.txt"
    tee = Tee(log_file)
    original_stdout = sys.stdout
    sys.stdout = tee
    
    print(f"\n[Logging initialized - output saved to {log_file}]")

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
    plot_metric_comparison(baseline_summary, full_summary, results_dir)
    plot_precision_at_k(baseline_summary, full_summary, results_dir)

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
    print("  • plots/model_metric_comparison.png")
    print("  • plots/precision_at_k_comparison.png")

    print("\n→ Proceed to Phase 3: 06_ablation_study.py")
    
    # Close logging
    sys.stdout = original_stdout
    tee.close()


if __name__ == "__main__":
    main()
