"""
Phase 4: Feature Pruning
========================
TCC Pipeline Script 07

Purpose: Identify the most efficient subset of features for operational deployment.

Logic: Apply the SHAP-RFE (Recursive Feature Elimination) loop strictly to the
       Unpruned Full Model. Filter out collinear features (Spearman > 0.80),
       protect temporally stable features, and iteratively drop the lowest SHAP
       contributors until absolute ΔAUPRC < -0.01 from the baseline.

Outputs:
    - data/results/pruned_features.json
    - data/results/rfe_history.csv
    - data/results/plots/rfe_degradation_trajectory.png
    - data/results/plots/rfe_metric_trajectories.png

Hardware Constraint: 8GB RAM - aggressive memory management required.
"""

import gc
import json
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import xgboost as xgb
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score, roc_auc_score

root_path = Path(__file__).resolve().parent.parent
sys.path.append(str(root_path))

from src.config import OUTPUT_FEATURES_FILE, DATA_PATH


CORRELATION_THRESHOLD = 0.80
DROP_PERCENTAGE = 10
DELTA_AUPRC_THRESHOLD = -0.01
TEMPORAL_PROTECTION_PERCENTILE = 60
MAX_ITERATIONS = 25
MIN_FEATURES = 5
SHAP_SAMPLE_SIZE = 800
MIN_TRAIN_WINDOWS = 2
UNDERSAMPLE_RATIO = 20
CLASSIFICATION_THRESHOLD = 0.5

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


def undersample_data(
    data: pd.DataFrame,
    target_col: str = "is_fraud",
    ratio: int = UNDERSAMPLE_RATIO
) -> pd.DataFrame:
    """Enforce strict 1:N asymmetric undersampling."""
    fraud = data[data[target_col] == 1]
    normal = data[data[target_col] == 0]
    target_count = len(fraud) * ratio

    if len(normal) > target_count:
        normal = normal.sample(n=target_count, random_state=42)

    result = pd.concat([fraud, normal]).sample(frac=1, random_state=42)
    return result


def compute_collinearity_filter(
    df: pd.DataFrame,
    feature_cols: List[str],
    shap_importance: Dict[str, float],
    threshold: float = CORRELATION_THRESHOLD
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
    print(f"Threshold: Spearman |ρ| > {threshold}")
    print("=" * 80)
    
    available_features = [f for f in feature_cols if f in df.columns]
    print(f"Computing Spearman correlation for {len(available_features)} features...")
    
    sample_size = min(10000, len(df))
    df_sample = df[available_features].sample(n=sample_size, random_state=42)
    
    corr_matrix = df_sample.corr(method='spearman').abs()
    
    del df_sample
    gc.collect()
    
    dropped_features = set()
    dropped_pairs = []
    
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            feat1 = corr_matrix.columns[i]
            feat2 = corr_matrix.columns[j]
            correlation = corr_matrix.iloc[i, j]
            
            if correlation > threshold and feat1 not in dropped_features and feat2 not in dropped_features:
                shap1 = shap_importance.get(feat1, 0)
                shap2 = shap_importance.get(feat2, 0)
                
                to_drop = feat1 if shap1 < shap2 else feat2
                
                dropped_features.add(to_drop)
                dropped_pairs.append((feat1, feat2, float(correlation)))
                print(f"  Dropping '{to_drop}' (SHAP={shap_importance.get(to_drop, 0):.4f}) - "
                      f"correlated with '{feat1 if to_drop == feat2 else feat2}' (ρ={correlation:.3f})")
    
    del corr_matrix
    gc.collect()
    
    retained_features = [f for f in available_features if f not in dropped_features]
    
    print(f"\n[COLLINEARITY FILTER] Dropped {len(dropped_features)} features")
    print(f"[COLLINEARITY FILTER] Retained {len(retained_features)} features")
    
    return retained_features, dropped_pairs


def compute_classification_metrics(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    threshold: float = CLASSIFICATION_THRESHOLD
) -> Dict[str, float]:
    """Compute thresholded and threshold-free classification metrics."""
    y_pred = (y_probs >= threshold).astype(int)
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_true, y_probs),
        'auprc': average_precision_score(y_true, y_probs),
    }


def evaluate_feature_set(
    df: pd.DataFrame,
    feature_cols: List[str],
    window_col: str,
    target_col: str
) -> Tuple[Dict[str, float], pd.DataFrame, Dict[str, float]]:
    """
    Run forward-chaining validation and compute SHAP importance.
    
    Returns:
        - overall_metrics: Dict of aggregate metrics
        - window_results: DataFrame with per-window metrics
        - shap_importance: Dict mapping feature -> mean |SHAP|
    """
    df = df.sort_values(by=window_col).reset_index(drop=True)
    unique_windows = sorted(df[window_col].unique())
    feature_cols = [col for col in feature_cols if col in df.columns]
    
    window_results = []
    all_y_true = []
    all_y_probs = []
    all_y_pred = []
    
    global_shap_values = []
    
    for test_window_idx in range(MIN_TRAIN_WINDOWS, len(unique_windows)):
        test_window_id = unique_windows[test_window_idx]
        train_window_ids = unique_windows[:test_window_idx]
        
        train_df = df[df[window_col].isin(train_window_ids)].copy()
        test_df = df[df[window_col] == test_window_id].copy()
        
        if len(train_df[target_col].unique()) < 2 or len(test_df[target_col].unique()) < 2:
            del train_df, test_df
            gc.collect()
            continue
        
        final_train_df = undersample_data(train_df, target_col)
        X_train = final_train_df[feature_cols]
        y_train = final_train_df[target_col].values
        X_test = test_df[feature_cols]
        y_test = test_df[target_col].values
        
        del train_df, final_train_df
        gc.collect()
        
        model = xgb.XGBClassifier(**XGBOOST_PARAMS)
        model.fit(X_train, y_train)
        
        del X_train, y_train
        gc.collect()
        
        y_probs = model.predict_proba(X_test)[:, 1]
        y_pred = (y_probs >= CLASSIFICATION_THRESHOLD).astype(int)
        window_metrics = compute_classification_metrics(y_test, y_probs)
        
        window_results.append({
            'window_id': test_window_id,
            'accuracy': window_metrics['accuracy'],
            'precision': window_metrics['precision'],
            'recall': window_metrics['recall'],
            'f1_score': window_metrics['f1_score'],
            'auprc': window_metrics['auprc'],
            'roc_auc': window_metrics['roc_auc'],
            'n_samples': len(y_test),
            'n_fraud': int(y_test.sum()),
        })
        
        all_y_true.extend(y_test)
        all_y_probs.extend(y_probs)
        all_y_pred.extend(y_pred)
        
        explainer = shap.TreeExplainer(model)
        sample_size = min(SHAP_SAMPLE_SIZE, len(X_test))
        X_test_sampled = X_test.sample(sample_size, random_state=42)
        shap_values = explainer.shap_values(X_test_sampled)
        global_shap_values.append(shap_values)
        
        del model, explainer, X_test, y_test, y_probs, test_df, shap_values, X_test_sampled
        gc.collect()
    
    results_df = pd.DataFrame(window_results)
    
    all_y_true = np.array(all_y_true)
    all_y_probs = np.array(all_y_probs)
    all_y_pred = np.array(all_y_pred)
    overall_metrics = compute_classification_metrics(all_y_true, all_y_probs)
    overall_metrics.update({
        'overall_accuracy': accuracy_score(all_y_true, all_y_pred),
        'overall_precision': precision_score(all_y_true, all_y_pred, zero_division=0),
        'overall_recall': recall_score(all_y_true, all_y_pred, zero_division=0),
        'overall_f1_score': f1_score(all_y_true, all_y_pred, zero_division=0),
        'overall_roc_auc': roc_auc_score(all_y_true, all_y_probs),
        'overall_auprc': average_precision_score(all_y_true, all_y_probs),
        'n_test_windows': len(window_results),
        'total_test_samples': len(all_y_true),
        'mean_window_auprc': results_df['auprc'].mean() if not results_df.empty else 0.0,
        'std_window_auprc': results_df['auprc'].std() if not results_df.empty else 0.0,
    })
    
    if global_shap_values:
        stacked_shap = np.vstack(global_shap_values)
        mean_abs_shap = np.abs(stacked_shap).mean(axis=0)
        shap_importance = dict(zip(feature_cols, mean_abs_shap))
        
        del stacked_shap, global_shap_values
        gc.collect()
    else:
        shap_importance = {f: 0.0 for f in feature_cols}
    
    return overall_metrics, results_df, shap_importance


def identify_protected_features(
    df: pd.DataFrame,
    feature_cols: List[str],
    shap_importance: Dict[str, float],
    percentile: float = TEMPORAL_PROTECTION_PERCENTILE
) -> Set[str]:
    """
    Identify features that should be protected from immediate elimination.
    
    Protection criteria:
    - Top percentile features by global SHAP importance
    - Features that are behavioral (core transactional features)
    """
    print("\n" + "=" * 80)
    print("STEP 2: TEMPORAL STABILITY PROTECTION")
    print(f"Protecting top {percentile}th percentile features by SHAP importance")
    print("=" * 80)
    
    shap_values = [shap_importance.get(f, 0) for f in feature_cols]
    threshold = np.percentile(shap_values, percentile)
    
    protected_features = set()
    
    for feat in feature_cols:
        feat_shap = shap_importance.get(feat, 0)
        if feat_shap >= threshold:
            protected_features.add(feat)
    
    for feat in BEHAVIORAL_COLS:
        if feat in feature_cols and shap_importance.get(feat, 0) > 0:
            protected_features.add(feat)
    
    print(f"\n[PROTECTION] {len(protected_features)} features protected:")
    protected_list = sorted(protected_features, key=lambda f: -shap_importance.get(f, 0))
    for feat in protected_list[:10]:
        print(f"  • {feat}: SHAP={shap_importance.get(feat, 0):.4f}")
    if len(protected_features) > 10:
        print(f"  ... and {len(protected_features) - 10} more")
    
    return protected_features


def run_shap_rfe_loop(
    df: pd.DataFrame,
    initial_features: List[str],
    protected_features: Set[str],
    initial_shap: Dict[str, float],
    window_col: str,
    target_col: str,
    baseline_metrics: Dict[str, float]
) -> Tuple[List[str], List[Dict], int]:
    """
    Iteratively eliminate lowest-performing features until AUPRC degrades.
    
    Returns:
        - Final retained feature list
        - Iteration history
        - Best iteration index
    """
    print("\n" + "=" * 80)
    print("STEP 3: ITERATIVE SHAP-RFE LOOP")
    baseline_auprc = baseline_metrics['overall_auprc']
    print(f"Stopping criterion: ΔAUPRC < {DELTA_AUPRC_THRESHOLD} from baseline ({baseline_auprc:.4f})")
    print("=" * 80)
    
    current_features = initial_features.copy()
    current_shap = initial_shap.copy()
    iteration_history = []
    
    iteration_history.append({
        'iteration': 0,
        'n_features': len(current_features),
        **baseline_metrics,
        'delta_from_baseline': 0.0,
        'features_dropped': [],
        'selected': False
    })
    
    print(
        f"\n[ITERATION 0] Baseline: {len(current_features)} features, "
        f"AUPRC={baseline_auprc:.4f}, Acc={baseline_metrics['overall_accuracy']:.4f}, "
        f"F1={baseline_metrics['overall_f1_score']:.4f}"
    )
    
    best_iteration = 0
    best_auprc = baseline_auprc
    
    for iteration in range(1, MAX_ITERATIONS + 1):
        droppable_features = [f for f in current_features if f not in protected_features]
        
        if len(droppable_features) == 0:
            print(f"\n[STOPPING] All remaining features are protected")
            break
        
        if len(current_features) <= MIN_FEATURES:
            print(f"\n[STOPPING] Minimum feature count ({MIN_FEATURES}) reached")
            break
        
        n_to_drop = max(1, int(len(droppable_features) * DROP_PERCENTAGE / 100))
        n_to_drop = min(n_to_drop, len(current_features) - MIN_FEATURES)
        
        droppable_sorted = sorted(droppable_features, key=lambda f: current_shap.get(f, 0))
        features_to_drop = droppable_sorted[:n_to_drop]
        
        print(f"\n[ITERATION {iteration}] Dropping {n_to_drop} lowest SHAP features:")
        for feat in features_to_drop:
            print(f"  - {feat} (SHAP={current_shap.get(feat, 0):.4f})")
        
        candidate_features = [f for f in current_features if f not in features_to_drop]
        
        print(f"  Retraining with {len(candidate_features)} features...")
        new_metrics, window_results, new_shap = evaluate_feature_set(
            df=df,
            feature_cols=candidate_features,
            window_col=window_col,
            target_col=target_col
        )
        
        delta_from_baseline = new_metrics['overall_auprc'] - baseline_auprc
        
        print(
            f"  AUPRC: {new_metrics['overall_auprc']:.4f} (Δ from baseline: {delta_from_baseline:+.4f})"
        )
        print(
            f"  Acc={new_metrics['overall_accuracy']:.4f}, "
            f"Prec={new_metrics['overall_precision']:.4f}, "
            f"Recall={new_metrics['overall_recall']:.4f}, "
            f"F1={new_metrics['overall_f1_score']:.4f}, "
            f"ROC-AUC={new_metrics['overall_roc_auc']:.4f}"
        )
        
        iteration_history.append({
            'iteration': iteration,
            'n_features': len(candidate_features),
            **new_metrics,
            'delta_from_baseline': delta_from_baseline,
            'features_dropped': features_to_drop,
            'selected': False
        })
        
        if delta_from_baseline < DELTA_AUPRC_THRESHOLD:
            print(f"\n[STOPPING] ΔAUPRC ({delta_from_baseline:+.4f}) < threshold ({DELTA_AUPRC_THRESHOLD})")
            print(f"[STOPPING] Reverting to iteration {iteration - 1}")
            break
        
        if new_metrics['overall_auprc'] >= best_auprc:
            best_auprc = new_metrics['overall_auprc']
            best_iteration = iteration
        
        current_features = candidate_features
        current_shap = new_shap
        
        del window_results
        gc.collect()
    
    iteration_history[best_iteration]['selected'] = True
    
    final_features = iteration_history[best_iteration].get('features', None)
    if final_features is None:
        all_dropped = []
        for i in range(1, best_iteration + 1):
            all_dropped.extend(iteration_history[i]['features_dropped'])
        final_features = [f for f in initial_features if f not in all_dropped]
    
    return final_features, iteration_history, best_iteration


def plot_rfe_trajectory(
    iteration_history: List[Dict],
    best_iteration: int,
    baseline_auprc: float,
    output_dir: Path
) -> None:
    """
    Generate RFE degradation trajectory visualization.
    
    Shows:
    - Number of features (X-axis) vs Overall AUPRC (Y-axis)
    - Highlights the selected iteration
    - Shows degradation threshold
    """
    print("\n" + "-" * 80)
    print("GENERATING RFE TRAJECTORY PLOT")
    print("-" * 80)
    
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True, parents=True)
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))
    
    n_features = [h['n_features'] for h in iteration_history]
    auprc_values = [h['overall_auprc'] for h in iteration_history]
    iterations = [h['iteration'] for h in iteration_history]
    
    ax.plot(n_features, auprc_values, 'o-', color='#3498db', linewidth=2,
            markersize=8, markerfacecolor='white', markeredgewidth=2,
            label='AUPRC Trajectory')
    
    for i, (nf, auprc, it) in enumerate(zip(n_features, auprc_values, iterations)):
        ax.annotate(f'It.{it}', (nf, auprc), textcoords="offset points",
                    xytext=(0, 10), ha='center', fontsize=8, color='#7f8c8d')
    
    best_n_features = iteration_history[best_iteration]['n_features']
    best_auprc = iteration_history[best_iteration]['overall_auprc']
    
    ax.scatter([best_n_features], [best_auprc], s=200, c='#27ae60', marker='*',
               zorder=5, label=f'Selected: {best_n_features} features', edgecolors='black')
    
    ax.axhline(y=baseline_auprc, color='#e74c3c', linestyle='--', linewidth=2,
               label=f'Baseline AUPRC: {baseline_auprc:.4f}')
    
    threshold_line = baseline_auprc + DELTA_AUPRC_THRESHOLD
    ax.axhline(y=threshold_line, color='#f39c12', linestyle=':', linewidth=2,
               label=f'Degradation Threshold: {threshold_line:.4f}')
    
    ax.fill_between(ax.get_xlim(), threshold_line, ax.get_ylim()[0],
                    alpha=0.1, color='#e74c3c', label='Unacceptable Region')
    
    ax.set_xlabel('Number of Features', fontsize=12, fontweight='bold')
    ax.set_ylabel('Overall AUPRC', fontsize=12, fontweight='bold')
    ax.set_title('SHAP-RFE Feature Elimination Trajectory\nAUPRC vs. Feature Count',
                 fontsize=14, fontweight='bold')
    
    ax.set_xlim(min(n_features) - 1, max(n_features) + 1)
    ax.invert_xaxis()
    
    ax.legend(loc='lower left', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    textstr = f'Best Iteration: {best_iteration}\n'
    textstr += f'Features: {best_n_features}\n'
    textstr += f'AUPRC: {best_auprc:.4f}\n'
    textstr += f'Δ from baseline: {best_auprc - baseline_auprc:+.4f}'
    
    props = dict(boxstyle='round', facecolor='#ecf0f1', edgecolor='#bdc3c7', alpha=0.9)
    ax.text(0.98, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right', bbox=props)
    
    plt.tight_layout()
    
    output_path = plots_dir / "rfe_degradation_trajectory.png"
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    print(f"✓ RFE trajectory plot saved to: {output_path}")


def plot_metric_trajectories(
    iteration_history: List[Dict],
    best_iteration: int,
    baseline_metrics: Dict[str, float],
    output_dir: Path
) -> None:
    """Generate a multi-panel metric trajectory plot across pruning iterations."""
    print("\n" + "-" * 80)
    print("GENERATING RFE METRIC TRAJECTORIES PLOT")
    print("-" * 80)

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True, parents=True)

    metric_specs = [
        ("Accuracy", "overall_accuracy"),
        ("Precision", "overall_precision"),
        ("Recall", "overall_recall"),
        ("F1", "overall_f1_score"),
        ("ROC-AUC", "overall_roc_auc"),
        ("AUPRC", "overall_auprc"),
    ]

    n_features = [h['n_features'] for h in iteration_history]
    best_n_features = iteration_history[best_iteration]['n_features']

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for ax, (label, key) in zip(axes, metric_specs):
        values = [h.get(key, 0.0) for h in iteration_history]
        baseline_value = baseline_metrics.get(key, values[0] if values else 0.0)
        best_value = iteration_history[best_iteration].get(key, 0.0)

        ax.plot(n_features, values, 'o-', color='#3498db', linewidth=2,
                markersize=7, markerfacecolor='white', markeredgewidth=1.5)
        ax.axhline(y=baseline_value, color='#e74c3c', linestyle='--', linewidth=1.8)
        ax.scatter([best_n_features], [best_value], s=140, c='#27ae60', marker='*',
                   zorder=5, edgecolors='black')
        ax.annotate(f'It.{best_iteration}', (best_n_features, best_value),
                    textcoords="offset points", xytext=(0, 10), ha='center', fontsize=8)

        if key == 'overall_auprc':
            threshold_line = baseline_value + DELTA_AUPRC_THRESHOLD
            ax.axhline(y=threshold_line, color='#f39c12', linestyle=':', linewidth=1.8)
            ax.fill_between(n_features, threshold_line, min(values + [threshold_line]),
                            alpha=0.08, color='#e74c3c')

        ax.set_title(label, fontsize=12, fontweight='bold')
        ax.set_xlabel('Number of Features', fontsize=10)
        ax.set_ylabel('Score', fontsize=10)
        ax.invert_xaxis()
        ax.grid(True, alpha=0.25)

    fig.suptitle('SHAP-RFE Metric Trajectories Across Feature Pruning', fontsize=15, fontweight='bold')
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])

    output_path = plots_dir / "rfe_metric_trajectories.png"
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    print(f"✓ RFE metric trajectories plot saved to: {output_path}")


def main():
    """
    Phase 4 Main Execution: Feature Pruning Pipeline
    """
    print("=" * 80)
    print("TCC PIPELINE - PHASE 4: FEATURE PRUNING")
    print("=" * 80)
    print("Purpose: Identify efficient feature subset for operational deployment")
    print("Hardware Constraint: 8GB RAM")
    print(f"Configuration:")
    print(f"  • Correlation threshold: {CORRELATION_THRESHOLD}")
    print(f"  • Drop percentage per iteration: {DROP_PERCENTAGE}%")
    print(f"  • ΔAUPRC threshold: {DELTA_AUPRC_THRESHOLD}")
    print(f"  • Protected percentile: {TEMPORAL_PROTECTION_PERCENTILE}%")
    
    results_dir = DATA_PATH / "results"
    results_dir.mkdir(exist_ok=True, parents=True)
    
    shap_file = results_dir / "shap_feature_importance.csv"
    if not shap_file.exists():
        raise FileNotFoundError(
            f"SHAP importance file not found at {shap_file}. "
            "Run 05_train_models.py first."
        )
    
    features_path = DATA_PATH / OUTPUT_FEATURES_FILE
    if not features_path.exists():
        raise FileNotFoundError(
            f"Features file not found at {features_path}. "
            "Run 03_extract_features.py first."
        )
    
    print(f"\nLoading SHAP importance from Phase 2...")
    shap_df = pd.read_csv(shap_file)
    initial_shap = dict(zip(shap_df['feature'], shap_df['mean_abs_shap']))
    initial_features = shap_df['feature'].tolist()
    
    print(f"Loading features from: {features_path}")
    df = pd.read_parquet(features_path)
    
    window_col = 'window_id' if 'window_id' in df.columns else 'date'
    target_col = 'is_fraud'
    
    print(f"\nDataset shape: {df.shape}")
    print(f"Initial features: {len(initial_features)}")
    
    features_after_collinearity, dropped_pairs = compute_collinearity_filter(
        df=df,
        feature_cols=initial_features,
        shap_importance=initial_shap,
        threshold=CORRELATION_THRESHOLD
    )
    
    print("\n[BASELINE EVALUATION] Computing baseline AUPRC after collinearity filter...")
    baseline_metrics, baseline_window_results, baseline_shap = evaluate_feature_set(
        df=df,
        feature_cols=features_after_collinearity,
        window_col=window_col,
        target_col=target_col
    )
    print(
        f"Baseline metrics (post-collinearity): "
        f"AUPRC={baseline_metrics['overall_auprc']:.4f}, "
        f"Acc={baseline_metrics['overall_accuracy']:.4f}, "
        f"Prec={baseline_metrics['overall_precision']:.4f}, "
        f"Recall={baseline_metrics['overall_recall']:.4f}, "
        f"F1={baseline_metrics['overall_f1_score']:.4f}, "
        f"ROC-AUC={baseline_metrics['overall_roc_auc']:.4f}"
    )
    
    protected_features = identify_protected_features(
        df=df,
        feature_cols=features_after_collinearity,
        shap_importance=baseline_shap,
        percentile=TEMPORAL_PROTECTION_PERCENTILE
    )
    
    final_features, iteration_history, best_iteration = run_shap_rfe_loop(
        df=df,
        initial_features=features_after_collinearity,
        protected_features=protected_features,
        initial_shap=baseline_shap,
        window_col=window_col,
        target_col=target_col,
        baseline_metrics=baseline_metrics
    )
    
    del df
    gc.collect()
    
    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    
    initial_count = len(initial_features)
    final_count = len(final_features)
    reduction_pct = (1 - final_count / initial_count) * 100
    
    final_metrics = iteration_history[best_iteration]
    delta = final_metrics['overall_auprc'] - iteration_history[0]['overall_auprc']

    print(f"\nFeature Count: {initial_count} → {final_count} ({reduction_pct:.1f}% reduction)")
    print(
        f"AUPRC: {iteration_history[0]['overall_auprc']:.4f} → {final_metrics['overall_auprc']:.4f} "
        f"(Δ={delta:+.4f})"
    )
    print(f"Best Iteration: {best_iteration}")

    print("\n" + "-" * 40)
    print("BASELINE VS FINAL METRICS:")
    print("-" * 40)
    metric_rows = [
        ("Accuracy", "overall_accuracy"),
        ("Precision", "overall_precision"),
        ("Recall", "overall_recall"),
        ("F1", "overall_f1_score"),
        ("ROC-AUC", "overall_roc_auc"),
        ("AUPRC", "overall_auprc"),
    ]
    print(f"{'Metric':<12} {'Baseline':>12} {'Final':>12} {'Lift':>12}")
    print("-" * 52)
    for label, key in metric_rows:
        baseline_val = baseline_metrics.get(key, 0.0)
        final_val = final_metrics.get(key, 0.0)
        lift_pct = ((final_val - baseline_val) / (baseline_val + 1e-9)) * 100
        print(f"{label:<12} {baseline_val:>12.4f} {final_val:>12.4f} {lift_pct:>+11.2f}%")
    
    print("\n" + "-" * 40)
    print("PRUNED FEATURES:")
    print("-" * 40)
    
    discarded_features = [f for f in initial_features if f not in final_features]
    print(f"Discarded ({len(discarded_features)} features):")
    for feat in sorted(discarded_features):
        print(f"  ✗ {feat} (SHAP={initial_shap.get(feat, 0):.4f})")
    
    print(f"\nRetained ({len(final_features)} features):")
    retained_sorted = sorted(final_features, key=lambda f: -baseline_shap.get(f, 0))
    for feat in retained_sorted:
        category = "behavioral" if feat in BEHAVIORAL_COLS else "topological"
        print(f"  ✓ {feat} (SHAP={baseline_shap.get(feat, 0):.4f}, {category})")
    
    pruned_features_output = {
        'n_features': len(final_features),
        'features': final_features,
        'initial_n_features': initial_count,
        'baseline_metrics': {k: baseline_metrics.get(k, None) for k in [
            'overall_accuracy', 'overall_precision', 'overall_recall',
            'overall_f1_score', 'overall_roc_auc', 'overall_auprc'
        ]},
        'final_metrics': {k: final_metrics.get(k, None) for k in [
            'overall_accuracy', 'overall_precision', 'overall_recall',
            'overall_f1_score', 'overall_roc_auc', 'overall_auprc'
        ]},
        'baseline_auprc': iteration_history[0]['overall_auprc'],
        'final_auprc': final_metrics['overall_auprc'],
        'delta_auprc': delta,
        'best_iteration': best_iteration,
        'discarded_features': discarded_features,
        'collinearity_pairs': [(f1, f2, corr) for f1, f2, corr in dropped_pairs]
    }
    
    pruned_path = results_dir / "pruned_features.json"
    with open(pruned_path, 'w') as f:
        json.dump(pruned_features_output, f, indent=2)
    print(f"\n✓ Pruned features saved to: {pruned_path}")
    
    history_df = pd.DataFrame([
        {
            'iteration': h['iteration'],
            'n_features': h['n_features'],
            'overall_auprc': h['overall_auprc'],
            'overall_accuracy': h.get('overall_accuracy', 0.0),
            'overall_precision': h.get('overall_precision', 0.0),
            'overall_recall': h.get('overall_recall', 0.0),
            'overall_f1_score': h.get('overall_f1_score', 0.0),
            'overall_roc_auc': h.get('overall_roc_auc', 0.0),
            'mean_window_auprc': h.get('mean_window_auprc', 0.0),
            'std_window_auprc': h.get('std_window_auprc', 0.0),
            'delta_from_baseline': h['delta_from_baseline'],
            'selected': h['selected']
        }
        for h in iteration_history
    ])
    history_path = results_dir / "rfe_history.csv"
    history_df.to_csv(history_path, index=False)
    print(f"✓ RFE history saved to: {history_path}")
    
    plot_rfe_trajectory(
        iteration_history=iteration_history,
        best_iteration=best_iteration,
        baseline_auprc=iteration_history[0]['overall_auprc'],
        output_dir=results_dir
    )
    plot_metric_trajectories(
        iteration_history=iteration_history,
        best_iteration=best_iteration,
        baseline_metrics=baseline_metrics,
        output_dir=results_dir
    )
    
    print("\n" + "=" * 80)
    print("PHASE 4 COMPLETE")
    print("=" * 80)
    print(f"\nOutputs saved to: {results_dir}")
    print("  • pruned_features.json")
    print("  • rfe_history.csv")
    print("  • plots/rfe_degradation_trajectory.png")
    print("  • plots/rfe_metric_trajectories.png")
    
    print("\n" + "-" * 40)
    print("SUMMARY FOR TCC DEFENSE:")
    print("-" * 40)
    print(f"Original feature set: {initial_count} features")
    print(f"Operational feature set: {final_count} features ({reduction_pct:.1f}% reduction)")
    print(f"Performance retention: {final_metrics['overall_auprc']/iteration_history[0]['overall_auprc']*100:.1f}% of baseline AUPRC")
    
    n_behavioral = sum(1 for f in final_features if f in BEHAVIORAL_COLS)
    n_topological = sum(1 for f in final_features if f in TOPOLOGICAL_COLS)
    print(f"Feature composition: {n_behavioral} behavioral, {n_topological} topological")
    
    print("\n✓ TCC Pipeline Complete - All 4 phases executed successfully")


if __name__ == "__main__":
    main()
