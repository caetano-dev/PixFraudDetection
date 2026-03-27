"""
Phase 1: Heuristic Evaluation
=============================
TCC Pipeline Script 04

Purpose: Prove the standalone predictive power of raw graph structures before
         passing them to a tree ensemble.

Logic: Calculate threshold-independent metrics (AUPRC, ROC-AUC) and Precision@K
       for every raw topological feature independently per temporal window.
       No F1 calculation for these continuous heuristics.

Outputs:
    - data/results/heuristic_metrics.parquet
    - data/results/plots/heuristic_auprc_comparison.png

Hardware Constraint: 8GB RAM - aggressive memory management required.
"""

import gc
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import average_precision_score, roc_auc_score

root_path = Path(__file__).resolve().parent.parent
sys.path.append(str(root_path))

from src.config import OUTPUT_FEATURES_FILE, DATA_PATH


HEURISTIC_FEATURES = [
    'pr_vol_deep',
    'pr_vol_shallow', 
    'pr_count',
    'hits_hub',
    'hits_auth',
    'betweenness',
    'k_core',
    'degree',
    'in_degree',
    'out_degree',
    'egonet_node_count',
    'egonet_edge_count',
    'egonet_density',
    'egonet_total_weight',
    'local_clustering_coefficient',
    'triangle_count',
    'average_neighbor_degree',
    'successor_avg_volume',
    'successor_max_volume',
]

K_VALUES = [10, 50, 100, 500]


def compute_precision_at_k(y_true: np.ndarray, y_scores: np.ndarray, k: int) -> float:
    """Compute Precision@K for a continuous heuristic score."""
    if k > len(y_true):
        k = len(y_true)
    sorted_indices = np.argsort(y_scores)[::-1]
    top_k_labels = y_true[sorted_indices[:k]]
    return float(np.sum(top_k_labels)) / k


def evaluate_single_heuristic(
    df_window: pd.DataFrame,
    feature_name: str,
    target_col: str = 'is_fraud'
) -> Dict:
    """
    Evaluate a single heuristic feature on a single temporal window.
    
    Returns threshold-independent metrics only:
    - AUPRC (Average Precision)
    - ROC-AUC
    - Precision@K for various K values
    """
    y_true = df_window[target_col].values
    scores = df_window[feature_name].values
    
    if len(np.unique(y_true)) < 2:
        return None
    
    scores_clean = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    try:
        auprc = average_precision_score(y_true, scores_clean)
    except Exception:
        auprc = np.nan
    
    try:
        roc_auc = roc_auc_score(y_true, scores_clean)
    except Exception:
        roc_auc = np.nan
    
    precision_at_k = {}
    for k in K_VALUES:
        precision_at_k[f'P@{k}'] = compute_precision_at_k(y_true, scores_clean, k)
    
    result = {
        'feature': feature_name,
        'auprc': auprc,
        'roc_auc': roc_auc,
        **precision_at_k
    }
    
    return result


def evaluate_all_heuristics_per_window(
    df: pd.DataFrame,
    window_col: str = 'window_id',
    target_col: str = 'is_fraud'
) -> pd.DataFrame:
    """
    Evaluate all heuristic features across all temporal windows.
    
    Memory-efficient: processes one window at a time and releases memory.
    """
    print("\n" + "=" * 80)
    print("PHASE 1: HEURISTIC EVALUATION")
    print("Evaluating raw graph features independently (no ML model)")
    print("=" * 80)
    
    unique_windows = sorted(df[window_col].unique())
    print(f"\nTemporal windows: {len(unique_windows)}")
    print(f"Heuristic features: {len(HEURISTIC_FEATURES)}")
    print(f"K values for P@K: {K_VALUES}")
    
    all_results = []
    
    for window_id in unique_windows:
        print(f"\n[Window {window_id}]", end=" ")
        
        df_window = df[df[window_col] == window_id].copy()
        n_fraud = df_window[target_col].sum()
        n_total = len(df_window)
        fraud_rate = n_fraud / n_total if n_total > 0 else 0
        
        print(f"N={n_total:,}, Fraud={n_fraud:,} ({fraud_rate:.2%})")
        
        if n_fraud == 0 or n_fraud == n_total:
            print(f"  Skipping: insufficient class variance")
            del df_window
            gc.collect()
            continue
        
        for feature_name in HEURISTIC_FEATURES:
            if feature_name not in df_window.columns:
                print(f"  Warning: {feature_name} not found in data")
                continue
            
            result = evaluate_single_heuristic(df_window, feature_name, target_col)
            
            if result is not None:
                result['window_id'] = window_id
                result['n_samples'] = n_total
                result['n_fraud'] = n_fraud
                result['fraud_rate'] = fraud_rate
                all_results.append(result)
        
        del df_window
        gc.collect()
    
    results_df = pd.DataFrame(all_results)
    
    return results_df


def compute_aggregate_statistics(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute aggregate statistics across all windows for each heuristic.
    """
    print("\n" + "-" * 80)
    print("AGGREGATE HEURISTIC PERFORMANCE (ACROSS ALL WINDOWS)")
    print("-" * 80)
    
    agg_stats = results_df.groupby('feature').agg({
        'auprc': ['mean', 'std', 'min', 'max'],
        'roc_auc': ['mean', 'std', 'min', 'max'],
        'P@100': ['mean', 'std'],
    }).round(4)
    
    agg_stats.columns = ['_'.join(col).strip() for col in agg_stats.columns.values]
    agg_stats = agg_stats.reset_index()
    agg_stats = agg_stats.sort_values('auprc_mean', ascending=False)
    
    print("\n" + agg_stats.to_string(index=False))
    
    return agg_stats


def plot_heuristic_auprc_comparison(
    results_df: pd.DataFrame,
    output_dir: Path
) -> None:
    """
    Generate a grouped bar chart comparing AUPRC of each heuristic across windows.
    Publication-quality visualization for TCC defense.
    """
    print("\n" + "-" * 80)
    print("GENERATING HEURISTIC COMPARISON PLOT")
    print("-" * 80)
    
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True, parents=True)
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(2, 1, figsize=(14, 12))
    
    # Plot 1: AUPRC per heuristic across windows (grouped bar chart)
    ax1 = axes[0]
    
    pivot_df = results_df.pivot(index='window_id', columns='feature', values='auprc')
    pivot_df = pivot_df[sorted(pivot_df.columns, 
                               key=lambda x: -results_df[results_df['feature']==x]['auprc'].mean())]
    
    n_windows = len(pivot_df)
    n_features = len(pivot_df.columns)
    bar_width = 0.8 / n_features
    x = np.arange(n_windows)
    
    colors = sns.color_palette("husl", n_features)
    
    for i, (feature, color) in enumerate(zip(pivot_df.columns, colors)):
        offset = (i - n_features / 2 + 0.5) * bar_width
        bars = ax1.bar(x + offset, pivot_df[feature].values, 
                       bar_width, label=feature, color=color, alpha=0.85)
    
    ax1.set_xlabel('Temporal Window ID', fontsize=12, fontweight='bold')
    ax1.set_ylabel('AUPRC', fontsize=12, fontweight='bold')
    ax1.set_title('Raw Heuristic Predictive Power: AUPRC per Temporal Window', 
                  fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(w) for w in pivot_df.index], rotation=45, ha='right')
    ax1.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)
    ax1.grid(axis='y', alpha=0.3)
    
    ax1.axhline(y=results_df['fraud_rate'].mean(), color='red', linestyle='--', 
                linewidth=1.5, label=f'Baseline (random): {results_df["fraud_rate"].mean():.4f}')
    
    # Plot 2: Mean AUPRC with error bars (summary view)
    ax2 = axes[1]
    
    agg_data = results_df.groupby('feature').agg({
        'auprc': ['mean', 'std']
    }).reset_index()
    agg_data.columns = ['feature', 'auprc_mean', 'auprc_std']
    agg_data = agg_data.sort_values('auprc_mean', ascending=True)
    
    y_pos = np.arange(len(agg_data))
    
    bars = ax2.barh(y_pos, agg_data['auprc_mean'].values, 
                    xerr=agg_data['auprc_std'].values,
                    color=sns.color_palette("viridis", len(agg_data)),
                    alpha=0.85, capsize=3)
    
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(agg_data['feature'].values)
    ax2.set_xlabel('Mean AUPRC (± 1 std)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Heuristic Feature', fontsize=12, fontweight='bold')
    ax2.set_title('Aggregate Heuristic Performance (Mean ± Std Across Windows)', 
                  fontsize=14, fontweight='bold')
    
    random_baseline = results_df['fraud_rate'].mean()
    ax2.axvline(x=random_baseline, color='red', linestyle='--', linewidth=2,
                label=f'Random Baseline: {random_baseline:.4f}')
    ax2.legend(loc='lower right', fontsize=10)
    
    for i, (mean, std) in enumerate(zip(agg_data['auprc_mean'], agg_data['auprc_std'])):
        ax2.text(mean + std + 0.01, i, f'{mean:.3f}', va='center', fontsize=9)
    
    plt.tight_layout()
    
    output_path = plots_dir / "heuristic_auprc_comparison.png"
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    print(f"✓ Plot saved to: {output_path}")
    
    del pivot_df, agg_data
    gc.collect()


def main():
    """
    Phase 1 Main Execution: Heuristic Evaluation Pipeline
    """
    print("=" * 80)
    print("TCC PIPELINE - PHASE 1: HEURISTIC EVALUATION")
    print("=" * 80)
    print("Purpose: Prove standalone predictive power of raw graph structures")
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
    
    results_df = evaluate_all_heuristics_per_window(
        df=df,
        window_col=window_col,
        target_col=target_col
    )
    
    del df
    gc.collect()
    
    agg_stats = compute_aggregate_statistics(results_df)
    
    results_dir = DATA_PATH / "results"
    results_dir.mkdir(exist_ok=True, parents=True)
    
    output_path = results_dir / "heuristic_metrics.parquet"
    results_df.to_parquet(output_path, index=False)
    print(f"\n✓ Heuristic metrics saved to: {output_path}")
    
    agg_path = results_dir / "heuristic_aggregate_stats.csv"
    agg_stats.to_csv(agg_path, index=False)
    print(f"✓ Aggregate statistics saved to: {agg_path}")
    
    plot_heuristic_auprc_comparison(results_df, results_dir)
    
    print("\n" + "=" * 80)
    print("PHASE 1 COMPLETE")
    print("=" * 80)
    print("\nKey Findings:")
    print("-" * 40)
    
    top_features = agg_stats.nlargest(3, 'auprc_mean')
    print("Top 3 Heuristics by Mean AUPRC:")
    for _, row in top_features.iterrows():
        print(f"  • {row['feature']}: {row['auprc_mean']:.4f} ± {row['auprc_std']:.4f}")
    
    random_baseline = results_df['fraud_rate'].mean()
    best_heuristic = agg_stats.iloc[0]
    lift = (best_heuristic['auprc_mean'] - random_baseline) / random_baseline * 100
    print(f"\nBest heuristic lift over random: {lift:+.1f}%")
    
    del results_df, agg_stats
    gc.collect()
    
    print("\n→ Proceed to Phase 2: 05_train_models.py")


if __name__ == "__main__":
    main()
