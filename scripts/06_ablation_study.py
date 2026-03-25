"""
Phase 3: Ablation Study
=======================
TCC Pipeline Script 06

Purpose: Quantify the exact performance delta caused by adding topological features.

Logic: Ingest the outputs from Phase 2. Calculate the paired differences in AUPRC,
       P@100, and ROC-AUC between the Baseline and Full models. Run a paired t-test
       on the AUPRC across the temporal windows to prove statistical significance.

Outputs:
    - data/results/ablation_results.csv
    - data/results/plots/ablation_lift.png

Hardware Constraint: 8GB RAM - minimal memory usage since this phase uses pre-computed predictions.
"""

import gc
import sys
from pathlib import Path
from typing import Dict, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.metrics import average_precision_score, roc_auc_score

root_path = Path(__file__).resolve().parent.parent
sys.path.append(str(root_path))

from src.config import DATA_PATH


K_VALUES = [10, 50, 100, 500]


def compute_precision_at_k(y_true: np.ndarray, y_probs: np.ndarray, k: int) -> float:
    """Compute Precision@K."""
    if k > len(y_true):
        k = len(y_true)
    sorted_indices = np.argsort(y_probs)[::-1]
    top_k_labels = y_true[sorted_indices[:k]]
    return float(np.sum(top_k_labels)) / k


def compute_window_metrics(predictions_df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-window metrics from predictions."""
    window_metrics = []
    
    for window_id in predictions_df['window_id'].unique():
        window_data = predictions_df[predictions_df['window_id'] == window_id]
        y_true = window_data['y_true'].values
        y_prob = window_data['y_prob'].values
        
        if len(np.unique(y_true)) < 2:
            continue
        
        metrics = {
            'window_id': window_id,
            'n_samples': len(y_true),
            'n_fraud': int(y_true.sum()),
            'fraud_rate': float(y_true.mean()),
            'auprc': average_precision_score(y_true, y_prob),
            'roc_auc': roc_auc_score(y_true, y_prob),
        }
        
        for k in K_VALUES:
            metrics[f'P@{k}'] = compute_precision_at_k(y_true, y_prob, k)
        
        window_metrics.append(metrics)
    
    return pd.DataFrame(window_metrics)


def compute_ablation_metrics(
    baseline_preds: pd.DataFrame,
    full_preds: pd.DataFrame
) -> Tuple[pd.DataFrame, Dict]:
    """
    Compute paired differences between baseline and full model.
    
    Returns:
        - comparison_df: Per-window comparison with deltas
        - statistical_tests: Dict with t-test results
    """
    print("\n" + "=" * 80)
    print("COMPUTING ABLATION METRICS")
    print("=" * 80)
    
    baseline_metrics = compute_window_metrics(baseline_preds)
    full_metrics = compute_window_metrics(full_preds)
    
    comparison_df = baseline_metrics.merge(
        full_metrics,
        on='window_id',
        suffixes=('_baseline', '_full')
    )
    
    comparison_df['delta_auprc'] = comparison_df['auprc_full'] - comparison_df['auprc_baseline']
    comparison_df['delta_roc_auc'] = comparison_df['roc_auc_full'] - comparison_df['roc_auc_baseline']
    comparison_df['delta_P@100'] = comparison_df['P@100_full'] - comparison_df['P@100_baseline']
    
    comparison_df['lift_auprc_pct'] = (
        comparison_df['delta_auprc'] / (comparison_df['auprc_baseline'] + 1e-9)
    ) * 100
    comparison_df['lift_roc_auc_pct'] = (
        comparison_df['delta_roc_auc'] / (comparison_df['roc_auc_baseline'] + 1e-9)
    ) * 100
    comparison_df['lift_P@100_pct'] = (
        comparison_df['delta_P@100'] / (comparison_df['P@100_baseline'] + 1e-9)
    ) * 100
    
    t_stat_auprc, p_value_auprc = stats.ttest_rel(
        comparison_df['auprc_baseline'].values,
        comparison_df['auprc_full'].values
    )
    
    t_stat_roc, p_value_roc = stats.ttest_rel(
        comparison_df['roc_auc_baseline'].values,
        comparison_df['roc_auc_full'].values
    )
    
    t_stat_p100, p_value_p100 = stats.ttest_rel(
        comparison_df['P@100_baseline'].values,
        comparison_df['P@100_full'].values
    )
    
    statistical_tests = {
        'auprc': {
            't_statistic': t_stat_auprc,
            'p_value': p_value_auprc,
            'significant': p_value_auprc < 0.05,
            'mean_delta': comparison_df['delta_auprc'].mean(),
            'std_delta': comparison_df['delta_auprc'].std(),
        },
        'roc_auc': {
            't_statistic': t_stat_roc,
            'p_value': p_value_roc,
            'significant': p_value_roc < 0.05,
            'mean_delta': comparison_df['delta_roc_auc'].mean(),
            'std_delta': comparison_df['delta_roc_auc'].std(),
        },
        'P@100': {
            't_statistic': t_stat_p100,
            'p_value': p_value_p100,
            'significant': p_value_p100 < 0.05,
            'mean_delta': comparison_df['delta_P@100'].mean(),
            'std_delta': comparison_df['delta_P@100'].std(),
        }
    }
    
    return comparison_df, statistical_tests


def print_ablation_report(
    comparison_df: pd.DataFrame,
    statistical_tests: Dict
) -> None:
    """Print formatted ablation study report."""
    print("\n" + "-" * 80)
    print("PER-WINDOW COMPARISON: Baseline vs Full Model")
    print("-" * 80)
    
    print(f"\n{'Window':<10} {'AUPRC (B/F)':<18} {'ROC-AUC (B/F)':<18} {'P@100 (B/F)':<18} {'AUPRC Lift':<12}")
    print("-" * 80)
    
    for _, row in comparison_df.iterrows():
        print(f"{row['window_id']:<10} "
              f"{row['auprc_baseline']:.3f} / {row['auprc_full']:.3f}  "
              f"{row['roc_auc_baseline']:.3f} / {row['roc_auc_full']:.3f}  "
              f"{row['P@100_baseline']:.3f} / {row['P@100_full']:.3f}  "
              f"{row['lift_auprc_pct']:+.1f}%")
    
    print("\n" + "-" * 80)
    print("STATISTICAL SIGNIFICANCE ANALYSIS")
    print("-" * 80)
    
    for metric, results in statistical_tests.items():
        print(f"\n{metric.upper()}:")
        print(f"  Mean Δ: {results['mean_delta']:+.4f} ± {results['std_delta']:.4f}")
        print(f"  t-statistic: {results['t_statistic']:.4f}")
        print(f"  p-value: {results['p_value']:.4e}")
        
        if results['significant']:
            direction = "improvement" if results['mean_delta'] > 0 else "degradation"
            print(f"  ✓ STATISTICALLY SIGNIFICANT {direction} at α=0.05")
        else:
            print(f"  ✗ Not statistically significant at α=0.05")
    
    mean_lift = comparison_df['lift_auprc_pct'].mean()
    print("\n" + "-" * 80)
    print("SUMMARY")
    print("-" * 80)
    print(f"Mean AUPRC Lift: {mean_lift:+.2f}%")
    print(f"Windows with positive lift: {(comparison_df['delta_auprc'] > 0).sum()}/{len(comparison_df)}")
    
    overall_baseline_auprc = comparison_df['auprc_baseline'].mean()
    overall_full_auprc = comparison_df['auprc_full'].mean()
    print(f"Overall Baseline AUPRC: {overall_baseline_auprc:.4f}")
    print(f"Overall Full Model AUPRC: {overall_full_auprc:.4f}")


def plot_ablation_lift(
    comparison_df: pd.DataFrame,
    statistical_tests: Dict,
    output_dir: Path
) -> None:
    """
    Generate comprehensive ablation visualization.
    
    Creates a multi-panel plot showing:
    - Panel 1: Baseline vs Full Model AUPRC per window (grouped bars)
    - Panel 2: Delta/Lift per window (waterfall-style)
    - Panel 3: Summary statistics with p-value annotation
    """
    print("\n" + "-" * 80)
    print("GENERATING ABLATION VISUALIZATION")
    print("-" * 80)
    
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True, parents=True)
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Panel 1: Grouped bar chart - AUPRC comparison
    ax1 = axes[0, 0]
    
    x = np.arange(len(comparison_df))
    width = 0.35
    
    bars_baseline = ax1.bar(
        x - width/2,
        comparison_df['auprc_baseline'].values,
        width,
        label='Baseline (Behavioral)',
        color='#e74c3c',
        alpha=0.85
    )
    bars_full = ax1.bar(
        x + width/2,
        comparison_df['auprc_full'].values,
        width,
        label='Full Model (+Topological)',
        color='#27ae60',
        alpha=0.85
    )
    
    for i, (b, f, lift) in enumerate(zip(
        comparison_df['auprc_baseline'],
        comparison_df['auprc_full'],
        comparison_df['lift_auprc_pct']
    )):
        y_pos = max(b, f) + 0.02
        ax1.annotate(
            f'{lift:+.1f}%',
            xy=(i, y_pos),
            ha='center',
            va='bottom',
            fontsize=9,
            fontweight='bold',
            color='#2c3e50'
        )
    
    ax1.set_xlabel('Temporal Window ID', fontsize=11, fontweight='bold')
    ax1.set_ylabel('AUPRC', fontsize=11, fontweight='bold')
    ax1.set_title('Ablation Study: AUPRC per Temporal Window', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(w) for w in comparison_df['window_id']], rotation=45, ha='right')
    ax1.legend(loc='lower right', fontsize=10)
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim(0, comparison_df['auprc_full'].max() * 1.25)
    
    # Panel 2: Delta waterfall
    ax2 = axes[0, 1]
    
    deltas = comparison_df['delta_auprc'].values
    colors = ['#27ae60' if d > 0 else '#e74c3c' for d in deltas]
    
    ax2.bar(x, deltas, color=colors, alpha=0.85, edgecolor='black', linewidth=0.5)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
    
    mean_delta = deltas.mean()
    ax2.axhline(y=mean_delta, color='#3498db', linestyle='--', linewidth=2,
                label=f'Mean Δ: {mean_delta:+.4f}')
    
    ax2.set_xlabel('Temporal Window ID', fontsize=11, fontweight='bold')
    ax2.set_ylabel('ΔAUPRC (Full - Baseline)', fontsize=11, fontweight='bold')
    ax2.set_title('Per-Window AUPRC Improvement', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(w) for w in comparison_df['window_id']], rotation=45, ha='right')
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(axis='y', alpha=0.3)
    
    # Panel 3: P@100 comparison
    ax3 = axes[1, 0]
    
    bars_baseline_p100 = ax3.bar(
        x - width/2,
        comparison_df['P@100_baseline'].values,
        width,
        label='Baseline',
        color='#e74c3c',
        alpha=0.85
    )
    bars_full_p100 = ax3.bar(
        x + width/2,
        comparison_df['P@100_full'].values,
        width,
        label='Full Model',
        color='#27ae60',
        alpha=0.85
    )
    
    ax3.set_xlabel('Temporal Window ID', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Precision@100', fontsize=11, fontweight='bold')
    ax3.set_title('Ablation Study: Precision@100 per Window', fontsize=12, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels([str(w) for w in comparison_df['window_id']], rotation=45, ha='right')
    ax3.legend(loc='lower right', fontsize=10)
    ax3.grid(axis='y', alpha=0.3)
    
    # Panel 4: Summary statistics with p-value
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    auprc_stats = statistical_tests['auprc']
    p100_stats = statistical_tests['P@100']
    roc_stats = statistical_tests['roc_auc']
    
    summary_text = f"""
╔══════════════════════════════════════════════════════════════╗
║              ABLATION STUDY: STATISTICAL SUMMARY              ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  AUPRC (Area Under Precision-Recall Curve)                   ║
║  ─────────────────────────────────────────                   ║
║    Mean Δ:        {auprc_stats['mean_delta']:+.4f} ± {auprc_stats['std_delta']:.4f}                       ║
║    t-statistic:   {auprc_stats['t_statistic']:.4f}                                   ║
║    p-value:       {auprc_stats['p_value']:.4e}                               ║
║    Significant:   {'✓ YES (α=0.05)' if auprc_stats['significant'] else '✗ NO'}                            ║
║                                                              ║
║  ROC-AUC                                                     ║
║  ───────                                                     ║
║    Mean Δ:        {roc_stats['mean_delta']:+.4f} ± {roc_stats['std_delta']:.4f}                       ║
║    p-value:       {roc_stats['p_value']:.4e}                               ║
║                                                              ║
║  Precision@100                                               ║
║  ─────────────                                               ║
║    Mean Δ:        {p100_stats['mean_delta']:+.4f} ± {p100_stats['std_delta']:.4f}                       ║
║    p-value:       {p100_stats['p_value']:.4e}                               ║
║                                                              ║
║  CONCLUSION:                                                 ║
║  {'Topological features provide STATISTICALLY SIGNIFICANT' if auprc_stats['significant'] else 'Difference is NOT statistically significant'}     ║
║  {'improvement in fraud detection performance.' if auprc_stats['significant'] else 'at the α=0.05 significance level.'}              ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
"""
    
    ax4.text(0.5, 0.5, summary_text, transform=ax4.transAxes,
             fontsize=10, fontfamily='monospace',
             verticalalignment='center', horizontalalignment='center',
             bbox=dict(boxstyle='round', facecolor='#f8f9fa', edgecolor='#dee2e6'))
    
    plt.tight_layout()
    
    output_path = plots_dir / "ablation_lift.png"
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    print(f"✓ Ablation visualization saved to: {output_path}")


def main():
    """
    Phase 3 Main Execution: Ablation Study Pipeline
    """
    print("=" * 80)
    print("TCC PIPELINE - PHASE 3: ABLATION STUDY")
    print("=" * 80)
    print("Purpose: Quantify performance delta from topological features")
    print("Hardware Constraint: 8GB RAM (minimal - using pre-computed predictions)")
    
    results_dir = DATA_PATH / "results"
    
    baseline_preds_path = results_dir / "baseline_predictions.parquet"
    full_preds_path = results_dir / "full_predictions.parquet"
    
    if not baseline_preds_path.exists():
        raise FileNotFoundError(
            f"Baseline predictions not found at {baseline_preds_path}. "
            "Run 05_train_models.py first."
        )
    
    if not full_preds_path.exists():
        raise FileNotFoundError(
            f"Full model predictions not found at {full_preds_path}. "
            "Run 05_train_models.py first."
        )
    
    print(f"\nLoading predictions from Phase 2...")
    baseline_preds = pd.read_parquet(baseline_preds_path)
    full_preds = pd.read_parquet(full_preds_path)
    
    print(f"Baseline predictions: {len(baseline_preds):,} samples")
    print(f"Full model predictions: {len(full_preds):,} samples")
    
    comparison_df, statistical_tests = compute_ablation_metrics(
        baseline_preds=baseline_preds,
        full_preds=full_preds
    )
    
    del baseline_preds, full_preds
    gc.collect()
    
    print_ablation_report(comparison_df, statistical_tests)
    
    ablation_results_path = results_dir / "ablation_results.csv"
    comparison_df.to_csv(ablation_results_path, index=False)
    print(f"\n✓ Ablation results saved to: {ablation_results_path}")
    
    stats_summary = []
    for metric, values in statistical_tests.items():
        stats_summary.append({
            'metric': metric,
            't_statistic': values['t_statistic'],
            'p_value': values['p_value'],
            'significant': values['significant'],
            'mean_delta': values['mean_delta'],
            'std_delta': values['std_delta']
        })
    stats_df = pd.DataFrame(stats_summary)
    stats_path = results_dir / "ablation_statistical_tests.csv"
    stats_df.to_csv(stats_path, index=False)
    print(f"✓ Statistical tests saved to: {stats_path}")
    
    plot_ablation_lift(comparison_df, statistical_tests, results_dir)
    
    print("\n" + "=" * 80)
    print("PHASE 3 COMPLETE")
    print("=" * 80)
    print(f"\nOutputs saved to: {results_dir}")
    print("  • ablation_results.csv")
    print("  • ablation_statistical_tests.csv")
    print("  • plots/ablation_lift.png")
    
    auprc_stats = statistical_tests['auprc']
    print("\n" + "-" * 40)
    print("KEY FINDING:")
    if auprc_stats['significant']:
        print(f"✓ Topological features provide a STATISTICALLY SIGNIFICANT")
        print(f"  improvement of {auprc_stats['mean_delta']:+.4f} AUPRC (p={auprc_stats['p_value']:.4e})")
    else:
        print(f"✗ The improvement of {auprc_stats['mean_delta']:+.4f} AUPRC is NOT")
        print(f"  statistically significant (p={auprc_stats['p_value']:.4e})")
    
    print("\n→ Proceed to Phase 4: 07_feature_pruning.py")


if __name__ == "__main__":
    main()
