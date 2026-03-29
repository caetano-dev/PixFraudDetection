"""
Phase 3: Ablation Study
=======================
TCC Pipeline Script 06

Purpose: Quantify the exact performance delta caused by adding topological features.

Logic: Ingest the outputs from Phase 2. Calculate the paired differences in AUPRC,
       P@100/200/300/500, ROC-AUC, accuracy, precision, recall, and F1 between the Baseline and
       Full models. Run paired t-tests across the temporal windows to prove
       statistical significance.

Outputs:
    - data/results/ablation_results.csv
    - data/results/plots/ablation_lift.png
    - data/results/plots/ablation_metric_comparison.png

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
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score, roc_auc_score

root_path = Path(__file__).resolve().parent.parent
sys.path.append(str(root_path))

from src.config import DATA_PATH


K_VALUES = [10, 50, 100, 200, 300, 500]


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
        if 'y_pred' in window_data.columns:
            y_pred = window_data['y_pred'].values
        elif 'threshold' in window_data.columns:
            threshold = float(window_data['threshold'].iloc[0])
            y_pred = (y_prob >= threshold).astype(int)
        else:
            y_pred = (y_prob >= 0.5).astype(int)
        
        if len(np.unique(y_true)) < 2:
            continue
        
        metrics = {
            'window_id': window_id,
            'n_samples': len(y_true),
            'n_fraud': int(y_true.sum()),
            'fraud_rate': float(y_true.mean()),
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
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
    comparison_df['delta_accuracy'] = comparison_df['accuracy_full'] - comparison_df['accuracy_baseline']
    comparison_df['delta_precision'] = comparison_df['precision_full'] - comparison_df['precision_baseline']
    comparison_df['delta_recall'] = comparison_df['recall_full'] - comparison_df['recall_baseline']
    comparison_df['delta_f1_score'] = comparison_df['f1_score_full'] - comparison_df['f1_score_baseline']
    comparison_df['delta_P@100'] = comparison_df['P@100_full'] - comparison_df['P@100_baseline']
    comparison_df['delta_P@200'] = comparison_df['P@200_full'] - comparison_df['P@200_baseline']
    comparison_df['delta_P@300'] = comparison_df['P@300_full'] - comparison_df['P@300_baseline']
    comparison_df['delta_P@500'] = comparison_df['P@500_full'] - comparison_df['P@500_baseline']
    
    comparison_df['lift_auprc_pct'] = (
        comparison_df['delta_auprc'] / (comparison_df['auprc_baseline'] + 1e-9)
    ) * 100
    comparison_df['lift_roc_auc_pct'] = (
        comparison_df['delta_roc_auc'] / (comparison_df['roc_auc_baseline'] + 1e-9)
    ) * 100
    comparison_df['lift_accuracy_pct'] = (
        comparison_df['delta_accuracy'] / (comparison_df['accuracy_baseline'] + 1e-9)
    ) * 100
    comparison_df['lift_precision_pct'] = (
        comparison_df['delta_precision'] / (comparison_df['precision_baseline'] + 1e-9)
    ) * 100
    comparison_df['lift_recall_pct'] = (
        comparison_df['delta_recall'] / (comparison_df['recall_baseline'] + 1e-9)
    ) * 100
    comparison_df['lift_f1_score_pct'] = (
        comparison_df['delta_f1_score'] / (comparison_df['f1_score_baseline'] + 1e-9)
    ) * 100
    comparison_df['lift_P@100_pct'] = (
        comparison_df['delta_P@100'] / (comparison_df['P@100_baseline'] + 1e-9)
    ) * 100
    comparison_df['lift_P@200_pct'] = (
        comparison_df['delta_P@200'] / (comparison_df['P@200_baseline'] + 1e-9)
    ) * 100
    comparison_df['lift_P@300_pct'] = (
        comparison_df['delta_P@300'] / (comparison_df['P@300_baseline'] + 1e-9)
    ) * 100
    comparison_df['lift_P@500_pct'] = (
        comparison_df['delta_P@500'] / (comparison_df['P@500_baseline'] + 1e-9)
    ) * 100
    
    metric_pairs = [
        ('accuracy', 'accuracy'),
        ('precision', 'precision'),
        ('recall', 'recall'),
        ('f1_score', 'f1_score'),
        ('roc_auc', 'roc_auc'),
        ('auprc', 'auprc'),
        ('P@100', 'P@100'),
        ('P@200', 'P@200'),
        ('P@300', 'P@300'),
        ('P@500', 'P@500'),
    ]

    statistical_tests = {}
    for metric_key, metric_col in metric_pairs:
        t_stat, p_value = stats.ttest_rel(
            comparison_df[f'{metric_col}_baseline'].values,
            comparison_df[f'{metric_col}_full'].values
        )
        statistical_tests[metric_key] = {
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'mean_delta': comparison_df[f'delta_{metric_col}'].mean(),
            'std_delta': comparison_df[f'delta_{metric_col}'].std(),
        }
    
    return comparison_df, statistical_tests


def print_ablation_report(
    comparison_df: pd.DataFrame,
    statistical_tests: Dict
) -> None:
    """Print formatted ablation study report."""
    print("\n" + "-" * 120)
    print("PER-WINDOW COMPARISON: Baseline vs Full Model")
    print("-" * 120)
    
    print(
        f"\n{'Window':<10} {'AUPRC (B/F)':<18} {'ROC-AUC (B/F)':<18} "
        f"{'P@100 (B/F)':<18} {'P@200 (B/F)':<18} {'P@300 (B/F)':<18} "
        f"{'P@500 (B/F)':<18} {'AUPRC Lift':<12}"
    )
    print("-" * 120)
    
    for _, row in comparison_df.iterrows():
        print(f"{row['window_id']:<10} "
              f"{row['auprc_baseline']:.3f} / {row['auprc_full']:.3f}  "
              f"{row['roc_auc_baseline']:.3f} / {row['roc_auc_full']:.3f}  "
              f"{row['P@100_baseline']:.3f} / {row['P@100_full']:.3f}  "
              f"{row['P@200_baseline']:.3f} / {row['P@200_full']:.3f}  "
              f"{row['P@300_baseline']:.3f} / {row['P@300_full']:.3f}  "
              f"{row['P@500_baseline']:.3f} / {row['P@500_full']:.3f}  "
              f"{row['lift_auprc_pct']:+.1f}%")
    
    print("\n" + "-" * 120)
    print("STATISTICAL SIGNIFICANCE ANALYSIS")
    print("-" * 120)
    
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
    print("\n" + "-" * 120)
    print("SUMMARY")
    print("-" * 120)
    print(f"Mean AUPRC Lift: {mean_lift:+.2f}%")
    print(f"Windows with positive lift: {(comparison_df['delta_auprc'] > 0).sum()}/{len(comparison_df)}")
    
    overall_baseline_auprc = comparison_df['auprc_baseline'].mean()
    overall_full_auprc = comparison_df['auprc_full'].mean()
    print(f"Overall Baseline AUPRC: {overall_baseline_auprc:.4f}")
    print(f"Overall Full Model AUPRC: {overall_full_auprc:.4f}")


def plot_metric_comparison(comparison_df: pd.DataFrame, output_dir: Path) -> None:
    """Generate a grouped metric comparison plot for baseline vs full model."""
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True, parents=True)

    metric_specs = [
        ("Accuracy", "accuracy"),
        ("Precision", "precision"),
        ("Recall", "recall"),
        ("F1", "f1_score"),
        ("ROC-AUC", "roc_auc"),
        ("AUPRC", "auprc"),
    ]

    baseline_values = [comparison_df[f"{col}_baseline"].mean() for _, col in metric_specs]
    full_values = [comparison_df[f"{col}_full"].mean() for _, col in metric_specs]

    x = np.arange(len(metric_specs))
    width = 0.36

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(13, 7))

    ax.bar(x - width / 2, baseline_values, width, label='Baseline', color='#e74c3c', alpha=0.85)
    ax.bar(x + width / 2, full_values, width, label='Full Model', color='#27ae60', alpha=0.85)

    for idx, (baseline_val, full_val) in enumerate(zip(baseline_values, full_values)):
        ax.text(idx - width / 2, baseline_val + 0.01, f"{baseline_val:.3f}", ha='center', va='bottom', fontsize=8)
        ax.text(idx + width / 2, full_val + 0.01, f"{full_val:.3f}", ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([name for name, _ in metric_specs], rotation=20, ha='right')
    ax.set_ylabel('Average Score Across Windows', fontsize=11, fontweight='bold')
    ax.set_title('Ablation Study: Baseline vs Full Model Metrics', fontsize=13, fontweight='bold')
    ax.set_ylim(0, min(1.05, max(baseline_values + full_values) * 1.2 if baseline_values or full_values else 1.0))
    ax.legend(loc='lower right')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    output_path = plots_dir / "ablation_metric_comparison.png"
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    print(f"✓ Metric comparison plot saved to: {output_path}")



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
    print("\n" + "-" * 120)
    print("GENERATING ABLATION VISUALIZATION")
    print("-" * 120)
    
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


def plot_cascade_funnel(
    baseline_results: Dict,
    full_results: Dict,
    output_dir: Path
) -> None:
    """
    Visualize Two-Stage Cascade Funnel metrics for Baseline vs Full Model.
    
    Creates a side-by-side grouped bar chart comparing:
    - Total Stage 1 Alerts
    - False Positives filtered by Stage 2 (Green - operational win)
    - True Positives lost by Stage 2 (Red - operational cost)
    - Final Stage 2 Alerts sent to investigators
    
    Args:
        baseline_results: Summary dict from baseline model training
        full_results: Summary dict from full model training
        output_dir: Directory to save visualization
    """
    print("\n" + "-" * 120)
    print("GENERATING CASCADE FUNNEL VISUALIZATION")
    print("-" * 120)
    
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True, parents=True)
    
    # Extract cascade metrics from both models
    models_data = {
        'Baseline Model': baseline_results,
        'Full Model': full_results
    }
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    for idx, (model_name, results) in enumerate(models_data.items()):
        ax = axes[idx]
        
        # Extract metrics
        s1_total = results.get('total_s1_flagged', 0)
        fp_filtered = results.get('total_fp_filtered_by_s2', 0)
        tp_lost = results.get('total_tp_lost_by_s2', 0)
        s2_total = results.get('total_s2_flagged', 0)
        
        # Calculate percentages
        fp_filtered_pct = (fp_filtered / s1_total * 100) if s1_total > 0 else 0
        tp_lost_pct = (tp_lost / s1_total * 100) if s1_total > 0 else 0
        
        # Create waterfall-style bars
        categories = ['Stage 1\nTotal Alerts', 'FPs Filtered\n(Win)', 'TPs Lost\n(Cost)', 'Stage 2\nFinal Alerts']
        values = [s1_total, fp_filtered, -tp_lost, s2_total]
        colors = ['#3498db', '#27ae60', '#e74c3c', '#9b59b6']
        
        bars = ax.bar(categories, values, color=colors, alpha=0.85, edgecolor='black', linewidth=1.5)
        
        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, values)):
            height = bar.get_height()
            label_y = height + (50 if height > 0 else -50)
            
            if i == 0 or i == 3:  # Stage 1 and Stage 2 totals
                label = f'{abs(int(height)):,}'
            else:  # FP filtered or TP lost
                label = f'{abs(int(height)):,}\n({abs(height)/s1_total*100:.1f}%)'
            
            ax.text(bar.get_x() + bar.get_width()/2, label_y,
                   label, ha='center', va='bottom' if height > 0 else 'top',
                   fontsize=11, fontweight='bold')
        
        # Styling
        ax.set_title(f'{model_name}\nTwo-Stage Cascade Funnel', fontsize=14, fontweight='bold', pad=20)
        ax.set_ylabel('Number of Transactions', fontsize=12, fontweight='bold')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add annotation box
        annotation_text = (
            f"FPs Eliminated: {fp_filtered_pct:.1f}%\n"
            f"TPs Sacrificed: {tp_lost_pct:.1f}%\n"
            f"Net Reduction: {(s1_total - s2_total)/s1_total*100:.1f}%"
        )
        ax.text(0.98, 0.97, annotation_text,
               transform=ax.transAxes,
               fontsize=10,
               verticalalignment='top',
               horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.9))
    
    plt.tight_layout()
    
    output_path = plots_dir / "cascade_funnel_comparison.png"
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    print(f"✓ Cascade funnel visualization saved to: {output_path}")
    
    # Print console report
    print("\n" + "=" * 80)
    print("CASCADE FUNNEL EFFICIENCY")
    print("=" * 80)
    
    for model_name, results in models_data.items():
        s1_total = results.get('total_s1_flagged', 0)
        s1_fp = results.get('total_s1_fp', 0)
        s1_tp = results.get('total_s1_tp', 0)
        fp_filtered = results.get('total_fp_filtered_by_s2', 0)
        tp_lost = results.get('total_tp_lost_by_s2', 0)
        s2_total = results.get('total_s2_flagged', 0)
        
        fp_elimination_rate = (fp_filtered / s1_fp * 100) if s1_fp > 0 else 0
        tp_sacrifice_rate = (tp_lost / s1_tp * 100) if s1_tp > 0 else 0
        alert_reduction = (s1_total - s2_total) / s1_total * 100 if s1_total > 0 else 0
        
        print(f"\n{model_name}:")
        print(f"  Stage 1 Alerts:              {s1_total:,} ({s1_tp:,} TP + {s1_fp:,} FP)")
        print(f"  Stage 2 Filters Out:         {fp_filtered + tp_lost:,} transactions")
        print(f"    ├─ False Positives Killed: {fp_filtered:,} ({fp_elimination_rate:.1f}% of S1 FPs) ✓")
        print(f"    └─ True Positives Lost:    {tp_lost:,} ({tp_sacrifice_rate:.1f}% of S1 TPs) ✗")
        print(f"  Stage 2 Final Alerts:        {s2_total:,}")
        print(f"  Net Alert Reduction:         {alert_reduction:.1f}%")
        print(f"  ")
        print(f"  OPERATIONAL IMPACT:")
        print(f"    • For every 1 TP sacrificed, Stage 2 eliminates {fp_filtered/tp_lost if tp_lost > 0 else float('inf'):.1f} FPs")
        print(f"    • Investigator workload reduced by {alert_reduction:.1f}%")


def plot_stage_evolution(
    baseline_results: Dict,
    full_results: Dict,
    output_dir: Path
) -> None:
    """
    Visualize Stage 1 vs Stage 2 metric evolution for both models.
    
    Creates a 1x2 multi-panel figure comparing core classification metrics
    (Precision, Recall, F1, AUPRC) at Stage 1 (The Net) vs Stage 2 (Final Cascade).
    
    Args:
        baseline_results: Summary dict from baseline model training
        full_results: Summary dict from full model training
        output_dir: Directory to save visualization
    """
    print("\n" + "-" * 120)
    print("GENERATING STAGE EVOLUTION VISUALIZATION")
    print("-" * 120)
    
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True, parents=True)
    
    # Metric names and display labels
    metrics = ['precision', 'recall', 'f1', 'auprc']
    metric_labels = ['Precision', 'Recall', 'F1', 'AUPRC']
    
    # Color scheme
    stage1_color = '#5DADE2'  # Light blue
    stage2_color = '#8E44AD'  # Dark purple
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    models_data = [
        ('Baseline Model', baseline_results, axes[0]),
        ('Full Model', full_results, axes[1])
    ]
    
    for model_name, results, ax in models_data:
        # Extract Stage 1 and Stage 2 metrics
        stage1_scores = [
            results.get(f's1_overall_{m}', 0) for m in metrics
        ]
        stage2_scores = [
            results.get(f'overall_{m}', 0) for m in metrics
        ]
        
        # Set up grouped bar chart
        x = np.arange(len(metric_labels))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, stage1_scores, width, 
                      label='Stage 1 (The Net)', color=stage1_color, 
                      alpha=0.9, edgecolor='black', linewidth=1.2)
        bars2 = ax.bar(x + width/2, stage2_scores, width, 
                      label='Stage 2 (Final)', color=stage2_color, 
                      alpha=0.9, edgecolor='black', linewidth=1.2)
        
        # Annotate bars with exact scores
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, height + 0.01,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Styling
        ax.set_title(f'{model_name}: Stage 1 vs Stage 2',
                    fontsize=14, fontweight='bold', pad=15)
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_xlabel('Metric', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(metric_labels, fontsize=11)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=11, loc='lower right', framealpha=0.95)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add delta annotations
        for i, (s1, s2) in enumerate(zip(stage1_scores, stage2_scores)):
            delta = s2 - s1
            delta_pct = (delta / s1 * 100) if s1 > 0 else 0
            color = 'green' if delta > 0 else 'red' if delta < 0 else 'gray'
            symbol = '▲' if delta > 0 else '▼' if delta < 0 else '='
            
            ax.text(i, max(s1, s2) + 0.08,
                   f'{symbol} {delta:+.3f}\n({delta_pct:+.1f}%)',
                   ha='center', va='bottom', fontsize=9, 
                   color=color, fontweight='bold')
    
    plt.tight_layout()
    
    output_path = plots_dir / "stage_evolution_comparison.png"
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    print(f"✓ Stage evolution visualization saved to: {output_path}")


def plot_entity_level_recall(
    baseline_preds: pd.DataFrame,
    full_preds: pd.DataFrame,
    output_dir: Path
) -> None:
    """
    Calculate and visualize the Entity-Level Recall.
    Measures the percentage of unique fraudulent accounts that were 
    successfully flagged at least once across all temporal test windows.
    """
    print("\n" + "-" * 120)
    print("OPERATIONAL IMPACT: ENTITY-LEVEL RECALL")
    print("-" * 120)
    
    # Baseline Calculation
    b_fraud = baseline_preds[baseline_preds['y_true'] == 1]
    b_total = b_fraud['entity_id'].nunique()
    b_caught = b_fraud[b_fraud['y_pred'] == 1]['entity_id'].nunique()
    b_recall = b_caught / b_total if b_total > 0 else 0
    
    # Full Model Calculation
    f_fraud = full_preds[full_preds['y_true'] == 1]
    f_total = f_fraud['entity_id'].nunique()
    f_caught = f_fraud[f_fraud['y_pred'] == 1]['entity_id'].nunique()
    f_recall = f_caught / f_total if f_total > 0 else 0
    
    print(f"Baseline Model: Caught {b_caught:,} out of {b_total:,} fraudulent entities ({b_recall:.2%})")
    print(f"Full Model:     Caught {f_caught:,} out of {f_total:,} fraudulent entities ({f_recall:.2%})")
    
    # Visualization
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True, parents=True)
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(8, 6))
    
    models = ['Baseline (Behavioral)', 'Full Model (+Topological)']
    recalls = [b_recall, f_recall]
    colors = ['#e74c3c', '#27ae60']
    
    bars = ax.bar(models, recalls, color=colors, alpha=0.85, edgecolor='black', linewidth=1.2, width=0.5)
    
    for bar, caught, total, rec in zip(bars, [b_caught, f_caught], [b_total, f_total], recalls):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.02,
               f'{rec:.1%}\n({caught:,} / {total:,} Accounts)',
               ha='center', va='bottom', fontsize=11, fontweight='bold')
        
    ax.set_ylabel('Entity-Level Recall (Accounts Caught)', fontsize=12, fontweight='bold')
    ax.set_title('Operational Impact: Entity-Level Recall', fontsize=14, fontweight='bold')
    ax.set_ylim(0, max(recalls) * 1.25 if max(recalls) > 0 else 1.0)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    output_path = plots_dir / "entity_level_recall.png"
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"✓ Entity-Level Recall plot saved to: {output_path}")


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
    
    # NEW INTEGRATION: Calculate and plot entity-level recall while predictions are in memory
    plot_entity_level_recall(baseline_preds, full_preds, results_dir)
    
    # Load model summaries for cascade metrics
    summaries_path = results_dir / "model_summaries.csv"
    if not summaries_path.exists():
        raise FileNotFoundError(
            f"Model summaries not found at {summaries_path}. "
            "Run 05_train_models.py first."
        )
    
    summaries_df = pd.read_csv(summaries_path)
    baseline_summary = summaries_df[summaries_df['model_name'] == 'Baseline'].iloc[0].to_dict()
    full_summary = summaries_df[summaries_df['model_name'] == 'Full Model'].iloc[0].to_dict()
    
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
    
    plot_metric_comparison(comparison_df, results_dir)
    plot_ablation_lift(comparison_df, statistical_tests, results_dir)
    plot_cascade_funnel(baseline_summary, full_summary, results_dir)
    plot_stage_evolution(baseline_summary, full_summary, results_dir)
    
    # Console report: CASCADE METRIC EVOLUTION
    print("\n" + "=" * 80)
    print("CASCADE METRIC EVOLUTION (STAGE 1 -> STAGE 2)")
    print("=" * 80)
    
    metrics_to_track = [
        ('precision', 'Precision'),
        ('recall', 'Recall'),
        ('f1', 'F1 Score'),
        ('auprc', 'AUPRC')
    ]
    
    for model_name, results in [('Baseline Model', baseline_summary), ('Full Model', full_summary)]:
        print(f"\n{model_name}:")
        print("-" * 60)
        
        for metric_key, metric_label in metrics_to_track:
            s1_value = results.get(f's1_overall_{metric_key}', 0)
            s2_value = results.get(f'overall_{metric_key}', 0)
            delta = s2_value - s1_value
            delta_pct = (delta / s1_value * 100) if s1_value > 0 else 0
            
            symbol = '▲' if delta > 0 else '▼' if delta < 0 else '='
            direction = 'LIFT' if delta > 0 else 'DROP' if delta < 0 else 'UNCHANGED'
            
            print(f"  {metric_label:12s}: S1={s1_value:.4f} → S2={s2_value:.4f}  "
                  f"{symbol} {delta:+.4f} ({delta_pct:+.2f}%) [{direction}]")
    
    print("\n" + "=" * 80)
    print("PHASE 3 COMPLETE")
    print("=" * 80)
    print(f"\nOutputs saved to: {results_dir}")
    print("  • ablation_results.csv")
    print("  • ablation_statistical_tests.csv")
    print("  • plots/ablation_metric_comparison.png")
    print("  • plots/ablation_lift.png")
    print("  • plots/cascade_funnel_comparison.png")
    print("  • plots/stage_evolution_comparison.png")
    print("  • plots/entity_level_recall.png")
    
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
