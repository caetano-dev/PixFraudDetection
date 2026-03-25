import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path
import sys
import importlib.util

root_path = Path(__file__).resolve().parent.parent
sys.path.append(str(root_path))

from src.config import OUTPUT_FEATURES_FILE, DATA_PATH

# Dynamically import 04_train_model_forward_chaining.py to handle numeric filenames
module_name = "04_train_model_forward_chaining"
file_path = Path(__file__).resolve().parent / f"{module_name}.py"
if not file_path.exists():
    raise FileNotFoundError(f"Missing {file_path}. Ensure script 04 is in the same directory.")

spec = importlib.util.spec_from_file_location(module_name, file_path)
train_module = importlib.util.module_from_spec(spec)
sys.modules[module_name] = train_module
spec.loader.exec_module(train_module)

forward_chaining_validation = train_module.forward_chaining_validation
generate_shap_analysis = train_module.generate_shap_analysis


# ============================================================================
# FEATURE SET DEFINITIONS
# ============================================================================

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
    'out_degree',
]

FULL_COLS = BEHAVIORAL_COLS + TOPOLOGICAL_COLS


# ============================================================================
# PERFORMANCE LIFT ANALYSIS
# ============================================================================

def compute_performance_lift(
    behavioral_results: pd.DataFrame,
    full_results: pd.DataFrame,
    behavioral_summary: dict,
    full_summary: dict
) -> pd.DataFrame:
    
    print("\n" + "=" * 80)
    print("PERFORMANCE LIFT FROM TOPOLOGICAL FEATURES")
    print("=" * 80)
    
    comparison = behavioral_results.merge(
        full_results,
        on='test_window_id',
        suffixes=('_beh', '_full')
    )
    
    comparison['auprc_lift_%'] = (
        (comparison['auprc_full'] - comparison['auprc_beh']) / 
        (comparison['auprc_beh'] + 1e-9) * 100
    ).fillna(0)
    
    print("\n" + "-" * 80)
    print("PER-WINDOW METRICS COMPARISON (Behavioral vs Full)")
    print("-" * 80)
    print(f"{'Win':<4} | {'AUPRC':<13} | {'ROC AUC':<13} | {'F1-Score':<13} | {'Precision':<13} | {'Recall':<13}")
    print("-" * 80)
    
    for _, row in comparison.iterrows():
        win = str(row['test_window_id'])
        print(f"{win:<4} | "
              f"{row['auprc_beh']:.3f}/{row['auprc_full']:.3f} | "
              f"{row['roc_auc_beh']:.3f}/{row['roc_auc_full']:.3f} | "
              f"{row['f1_score_beh']:.3f}/{row['f1_score_full']:.3f} | "
              f"{row['precision_beh']:.3f}/{row['precision_full']:.3f} | "
              f"{row['recall_beh']:.3f}/{row['recall_full']:.3f}")
    
    print("\n" + "-" * 80)
    print("AGGREGATE METRICS COMPARISON")
    print("-" * 80)
    
    # Calculate overall F1 for summaries (Harmonic mean of overall Precision and Recall)
    def calc_f1(p, r):
        return 2 * (p * r) / (p + r) if (p + r) > 0 else 0.0

    beh_f1 = calc_f1(behavioral_summary['overall_precision'], behavioral_summary['overall_recall'])
    full_f1 = calc_f1(full_summary['overall_precision'], full_summary['overall_recall'])

    metrics_comparison = [
        ("AUPRC", behavioral_summary['overall_auprc'], full_summary['overall_auprc']),
        ("ROC AUC", behavioral_summary['overall_roc_auc'], full_summary['overall_roc_auc']),
        ("F1-Score", beh_f1, full_f1),
        ("Precision", behavioral_summary['overall_precision'], full_summary['overall_precision']),
        ("Recall", behavioral_summary['overall_recall'], full_summary['overall_recall'])
    ]

    for name, beh_val, full_val in metrics_comparison:
        lift = ((full_val - beh_val) / (beh_val + 1e-9)) * 100
        print(f"{name:<12}: Behavioral = {beh_val:.4f} | Full = {full_val:.4f} | Lift = {lift:+.2f}%")
    
    # Statistical significance (paired t-test on AUPRC)
    t_stat, p_value = stats.ttest_rel(
        comparison['auprc_beh'].values,
        comparison['auprc_full'].values
    )
    print(f"\nStatistical Significance (paired t-test on AUPRC):")
    print(f"  t-statistic: {t_stat:.4f}")
    print(f"  p-value:     {p_value:.4e}")
    if p_value < 0.05:
        print(f"  ✓ Topological features provide STATISTICALLY SIGNIFICANT difference (p < 0.05)")
    else:
        print(f"  ✗ Difference not statistically significant at α=0.05")
    
    return comparison


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_ablation_comparison(comparison_df: pd.DataFrame, output_dir: Path):
    
    print("\n" + "=" * 80)
    print("GENERATING ABLATION STUDY VISUALIZATIONS")
    print("=" * 80)
    
    fig, ax = plt.subplots(figsize=(14, 6))
   
    x = comparison_df['test_window_id'].astype(str).values
    width = 0.35
    x_pos = np.arange(len(x))
    
    behavioral_bars = ax.bar(
        x_pos - width/2, comparison_df['auprc_beh'].values,
        width, label='Behavioral-only (Baseline)', color='#e74c3c', alpha=0.8
    )
    full_bars = ax.bar(
        x_pos + width/2, comparison_df['auprc_full'].values,
        width, label='Full Model (Behavioral + Topological)', color='#27ae60', alpha=0.8
    )
    
    # Auto-label the bar heights using the previously unused variables
    ax.bar_label(behavioral_bars, fmt='%.3f', padding=3, fontsize=8)
    ax.bar_label(full_bars, fmt='%.3f', padding=3, fontsize=8)
    
    for i, row in comparison_df.iterrows():
        lift = row['auprc_lift_%']
        y_pos = max(row['auprc_beh'], row['auprc_full']) + 0.05
        ax.text(i, y_pos, f'{lift:+.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Test Window ID', fontsize=12, fontweight='bold')
    ax.set_ylabel('AUPRC', fontsize=12, fontweight='bold')
    ax.set_title('Ablation Study: Impact of Topological Features on Fraud Detection', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x)
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, max(comparison_df['auprc_full'].max(), comparison_df['auprc_beh'].max()) * 1.25)
    
    fig.tight_layout()
    output_dir.mkdir(exist_ok=True, parents=True)
    ablation_plot_path = output_dir / "ablation_study_results.png"
    
    # Using the bound fig variable to save
    fig.savefig(ablation_plot_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"✓ Ablation study plot saved to: {ablation_plot_path}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    features_path = DATA_PATH / OUTPUT_FEATURES_FILE
    if not features_path.exists():
        raise FileNotFoundError(f"Missing {features_path}.")
    
    print("=" * 80)
    print("ABLATION STUDY: EMPIRICAL VALIDATION OF TOPOLOGICAL FEATURES")
    print("=" * 80)
    df = pd.read_parquet(features_path)
    
    window_col = 'window_id' if 'window_id' in df.columns else 'date'
    
    # Run Baseline
    print("\n" + "-" * 80)
    print(f"PHASE 1: BEHAVIORAL BASELINE ({len(BEHAVIORAL_COLS)} features)")
    print("-" * 80)
    beh_results_df, beh_summary, _, _ = forward_chaining_validation(
        df=df, feature_cols=BEHAVIORAL_COLS, window_col=window_col,
        target_col="is_fraud", min_train_windows=2, n_trials=20, verbose=True
    )
    
    # Run Proposed Model
    print("\n" + "-" * 80)
    print(f"PHASE 2: FULL MODEL ({len(FULL_COLS)} features)")
    print("-" * 80)
    full_results_df, full_summary, full_global_shap, full_global_x_test = forward_chaining_validation(
        df=df, feature_cols=FULL_COLS, window_col=window_col,
        target_col="is_fraud", min_train_windows=2, n_trials=20, verbose=True
    )
    
    # Analyze Lift
    comparison_df = compute_performance_lift(
        beh_results_df, full_results_df, beh_summary, full_summary
    )
    
    # Save Results & Plots
    output_dir = DATA_PATH / "results"
    output_dir.mkdir(exist_ok=True)
    
    comparison_df.to_csv(output_dir / "ablation_comparison.csv", index=False)
    plot_ablation_comparison(comparison_df, output_dir)
    
    # Interpretability
    print("\n" + "-" * 80)
    print("PHASE 3: GLOBAL INTERPRETABILITY (SHAP)")
    print("-" * 80)
    shap_values, feature_importance = generate_shap_analysis(
        global_shap_values=full_global_shap,
        global_x_test_df=full_global_x_test,
        output_dir=output_dir
    )
    
    feature_importance.to_csv(output_dir / "shap_feature_importance.csv", index=False)
    
    # Serialize the raw SHAP array using the previously unused variable
    np.save(output_dir / "raw_shap_values.npy", shap_values)
    print(f"✓ Raw SHAP matrix serialized to: {output_dir / 'raw_shap_values.npy'}")
    
    print("\n" + "=" * 80)
    print("PIPELINE EXECUTION COMPLETE.")
    print("=" * 80)

if __name__ == "__main__":
    main()
