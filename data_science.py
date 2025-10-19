import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
from scipy import stats
from scipy.stats import mannwhitneyu, wilcoxon, friedmanchisquare
from sklearn.preprocessing import StandardScaler
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

warnings.filterwarnings('ignore')

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 16

# ============================================================================
# CONFIGURATION FOR GOOGLE COLAB
# ============================================================================

# Mount Google Drive (uncomment when running in Colab)
from google.colab import drive
try:
  drive.mount('/content/drive', force_remount=False)
except ValueError:
  print("Drive already mounted")

# Path configuration - MODIFY THIS TO YOUR GOOGLE DRIVE PATH
BASE_PATH = Path('/content/drive/MyDrive/AML/processed/LI_Small/US_Dollar')

CSV_PATH = BASE_PATH / 'metrics' / 'window_metrics.csv'
print("csv path")
print(CSV_PATH)

# Alternative: Use local path if file is uploaded to Colab
# CSV_PATH = 'window_metrics.csv'

# Output directory for saving plots
OUTPUT_DIR = BASE_PATH / 'metrics' / 'analysis_plots'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("="*80)
print("MONEY LAUNDERING DETECTION - DATA SCIENCE ANALYSIS")
print("="*80)
print(f"\nData path: {CSV_PATH}")
print(f"Output directory: {OUTPUT_DIR}")
print()

# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================

def load_and_prepare_data(csv_path):
    """
    Load and prepare the metrics data for analysis.
    
    Parameters:
    -----------
    csv_path : Path
        Path to the window_metrics.csv file
        
    Returns:
    --------
    df : pd.DataFrame
        Prepared dataframe with additional computed columns
    """
    print("Loading data...")
    df = pd.read_csv(csv_path)
    
    # Convert date columns to datetime
    df['ws'] = pd.to_datetime(df['ws'])
    df['we'] = pd.to_datetime(df['we'])
    
    # Create additional useful columns
    df['window_duration'] = (df['we'] - df['ws']).dt.days
    df['window_id'] = df.groupby(['window_days', 'ws']).ngroup()
    
    # Calculate false positive rate at different k values
    for k in ['0.5pct', '1.0pct', '2.0pct', '5.0pct']:
        precision_col = f'p_at_{k}'
        if precision_col in df.columns:
            df[f'fpr_at_{k}'] = 1 - df[precision_col]
    
    # Calculate F1 score approximation (using prevalence as recall proxy)
    if 'p_at_1.0pct' in df.columns and 'prevalence_eval' in df.columns:
        precision = df['p_at_1.0pct']
        recall = df['prevalence_eval']
        df['f1_approx'] = 2 * (precision * recall) / (precision + recall + 1e-10)
    
    print(f"✓ Loaded {len(df):,} records")
    print(f"✓ Date range: {df['ws'].min()} to {df['we'].max()}")
    print(f"✓ Methods: {df['method'].nunique()}")
    print(f"✓ Window configurations: {df['window_days'].nunique()}")
    
    return df

# ============================================================================
# EXPLORATORY DATA ANALYSIS
# ============================================================================

def print_data_summary(df):
    """Print comprehensive data summary statistics."""
    print("\n" + "="*80)
    print("DATA SUMMARY")
    print("="*80)
    
    print("\n1. Dataset Overview")
    print("-" * 80)
    print(f"Total records: {len(df):,}")
    print(f"Methods analyzed: {df['method'].nunique()}")
    print(f"Time windows: {df['window_days'].nunique()} configurations ({df['window_days'].unique()})")
    print(f"Total unique windows: {df['window_id'].nunique()}")
    
    print("\n2. Methods Evaluated")
    print("-" * 80)
    methods = sorted(df['method'].unique())
    for i, method in enumerate(methods, 1):
        count = len(df[df['method'] == method])
        print(f"{i:2d}. {method:30s} ({count:3d} records)")
    
    print("\n3. Key Metrics Available")
    print("-" * 80)
    metrics_info = {
        'ap': 'Average Precision',
        'p_at_1.0pct': 'Precision @ 1%',
        'attcov_at_1.0pct': 'Attempt Coverage @ 1%',
        'lift_p_at_1.0pct': 'Lift @ 1%',
        'prevalence_eval': 'Fraud Prevalence',
        'nodes': 'Graph Nodes',
        'edges': 'Graph Edges',
        'pos_nodes': 'Positive Nodes'
    }
    
    for col, desc in metrics_info.items():
        if col in df.columns:
            print(f"  • {desc:30s}: {col}")
    
    print("\n4. Graph Statistics (Average)")
    print("-" * 80)
    print(f"Nodes per window: {df['nodes'].mean():,.0f} (±{df['nodes'].std():,.0f})")
    print(f"Edges per window: {df['edges'].mean():,.0f} (±{df['edges'].std():,.0f})")
    print(f"Positive nodes: {df['pos_nodes'].mean():,.0f} (±{df['pos_nodes'].std():,.0f})")
    print(f"Fraud prevalence: {df['prevalence_eval'].mean()*100:.3f}% (±{df['prevalence_eval'].std()*100:.3f}%)")
    
    print("\n5. Performance Ranges")
    print("-" * 80)
    for metric in ['ap', 'p_at_1.0pct', 'attcov_at_1.0pct']:
        if metric in df.columns:
            print(f"{metric:20s}: {df[metric].min():.4f} to {df[metric].max():.4f}")

def analyze_missing_data(df):
    """Analyze and visualize missing data patterns."""
    print("\n" + "="*80)
    print("MISSING DATA ANALYSIS")
    print("="*80)
    
    missing = df.isnull().sum()
    missing_pct = 100 * missing / len(df)
    
    missing_df = pd.DataFrame({
        'Missing_Count': missing,
        'Percentage': missing_pct
    }).sort_values('Percentage', ascending=False)
    
    missing_df = missing_df[missing_df['Missing_Count'] > 0]
    
    if len(missing_df) > 0:
        print(f"\nFound {len(missing_df)} columns with missing data:")
        print(missing_df)
        
        # Visualize
        if len(missing_df) <= 20:
            fig, ax = plt.subplots(figsize=(10, 6))
            missing_df['Percentage'].plot(kind='barh', ax=ax, color='coral')
            ax.set_xlabel('Percentage Missing (%)')
            ax.set_title('Missing Data by Column')
            ax.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            plt.savefig(OUTPUT_DIR / '01_missing_data.png', dpi=300, bbox_inches='tight')
            plt.show()
    else:
        print("\n✓ No missing data found!")

# ============================================================================
# PERFORMANCE ANALYSIS
# ============================================================================

def analyze_method_performance(df, metric='ap', top_n=10):
    """
    Analyze and visualize method performance.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Metrics dataframe
    metric : str
        Performance metric to analyze
    top_n : int
        Number of top methods to highlight
    """
    print(f"\n{'='*80}")
    print(f"METHOD PERFORMANCE ANALYSIS - {metric.upper()}")
    print("="*80)
    
    # Calculate statistics by method
    method_stats = df.groupby('method')[metric].agg([
        'mean', 'median', 'std', 'min', 'max', 'count'
    ]).sort_values('median', ascending=False)
    
    print(f"\nTop {top_n} Methods by Median {metric.upper()}:")
    print("-" * 80)
    print(method_stats.head(top_n).to_string())
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Box plot of top methods
    ax1 = axes[0, 0]
    top_methods = method_stats.head(top_n).index
    df_top = df[df['method'].isin(top_methods)]
    df_top_sorted = df_top.copy()
    df_top_sorted['method'] = pd.Categorical(
        df_top_sorted['method'], 
        categories=top_methods, 
        ordered=True
    )
    sns.boxplot(data=df_top_sorted, y='method', x=metric, ax=ax1, palette='Set2')
    ax1.set_title(f'Distribution of {metric.upper()} - Top {top_n} Methods')
    ax1.set_xlabel(metric.upper())
    ax1.set_ylabel('Method')
    ax1.axvline(df[df['method']=='random'][metric].median(), 
                color='red', linestyle='--', label='Random Baseline', linewidth=2)
    ax1.legend()
    ax1.grid(axis='x', alpha=0.3)
    
    # 2. Bar plot with error bars (median + IQR)
    ax2 = axes[0, 1]
    method_stats_top = method_stats.head(top_n)
    x_pos = np.arange(len(method_stats_top))
    ax2.barh(x_pos, method_stats_top['median'], 
             xerr=method_stats_top['std'], 
             alpha=0.7, capsize=5, color='steelblue')
    ax2.set_yticks(x_pos)
    ax2.set_yticklabels(method_stats_top.index)
    ax2.set_xlabel(f'Median {metric.upper()}')
    ax2.set_title(f'Method Ranking by Median {metric.upper()}')
    ax2.axvline(df[df['method']=='random'][metric].median(), 
                color='red', linestyle='--', label='Random', linewidth=2)
    ax2.legend()
    ax2.grid(axis='x', alpha=0.3)
    ax2.invert_yaxis()
    
    # 3. Violin plot
    ax3 = axes[1, 0]
    sns.violinplot(data=df_top_sorted, y='method', x=metric, ax=ax3, palette='Set3')
    ax3.set_title(f'Distribution Density - {metric.upper()}')
    ax3.set_xlabel(metric.upper())
    ax3.set_ylabel('')
    ax3.grid(axis='x', alpha=0.3)
    
    # 4. Performance vs Random comparison
    ax4 = axes[1, 1]
    random_median = df[df['method']=='random'][metric].median()
    improvements = ((method_stats_top['median'] - random_median) / random_median * 100)
    colors = ['green' if x > 0 else 'red' for x in improvements]
    ax4.barh(range(len(improvements)), improvements, color=colors, alpha=0.7)
    ax4.set_yticks(range(len(improvements)))
    ax4.set_yticklabels(improvements.index)
    ax4.set_xlabel('Improvement over Random (%)')
    ax4.set_title(f'Relative Performance - {metric.upper()}')
    ax4.axvline(0, color='black', linestyle='-', linewidth=1)
    ax4.grid(axis='x', alpha=0.3)
    ax4.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'02_method_performance_{metric}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return method_stats

def compare_multiple_metrics(df, metrics=['ap', 'p_at_1.0pct', 'attcov_at_1.0pct'], top_n=10):
    """Compare methods across multiple metrics simultaneously."""
    print(f"\n{'='*80}")
    print("MULTI-METRIC COMPARISON")
    print("="*80)
    
    # Calculate median for each metric and method
    results = {}
    for metric in metrics:
        if metric in df.columns:
            results[metric] = df.groupby('method')[metric].median()
    
    comparison_df = pd.DataFrame(results)
    
    # Normalize to 0-100 scale for comparison
    scaler = StandardScaler()
    comparison_normalized = pd.DataFrame(
        scaler.fit_transform(comparison_df),
        index=comparison_df.index,
        columns=comparison_df.columns
    )
    
    # Calculate composite score
    comparison_normalized['composite_score'] = comparison_normalized.mean(axis=1)
    comparison_sorted = comparison_normalized.sort_values('composite_score', ascending=False)
    
    print("\nTop methods by composite score (normalized):")
    print(comparison_sorted.head(top_n))
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Heatmap
    ax1 = axes[0]
    top_methods = comparison_sorted.head(top_n).index
    sns.heatmap(comparison_df.loc[top_methods], annot=True, fmt='.4f', 
                cmap='YlGnBu', ax=ax1, cbar_kws={'label': 'Metric Value'})
    ax1.set_title(f'Performance Heatmap - Top {top_n} Methods')
    ax1.set_xlabel('Metric')
    ax1.set_ylabel('Method')
    
    # Radar/Spider chart
    ax2 = axes[1]
    top_5 = comparison_sorted.head(5).drop('composite_score', axis=1)
    
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    ax2 = plt.subplot(122, projection='polar')
    
    for idx, method in enumerate(top_5.index):
        values = top_5.loc[method].values.tolist()
        values += values[:1]  # Complete the circle
        ax2.plot(angles, values, 'o-', linewidth=2, label=method)
        ax2.fill(angles, values, alpha=0.15)
    
    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels(metrics)
    ax2.set_ylim(-2, 2)
    ax2.set_title('Multi-Metric Performance (Normalized)', y=1.08)
    ax2.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '03_multi_metric_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return comparison_df, comparison_normalized

# ============================================================================
# TEMPORAL ANALYSIS
# ============================================================================

def analyze_temporal_trends(df, methods=['pagerank_wlog', 'in_deg', 'ensemble_top3', 'random']):
    """Analyze how method performance changes over time windows."""
    print(f"\n{'='*80}")
    print("TEMPORAL TREND ANALYSIS")
    print("="*80)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    metrics_to_plot = ['ap', 'p_at_1.0pct', 'attcov_at_1.0pct', 'lift_p_at_1.0pct']
    
    for idx, metric in enumerate(metrics_to_plot):
        ax = axes[idx // 2, idx % 2]
        
        for method in methods:
            method_data = df[df['method'] == method].sort_values('ws')
            
            # Separate by window size
            for window_days in sorted(df['window_days'].unique()):
                window_data = method_data[method_data['window_days'] == window_days]
                if not window_data.empty:
                    label = f"{method} ({window_days}d)"
                    marker = 'o' if window_days == 3 else 's'
                    ax.plot(window_data['ws'], window_data[metric], 
                           marker=marker, label=label, alpha=0.7, linewidth=2)
        
        ax.set_xlabel('Time Window Start')
        ax.set_ylabel(metric.upper())
        ax.set_title(f'Temporal Evolution - {metric.upper()}')
        ax.legend(fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '04_temporal_trends.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Stability analysis
    print("\nStability Analysis (Standard Deviation across windows):")
    print("-" * 80)
    stability = df.groupby('method')['ap'].std().sort_values()
    print(stability.head(10))
    
    return stability

def analyze_window_size_effect(df):
    """Compare performance between different window sizes."""
    print(f"\n{'='*80}")
    print("WINDOW SIZE EFFECT ANALYSIS")
    print("="*80)
    
    # Statistical comparison
    metrics = ['ap', 'p_at_1.0pct', 'attcov_at_1.0pct']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        # Create comparison data
        window_comparison = []
        for window_days in sorted(df['window_days'].unique()):
            window_data = df[df['window_days'] == window_days]
            window_comparison.append({
                'window_days': f"{window_days}-day",
                'median': window_data[metric].median(),
                'mean': window_data[metric].mean(),
                'std': window_data[metric].std()
            })
        
        comp_df = pd.DataFrame(window_comparison)
        
        # Violin plot by window size
        df_plot = df[['window_days', metric]].copy()
        df_plot['window_days'] = df_plot['window_days'].astype(str) + '-day'
        
        sns.violinplot(data=df_plot, x='window_days', y=metric, ax=ax, palette='Set2')
        ax.set_xlabel('Window Size')
        ax.set_ylabel(metric.upper())
        ax.set_title(f'{metric.upper()} by Window Size')
        ax.grid(axis='y', alpha=0.3)
        
        # Add mean markers
        for i, row in comp_df.iterrows():
            ax.scatter(i, row['mean'], color='red', s=100, zorder=10, 
                      marker='D', edgecolors='black', label='Mean' if i == 0 else '')
        
        if idx == 0:
            ax.legend()
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '05_window_size_effect.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Statistical tests
    print("\nStatistical Significance Tests (Mann-Whitney U):")
    print("-" * 80)
    
    window_sizes = sorted(df['window_days'].unique())
    if len(window_sizes) == 2:
        for metric in metrics:
            data1 = df[df['window_days'] == window_sizes[0]][metric].dropna()
            data2 = df[df['window_days'] == window_sizes[1]][metric].dropna()
            
            statistic, pvalue = mannwhitneyu(data1, data2, alternative='two-sided')
            
            print(f"\n{metric}:")
            print(f"  {window_sizes[0]}-day median: {data1.median():.6f}")
            print(f"  {window_sizes[1]}-day median: {data2.median():.6f}")
            print(f"  U-statistic: {statistic:.2f}")
            print(f"  p-value: {pvalue:.4f}")
            print(f"  Significant: {'Yes' if pvalue < 0.05 else 'No'} (α=0.05)")

# ============================================================================
# CORRELATION AND FEATURE ANALYSIS
# ============================================================================

def analyze_correlations(df):
    """Analyze correlations between different metrics."""
    print(f"\n{'='*80}")
    print("CORRELATION ANALYSIS")
    print("="*80)
    
    # Select numeric columns for correlation
    numeric_cols = [
        'ap', 'p_at_1.0pct', 'p_at_2.0pct', 'p_at_5.0pct',
        'attcov_at_1.0pct', 'attcov_at_2.0pct', 'attcov_at_5.0pct',
        'lift_p_at_1.0pct', 'prevalence_eval', 'nodes', 'edges', 'pos_nodes'
    ]
    
    # Filter existing columns
    numeric_cols = [col for col in numeric_cols if col in df.columns]
    
    corr_matrix = df[numeric_cols].corr()
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    
    # Full correlation heatmap
    ax1 = axes[0]
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, ax=ax1, cbar_kws={'label': 'Correlation'})
    ax1.set_title('Correlation Matrix - All Metrics')
    
    # Focus on key metrics
    key_metrics = ['ap', 'p_at_1.0pct', 'attcov_at_1.0pct', 'lift_p_at_1.0pct', 
                   'nodes', 'edges', 'pos_nodes']
    key_metrics = [col for col in key_metrics if col in df.columns]
    
    ax2 = axes[1]
    key_corr = df[key_metrics].corr()
    sns.heatmap(key_corr, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, ax=ax2, cbar_kws={'label': 'Correlation'}, 
                square=True, linewidths=1)
    ax2.set_title('Correlation Matrix - Key Metrics')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '06_correlations.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print strongest correlations
    print("\nStrongest Positive Correlations:")
    print("-" * 80)
    corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_pairs.append({
                'metric1': corr_matrix.columns[i],
                'metric2': corr_matrix.columns[j],
                'correlation': corr_matrix.iloc[i, j]
            })
    
    corr_df = pd.DataFrame(corr_pairs).sort_values('correlation', ascending=False)
    print(corr_df.head(10).to_string(index=False))
    
    return corr_matrix

def analyze_graph_statistics(df):
    """Analyze relationship between graph properties and performance."""
    print(f"\n{'='*80}")
    print("GRAPH STATISTICS ANALYSIS")
    print("="*80)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    top_methods = df.groupby('method')['ap'].median().nlargest(5).index
    
    # Scatter plots
    scatter_configs = [
        ('nodes', 'ap', 'Number of Nodes vs AP'),
        ('edges', 'ap', 'Number of Edges vs AP'),
        ('pos_nodes', 'ap', 'Positive Nodes vs AP'),
        ('nodes', 'p_at_1.0pct', 'Nodes vs Precision@1%'),
        ('prevalence_eval', 'ap', 'Prevalence vs AP'),
        ('prevalence_eval', 'p_at_1.0pct', 'Prevalence vs Precision@1%')
    ]
    
    for idx, (x_col, y_col, title) in enumerate(scatter_configs):
        ax = axes[idx // 3, idx % 3]
        
        for method in top_methods:
            method_data = df[df['method'] == method]
            ax.scatter(method_data[x_col], method_data[y_col], 
                      label=method, alpha=0.6, s=50)
        
        ax.set_xlabel(x_col.replace('_', ' ').title())
        ax.set_ylabel(y_col.replace('_', ' ').upper())
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '07_graph_statistics.png', dpi=300, bbox_inches='tight')
    plt.show()

# ============================================================================
# ENSEMBLE AND METHOD COMPARISON
# ============================================================================

def analyze_ensemble_methods(df):
    """Analyze ensemble methods vs individual methods."""
    print(f"\n{'='*80}")
    print("ENSEMBLE METHODS ANALYSIS")
    print("="*80)
    
    # Identify ensemble vs individual methods
    ensemble_methods = [m for m in df['method'].unique() if 'ensemble' in m.lower()]
    individual_methods = [m for m in df['method'].unique() 
                         if 'ensemble' not in m.lower() and m != 'random']
    
    print(f"\nEnsemble methods: {len(ensemble_methods)}")
    print(f"Individual methods: {len(individual_methods)}")
    
    # Performance comparison
    metrics = ['ap', 'p_at_1.0pct', 'attcov_at_1.0pct']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        # Prepare data
        ensemble_data = df[df['method'].isin(ensemble_methods)]
        individual_data = df[df['method'].isin(individual_methods)]
        
        plot_data = pd.DataFrame({
            'Ensemble': ensemble_data[metric],
            'Individual': individual_data[metric]
        })
        
        # Box plot
        bp = ax.boxplot([ensemble_data[metric].dropna(), 
                         individual_data[metric].dropna()],
                        labels=['Ensemble', 'Individual'],
                        patch_artist=True,
                        showmeans=True)
        
        # Color boxes
        colors = ['lightblue', 'lightgreen']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        ax.set_ylabel(metric.upper())
        ax.set_title(f'{metric.upper()}: Ensemble vs Individual')
        ax.grid(axis='y', alpha=0.3)
        
        # Add mean values as text
        ensemble_mean = ensemble_data[metric].mean()
        individual_mean = individual_data[metric].mean()
        ax.text(1, ensemble_mean, f'{ensemble_mean:.4f}', 
               ha='center', va='bottom', fontweight='bold')
        ax.text(2, individual_mean, f'{individual_mean:.4f}', 
               ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '08_ensemble_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Statistical test
    print("\nStatistical Tests (Mann-Whitney U):")
    print("-" * 80)
    
    for metric in metrics:
        ensemble_vals = df[df['method'].isin(ensemble_methods)][metric].dropna()
        individual_vals = df[df['method'].isin(individual_methods)][metric].dropna()
        
        if len(ensemble_vals) > 0 and len(individual_vals) > 0:
            statistic, pvalue = mannwhitneyu(ensemble_vals, individual_vals)
            
            print(f"\n{metric}:")
            print(f"  Ensemble median: {ensemble_vals.median():.6f}")
            print(f"  Individual median: {individual_vals.median():.6f}")
            print(f"  p-value: {pvalue:.4f}")
            print(f"  Significant: {'Yes' if pvalue < 0.05 else 'No'}")

def analyze_method_categories(df):
    """Categorize and analyze methods by type."""
    print(f"\n{'='*80}")
    print("METHOD CATEGORY ANALYSIS")
    print("="*80)
    
    # Define categories
    categories = {
        'Graph Structure': ['pagerank_wlog', 'in_deg', 'out_deg', 'hits_hub', 'hits_auth'],
        'Graph Decomposition': ['kcore_in', 'kcore_out', 'kcore_und'],
        'Transaction Volume': ['in_tx', 'out_tx', 'in_amt', 'out_amt'],
        'Heuristic': ['collector', 'distributor', 'pattern_features'],
        'Community': ['communities_unsup_louvain', 'communities_unsup_leiden'],
        'Ensemble': ['ensemble_top3', 'ensemble_ultimate', 'ensemble_diverse', 
                     'ensemble_pattern', 'ensemble_seeded'],
        'Advanced': ['seeded_pr'],
        'Baseline': ['random']
    }
    
    # Add category column
    def get_category(method):
        for cat, methods in categories.items():
            if method in methods:
                return cat
        return 'Other'
    
    df['category'] = df['method'].apply(get_category)
    
    # Performance by category
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    metrics = ['ap', 'p_at_1.0pct', 'attcov_at_1.0pct', 'lift_p_at_1.0pct']
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        
        category_stats = df.groupby('category')[metric].median().sort_values(ascending=False)
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(category_stats)))
        bars = ax.barh(range(len(category_stats)), category_stats.values, color=colors)
        ax.set_yticks(range(len(category_stats)))
        ax.set_yticklabels(category_stats.index)
        ax.set_xlabel(f'Median {metric.upper()}')
        ax.set_title(f'Performance by Category - {metric.upper()}')
        ax.grid(axis='x', alpha=0.3)
        ax.invert_yaxis()
        
        # Add values on bars
        for i, (cat, val) in enumerate(category_stats.items()):
            ax.text(val, i, f' {val:.4f}', va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '09_category_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print category summary
    print("\nCategory Performance Summary:")
    print("-" * 80)
    for metric in ['ap', 'p_at_1.0pct']:
        print(f"\n{metric.upper()}:")
        cat_stats = df.groupby('category')[metric].agg(['median', 'mean', 'std', 'count'])
        print(cat_stats.sort_values('median', ascending=False).to_string())

# ============================================================================
# COMMUNITY DETECTION ANALYSIS - GLOBAL (ENTIRE GRAPH) vs WINDOW-BASED
# ============================================================================

def analyze_community_methods(base_path):
    """
    Analyze GLOBAL community detection (entire graph) for Louvain and Leiden.
    
    This analysis examines communities detected on the COMPLETE graph (all time periods),
    as opposed to the window-based community detection in the main metrics CSV.
    
    Key Distinction:
    - GLOBAL: Communities detected on entire transaction network (persistent structures)
    - WINDOWS: Communities detected in time windows (temporal patterns, in window_metrics.csv)
    
    Parameters:
    -----------
    base_path : Path
        Base path to the data directory (contains communities folder)
    """
    print(f"\n{'='*80}")
    print("COMMUNITY DETECTION: GLOBAL GRAPH ANALYSIS")
    print("="*80)
    print("\nNote: This analyzes communities detected on the ENTIRE graph,")
    print("      not the window-based detection (see window_metrics.csv for that).\n")
    
    # Define paths for community metrics
    louvain_path = base_path / 'communities' / 'louvain' / 'community_summary.csv'
    leiden_path = base_path / 'communities' / 'leiden' / 'community_summary.csv'
    
    # Load community data
    try:
        louvain_df = pd.read_csv(louvain_path)
        louvain_df['method'] = 'Louvain'
        print(f"✓ Loaded Louvain: {len(louvain_df):,} communities")
    except FileNotFoundError:
        print(f"⚠ Warning: Could not find {louvain_path}")
        louvain_df = None
    
    try:
        leiden_df = pd.read_csv(leiden_path)
        leiden_df['method'] = 'Leiden'
        print(f"✓ Loaded Leiden: {len(leiden_df):,} communities")
    except FileNotFoundError:
        print(f"⚠ Warning: Could not find {leiden_path}")
        leiden_df = None
    
    # If no data, return early
    if louvain_df is None and leiden_df is None:
        print("\n❌ No community detection data found. Skipping community analysis.")
        return
    
    # Combine data
    if louvain_df is not None and leiden_df is not None:
        combined_df = pd.concat([louvain_df, leiden_df], ignore_index=True)
    elif louvain_df is not None:
        combined_df = louvain_df
    else:
        combined_df = leiden_df
    
    # Print summary statistics
    print("\n" + "-"*80)
    print("COMMUNITY DETECTION SUMMARY")
    print("-"*80)
    
    for method in combined_df['method'].unique():
        method_df = combined_df[combined_df['method'] == method]
        print(f"\n{method} Communities (GLOBAL - Entire Graph):")
        print(f"  Total communities: {len(method_df):,}")
        print(f"  Avg community size: {method_df['size'].mean():.2f}")
        print(f"  Median community size: {method_df['size'].median():.0f}")
        print(f"  Avg density: {method_df['density'].mean():.4f}")
        print(f"  Communities with laundering: {(method_df['laundering_nodes'] > 0).sum():,}")
        print(f"  Total laundering nodes detected: {method_df['laundering_nodes'].sum():,.0f}")
        print(f"  Avg laundering %: {method_df['laundering_pct'].mean():.4f}")
        print(f"  Total attempts covered: {method_df['num_attempts'].sum():,.0f}")
        print(f"  >>> This represents persistent fraud structures across ALL time periods")
    
    # Create comprehensive visualization
    fig = plt.figure(figsize=(18, 14))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # 1. Score Distribution Comparison
    ax1 = fig.add_subplot(gs[0, 0])
    for method in combined_df['method'].unique():
        method_df = combined_df[combined_df['method'] == method]
        ax1.hist(method_df['score'], bins=30, alpha=0.6, label=method, edgecolor='black')
    ax1.set_xlabel('Community Score')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Score Distribution by Method')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # 2. Community Size Distribution
    ax2 = fig.add_subplot(gs[0, 1])
    for method in combined_df['method'].unique():
        method_df = combined_df[combined_df['method'] == method]
        ax2.hist(method_df['size'], bins=50, alpha=0.6, label=method, edgecolor='black', range=(0, 100))
    ax2.set_xlabel('Community Size (nodes)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Community Size Distribution (0-100 nodes)')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    # 3. Density Distribution
    ax3 = fig.add_subplot(gs[0, 2])
    for method in combined_df['method'].unique():
        method_df = combined_df[combined_df['method'] == method]
        ax3.hist(method_df['density'], bins=30, alpha=0.6, label=method, edgecolor='black')
    ax3.set_xlabel('Community Density')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Density Distribution by Method')
    ax3.legend()
    ax3.grid(alpha=0.3)
    
    # 4. Laundering Detection Rate
    ax4 = fig.add_subplot(gs[1, 0])
    method_stats = []
    for method in combined_df['method'].unique():
        method_df = combined_df[combined_df['method'] == method]
        total_comm = len(method_df)
        laundering_comm = (method_df['laundering_nodes'] > 0).sum()
        pct = (laundering_comm / total_comm) * 100
        method_stats.append({'method': method, 'pct': pct, 'count': laundering_comm})
    
    stats_df = pd.DataFrame(method_stats)
    bars = ax4.bar(stats_df['method'], stats_df['pct'], color=['#FF6B6B', '#4ECDC4'], edgecolor='black')
    ax4.set_ylabel('% of Communities')
    ax4.set_title('Communities with Laundering Nodes')
    ax4.grid(axis='y', alpha=0.3)
    for i, (method, pct, count) in enumerate(zip(stats_df['method'], stats_df['pct'], stats_df['count'])):
        ax4.text(i, pct + 0.5, f'{pct:.2f}%\n({count})', ha='center', fontweight='bold')
    
    # 5. Laundering Percentage by Community Size
    ax5 = fig.add_subplot(gs[1, 1])
    for method in combined_df['method'].unique():
        method_df = combined_df[combined_df['method'] == method]
        # Only plot communities with laundering
        laundering_df = method_df[method_df['laundering_nodes'] > 0]
        if len(laundering_df) > 0:
            ax5.scatter(laundering_df['size'], laundering_df['laundering_pct'] * 100, 
                       alpha=0.6, label=method, s=50, edgecolor='black', linewidth=0.5)
    ax5.set_xlabel('Community Size')
    ax5.set_ylabel('Laundering % in Community')
    ax5.set_title('Laundering Concentration vs Community Size')
    ax5.legend()
    ax5.grid(alpha=0.3)
    
    # 6. Top Communities Comparison
    ax6 = fig.add_subplot(gs[1, 2])
    top_n = 20
    top_communities = []
    for method in combined_df['method'].unique():
        method_df = combined_df[combined_df['method'] == method].head(top_n)
        top_communities.append({
            'method': method,
            'avg_score': method_df['score'].mean(),
            'avg_size': method_df['size'].mean(),
            'avg_laundering_pct': method_df['laundering_pct'].mean() * 100,
            'total_laundering': method_df['laundering_nodes'].sum()
        })
    
    top_df = pd.DataFrame(top_communities)
    x = np.arange(len(top_df))
    width = 0.35
    
    bars1 = ax6.bar(x - width/2, top_df['avg_score'] * 100, width, label='Avg Score (×100)', 
                    color='#FF6B6B', edgecolor='black')
    bars2 = ax6.bar(x + width/2, top_df['avg_laundering_pct'], width, label='Avg Laundering %',
                    color='#4ECDC4', edgecolor='black')
    
    ax6.set_ylabel('Value')
    ax6.set_title(f'Top {top_n} Communities Comparison')
    ax6.set_xticks(x)
    ax6.set_xticklabels(top_df['method'])
    ax6.legend()
    ax6.grid(axis='y', alpha=0.3)
    
    # 7. Attempt Coverage
    ax7 = fig.add_subplot(gs[2, 0])
    attempt_stats = []
    for method in combined_df['method'].unique():
        method_df = combined_df[combined_df['method'] == method]
        total_attempts = method_df['num_attempts'].sum()
        communities_with_attempts = (method_df['num_attempts'] > 0).sum()
        attempt_stats.append({
            'method': method,
            'total_attempts': total_attempts,
            'communities': communities_with_attempts
        })
    
    attempt_df = pd.DataFrame(attempt_stats)
    bars = ax7.bar(attempt_df['method'], attempt_df['total_attempts'], 
                   color=['#FF6B6B', '#4ECDC4'], edgecolor='black')
    ax7.set_ylabel('Total Laundering Attempts')
    ax7.set_title('Laundering Attempts Coverage')
    ax7.grid(axis='y', alpha=0.3)
    for i, (method, attempts, comm) in enumerate(zip(attempt_df['method'], 
                                                      attempt_df['total_attempts'],
                                                      attempt_df['communities'])):
        ax7.text(i, attempts + max(attempt_df['total_attempts'])*0.02, 
                f'{attempts}\n({comm} comm)', ha='center', fontweight='bold')
    
    # 8. Density vs Score (Quality Assessment)
    ax8 = fig.add_subplot(gs[2, 1])
    for method in combined_df['method'].unique():
        method_df = combined_df[combined_df['method'] == method].head(100)  # Top 100
        ax8.scatter(method_df['density'], method_df['score'], 
                   alpha=0.5, label=method, s=30, edgecolor='black', linewidth=0.5)
    ax8.set_xlabel('Density')
    ax8.set_ylabel('Score')
    ax8.set_title('Community Quality: Density vs Score (Top 100)')
    ax8.legend()
    ax8.grid(alpha=0.3)
    
    # 9. Overall Performance Summary Table
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis('off')
    
    summary_data = []
    for method in combined_df['method'].unique():
        method_df = combined_df[combined_df['method'] == method]
        summary_data.append([
            method,
            f"{len(method_df):,}",
            f"{method_df['size'].median():.0f}",
            f"{method_df['density'].mean():.3f}",
            f"{(method_df['laundering_nodes'] > 0).sum():,}",
            f"{method_df['laundering_pct'].mean()*100:.2f}%"
        ])
    
    table = ax9.table(cellText=summary_data,
                     colLabels=['Method', 'Total\nCommunities', 'Median\nSize', 
                               'Avg\nDensity', 'Laundering\nCommunities', 'Avg\nLaundering %'],
                     cellLoc='center',
                     loc='center',
                     bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Style header
    for i in range(6):
        table[(0, i)].set_facecolor('#4ECDC4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style rows
    for i in range(1, len(summary_data) + 1):
        for j in range(6):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#F0F0F0')
    
    ax9.set_title('Community Detection Performance Summary', 
                 fontweight='bold', fontsize=12, pad=20)
    
    plt.suptitle('Community Detection: Louvain vs Leiden on ENTIRE GRAPH (Global Analysis)',
                fontsize=16, fontweight='bold', y=0.995)
    
    plt.savefig(OUTPUT_DIR / '09b_community_global_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print detailed comparison
    print("\n" + "-"*80)
    print("DETAILED COMPARISON")
    print("-"*80)
    
    if len(combined_df['method'].unique()) == 2:
        louvain_data = combined_df[combined_df['method'] == 'Louvain']
        leiden_data = combined_df[combined_df['method'] == 'Leiden']
        
        print("\nKey Differences (GLOBAL - Entire Graph):")
        print(f"  Community count: Louvain={len(louvain_data)}, Leiden={len(leiden_data)}")
        print(f"  Avg score: Louvain={louvain_data['score'].mean():.4f}, Leiden={leiden_data['score'].mean():.4f}")
        print(f"  Median size: Louvain={louvain_data['size'].median():.0f}, Leiden={leiden_data['size'].median():.0f}")
        print(f"  Total laundering nodes: Louvain={louvain_data['laundering_nodes'].sum():.0f}, Leiden={leiden_data['laundering_nodes'].sum():.0f}")
        print(f"  Communities with laundering: Louvain={((louvain_data['laundering_nodes'] > 0).sum())}, Leiden={((leiden_data['laundering_nodes'] > 0).sum())}")
    
    print("\n" + "-"*80)
    print("GLOBAL vs WINDOW-BASED COMPARISON")
    print("-"*80)
    print("\nTo compare with window-based community detection:")
    print("  - Global (this analysis): Persistent structures across entire graph")
    print("  - Windows (in main analysis): Temporal patterns in window_metrics.csv")
    print("  - Look for 'communities_unsup_louvain' and 'communities_unsup_leiden' in Section 4")
    
    print("\n✓ Community analysis complete")
    
    return combined_df

# ============================================================================
# STATISTICAL SIGNIFICANCE TESTING
# ============================================================================

def perform_statistical_tests(df, top_n=5):
    """Perform comprehensive statistical significance tests."""
    print(f"\n{'='*80}")
    print("STATISTICAL SIGNIFICANCE TESTING")
    print("="*80)
    
    # Get top methods
    top_methods = df.groupby('method')['ap'].median().nlargest(top_n).index.tolist()
    
    if 'random' not in top_methods:
        top_methods.append('random')
    
    print(f"\nComparing top {top_n} methods against random baseline")
    print(f"Methods: {', '.join(top_methods)}")
    
    # Pairwise comparisons
    print("\n" + "-" * 80)
    print("PAIRWISE COMPARISONS (Mann-Whitney U Test)")
    print("-" * 80)
    
    metrics = ['ap', 'p_at_1.0pct', 'attcov_at_1.0pct']
    
    results = []
    
    for metric in metrics:
        print(f"\n{metric.upper()}:")
        
        random_data = df[df['method'] == 'random'][metric].dropna()
        
        for method in top_methods:
            if method == 'random':
                continue
                
            method_data = df[df['method'] == method][metric].dropna()
            
            if len(method_data) > 0:
                statistic, pvalue = mannwhitneyu(method_data, random_data, 
                                                 alternative='greater')
                
                improvement = ((method_data.median() - random_data.median()) / 
                              random_data.median() * 100)
                
                results.append({
                    'metric': metric,
                    'method': method,
                    'method_median': method_data.median(),
                    'random_median': random_data.median(),
                    'improvement_%': improvement,
                    'p_value': pvalue,
                    'significant': pvalue < 0.05
                })
                
                sig_marker = '***' if pvalue < 0.001 else '**' if pvalue < 0.01 else '*' if pvalue < 0.05 else ''
                
                print(f"  {method:30s}: p={pvalue:.6f} {sig_marker:3s} "
                      f"(improvement: {improvement:+.1f}%)")
    
    results_df = pd.DataFrame(results)
    
    # Visualize significance
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        metric_results = results_df[results_df['metric'] == metric]
        
        # Bar plot with significance markers
        colors = ['green' if sig else 'orange' 
                 for sig in metric_results['significant']]
        
        bars = ax.barh(range(len(metric_results)), 
                      metric_results['improvement_%'], 
                      color=colors, alpha=0.7)
        
        ax.set_yticks(range(len(metric_results)))
        ax.set_yticklabels(metric_results['method'])
        ax.set_xlabel('Improvement over Random (%)')
        ax.set_title(f'{metric.upper()} - Statistical Significance')
        ax.axvline(0, color='black', linestyle='-', linewidth=1)
        ax.grid(axis='x', alpha=0.3)
        ax.invert_yaxis()
        
        # Add p-value annotations
        for i, row in metric_results.iterrows():
            sig_text = '***' if row['p_value'] < 0.001 else \
                      '**' if row['p_value'] < 0.01 else \
                      '*' if row['p_value'] < 0.05 else 'ns'
            x_pos = row['improvement_%'] + (5 if row['improvement_%'] > 0 else -5)
            ax.text(x_pos, i - metric_results.index[0], sig_text, 
                   va='center', ha='left' if row['improvement_%'] > 0 else 'right',
                   fontweight='bold', fontsize=12)
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', alpha=0.7, label='Significant (p<0.05)'),
        Patch(facecolor='orange', alpha=0.7, label='Not Significant')
    ]
    axes[2].legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '10_statistical_significance.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return results_df

# ============================================================================
# PRECISION-RECALL ANALYSIS
# ============================================================================

def analyze_precision_coverage_tradeoff(df):
    """Analyze the tradeoff between precision and coverage."""
    print(f"\n{'='*80}")
    print("PRECISION-COVERAGE TRADEOFF ANALYSIS")
    print("="*80)
    
    top_methods = df.groupby('method')['ap'].median().nlargest(8).index
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 1. Precision vs Coverage scatter
    ax1 = axes[0]
    
    for method in top_methods:
        method_data = df[df['method'] == method]
        ax1.scatter(method_data['attcov_at_1.0pct'], 
                   method_data['p_at_1.0pct'],
                   label=method, s=100, alpha=0.6)
    
    ax1.set_xlabel('Attempt Coverage @ 1% (Higher = More Schemes Detected)')
    ax1.set_ylabel('Precision @ 1% (Higher = Fewer False Positives)')
    ax1.set_title('Precision-Coverage Tradeoff')
    ax1.legend(fontsize=9, loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Add diagonal reference lines
    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=1)
    
    # 2. Efficiency frontier
    ax2 = axes[1]
    
    # Calculate aggregate score (harmonic mean of precision and coverage)
    efficiency_data = []
    for method in df['method'].unique():
        method_data = df[df['method'] == method]
        precision = method_data['p_at_1.0pct'].median()
        coverage = method_data['attcov_at_1.0pct'].median()
        
        if precision > 0 and coverage > 0:
            harmonic_mean = 2 * (precision * coverage) / (precision + coverage)
            efficiency_data.append({
                'method': method,
                'precision': precision,
                'coverage': coverage,
                'efficiency': harmonic_mean
            })
    
    efficiency_df = pd.DataFrame(efficiency_data).sort_values('efficiency', ascending=False)
    
    # Color by efficiency
    scatter = ax2.scatter(efficiency_df['coverage'], 
                         efficiency_df['precision'],
                         c=efficiency_df['efficiency'],
                         s=200, alpha=0.7, cmap='viridis',
                         edgecolors='black', linewidth=1)
    
    # Annotate top methods
    for i, row in efficiency_df.head(5).iterrows():
        ax2.annotate(row['method'], 
                    (row['coverage'], row['precision']),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=8, fontweight='bold')
    
    ax2.set_xlabel('Median Coverage @ 1%')
    ax2.set_ylabel('Median Precision @ 1%')
    ax2.set_title('Efficiency Frontier (colored by harmonic mean)')
    ax2.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax2, label='Efficiency Score')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '11_precision_coverage_tradeoff.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\nTop 10 Methods by Efficiency (Harmonic Mean):")
    print("-" * 80)
    print(efficiency_df.head(10).to_string(index=False))
    
    return efficiency_df

# ============================================================================
# LIFT ANALYSIS
# ============================================================================

def analyze_lift_metrics(df):
    """Analyze lift metrics in detail."""
    print(f"\n{'='*80}")
    print("LIFT METRICS ANALYSIS")
    print("="*80)
    
    lift_cols = [col for col in df.columns if 'lift' in col.lower()]
    
    if not lift_cols:
        print("No lift metrics found!")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Get top methods
    top_methods = df.groupby('method')['lift_p_at_1.0pct'].median().nlargest(10).index
    
    # 1. Lift distribution
    ax1 = axes[0, 0]
    df_top = df[df['method'].isin(top_methods)]
    sns.boxplot(data=df_top, y='method', x='lift_p_at_1.0pct', 
               ax=ax1, palette='Set2', order=top_methods)
    ax1.set_xlabel('Lift @ 1%')
    ax1.set_ylabel('Method')
    ax1.set_title('Lift Distribution - Top 10 Methods')
    ax1.axvline(1.0, color='red', linestyle='--', label='No improvement', linewidth=2)
    ax1.legend()
    ax1.grid(axis='x', alpha=0.3)
    
    # 2. Lift across different k values
    ax2 = axes[0, 1]
    k_values = ['0.5pct', '1.0pct', '2.0pct', '5.0pct']
    lift_k_cols = [f'lift_p_at_{k}' for k in k_values if f'lift_p_at_{k}' in df.columns]
    
    top_5 = top_methods[:5]
    x_pos = np.arange(len(lift_k_cols))
    width = 0.15
    
    for i, method in enumerate(top_5):
        method_data = df[df['method'] == method]
        lifts = [method_data[col].median() for col in lift_k_cols]
        ax2.bar(x_pos + i * width, lifts, width, label=method, alpha=0.8)
    
    ax2.set_xlabel('K Value')
    ax2.set_ylabel('Median Lift')
    ax2.set_title('Lift @ Different K Values')
    ax2.set_xticks(x_pos + width * 2)
    ax2.set_xticklabels(k_values)
    ax2.legend(fontsize=8)
    ax2.axhline(1.0, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax2.grid(axis='y', alpha=0.3)
    
    # 3. Investigation efficiency
    ax3 = axes[1, 0]
    efficiency_data = []
    for method in top_methods:
        method_data = df[df['method'] == method]
        lift = method_data['lift_p_at_1.0pct'].median()
        efficiency_data.append({'method': method, 'lift': lift})
    
    eff_df = pd.DataFrame(efficiency_data).sort_values('lift', ascending=True)
    
    colors = ['green' if lift > 3 else 'orange' if lift > 2 else 'red' 
             for lift in eff_df['lift']]
    
    ax3.barh(range(len(eff_df)), eff_df['lift'], color=colors, alpha=0.7)
    ax3.set_yticks(range(len(eff_df)))
    ax3.set_yticklabels(eff_df['method'])
    ax3.set_xlabel('Investigation Efficiency (Lift @ 1%)')
    ax3.set_title('How Much More Efficient Than Random?')
    ax3.axvline(1.0, color='black', linestyle='-', linewidth=1)
    ax3.grid(axis='x', alpha=0.3)
    
    # Add labels
    for i, lift in enumerate(eff_df['lift']):
        ax3.text(lift + 0.1, i, f'{lift:.1f}x', va='center', fontweight='bold')
    
    # 4. Lift vs AP correlation
    ax4 = axes[1, 1]
    
    for method in top_methods:
        method_data = df[df['method'] == method]
        ax4.scatter(method_data['ap'], method_data['lift_p_at_1.0pct'],
                   label=method, s=100, alpha=0.6)
    
    ax4.set_xlabel('Average Precision (AP)')
    ax4.set_ylabel('Lift @ 1%')
    ax4.set_title('Correlation: AP vs Lift')
    ax4.legend(fontsize=8, ncol=2)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '12_lift_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print efficiency interpretation
    print("\nInvestigation Efficiency Interpretation:")
    print("-" * 80)
    print("Lift = How many times more efficient than random investigation")
    print("\nTop 10 Methods:")
    for i, row in eff_df.tail(10).iterrows():
        efficiency_class = ("⭐ Excellent" if row['lift'] > 5 else 
                          "✅ Good" if row['lift'] > 3 else 
                          "⚠️  Moderate" if row['lift'] > 2 else 
                          "❌ Poor")
        print(f"  {row['method']:30s}: {row['lift']:6.2f}x  {efficiency_class}")

# ============================================================================
# COMPREHENSIVE SUMMARY REPORT
# ============================================================================

def generate_summary_report(df, method_stats, efficiency_df):
    """Generate a comprehensive summary report."""
    print(f"\n{'='*80}")
    print("COMPREHENSIVE SUMMARY REPORT")
    print("="*80)
    
    # Overall statistics
    print("\n1. DATASET OVERVIEW")
    print("-" * 80)
    print(f"Total measurements: {len(df):,}")
    print(f"Methods evaluated: {df['method'].nunique()}")
    print(f"Time windows: {df['window_id'].nunique()}")
    print(f"Window configurations: {df['window_days'].unique()}")
    print(f"Date range: {df['ws'].min().date()} to {df['we'].max().date()}")
    
    # Best performing methods
    print("\n2. TOP PERFORMING METHODS")
    print("-" * 80)
    
    print("\nBy Average Precision (AP):")
    top_ap = method_stats.nlargest(5, 'median')[['median', 'mean', 'std']]
    for method, row in top_ap.iterrows():
        print(f"  {method:30s}: {row['median']:.6f} (±{row['std']:.6f})")
    
    print("\nBy Precision @ 1%:")
    p1_stats = df.groupby('method')['p_at_1.0pct'].agg(['median', 'mean', 'std'])
    top_p1 = p1_stats.nlargest(5, 'median')
    for method, row in top_p1.iterrows():
        print(f"  {method:30s}: {row['median']:.6f} (±{row['std']:.6f})")
    
    print("\nBy Efficiency (Precision-Coverage Balance):")
    for i, row in efficiency_df.head(5).iterrows():
        print(f"  {row['method']:30s}: {row['efficiency']:.6f} "
              f"(P={row['precision']:.4f}, C={row['coverage']:.4f})")
    
    # Improvement over baseline
    print("\n3. IMPROVEMENT OVER RANDOM BASELINE")
    print("-" * 80)
    random_ap = df[df['method'] == 'random']['ap'].median()
    random_p1 = df[df['method'] == 'random']['p_at_1.0pct'].median()
    
    best_method = method_stats.nlargest(1, 'median').index[0]
    best_ap = method_stats.loc[best_method, 'median']
    best_p1 = df[df['method'] == best_method]['p_at_1.0pct'].median()
    
    ap_improvement = ((best_ap - random_ap) / random_ap) * 100
    p1_improvement = ((best_p1 - random_p1) / random_p1) * 100
    
    print(f"Best method: {best_method}")
    print(f"  AP improvement: {ap_improvement:+.1f}%")
    print(f"  P@1% improvement: {p1_improvement:+.1f}%")
    print(f"  Investigation efficiency: {best_p1/random_p1:.1f}x better")
    
    # Window size effect
    print("\n4. WINDOW SIZE ANALYSIS")
    print("-" * 80)
    for window_days in sorted(df['window_days'].unique()):
        window_data = df[df['window_days'] == window_days]
        print(f"\n{window_days}-day windows:")
        print(f"  Median AP: {window_data['ap'].median():.6f}")
        print(f"  Median P@1%: {window_data['p_at_1.0pct'].median():.6f}")
        print(f"  Median Coverage@1%: {window_data['attcov_at_1.0pct'].median():.6f}")
    
    # Method categories
    print("\n5. PERFORMANCE BY METHOD CATEGORY")
    print("-" * 80)
    
    if 'category' in df.columns:
        cat_performance = df.groupby('category')['ap'].median().sort_values(ascending=False)
        for category, ap in cat_performance.items():
            print(f"  {category:25s}: {ap:.6f}")
    
    # Recommendations
    print("\n6. RECOMMENDATIONS")
    print("-" * 80)
    print("\n✅ DEPLOY IMMEDIATELY:")
    for method in top_ap.head(3).index:
        ap_val = method_stats.loc[method, 'median']
        print(f"  • {method:30s} (AP: {ap_val:.6f})")
    
    print("\n⚠️  REQUIRES IMPROVEMENT:")
    worst_methods = method_stats[method_stats['median'] < random_ap].sort_values('median')
    for method, row in worst_methods.head(3).iterrows():
        if method != 'random':
            print(f"  • {method:30s} (AP: {row['median']:.6f})")
    
    print("\n7. KEY INSIGHTS")
    print("-" * 80)
    
    # Statistical insights
    ap_range = df['ap'].max() - df['ap'].min()
    p1_range = df['p_at_1.0pct'].max() - df['p_at_1.0pct'].min()
    
    print(f"  • Performance range: {ap_range:.6f} (AP), {p1_range:.6f} (P@1%)")
    print(f"  • Best method is {ap_improvement:.0f}% better than random")
    print(f"  • Top method achieves {best_p1*100:.2f}% precision at 1%")
    
    # Stability
    most_stable = method_stats.nsmallest(5, 'std')['std']
    print(f"  • Most stable methods have std < {most_stable.max():.6f}")
    
    # Coverage
    max_coverage = df['attcov_at_1.0pct'].max()
    print(f"  • Best coverage achieves {max_coverage*100:.1f}% scheme detection at 1%")

def create_final_summary_plot(df, method_stats):
    """Create a final comprehensive summary visualization."""
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(3, 3, figure=fig)
    
    # Title
    fig.suptitle('Money Laundering Detection - Complete Performance Summary', 
                 fontsize=20, fontweight='bold')
    
    top_10 = method_stats.nlargest(10, 'median').index
    
    # 1. Overall rankings
    ax1 = fig.add_subplot(gs[0, :2])
    metrics = ['ap', 'p_at_1.0pct', 'attcov_at_1.0pct']
    
    x = np.arange(len(top_10))
    width = 0.25
    
    for i, metric in enumerate(metrics):
        values = [df[df['method'] == m][metric].median() for m in top_10]
        ax1.bar(x + i * width, values, width, label=metric.upper(), alpha=0.8)
    
    ax1.set_xticks(x + width)
    ax1.set_xticklabels(top_10, rotation=45, ha='right')
    ax1.set_ylabel('Performance')
    ax1.set_title('Top 10 Methods - Multi-Metric Performance')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # 2. Performance distribution
    ax2 = fig.add_subplot(gs[0, 2])
    all_methods_ap = [df[df['method'] == m]['ap'].median() 
                     for m in df['method'].unique()]
    ax2.hist(all_methods_ap, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
    ax2.axvline(df[df['method']=='random']['ap'].median(), 
               color='red', linestyle='--', linewidth=2, label='Random')
    ax2.set_xlabel('Average Precision')
    ax2.set_ylabel('Number of Methods')
    ax2.set_title('AP Distribution (All Methods)')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    # 3. Temporal consistency
    ax3 = fig.add_subplot(gs[1, :])
    top_5 = top_10[:5]
    
    for method in top_5:
        method_data = df[df['method'] == method].sort_values('ws')
        ax3.plot(method_data['ws'], method_data['ap'], 
                marker='o', label=method, linewidth=2, markersize=4)
    
    ax3.set_xlabel('Time Window Start')
    ax3.set_ylabel('Average Precision')
    ax3.set_title('Temporal Stability - Top 5 Methods')
    ax3.legend(ncol=5)
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(axis='x', rotation=45)
    
    # 4. Precision vs Coverage
    ax4 = fig.add_subplot(gs[2, 0])
    
    for method in top_10:
        method_data = df[df['method'] == method]
        ax4.scatter(method_data['attcov_at_1.0pct'].median(),
                   method_data['p_at_1.0pct'].median(),
                   s=200, alpha=0.7, label=method)
    
    ax4.set_xlabel('Coverage @ 1%')
    ax4.set_ylabel('Precision @ 1%')
    ax4.set_title('Precision-Coverage Balance')
    ax4.legend(fontsize=8, ncol=2)
    ax4.grid(True, alpha=0.3)
    
    # 5. Improvement over random
    ax5 = fig.add_subplot(gs[2, 1])
    
    random_ap = df[df['method']=='random']['ap'].median()
    improvements = []
    
    for method in top_10:
        method_ap = df[df['method']==method]['ap'].median()
        improvement = ((method_ap - random_ap) / random_ap) * 100
        improvements.append(improvement)
    
    colors = ['green' if x > 50 else 'orange' if x > 20 else 'yellow' 
             for x in improvements]
    ax5.barh(range(len(top_10)), improvements, color=colors, alpha=0.7)
    ax5.set_yticks(range(len(top_10)))
    ax5.set_yticklabels(top_10)
    ax5.set_xlabel('Improvement over Random (%)')
    ax5.set_title('Relative Performance Gains')
    ax5.axvline(0, color='black', linestyle='-', linewidth=1)
    ax5.grid(axis='x', alpha=0.3)
    ax5.invert_yaxis()
    
    # 6. Summary statistics table
    ax6 = fig.add_subplot(gs[2, 2])
    ax6.axis('tight')
    ax6.axis('off')
    
    summary_data = []
    for method in top_10[:5]:
        method_data = df[df['method'] == method]
        summary_data.append([
            method[:20],
            f"{method_data['ap'].median():.4f}",
            f"{method_data['p_at_1.0pct'].median():.4f}",
            f"{method_data['attcov_at_1.0pct'].median():.4f}"
        ])
    
    table = ax6.table(cellText=summary_data,
                     colLabels=['Method', 'AP', 'P@1%', 'Cov@1%'],
                     cellLoc='center',
                     loc='center',
                     bbox=[0, 0, 1, 1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Style table
    for i in range(len(summary_data) + 1):
        for j in range(4):
            cell = table[(i, j)]
            if i == 0:
                cell.set_facecolor('#4CAF50')
                cell.set_text_props(weight='bold', color='white')
            else:
                cell.set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
    
    ax6.set_title('Top 5 Methods Summary', fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '13_final_summary.png', dpi=300, bbox_inches='tight')
    plt.show()

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    
    print("\n" + "="*80)
    print("STARTING COMPREHENSIVE ANALYSIS")
    print("="*80)
    
    # Load data
    try:
        df = load_and_prepare_data(CSV_PATH)
    except FileNotFoundError:
        print(f"\n❌ ERROR: Could not find {CSV_PATH}")
        print("\nPlease ensure:")
        print("1. Google Drive is mounted (uncomment drive.mount line)")
        print("2. Path is correct for your setup")
        print("3. File exists at the specified location")
        return
    
    # 1. Data Summary
    print_data_summary(df)
    analyze_missing_data(df)
    
    # 2. Performance Analysis
    print("\n" + "="*80)
    print("SECTION 1: PERFORMANCE ANALYSIS")
    print("="*80)
    
    method_stats_ap = analyze_method_performance(df, metric='ap', top_n=10)
    method_stats_p1 = analyze_method_performance(df, metric='p_at_1.0pct', top_n=10)
    
    comparison_df, comparison_normalized = compare_multiple_metrics(
        df, 
        metrics=['ap', 'p_at_1.0pct', 'attcov_at_1.0pct'],
        top_n=10
    )
    
    # 3. Temporal Analysis
    print("\n" + "="*80)
    print("SECTION 2: TEMPORAL ANALYSIS")
    print("="*80)
    
    stability = analyze_temporal_trends(
        df,
        methods=['pagerank_wlog', 'in_deg', 'ensemble_top3', 'random']
    )
    
    analyze_window_size_effect(df)
    
    # 4. Correlations
    print("\n" + "="*80)
    print("SECTION 3: CORRELATION & FEATURE ANALYSIS")
    print("="*80)
    
    corr_matrix = analyze_correlations(df)
    analyze_graph_statistics(df)
    
    # 5. Method Comparisons
    print("\n" + "="*80)
    print("SECTION 4: METHOD COMPARISON")
    print("="*80)
    
    analyze_ensemble_methods(df)
    analyze_method_categories(df)
    
    # 5b. Community Detection Analysis - GLOBAL (Entire Graph)
    print("\n" + "="*80)
    print("SECTION 4B: COMMUNITY DETECTION - GLOBAL GRAPH ANALYSIS")
    print("="*80)
    print("\nThis section analyzes communities detected on the ENTIRE graph.")
    print("For window-based community detection, see Section 4 (Method Categories).")
    
    try:
        community_df = analyze_community_methods(BASE_PATH)
    except Exception as e:
        print(f"⚠ Warning: Could not complete community analysis: {e}")
        community_df = None
    
    # 6. Statistical Testing
    print("\n" + "="*80)
    print("SECTION 5: STATISTICAL TESTING")
    print("="*80)
    
    significance_results = perform_statistical_tests(df, top_n=5)
    
    # 7. Precision-Coverage Analysis
    print("\n" + "="*80)
    print("SECTION 6: PRECISION-COVERAGE ANALYSIS")
    print("="*80)
    
    efficiency_df = analyze_precision_coverage_tradeoff(df)
    
    # 8. Lift Analysis
    print("\n" + "="*80)
    print("SECTION 7: LIFT ANALYSIS")
    print("="*80)
    
    analyze_lift_metrics(df)
    
    # 9. Final Summary
    print("\n" + "="*80)
    print("SECTION 8: FINAL SUMMARY")
    print("="*80)
    
    generate_summary_report(df, method_stats_ap, efficiency_df)
    create_final_summary_plot(df, method_stats_ap)
    
    # Save summary statistics to CSV
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)
    
    # Save method statistics
    method_stats_ap.to_csv(OUTPUT_DIR / 'method_statistics_ap.csv')
    print(f"✓ Saved method statistics to {OUTPUT_DIR / 'method_statistics_ap.csv'}")
    
    # Save efficiency rankings
    efficiency_df.to_csv(OUTPUT_DIR / 'efficiency_rankings.csv', index=False)
    print(f"✓ Saved efficiency rankings to {OUTPUT_DIR / 'efficiency_rankings.csv'}")
    
    # Save significance test results
    if significance_results is not None:
        significance_results.to_csv(OUTPUT_DIR / 'significance_tests.csv', index=False)
        print(f"✓ Saved significance tests to {OUTPUT_DIR / 'significance_tests.csv'}")
    
    # Save comparison matrix
    comparison_df.to_csv(OUTPUT_DIR / 'multi_metric_comparison.csv')
    print(f"✓ Saved multi-metric comparison to {OUTPUT_DIR / 'multi_metric_comparison.csv'}")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\nAll plots saved to: {OUTPUT_DIR}")
    print(f"Total plots generated: 14")
    print("\nPlot files:")
    for i, name in enumerate([
        '01_missing_data.png',
        '02_method_performance_ap.png',
        '03_multi_metric_comparison.png',
        '04_temporal_trends.png',
        '05_window_size_effect.png',
        '06_correlations.png',
        '07_graph_statistics.png',
        '08_ensemble_comparison.png',
        '09_category_analysis.png',
        '09b_community_global_analysis.png',
        '10_statistical_significance.png',
        '11_precision_coverage_tradeoff.png',
        '12_lift_analysis.png',
        '13_final_summary.png'
    ], 1):
        print(f"  {i:2d}. {name}")
    
    print("\n✅ Ready for final project presentation!")

if __name__ == "__main__":
    main()