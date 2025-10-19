# Data Science Analysis Guide

## Overview

This document explains how to use the `data_science.py` script for comprehensive analysis of money laundering detection metrics.

## Purpose

The `data_science.py` script performs an in-depth statistical and visual analysis of the fraud detection system's performance, generating:
- 13 high-quality publication-ready plots
- Statistical significance tests
- Performance comparisons across methods
- Temporal trend analysis
- Comprehensive summary reports

## Requirements

### Python Libraries

```python
pandas
numpy
matplotlib
seaborn
scipy
scikit-learn
```

Install with:
```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn
```

## Setup for Google Colab

### 1. Upload the Script

Upload `data_science.py` to your Google Colab session or GitHub repository.

### 2. Mount Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

### 3. Configure Paths

Edit the `BASE_PATH` variable in `data_science.py` to match your Google Drive structure:

```python
# Default path (modify to match your setup)
BASE_PATH = Path('/content/drive/MyDrive/AML/processed/LI_Small/US_Dollar')
```

The script expects to find:
```
/content/drive/MyDrive/AML/processed/LI_Small/US_Dollar/
└── metrics/
    └── window_metrics.csv
```

### 4. Run the Analysis

In Google Colab:

```python
# Run the entire analysis
%run data_science.py
```

Or import specific functions:

```python
from data_science import *

# Load data
df = load_and_prepare_data(CSV_PATH)

# Run specific analyses
method_stats = analyze_method_performance(df, metric='ap', top_n=10)
analyze_temporal_trends(df)
```

## Output Files

The script generates 13 PNG plots in the `metrics/analysis_plots/` directory:

### Performance Analysis
1. **01_missing_data.png** - Missing data visualization
2. **02_method_performance_ap.png** - Method performance by AP
3. **03_multi_metric_comparison.png** - Multi-metric heatmap and radar chart

### Temporal Analysis
4. **04_temporal_trends.png** - Performance over time windows
5. **05_window_size_effect.png** - 3-day vs 7-day window comparison

### Correlation Analysis
6. **06_correlations.png** - Correlation matrices
7. **07_graph_statistics.png** - Graph properties vs performance

### Method Comparisons
8. **08_ensemble_comparison.png** - Ensemble vs individual methods
9. **09_category_analysis.png** - Performance by method category
10. **10_statistical_significance.png** - Significance tests results

### Advanced Analysis
11. **11_precision_coverage_tradeoff.png** - Precision-coverage balance
12. **12_lift_analysis.png** - Investigation efficiency analysis
13. **13_final_summary.png** - Comprehensive summary dashboard

### CSV Exports
- `method_statistics_ap.csv` - Detailed method statistics
- `efficiency_rankings.csv` - Methods ranked by efficiency
- `significance_tests.csv` - Statistical test results
- `multi_metric_comparison.csv` - Multi-metric comparison matrix

## Analysis Sections

### Section 1: Performance Analysis
- Method ranking by multiple metrics
- Distribution analysis (box plots, violin plots)
- Improvement over random baseline
- Multi-metric comparison

**Key Metrics:**
- Average Precision (AP)
- Precision @ 1%
- Attempt Coverage @ 1%

### Section 2: Temporal Analysis
- Performance evolution over time windows
- Stability analysis (variance across windows)
- Window size effect (3-day vs 7-day)
- Statistical tests for window size differences

### Section 3: Correlation & Feature Analysis
- Correlation matrices between all metrics
- Relationship between graph properties and performance
- Feature importance analysis

### Section 4: Method Comparison
- Ensemble vs individual methods
- Performance by method category:
  - Graph Structure (PageRank, degree)
  - Graph Decomposition (k-core)
  - Transaction Volume (amounts, counts)
  - Heuristics (patterns, ratios)
  - Community Detection
  - Ensembles

### Section 5: Statistical Testing
- Mann-Whitney U tests vs random baseline
- Pairwise method comparisons
- Significance visualization

### Section 6: Precision-Coverage Analysis
- Tradeoff between precision and coverage
- Efficiency frontier
- Harmonic mean optimization

### Section 7: Lift Analysis
- Investigation efficiency metrics
- Lift across different K values
- Efficiency classification

### Section 8: Final Summary
- Comprehensive performance report
- Top method recommendations
- Key insights and findings

## Key Functions

### Data Loading
```python
df = load_and_prepare_data(csv_path)
```
Loads CSV and adds computed columns (window duration, FPR, F1, etc.)

### Performance Analysis
```python
method_stats = analyze_method_performance(df, metric='ap', top_n=10)
```
Analyzes top N methods by specified metric with multiple visualizations.

### Temporal Analysis
```python
stability = analyze_temporal_trends(df, methods=['pagerank_wlog', 'in_deg'])
```
Shows how methods perform over time.

### Statistical Testing
```python
results_df = perform_statistical_tests(df, top_n=5)
```
Performs Mann-Whitney U tests comparing methods to random baseline.

### Correlation Analysis
```python
corr_matrix = analyze_correlations(df)
```
Generates correlation heatmaps for all metrics.

### Ensemble Analysis
```python
analyze_ensemble_methods(df)
```
Compares ensemble methods to individual methods.

### Summary Report
```python
generate_summary_report(df, method_stats, efficiency_df)
```
Prints comprehensive text summary of findings.

## Customization

### Change Output Directory
```python
OUTPUT_DIR = Path('/your/custom/path')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
```

### Analyze Different Metrics
```python
analyze_method_performance(df, metric='p_at_2.0pct', top_n=15)
```

### Focus on Specific Methods
```python
analyze_temporal_trends(df, methods=['your_method_1', 'your_method_2'])
```

### Adjust Plot Style
```python
plt.style.use('seaborn-v0_8-whitegrid')  # or other styles
sns.set_palette("Set2")  # or other palettes
```

## Interpretation Guide

### Average Precision (AP)
- **Range:** 0.0 to 1.0
- **Good:** > 0.025 (for 1.25% prevalence)
- **Excellent:** > 0.030
- Your best: **0.0272** ✅

### Precision @ 1%
- **Good:** > 5.0%
- **Excellent:** > 7.0%
- Your best: **7.32%** ✅

### Lift
- **Good:** > 3x
- **Excellent:** > 5x
- Your best: **6.0x** ✅

### Coverage @ 1%
- **Moderate:** 5-10%
- **Good:** 10-20%
- **Excellent:** > 20%
- Your best: **9.42%** ⚠️

## Troubleshooting

### File Not Found
```
❌ ERROR: Could not find /content/drive/MyDrive/...
```
**Solution:** 
1. Ensure Google Drive is mounted
2. Check path in `BASE_PATH` variable
3. Verify `window_metrics.csv` exists at location

### Memory Error
```
MemoryError: Unable to allocate array
```
**Solution:**
1. Use Colab Pro for more RAM
2. Reduce `top_n` parameter in functions
3. Analyze subsets of data

### Import Error
```
ModuleNotFoundError: No module named 'seaborn'
```
**Solution:**
```python
!pip install seaborn
```

### Plot Not Showing
In Colab, ensure:
```python
%matplotlib inline
```

## Tips for Final Project

### 1. Focus on Key Plots
For presentations, prioritize:
- `02_method_performance_ap.png` - Core results
- `10_statistical_significance.png` - Proves it works
- `13_final_summary.png` - Overview

### 2. Customize for Your Needs
Modify plot titles, labels, and colors to match your project theme.

### 3. Export High-Resolution
All plots saved at 300 DPI (publication quality).

### 4. Use Summary Statistics
CSV exports provide exact numbers for tables in your report.

### 5. Cite Statistical Tests
Include p-values from significance tests to prove performance.

## Example Workflow

```python
# 1. Setup
from google.colab import drive
drive.mount('/content/drive')

# 2. Run complete analysis
%run data_science.py

# 3. Or run step-by-step
from data_science import *

df = load_and_prepare_data(CSV_PATH)
print_data_summary(df)

# 4. Focus on specific analysis
method_stats = analyze_method_performance(df, 'ap', top_n=10)
analyze_temporal_trends(df)
results_df = perform_statistical_tests(df)

# 5. Export for report
generate_summary_report(df, method_stats, efficiency_df)
```

## Expected Runtime

- **Full analysis:** 3-5 minutes
- **Individual sections:** 10-30 seconds each
- **Single plot:** < 5 seconds

## Questions?

Common questions answered:

**Q: Can I analyze my own data?**
A: Yes! Just ensure CSV has the same column structure as `window_metrics.csv`.

**Q: How do I add more methods to compare?**
A: Edit the `methods` parameter in relevant functions.

**Q: Can I change color schemes?**
A: Yes! Modify `sns.set_palette()` and matplotlib color parameters.

**Q: How do I get p-values for my report?**
A: Run `perform_statistical_tests()` and check `significance_tests.csv`.

## Citation

If using this analysis in your final project, consider citing:

```
Statistical analysis performed using custom Python analysis pipeline
with scipy, pandas, matplotlib, and seaborn libraries.
Mann-Whitney U tests used for significance testing (α=0.05).
```

## Additional Resources

- Matplotlib documentation: https://matplotlib.org/
- Seaborn tutorial: https://seaborn.pydata.org/
- Scipy stats: https://docs.scipy.org/doc/scipy/reference/stats.html
- Pandas guide: https://pandas.pydata.org/docs/

---

**Version:** 1.0  
**Last Updated:** 2025  
**Tested On:** Google Colab with Python 3.10+