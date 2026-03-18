#!/usr/bin/env python3
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr, pointbiserialr, ks_2samp
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

DATA_PATH = Path(__file__).parent.parent / "data/HI_Small" / "sliding_window_features.parquet"

def load_data():
    df = pd.read_parquet(DATA_PATH)
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}\n")
    return df

def check_target_integrity(df):
    print("="*80)
    print("1. TARGET INTEGRITY & TEMPORAL DRIFT")
    print("="*80)
    
    fraud_rate = df['is_fraud'].mean()
    fraud_count = df['is_fraud'].sum()
    total_count = len(df)
    
    print(f"Global Fraud Rate: {fraud_rate:.4%} ({fraud_count:,} / {total_count:,})")
    
    if 'window_date' in df.columns:
        temporal = df.groupby('window_date').agg({
            'is_fraud': ['sum', 'count', 'mean']
        }).reset_index()
        temporal.columns = ['window_date', 'fraud_count', 'total_count', 'fraud_rate']
        temporal = temporal.sort_values('window_date')
        
        print(f"\nTemporal Fraud Distribution:")
        print(temporal.to_string(index=False))
        
        zero_fraud_windows = (temporal['fraud_rate'] == 0).sum()
        print(f"\nWindows with 0% fraud: {zero_fraud_windows} / {len(temporal)}")
        
        if zero_fraud_windows > 0:
            print("WARNING: Zero-fraud windows detected - TimeSeriesSplit will fail on these folds!")
    
    print()

def check_feature_sparsity(df):
    print("="*80)
    print("2. FEATURE SPARSITY & DEFAULT VALUE BUGS")
    print("="*80)
    
    exclude_cols = ['is_fraud', 'window_date', 'node_id', 'account_id']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    sparsity_report = []
    
    for col in feature_cols:
        if df[col].dtype in ['float64', 'float32', 'int64', 'int32']:
            nan_pct = df[col].isna().mean() * 100
            zero_pct = (df[col] == 0).mean() * 100
            inf_pct = np.isinf(df[col]).mean() * 100
            sparsity_report.append({
                'feature': col,
                'nan_pct': nan_pct,
                'zero_pct': zero_pct,
                'inf_pct': inf_pct,
                'total_sparse': nan_pct + zero_pct
            })
    
    sparsity_df = pd.DataFrame(sparsity_report).sort_values('total_sparse', ascending=False)
    
    print("\nTop 20 Sparsest Features:")
    print(sparsity_df.head(20).to_string(index=False))
    
    critical_sparse = sparsity_df[sparsity_df['total_sparse'] > 95]
    if len(critical_sparse) > 0:
        print(f"\nCRITICAL: {len(critical_sparse)} features are >95% sparse (NaN or zero)")
        print("These features provide minimal signal and bloat the feature space.")
    
    print()

def check_power_law_skew(df):
    print("="*80)
    print("3. POWER-LAW SKEW & OUTLIER MASKING")
    print("="*80)
    
    exclude_cols = ['is_fraud', 'window_date', 'node_id', 'account_id']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    skew_report = []
    
    for col in feature_cols:
        if df[col].dtype in ['float64', 'float32', 'int64', 'int32']:
            clean_vals = df[col].dropna()
            if len(clean_vals) > 0 and clean_vals.std() > 0:
                skewness = clean_vals.skew()
                p99 = clean_vals.quantile(0.99)
                max_val = clean_vals.max()
                outlier_ratio = max_val / p99 if p99 > 0 else np.inf
                
                skew_report.append({
                    'feature': col,
                    'skewness': skewness,
                    'max': max_val,
                    'p99': p99,
                    'outlier_ratio': outlier_ratio
                })
    
    skew_df = pd.DataFrame(skew_report).sort_values('skewness', ascending=False, key=abs)
    
    print("\nTop 20 Most Skewed Features:")
    print(skew_df.head(20).to_string(index=False))
    
    extreme_skew = skew_df[abs(skew_df['skewness']) > 10]
    if len(extreme_skew) > 0:
        print(f"\nWARNING: {len(extreme_skew)} features have |skewness| > 10")
        print("Power-law distributions compress feature space. Consider log-transform or rank-based normalization.")
    
    print()

def check_multicollinearity(df):
    print("="*80)
    print("4. INFORMATION OVERLAP (MULTICOLLINEARITY)")
    print("="*80)
    
    exclude_cols = ['is_fraud', 'window_date', 'node_id', 'account_id']
    feature_cols = [c for c in df.columns if c not in exclude_cols and df[c].dtype in ['float64', 'float32', 'int64', 'int32']]
    
    feature_df = df[feature_cols].dropna(axis=1, how='all')
    feature_df = feature_df.fillna(0)
    
    if len(feature_df.columns) > 100:
        print(f"WARNING: {len(feature_df.columns)} features detected. Computing correlation for top variance features only...")
        variances = feature_df.var().sort_values(ascending=False)
        top_features = variances.head(50).index.tolist()
        feature_df = feature_df[top_features]
    
    corr_matrix = feature_df.corr(method='spearman').abs()
    
    upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    high_corr_pairs = []
    for col in upper_triangle.columns:
        for idx in upper_triangle.index:
            val = upper_triangle.loc[idx, col]
            if pd.notna(val) and val > 0.9:
                high_corr_pairs.append({
                    'feature_1': idx,
                    'feature_2': col,
                    'correlation': val
                })
    
    if high_corr_pairs:
        high_corr_df = pd.DataFrame(high_corr_pairs).sort_values('correlation', ascending=False)
        print(f"\nHigh Correlation Pairs (|r| > 0.9): {len(high_corr_df)}")
        print(high_corr_df.head(20).to_string(index=False))
        print("\nRedundant features detected. Consider removing one from each pair.")
    else:
        print("\nNo extreme multicollinearity detected (threshold: |r| > 0.9)")
    
    print()

def check_target_separability(df):
    print("="*80)
    print("5. TARGET SEPARABILITY (GROUND TRUTH CHECK)")
    print("="*80)
    
    exclude_cols = ['is_fraud', 'window_date', 'node_id', 'account_id']
    feature_cols = [c for c in df.columns if c not in exclude_cols and df[c].dtype in ['float64', 'float32', 'int64', 'int32']]
    
    separability_report = []
    
    y = df['is_fraud'].values
    
    for col in feature_cols:
        X = df[col].fillna(0).values
        
        if len(np.unique(X)) > 1:
            try:
                corr, p_value = pointbiserialr(y, X)
                
                fraud_vals = X[y == 1]
                legit_vals = X[y == 0]
                
                if len(fraud_vals) > 0 and len(legit_vals) > 0:
                    ks_stat, ks_p = ks_2samp(fraud_vals, legit_vals)
                else:
                    ks_stat, ks_p = 0, 1
                
                separability_report.append({
                    'feature': col,
                    'point_biserial_r': corr,
                    'pb_pvalue': p_value,
                    'ks_statistic': ks_stat,
                    'ks_pvalue': ks_p
                })
            except:
                pass
    
    sep_df = pd.DataFrame(separability_report).sort_values('ks_statistic', ascending=False)
    
    print("\nTop 20 Most Separable Features (by KS statistic):")
    print(sep_df.head(20).to_string(index=False))
    
    weak_features = sep_df[sep_df['ks_statistic'] < 0.05]
    print(f"\nFeatures with KS < 0.05 (weak separability): {len(weak_features)} / {len(sep_df)}")
    
    if len(sep_df) > 0 and sep_df['ks_statistic'].max() < 0.1:
        print("\nCRITICAL: ALL features show weak target separability (KS < 0.1)")
        print("The models fail because features have ZERO predictive power.")
        print("Root cause: Feature extraction logic or data aggregation destroyed the fraud signal.")
    
    print()

def check_target_leakage(df):
    print("="*80)
    print("6. TARGET LEAKAGE DETECTION")
    print("="*80)
    
    exclude_cols = ['is_fraud', 'window_date', 'node_id', 'account_id']
    feature_cols = [c for c in df.columns if c not in exclude_cols and df[c].dtype in ['float64', 'float32', 'int64', 'int32']]
    
    leakage_report = []
    
    y = df['is_fraud'].values
    
    for col in feature_cols:
        X = df[col].fillna(0).values
        
        if len(np.unique(X)) > 1 and len(np.unique(y)) > 1:
            try:
                auc = roc_auc_score(y, X)
                leakage_report.append({
                    'feature': col,
                    'single_feature_auc': auc
                })
            except:
                pass
    
    leak_df = pd.DataFrame(leakage_report).sort_values('single_feature_auc', ascending=False)
    
    print("\nTop 20 Features by Single-Feature AUC:")
    print(leak_df.head(20).to_string(index=False))
    
    perfect_leaks = leak_df[leak_df['single_feature_auc'] > 0.99]
    if len(perfect_leaks) > 0:
        print(f"\nCRITICAL LEAKAGE DETECTED: {len(perfect_leaks)} features have AUC > 0.99")
        print("These features perfectly predict the target - temporal leakage confirmed.")
        print(perfect_leaks.to_string(index=False))
    else:
        print("\nNo obvious target leakage detected (no single feature with AUC > 0.99)")
    
    print()

def main():
    print("\n" + "="*80)
    print("FRAUD DETECTION PIPELINE DIAGNOSTIC REPORT")
    print("="*80 + "\n")
    
    df = load_data()
    
    check_target_integrity(df)
    check_feature_sparsity(df)
    check_power_law_skew(df)
    check_multicollinearity(df)
    check_target_separability(df)
    check_target_leakage(df)
    
    print("="*80)
    print("DIAGNOSTIC COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()
