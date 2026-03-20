"""
Forward-Chaining Time-Series Cross-Validation for Money Laundering Detection.
"""
import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, precision_recall_curve
from pathlib import Path
import sys
from typing import Dict, Tuple

root_path = Path(__file__).resolve().parent.parent
sys.path.append(str(root_path))

from src.config import OUTPUT_FEATURES_FILE, DATA_PATH

def compute_precision_at_k(y_true: np.ndarray, y_probs: np.ndarray, k_values: list) -> dict:
    results = {}
    sorted_indices = np.argsort(y_probs)[::-1]
    for k in k_values:
        if k > len(y_true): continue
        precision = np.sum(y_true[sorted_indices[:k]]) / k
        results[k] = precision
    return results

def forward_chaining_validation(
    df: pd.DataFrame, window_col: str = "window_id", target_col: str = "is_fraud",
    min_train_windows: int = 2, model_type: str = "xgboost"
) -> Tuple[pd.DataFrame, Dict]:
    
    df = df.sort_values(by=window_col).reset_index(drop=True)
    unique_windows = sorted(df[window_col].unique())
    
    # Keeping topological features. Only dropping metadata.
    cols_to_drop = [
        window_col, "window_start", "window_end", "entity_id", target_col,
        "leiden_macro_id", "leiden_micro_id", "is_rank_anomaly",
        'pr_vol_deep', 'pr_vol_shallow', 'pr_count', 'hits_hub', 'hits_auth',
        'leiden_macro_size', 'leiden_macro_modularity', 'leiden_micro_size', 
        'leiden_micro_modularity', 'betweenness', 'k_core', 'degree', 'in_degree', 
        'out_degree', 'fan_out_count', 'fan_in_count', 'scatter_gather_count', 
        'gather_scatter_count', 'cycle_count'
    ]
    cols_to_drop = [window_col, "window_start", "window_end", "entity_id", "is_fraud", "leiden_macro_id", "leiden_micro_id", "is_rank_anomaly"]
    drop_cols = [col for col in cols_to_drop if col in df.columns]
    
    window_results, all_y_true, all_y_probs = [], [], []
    
    for test_window_idx in range(min_train_windows, len(unique_windows)):
        test_window_id = unique_windows[test_window_idx]
        train_window_ids = unique_windows[:test_window_idx]
        
        train_df = df[df[window_col].isin(train_window_ids)]
        test_df = df[df[window_col] == test_window_id]
        
        X_train = train_df.drop(columns=drop_cols)
        y_train = train_df[target_col].values
        X_test = test_df.drop(columns=drop_cols)
        y_test = test_df[target_col].values
        
        if len(np.unique(y_test)) < 2 or len(np.unique(y_train)) < 2:
            continue
            
        # --- INTRA-LOOP ASYMMETRIC UNDER-SAMPLING ---
        fraud_indices = np.where(y_train == 1)[0]
        normal_indices = np.where(y_train == 0)[0]
        target_normal_count = len(fraud_indices) * 20 
        
        if len(normal_indices) > target_normal_count:
            np.random.seed(int(test_window_id)) 
            sampled_normal_indices = np.random.choice(normal_indices, target_normal_count, replace=False)
            keep_indices = np.concatenate([fraud_indices, sampled_normal_indices])
            np.random.shuffle(keep_indices)
            
            X_train = X_train.iloc[keep_indices]
            y_train = y_train[keep_indices]
        # ---------------------------------------------

        if model_type == "xgboost":
            model = xgb.XGBClassifier(
              # best for no graph features
                n_estimators=209,
                max_depth=8,
                learning_rate=0.0654005293991036,
                objective='binary:logistic', eval_metric='aucpr', tree_method='hist',
                random_state=42, n_jobs=-1, verbosity=0,
                min_child_weight=3,
                subsample=0.8976631049318831,
                colsample_bytree=0.6864151054104919,
                gamma=3.1410358866866583,
            )
        elif model_type == "lightgbm":
            model = lgb.LGBMClassifier(
                n_estimators=119, #best parameter for no graph features
                objective='binary', metric='auc', random_state=42, n_jobs=-1, verbosity=-1,
                max_depth=6,
                num_leaves=21,
                learning_rate=0.11273425559846745,
                min_child_samples=27,
                subsample=0.5549387649993478,
                colsample_bytree=0.5184421783713322,
                reg_alpha=4.540774375813507,
                reg_lambda=2.9925453541464027,
            )
        
        model.fit(X_train, y_train)
        y_probs = model.predict_proba(X_test)[:, 1]
        
        all_y_true.extend(y_test)
        all_y_probs.extend(y_probs)
        
        auprc = average_precision_score(y_test, y_probs)
        roc_auc = roc_auc_score(y_test, y_probs)
        precisions, recalls, thresholds = precision_recall_curve(y_test, y_probs)
        f1_scores = np.divide(2 * precisions * recalls, precisions + recalls, out=np.zeros_like(precisions), where=(precisions + recalls) != 0)
        best_threshold_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[best_threshold_idx] if best_threshold_idx < len(thresholds) else 0.5
        
        f1 = f1_score(y_test, (y_probs >= optimal_threshold).astype(int))
        
        result = {
            "test_window_id": test_window_id, "n_train_samples": len(X_train),
            "auprc": auprc, "roc_auc": roc_auc, "f1_score": f1, "optimal_threshold": optimal_threshold,
        }
        for k, prec in compute_precision_at_k(y_test, y_probs, [10, 50, 100, 500]).items():
            result[f"precision_at_{k}"] = prec
        window_results.append(result)
        print(f"  Window {test_window_id:3d}: AUPRC={auprc:.4f}, F1={f1:.4f}, Train_size={len(X_train):,}")
    
    all_y_true, all_y_probs = np.array(all_y_true), np.array(all_y_probs)
    precisions, recalls, thresholds = precision_recall_curve(all_y_true, all_y_probs)
    f1_scores = np.divide(2 * precisions * recalls, precisions + recalls, out=np.zeros_like(precisions), where=(precisions + recalls) != 0)
    overall_threshold = thresholds[np.argmax(f1_scores)] if np.argmax(f1_scores) < len(thresholds) else 0.5
    
    summary = {
        "n_test_windows": len(window_results), "total_test_samples": len(all_y_true),
        "overall_fraud_rate": np.mean(all_y_true), "overall_auprc": average_precision_score(all_y_true, all_y_probs),
        "overall_roc_auc": roc_auc_score(all_y_true, all_y_probs), "overall_threshold": overall_threshold,
        "overall_f1": f1_score(all_y_true, (all_y_probs >= overall_threshold).astype(int)),
        "mean_window_auprc": pd.DataFrame(window_results)["auprc"].mean()
    }
    return pd.DataFrame(window_results), summary

def main():
    df = pd.read_parquet(DATA_PATH / OUTPUT_FEATURES_FILE)
    window_col = "window_id" if "window_id" in df.columns else "date"
    
    xgb_results, xgb_summary = forward_chaining_validation(df, window_col=window_col, model_type="xgboost")
    
    print("\n" + "-" * 80)
    print("XGBOOST AGGREGATE RESULTS")
    print(f"Total test samples: {xgb_summary['total_test_samples']:,}")
    print(f"Overall AUPRC:      {xgb_summary['overall_auprc']:.4f}")
    print(f"Overall F1-Score:   {xgb_summary['overall_f1']:.4f} (threshold={xgb_summary['overall_threshold']:.4f})")
    print(f"Overall ROC-AUC:    {xgb_summary['overall_roc_auc']:.4f}")

if __name__ == "__main__":
    main()
