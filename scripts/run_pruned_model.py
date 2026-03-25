import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, precision_recall_curve, precision_score, recall_score
from pathlib import Path
import sys
from typing import Dict, List, Tuple

root_path = Path(__file__).resolve().parent.parent
sys.path.append(str(root_path))

# Override config to use SMALL dataset
import src.config as config
config.DATASET_SIZE = "SMALL"
config.DATA_PATH = Path("data/HI_Small")

from src.config import OUTPUT_FEATURES_FILE, DATA_PATH

# Features discarded by the pruner
DISCARDED_FEATURES = [
    "flow_ratio",
    "out_degree",
    "distinct_currencies_sent",
    "credit_card_count_sent",
    "time_variance",
    "k_core",
    "pr_vol_shallow",
    "leiden_micro_modularity",
    "gather_scatter_count",
    "reinvestment_count_recv"
]

def compute_precision_at_k(y_true: np.ndarray, y_probs: np.ndarray, k_values: list) -> dict:
    """Computes Precision@K to measure real-world investigator efficiency."""
    results = {}
    sorted_indices = np.argsort(y_probs)[::-1]
    for k in k_values:
        if k > len(y_true): continue
        precision = np.sum(y_true[sorted_indices[:k]]) / k
        results[k] = precision
    return results

def undersample_data(data: pd.DataFrame, target_col: str = "is_fraud", ratio: int = 20) -> pd.DataFrame:
    """Enforces strict 1:N asymmetric undersampling to stabilize training distributions."""
    fraud = data[data[target_col] == 1]
    normal = data[data[target_col] == 0]
    target_count = len(fraud) * ratio
    
    if len(normal) > target_count:
        normal = normal.sample(n=target_count, random_state=42)
    
    return pd.concat([fraud, normal]).sample(frac=1, random_state=42)

def forward_chaining_validation(
    df: pd.DataFrame,
    feature_cols: List[str],
    window_col: str = "window_id",
    target_col: str = "is_fraud",
    min_train_windows: int = 2,
    n_trials: int = 20,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, Dict]:
    
    df = df.sort_values(by=window_col).reset_index(drop=True)
    unique_windows = sorted(df[window_col].unique())
    feature_cols = [col for col in feature_cols if col in df.columns]
    
    window_results = []
    all_y_true = []
    all_y_probs = []
    all_y_pred = []
    
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    for test_window_idx in range(min_train_windows, len(unique_windows)):
        test_window_id = unique_windows[test_window_idx]
        train_window_ids = unique_windows[:test_window_idx]
        
        train_df = df[df[window_col].isin(train_window_ids)].copy()
        test_df = df[df[window_col] == test_window_id].copy()
        
        if len(train_df[target_col].unique()) < 2 or len(test_df[target_col].unique()) < 2:
            if verbose: print(f"Window {test_window_id}: Skipped (insufficient variance)")
            continue

        # 1. Inner Validation Split (Last 20% of training data)
        split_idx = int(len(train_df) * 0.8)
        inner_train_df = train_df.iloc[:split_idx]
        inner_val_df = train_df.iloc[split_idx:]
        
        # Apply undersampling strictly to the training split
        inner_train_df = undersample_data(inner_train_df, target_col, ratio=20)
        
        X_inner_train, y_inner_train = inner_train_df[feature_cols], inner_train_df[target_col].values
        X_inner_val, y_inner_val = inner_val_df[feature_cols], inner_val_df[target_col].values
        X_test, y_test = test_df[feature_cols], test_df[target_col].values

        # 2. Inner-Loop Hyperparameter Optimization
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 300),
                'max_depth': trial.suggest_int('max_depth', 4, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'gamma': trial.suggest_float('gamma', 0.0, 5.0),
                'objective': 'binary:logistic',
                'eval_metric': 'aucpr',
                'tree_method': 'hist',
                'random_state': 42,
                'device': 'cuda',
                'n_jobs': -1
            }
            model = xgb.XGBClassifier(**params)
            model.fit(X_inner_train, y_inner_train)
            preds = model.predict_proba(X_inner_val)[:, 1]
            return average_precision_score(y_inner_val, preds)

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        
        # 3. Out-of-Sample Threshold Calibration (using model trained ONLY on inner_train)
        best_params = study.best_params
        best_params.update({'objective': 'binary:logistic', 'eval_metric': 'aucpr', 'tree_method': 'hist', 'random_state': 42, 'n_jobs': -1})
        
        calib_model = xgb.XGBClassifier(**best_params)
        calib_model.fit(X_inner_train, y_inner_train)
        val_probs = calib_model.predict_proba(X_inner_val)[:, 1]
        precisions, recalls, thresholds = precision_recall_curve(y_inner_val, val_probs)
        f1_scores = np.divide(2 * precisions * recalls, precisions + recalls, out=np.zeros_like(precisions), where=(precisions + recalls) != 0)
        optimal_threshold = thresholds[np.argmax(f1_scores)] if len(thresholds) > 0 else 0.5

        # 4. Train final model on ALL historical data (undersampled)
        final_train_df = undersample_data(train_df, target_col, ratio=20)
        X_train_final, y_train_final = final_train_df[feature_cols], final_train_df[target_col].values
        
        final_model = xgb.XGBClassifier(**best_params)
        final_model.fit(X_train_final, y_train_final)

        # 5. Out-of-Sample Prediction
        y_probs = final_model.predict_proba(X_test)[:, 1]
        y_pred = (y_probs >= optimal_threshold).astype(int)
        
        all_y_true.extend(y_test)
        all_y_probs.extend(y_probs)
        all_y_pred.extend(y_pred)
        
        auprc = average_precision_score(y_test, y_probs)
        roc_auc = roc_auc_score(y_test, y_probs)
        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        
        result = {
            "test_window_id": test_window_id,
            "train_samples": len(X_train_final),
            "test_fraud_rate": np.mean(y_test),
            "auprc": auprc,
            "roc_auc": roc_auc,
            "f1_score": f1,
            "precision": precision,
            "recall": recall,
            "optimal_threshold": optimal_threshold,
        }
        for k, prec in compute_precision_at_k(y_test, y_probs, [1, 2, 10, 50, 100, 500]).items():
            result[f"precision_at_{k}"] = prec
            
        window_results.append(result)
        
        if verbose:
            print(f"Window {test_window_id}: AUPRC={auprc:.4f}, ROC-AUC={roc_auc:.4f}, F1={f1:.4f}, P@100={result.get('precision_at_100', 0):.4f}")

    results_df = pd.DataFrame(window_results)
    
    all_y_true = np.array(all_y_true)
    all_y_probs = np.array(all_y_probs)
    all_y_pred = np.array(all_y_pred)
    
    # Calculate precision@k values for summary
    k_values = [1, 2, 10, 50, 100, 500]
    precision_at_k_dict = compute_precision_at_k(all_y_true, all_y_probs, k_values)
    
    summary = {
        "n_test_windows": len(window_results),
        "total_test_samples": len(all_y_true),
        "overall_fraud_rate": np.mean(all_y_true),
        "overall_auprc": average_precision_score(all_y_true, all_y_probs),
        "overall_roc_auc": roc_auc_score(all_y_true, all_y_probs),
        "overall_precision": precision_score(all_y_true, all_y_pred, zero_division=0),
        "overall_recall": recall_score(all_y_true, all_y_pred, zero_division=0),
        "mean_window_auprc": results_df["auprc"].mean(),
        "median_window_auprc": results_df["auprc"].median(),
        "mean_window_roc_auc": results_df["roc_auc"].mean(),
        "median_window_roc_auc": results_df["roc_auc"].median(),
        "std_window_auprc": results_df["auprc"].std()
    }
    
    # Add mean and median precision@k to summary
    for k in k_values:
        col_name = f"precision_at_{k}"
        if col_name in results_df.columns:
            summary[f"mean_{col_name}"] = results_df[col_name].mean()
            summary[f"median_{col_name}"] = results_df[col_name].median()
    
    return results_df, summary


def main():
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
        'gather_scatter_count', 'cycle_count'
    ]
    FULL_COLS = BEHAVIORAL_COLS + TOPOLOGICAL_COLS
    
    # Remove discarded features
    PRUNED_COLS = [col for col in FULL_COLS if col not in DISCARDED_FEATURES]

    df = pd.read_parquet(DATA_PATH / OUTPUT_FEATURES_FILE)
    window_col = "window_id" if "window_id" in df.columns else "date"
    
    print("\n" + "=" * 80)
    print(f"FORWARD-CHAINING VALIDATION: PRUNED MODEL")
    print(f"Original features: {len(FULL_COLS)}")
    print(f"Discarded features: {len(DISCARDED_FEATURES)}")
    print(f"Remaining features: {len(PRUNED_COLS)}")
    print("=" * 80)
    print("\nDiscarded features:")
    for feat in DISCARDED_FEATURES:
        print(f"  - {feat}")
    print("=" * 80)
    
    # Run the validation loop
    results_df, summary = forward_chaining_validation(
        df=df,
        feature_cols=PRUNED_COLS,
        window_col=window_col,
        target_col="is_fraud",
        min_train_windows=2,
        n_trials=20,
        verbose=True
    )
    
    print("\n" + "=" * 80)
    print("PRUNED MODEL RESULTS")
    print("=" * 80)
    print(f"Test Windows:       {summary['n_test_windows']}")
    print(f"Total test samples: {summary['total_test_samples']:,}")
    print(f"Overall Fraud Rate: {summary['overall_fraud_rate']:.4%}")
    print(f"Overall AUPRC:      {summary['overall_auprc']:.4f}")
    print(f"Mean Window AUPRC:  {summary['mean_window_auprc']:.4f} ± {summary['std_window_auprc']:.4f}")
    print(f"Median Window AUPRC:{summary['median_window_auprc']:.4f}")
    print(f"Overall ROC-AUC:    {summary['overall_roc_auc']:.4f}")
    print(f"Mean Window ROC-AUC:{summary['mean_window_roc_auc']:.4f}")
    print(f"Median ROC-AUC:     {summary['median_window_roc_auc']:.4f}")
    
    print("\n" + "-" * 80)
    print("MEAN PRECISION@K")
    print("-" * 80)
    baseline_fraud_rate = summary['overall_fraud_rate']
    for k in [1, 2, 10, 50, 100, 500]:
        prec = summary.get(f"mean_precision_at_{k}", 0)
        lift = prec / baseline_fraud_rate if baseline_fraud_rate > 0 else 0
        print(f"  @ {k:4d}: {prec*100:6.4f}% (lift: {lift:.2f}x)")
    
    print("\n" + "-" * 80)
    print("MEDIAN PRECISION@K")
    print("-" * 80)
    for k in [1, 2, 10, 50, 100, 500]:
        prec = summary.get(f"median_precision_at_{k}", 0)
        lift = prec / baseline_fraud_rate if baseline_fraud_rate > 0 else 0
        print(f"  @ {k:4d}: {prec*100:6.4f}% (lift: {lift:.2f}x)")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
