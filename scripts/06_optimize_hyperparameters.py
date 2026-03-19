"""
Bayesian Hyperparameter Optimization using Optuna and Intra-Loop Under-sampling.
Includes Topological Features and LightGBM support.
"""
import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
import optuna
from sklearn.metrics import average_precision_score, precision_recall_curve, f1_score
from pathlib import Path
import sys
import gc

root_path = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(root_path))
from src.config import DATA_PATH, OUTPUT_FEATURES_FILE

optuna.logging.set_verbosity(optuna.logging.WARNING)

def objective(trial, df, window_col, target_col, drop_cols, model_type):
    if model_type == 'xgboost':
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 250),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 15),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'gamma': trial.suggest_float('gamma', 0.0, 5.0),
            'objective': 'binary:logistic',
            'eval_metric': 'aucpr',
            'tree_method': 'hist',
            'random_state': 42,
            'n_jobs': -1 
        }
    elif model_type == 'lightgbm':
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 250),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'num_leaves': trial.suggest_int('num_leaves', 20, 100),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 5.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 5.0),
            'objective': 'binary',
            'metric': 'auc',
            'random_state': 42,
            'n_jobs': -1,
            'verbosity': -1
        }

    unique_windows = sorted(df[window_col].unique())
    min_train_windows = 2
    
    all_y_true = []
    all_y_probs = []

    for test_window_idx in range(min_train_windows, len(unique_windows)):
        test_window_id = unique_windows[test_window_idx]
        train_window_ids = unique_windows[:test_window_idx]
        
        train_df = df[df[window_col].isin(train_window_ids)]
        test_df = df[df[window_col] == test_window_id]
        
        X_train = train_df.drop(columns=drop_cols, errors='ignore')
        y_train = train_df[target_col].values
        X_test = test_df.drop(columns=drop_cols, errors='ignore')
        y_test = test_df[target_col].values
        
        if len(np.unique(y_test)) < 2 or len(np.unique(y_train)) < 2:
            continue

        # --- DETERMINISTIC INTRA-LOOP ASYMMETRIC UNDER-SAMPLING ---
        fraud_indices = np.where(y_train == 1)[0]
        normal_indices = np.where(y_train == 0)[0]
        
        target_normal_count = len(fraud_indices) * 20 # 1:20 ratio
        
        if len(normal_indices) > target_normal_count:
            np.random.seed(int(test_window_id) * 42) 
            sampled_normal_indices = np.random.choice(normal_indices, target_normal_count, replace=False)
            
            keep_indices = np.concatenate([fraud_indices, sampled_normal_indices])
            np.random.shuffle(keep_indices)
            
            X_train = X_train.iloc[keep_indices]
            y_train = y_train[keep_indices]
        # ---------------------------------------------------------
            
        if model_type == 'xgboost':
            model = xgb.XGBClassifier(**params)
        else:
            model = lgb.LGBMClassifier(**params)
            
        model.fit(X_train, y_train)
        y_probs = model.predict_proba(X_test)[:, 1]
        
        all_y_true.extend(y_test)
        all_y_probs.extend(y_probs)
        
        del train_df, test_df, X_train, X_test, model
        gc.collect()

    if not all_y_true:
        raise optuna.exceptions.TrialPruned()

    auprc = average_precision_score(all_y_true, all_y_probs)
    
    precisions, recalls, thresholds = precision_recall_curve(all_y_true, all_y_probs)
    f1_scores = np.divide(2 * precisions * recalls, precisions + recalls, out=np.zeros_like(precisions), where=(precisions + recalls) != 0)
    best_f1 = np.max(f1_scores)
    
    trial.set_user_attr("best_f1", float(best_f1))
    
    return auprc

def main():
    features_path = DATA_PATH / OUTPUT_FEATURES_FILE
    df = pd.read_parquet(features_path)
    window_col = 'window_id' if 'window_id' in df.columns else 'date'
    
    # Keeping topological features. Only dropping metadata.
    cols_to_drop = [
        window_col, "window_start", "window_end", "entity_id", "is_fraud",
        "leiden_macro_id", "leiden_micro_id", "is_rank_anomaly"
    ]
    cols_to_drop = [
          window_col, "window_start", "window_end", "entity_id",
          "leiden_macro_id", "leiden_micro_id", "is_rank_anomaly",
          'pr_vol_deep', 'pr_vol_shallow', 'pr_count', 'hits_hub', 'hits_auth',
          'leiden_macro_size', 'leiden_macro_modularity', 'leiden_micro_size', 
          'leiden_micro_modularity', 'betweenness', 'k_core', 'degree', 'in_degree', 
          'out_degree', 'fan_out_count', 'fan_in_count', 'scatter_gather_count', 
          'gather_scatter_count', 'cycle_count'
      ]
    
    model_choice = "xgboost" # Toggle to 'xgboost' if needed
    
    print(f"Starting Bayesian Optimization for {model_choice.upper()} (50 trials)...")
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, df, window_col, "is_fraud", cols_to_drop, model_choice), n_trials=50, show_progress_bar=True)
    
    print("\n" + "="*50)
    print("OPTIMIZATION COMPLETE")
    print(f"Best AUPRC:        {study.best_value:.4f}")
    print(f"Associated F1:     {study.best_trial.user_attrs['best_f1']:.4f}")
    print("\nBest Hyperparameters:")
    for key, value in study.best_trial.params.items():
        print(f"    {key}={value},")

if __name__ == "__main__":
    main()