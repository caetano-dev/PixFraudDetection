"""
Supervised Machine Learning and Comparative Analysis Pipeline.

This script executes the final phase of the TCC:
1. Loads the extracted temporal graph features.
2. Performs a strict chronological Train/Validation/Test split.
3. Trains an XGBoost classifier with dynamic class-imbalance weighting and early stopping.
4. Evaluates performance using Precision@K and AUPRC.
5. Generates SHAP values to empirically rank algorithm effectiveness.
"""
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score
from sklearn.ensemble import IsolationForest
import shap
import matplotlib.pyplot as plt
from pathlib import Path
import sys

root_path = Path(__file__).resolve().parent.parent
sys.path.append(str(root_path))

from src.config import OUTPUT_FEATURES_FILE, DATA_PATH

def compute_precision_at_k(y_true: np.ndarray, y_probs: np.ndarray, k_values: list) -> dict:
    """Computes Precision@K to measure real-world investigator efficiency."""
    results = {}
    # Sort indices by highest predicted probability
    sorted_indices = np.argsort(y_probs)[::-1]
    
    for k in k_values:
        if k > len(y_true):
            continue
        top_k_indices = sorted_indices[:k]
        top_k_labels = y_true[top_k_indices]
        precision = np.sum(top_k_labels) / k
        results[k] = precision
    return results

def main():
    features_path = DATA_PATH / OUTPUT_FEATURES_FILE
    if not features_path.exists():
        raise FileNotFoundError(f"Missing {features_path}. Run 03_extract_features.py first.")

    print(f"Loading feature matrix from {features_path}...")
    df = pd.read_parquet(features_path)

    # 1. Feature Engineering & Cleanup
    # Ensure data is sorted by time to prevent any future leakage
    df = df.sort_values(by=["date", "entity_id"]).reset_index(drop=True)
    
    # 2. Strict Chronological Split (60% Train / 20% Val / 20% Test)
    print("\n[Temporal Split Enforced]")
    n_rows = len(df)
    train_end_idx = int(n_rows * 0.6)
    val_end_idx = int(n_rows * 0.8)
    
    train_df = df.iloc[:train_end_idx]
    val_df = df.iloc[train_end_idx:val_end_idx]
    test_df = df.iloc[val_end_idx:]

    print(f"Training set:   {len(train_df):,} rows (up to {train_df['date'].max()})")
    print(f"Validation set: {len(val_df):,} rows (up to {val_df['date'].max()})")
    print(f"Testing set:    {len(test_df):,} rows (up to {test_df['date'].max()})")

    # Drop non-predictive columns and arbitrary integer IDs
    cols_to_drop = [
        "date", "entity_id", "is_fraud", 
        "leiden_macro_id", "leiden_micro_id", "is_rank_anomaly"
    ]
    drop_cols = [col for col in cols_to_drop if col in df.columns]
    
    X_train = train_df.drop(columns=drop_cols)
    y_train = train_df["is_fraud"].values
    
    X_val = val_df.drop(columns=drop_cols)
    y_val = val_df["is_fraud"].values
    
    X_test = test_df.drop(columns=drop_cols)
    y_test = test_df["is_fraud"].values

    # 3. Class Imbalance Handling
    # Use exact ratio weighting for the XGBoost objective function on Train set only
    num_neg = np.sum(y_train == 0)
    num_pos = np.sum(y_train == 1)
    pos_weight = num_neg / num_pos if num_pos > 0 else 1.0
    print(f"\nClass Imbalance Ratio (Neg/Pos) in Train: {pos_weight:.2f}")

    # 4. Train XGBoost Model with Validation Early Stopping
    # 
    # --- 1. OPTIMIZED HYPERPARAMETERS ---
    # Use sqrt of imbalance to prevent Precision collapse while aiding Recall
    imbalance_ratio = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
    tuned_scale_pos_weight = np.sqrt(imbalance_ratio) 

    print("\nTraining XGBoost Classifier with Tuned Graph Parameters...")
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='aucpr',
        scale_pos_weight=tuned_scale_pos_weight,
        learning_rate=0.05,          # Slower, more robust convergence
        max_depth=5,                 # Restrict depth to prevent noise memorization
        subsample=0.8,               # Row stochasticity
        colsample_bytree=0.8,        # Feature stochasticity
        min_child_weight=3,          # Require more evidence to create a leaf
        n_estimators=1000,
        random_state=42,
        tree_method='hist',           # Faster execution for large datasets
        early_stopping_rounds=50
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=50
    )

    # --- 2. DYNAMIC THRESHOLD TUNING (ON VALIDATION SET) ---
    from sklearn.metrics import precision_recall_curve
    
    print("\nExtracting Optimal F1 Threshold from Validation Set...")
    val_probs = model.predict_proba(X_val)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_val, val_probs)
    
    # Calculate F1 for all thresholds, ignoring division by zero
    f1_scores = np.divide(
        2 * precisions * recalls, 
        precisions + recalls, 
        out=np.zeros_like(precisions), 
        where=(precisions + recalls) != 0
    )
    best_threshold_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[best_threshold_idx] if best_threshold_idx < len(thresholds) else 0.5
    print(f"Optimal Threshold Found: {optimal_threshold:.4f}")

    # --- 3. EVALUATION (ON TEST SET) ---
    print("\nEvaluating on Future Temporal Window (Test Set)...")
    y_probs = model.predict_proba(X_test)[:, 1]
    
    auprc = average_precision_score(y_test, y_probs)
    roc_auc = roc_auc_score(y_test, y_probs)
    
    # Apply the empirically proven threshold instead of 0.5
    y_pred = (y_probs >= optimal_threshold).astype(int)
    f1 = f1_score(y_test, y_pred)

    print(f"Test AUPRC:   {auprc:.4f}")
    print(f"Test ROC-AUC: {roc_auc:.4f}")
    print(f"Test F1-Score: {f1:.4f} (at {optimal_threshold:.4f} threshold)")

    
    print("\nTest Precision@K:")
    baseline_fraud_rate = np.mean(y_test)
    k_vals = [10, 50, 100, 500]
    prec_at_k = compute_precision_at_k(y_test, y_probs, k_vals)
    for k, prec in prec_at_k.items():
        lift = prec / baseline_fraud_rate if baseline_fraud_rate > 0 else 0
        print(f"  @ {k:>3}: {prec:.4%} (Lift: {lift:.2f}x)")

    # 5. UNSUPERVISED BENCHMARK — Isolation Forest
    print("\n========================================")
    print("[UNSUPERVISED BENCHMARK] Training Isolation Forest...")
    print("========================================")

    train_fraud_rate = float(np.mean(y_train))

    iso_forest = IsolationForest(
        n_estimators=200,
        contamination=train_fraud_rate,
        random_state=42,
        n_jobs=-1
    )
    iso_forest.fit(X_train)
    iso_scores = -iso_forest.decision_function(X_test)

    iso_auprc  = average_precision_score(y_test, iso_scores)
    iso_roc_auc = roc_auc_score(y_test, iso_scores)

    print(f"Isolation Forest Test AUPRC:   {iso_auprc:.4f}")
    print(f"Isolation Forest Test ROC-AUC: {iso_roc_auc:.4f}")

    print("\nIsolation Forest Test Precision@K:")
    iso_prec_at_k = compute_precision_at_k(y_test, iso_scores, k_vals)
    for k, prec in iso_prec_at_k.items():
        lift = prec / baseline_fraud_rate if baseline_fraud_rate > 0 else 0
        print(f"  @ {k:>3}: {prec:.4%} (Lift: {lift:.2f}x)")

    # 6. SHAP Comparative Analysis (The TCC Defense)
    print("\nExtracting SHAP Values for Comparative Algorithm Analysis...")
    # Use TreeExplainer on a sample of the test set to avoid memory overload
    X_test_sample = X_test.sample(n=min(10000, len(X_test)), random_state=42)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test_sample)

    # Save the SHAP summary plot for the thesis document
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test_sample, show=False)
    
    output_dir = Path("notebooks")
    output_dir.mkdir(exist_ok=True)
    plot_path = output_dir / "shap_feature_importance.png"
    plt.savefig(plot_path, bbox_inches='tight', dpi=300)
    print(f"\n[Success] SHAP comparative analysis plot saved to {plot_path}")

if __name__ == "__main__":
    main()
