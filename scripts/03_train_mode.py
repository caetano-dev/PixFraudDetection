"""
Supervised Machine Learning and Comparative Analysis Pipeline.

This script executes the final phase of the TCC:
1. Loads the extracted temporal graph features.
2. Performs a strict chronological Train/Test split.
3. Trains an XGBoost classifier with dynamic class-imbalance weighting.
4. Evaluates performance using Precision@K and AUPRC.
5. Generates SHAP values to empirically rank algorithm effectiveness.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score
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
        raise FileNotFoundError(f"Missing {features_path}. Run 02_extract_features.py first.")

    print(f"Loading feature matrix from {features_path}...")
    df = pd.read_parquet(features_path)

    # 1. Feature Engineering & Cleanup
    # Sort chronologically to prevent future data leakage
    df = df.sort_values(by=["date", "entity_id"]).reset_index(drop=True)
    
    # Isolate targets and identifiers
    y = df["is_fraud"].values
    
    # Drop non-predictive columns and arbitrary integer IDs
    cols_to_drop = [
        "date", "entity_id", "is_fraud", 
        "leiden_macro_id", "leiden_micro_id", "is_rank_anomaly"
    ]
    X = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
    
    # 2. Strict Chronological Split (70% Train, 30% Test)
    # Splitting by unique dates ensures we don't sever a single day's graph
    unique_dates = df['date'].unique()
    split_index = int(len(unique_dates) * 0.70)
    split_date = unique_dates[split_index]
    
    train_mask = df['date'] <= split_date
    test_mask = df['date'] > split_date
    
    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    
    print(f"\n[Temporal Split Enforced]")
    print(f"Training set: {len(X_train):,} rows (<= {split_date})")
    print(f"Testing set:  {len(X_test):,} rows (> {split_date})")

    # 3. Class Imbalance Handling
    # Do not use SMOTE. Use exact ratio weighting for the XGBoost objective function.
    num_neg = np.sum(y_train == 0)
    num_pos = np.sum(y_train == 1)
    pos_weight = num_neg / num_pos if num_pos > 0 else 1.0
    print(f"Class Imbalance Ratio (Neg/Pos): {pos_weight:.2f}")

    # 4. Train XGBoost Model
    print("\nTraining XGBoost Classifier...")
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=pos_weight,
        eval_metric="aucpr",
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)

    # 5. Evaluation Metrics
    print("\nEvaluating on Future Temporal Window (Test Set)...")
    y_probs = model.predict_proba(X_test)[:, 1]
    
    auprc = average_precision_score(y_test, y_probs)
    roc_auc = roc_auc_score(y_test, y_probs)
    
    print(f"Test AUPRC:   {auprc:.4f}")
    print(f"Test ROC-AUC: {roc_auc:.4f}")
    
    print("\nTest Precision@K:")
    baseline_fraud_rate = np.mean(y_test)
    k_vals = [10, 50, 100, 500]
    prec_at_k = compute_precision_at_k(y_test, y_probs, k_vals)
    for k, prec in prec_at_k.items():
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