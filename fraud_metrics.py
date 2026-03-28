import pandas as pd

# Load your final Full Model predictions
# Note: These predictions are already strictly from the out-of-sample test windows
df = pd.read_parquet("data/HI_Small/results/full_predictions.parquet")

# Isolate only the actual fraudulent instances
fraud_df = df[df['y_true'] == 1]

# 1. Count total unique fraudulent entities in the test timeline
total_fraud_accounts = fraud_df['entity_id'].nunique()

# 2. Count unique fraudulent entities that triggered AT LEAST ONE alert (y_pred == 1)
caught_fraud_accounts = fraud_df[fraud_df['y_pred'] == 1]['entity_id'].nunique()

# 3. Calculate Entity-Level Recall
entity_recall = caught_fraud_accounts / total_fraud_accounts

print(f"Total Fraudulent Accounts in Test Set: {total_fraud_accounts:,}")
print(f"Fraudulent Accounts Caught: {caught_fraud_accounts:,}")
print(f"Entity-Level Recall: {entity_recall:.2%}")
