import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
import joblib
import os

def main():
    """
    This script trains a supervised RandomForestClassifier to predict
    fraudulent transactions in real-time.
    """
    print("Starting supervised model training...")

    # --- 1. Data Loading and Preparation ---
    try:
        # Use low_memory=False to handle mixed types more efficiently
        transactions_df = pd.read_csv("pix_transactions.csv", low_memory=False)
        account_features_df = pd.read_csv("account_features.csv")
    except FileNotFoundError as e:
        print(f"Error: {e}. Make sure 'pix_transactions.csv' and 'account_features.csv' exist.")
        print("Please run data_generator.py and feature_engineering.py first.")
        return

    print(f"Loaded {len(transactions_df)} transactions and features for {len(account_features_df)} accounts.")

    # --- 2. Feature Engineering for Supervised Learning ---
    # Create the target variable: 1 for fraud, 0 for normal
    normal_flags = ['NORMAL', 'NORMAL_MICRO', 'NORMAL_SALARY', 'NORMAL_B2B']
    transactions_df['is_fraud'] = transactions_df['fraud_flag'].apply(
        lambda x: 0 if x in normal_flags else 1
    )

    # Rename account features to avoid column name conflicts after merging
    # These represent the features of the SENDER of the transaction
    sender_features_df = account_features_df.add_prefix('sender_')
    
    # Merge transaction data with sender's features
    # This simulates the data we'll have at prediction time
    model_df = pd.merge(
        transactions_df,
        sender_features_df,
        left_on='sender_id',
        right_on='sender_accountId',
        how='left'
    )

    # Fill any missing values that might result from the join
    model_df.fillna(0, inplace=True)

    # Convert timestamp, coercing errors to NaT (Not a Time)
    model_df['timestamp'] = pd.to_datetime(model_df['timestamp'], errors='coerce')

    # Drop rows where the timestamp could not be parsed
    model_df.dropna(subset=['timestamp'], inplace=True)
    
    model_df['hour'] = model_df['timestamp'].dt.hour
    model_df['day'] = model_df['timestamp'].dt.day
    model_df['month'] = model_df['timestamp'].dt.month
    model_df['day_of_week'] = model_df['timestamp'].dt.dayofweek

    # --- 3. Model Training ---
    # Define the feature set for the model
    # These are features that would be available in a real-time scenario
    features_list = [
        'amount',
        'hour',
        'day',
        'month',
        'day_of_week',
        'sender_inDegree',
        'sender_outDegree',
        'sender_totalDegree',
        'sender_totalAmount',
        'sender_avgAmount',
        'sender_maxAmount',
        'sender_transactionCount',
        'sender_riskScore'
    ]
    
    X = model_df[features_list]
    y = model_df['is_fraud']

    print(f"\nTraining model with {len(features_list)} features.")
    print("Feature list:", features_list)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train the RandomForestClassifier
    print("\nTraining RandomForestClassifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train_scaled, y_train)

    # --- 4. Model Evaluation ---
    print("\nEvaluating model performance...")
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    auc_score = roc_auc_score(y_test, y_pred_proba)
    print(f"ROC AUC Score: {auc_score:.4f}")

    # --- 5. Save the Model, Scaler, and Feature List ---
    model_path = 'realtime_fraud_model.joblib'
    scaler_path = 'realtime_scaler.joblib'
    features_path = 'realtime_features.joblib'
    
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(features_list, features_path)
    print(f"\nTrained model saved to '{model_path}'")
    print(f"Scaler saved to '{scaler_path}'")
    print(f"Feature list saved to '{features_path}'")


if __name__ == "__main__":
    main()
