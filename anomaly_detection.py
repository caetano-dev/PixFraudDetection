import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib

def main():
    # Load features
    try:
        features_df = pd.read_csv("src/data/account_features.csv")
    except FileNotFoundError:
        print("Error: src/data/account_features.csv not found.")
        print("Please run feature_engineering.py first to generate the features.")
        return

    print("Loaded features for {} accounts.".format(len(features_df)))

    # Define features to be used for anomaly detection
    # We exclude accountId as it's an identifier, not a feature
    features_list = [
        'inDegree', 'outDegree', 'totalDegree',
        'totalAmount', 'avgAmount', 'maxAmount',
        'transactionCount', 'riskScore'
    ]
    
    X = features_df[features_list]

    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply Isolation Forest model
    # contamination='auto' is a good starting point
    # n_estimators can be tuned
    print("Applying Isolation Forest model...")
    model = IsolationForest(n_estimators=100, contamination='auto', random_state=42)
    predictions = model.fit_predict(X_scaled)

    # The model's decision_function gives the anomaly score
    # Lower scores are more anomalous
    anomaly_scores = model.decision_function(X_scaled)

    # Add results back to the DataFrame
    # The prediction is -1 for outliers and 1 for inliers.
    features_df['anomaly_score'] = anomaly_scores
    features_df['is_outlier'] = predictions

    # Convert prediction to a more intuitive 0/1 format (1 for outlier)
    features_df['is_outlier'] = features_df['is_outlier'].apply(lambda x: 1 if x == -1 else 0)

    # Save the results
    output_path = "src/data/anomaly_scores.csv"
    features_df.to_csv(output_path, index=False)
    print(f"Anomaly detection complete. Results saved to {output_path}")

    # Print summary
    outlier_count = features_df['is_outlier'].sum()
    print(f"\nDetected {outlier_count} potential outliers.")
    print("\nTop 10 most anomalous accounts:")
    print(features_df.sort_values('anomaly_score').head(10))
    
    # Save the model and the scaler for later use (e.g., in a real-time pipeline)
    joblib.dump(model, 'src/models/isolation_forest_model.joblib')
    joblib.dump(scaler, 'src/models/scaler.joblib')
    print("\nModel and scaler saved to 'src/models/isolation_forest_model.joblib' and 'src/models/scaler.joblib'")


if __name__ == "__main__":
    main()
