import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
import joblib
import argparse

def apply_isolation_forest(features_df, features_list):
    """
    Applies Isolation Forest for anomaly detection.
    """
    print("Applying Isolation Forest model...")
    
    X = features_df[features_list]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(n_estimators=100, contamination='auto', random_state=42)
    predictions = model.fit_predict(X_scaled)
    anomaly_scores = model.decision_function(X_scaled)

    features_df['anomaly_score'] = anomaly_scores
    features_df['is_outlier'] = (predictions == -1).astype(int)
    
    # Save the model and the scaler for later use
    joblib.dump(model, './models/isolation_forest_model.joblib')
    joblib.dump(scaler, './models/scaler.joblib')
    
    return features_df

def apply_local_outlier_factor(features_df, features_list, n_neighbors=20, contamination=0.1):
    """
    Applies Local Outlier Factor for anomaly detection.
    """
    print("Applying Local Outlier Factor model...")
    
    X = features_df[features_list]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Adjust n_neighbors if dataset is small
    actual_n_neighbors = min(n_neighbors, len(features_df) - 1)
    
    model = LocalOutlierFactor(n_neighbors=actual_n_neighbors, contamination=contamination)
    predictions = model.fit_predict(X_scaled)
    anomaly_scores = model.negative_outlier_factor_
    
    features_df['anomaly_score'] = anomaly_scores
    features_df['is_outlier'] = (predictions == -1).astype(int)
    
    # Save the scaler for later use (LOF doesn't have a save method as it doesn't support predict)
    joblib.dump(scaler, './models/lof_scaler.joblib')
    
    return features_df

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Anomaly Detection for PIX Fraud Detection')
    parser.add_argument('--algorithm', choices=['isolation_forest', 'lof'], default='lof',
                       help='Algorithm to use for anomaly detection (default: lof)')
    parser.add_argument('--n_neighbors', type=int, default=20,
                       help='Number of neighbors for LOF algorithm (default: 20)')
    parser.add_argument('--contamination', type=float, default=0.1,
                       help='Contamination parameter (default: 0.1)')
    
    args = parser.parse_args()
    
    # Load features
    try:
        features_df = pd.read_csv("./data/account_features.csv")
    except FileNotFoundError:
        print("Error: account_features.csv not found.")
        print("Please run feature_engineering.py first to generate the features.")
        return

    print("Loaded features for {} accounts.".format(len(features_df)))

    # Define features to be used for anomaly detection (production-ready features only)
    # These features are all observable without knowing fraud labels
    features_list = [
        # Transaction volume and patterns
        'totalTransactions', 'sentCount', 'receivedCount', 'uniqueCounterparties',
        'totalAmount', 'totalSentAmount', 'totalReceivedAmount',
        'avgTransactionAmount', 'maxTransactionAmount', 'minTransactionAmount',
        'sentToReceivedRatio', 'amountVariance',
        
        # Temporal behavior patterns
        'weekendTransactionRatio', 'nightTransactionRatio', 'dayOfWeekEntropy',
        'hourOfDayEntropy', 'transactionTimeSpread',
        
        # Device and channel patterns
        'uniqueDevices', 'uniqueChannels', 'deviceDiversityRatio', 'channelDiversityRatio',
        
        # Money flow patterns
        'outFlowCount', 'inFlowCount', 'totalFlowCount', 'totalOutFlowAmount',
        'totalInFlowAmount', 'totalFlowAmount', 'avgOutFlowAmount', 'avgInFlowAmount',
        'maxOutFlowAmount', 'maxInFlowAmount', 'uniqueOutFlowPartners',
        'uniqueInFlowPartners', 'totalUniquePartners', 'circularFlowPartners',
        'inFlowRatio', 'outToInFlowAmountRatio', 'circularityRatio',
        
        # Velocity features
        'avgTimeBetweenTransactions', 'minTimeBetweenTransactions',
        'transactionsWithin1Min', 'transactionsWithin5Min', 'rapidTransactionRatio', 'totalVelocityTransactions'
    ]
    
    # Filter to only include features that exist in the dataset
    available_features = [f for f in features_list if f in features_df.columns]
    if len(available_features) < len(features_list):
        missing_features = [f for f in features_list if f not in features_df.columns]
        print(f"Warning: Missing features: {missing_features}")
        print(f"Using {len(available_features)} out of {len(features_list)} features.")
    
    if not available_features:
        print("Error: No valid features found in the dataset.")
        print("Available columns:", list(features_df.columns))
        return
    
    # Apply the selected algorithm
    if args.algorithm == 'isolation_forest':
        features_df = apply_isolation_forest(features_df, available_features)
        model_name = "Isolation Forest"
    else:
        features_df = apply_local_outlier_factor(features_df, available_features, 
                                                args.n_neighbors, args.contamination)
        model_name = "Local Outlier Factor"

    # Save the results
    output_path = "./data/anomaly_scores.csv"
    features_df.to_csv(output_path, index=False)
    print(f"{model_name} anomaly detection complete. Results saved to {output_path}")

    # Print summary
    outlier_count = features_df['is_outlier'].sum()
    print(f"\nDetected {outlier_count} potential outliers using {model_name}.")
    print(f"Outlier rate: {outlier_count/len(features_df)*100:.2f}%")
    
    print("\nTop 10 most anomalous accounts:")
    top_anomalous = features_df.sort_values('anomaly_score').head(10)
    display_columns = ['accountId', 'anomaly_score', 'is_outlier']
    
    # Add other interesting observable columns if they exist
    for col in ['totalTransactions', 'totalAmount', 'circularityRatio', 'rapidTransactionRatio', 'deviceDiversityRatio', 'totalVelocityTransactions']:
        if col in features_df.columns:
            display_columns.append(col)
    
    print(top_anomalous[display_columns])
    
    # Show feature importance for Isolation Forest (if applicable)
    if args.algorithm == 'isolation_forest':
        print("\nMost important features for anomaly detection:")
        try:
            feature_importance = pd.DataFrame({
                'feature': available_features,
                'importance': abs(features_df[available_features].corrwith(features_df['anomaly_score']))
            }).sort_values('importance', ascending=False)
            print(feature_importance.head(10))
        except:
            print("Could not calculate feature importance")
    
    print(f"\nModel and scaler saved for {model_name.lower().replace(' ', '_')}")
    print("Note: These models use only observable behavioral patterns (no fraud labels)")
    print("For model evaluation, use the evaluation_dataset.csv with ground truth labels")


if __name__ == "__main__":
    main()
