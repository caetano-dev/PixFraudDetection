import os
import pandas as pd
import numpy as np
from neo4j import GraphDatabase
from dotenv import load_dotenv
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
import joblib

# Load environment variables from .env file
load_dotenv()

class Neo4jConnection:
    def __init__(self, uri, user, password):
        self._driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self._driver.close()

    def query(self, query, db=None, **params):
        with self._driver.session(database=db) as session:
            result = session.run(query, **params)
            return pd.DataFrame([r.data() for r in result])

def get_community_features(conn):
    """
    Extracts features for each account within their respective communities.
    """
    print("Extracting community-based features...")
    
    query = """
    MATCH (a:Account)
    WHERE a.communityId IS NOT NULL
    
    // Get basic account features
    OPTIONAL MATCH (a)-[:SENT]->(t:Transaction)
    WITH a, 
         count(t) AS sentTransactions,
         COALESCE(sum(t.amount), 0) AS totalSentAmount,
         COALESCE(avg(t.amount), 0) AS avgSentAmount,
         COALESCE(max(t.amount), 0) AS maxSentAmount
    
    // Get received transactions
    OPTIONAL MATCH (t2:Transaction)-[:RECEIVED_BY]->(a)
    WITH a, sentTransactions, totalSentAmount, avgSentAmount, maxSentAmount,
         count(t2) AS receivedTransactions,
         COALESCE(sum(t2.amount), 0) AS totalReceivedAmount,
         COALESCE(avg(t2.amount), 0) AS avgReceivedAmount
    
    // Get fraud-related transactions
    OPTIONAL MATCH (a)-[:SENT]->(ft:Transaction)
    WHERE ft.fraudFlag STARTS WITH 'SMURFING' OR ft.fraudFlag = 'CIRCULAR_PAYMENT'
    WITH a, sentTransactions, totalSentAmount, avgSentAmount, maxSentAmount,
         receivedTransactions, totalReceivedAmount, avgReceivedAmount,
         count(ft) AS fraudulentTransactions,
         COALESCE(sum(ft.amount), 0) AS totalFraudAmount
    
    // Get community connections (internal vs external)
    OPTIONAL MATCH (a)-[r:MONEY_FLOW]->(other:Account)
    WHERE other.communityId = a.communityId
    WITH a, sentTransactions, totalSentAmount, avgSentAmount, maxSentAmount,
         receivedTransactions, totalReceivedAmount, avgReceivedAmount,
         fraudulentTransactions, totalFraudAmount,
         count(r) AS internalConnections,
         COALESCE(sum(r.totalAmount), 0) AS internalTransactionAmount
    
    OPTIONAL MATCH (a)-[r2:MONEY_FLOW]->(external:Account)
    WHERE external.communityId <> a.communityId
    WITH a, sentTransactions, totalSentAmount, avgSentAmount, maxSentAmount,
         receivedTransactions, totalReceivedAmount, avgReceivedAmount,
         fraudulentTransactions, totalFraudAmount,
         internalConnections, internalTransactionAmount,
         count(r2) AS externalConnections,
         COALESCE(sum(r2.totalAmount), 0) AS externalTransactionAmount
    
    RETURN a.accountId AS accountId,
           a.communityId AS community,
           a.riskScore AS riskScore,
           CASE WHEN a.isVerified THEN 1 ELSE 0 END AS isVerified,
           sentTransactions,
           receivedTransactions,
           sentTransactions + receivedTransactions AS totalTransactions,
           totalSentAmount,
           totalReceivedAmount,
           totalSentAmount + totalReceivedAmount AS totalTransactionAmount,
           avgSentAmount,
           avgReceivedAmount,
           maxSentAmount,
           fraudulentTransactions,
           totalFraudAmount,
           internalConnections,
           externalConnections,
           internalConnections + externalConnections AS totalConnections,
           internalTransactionAmount,
           externalTransactionAmount,
           CASE WHEN (internalConnections + externalConnections) > 0 
                THEN toFloat(internalConnections) / (internalConnections + externalConnections) 
                ELSE 0 END AS internalConnectionRatio,
           CASE WHEN (internalTransactionAmount + externalTransactionAmount) > 0 
                THEN toFloat(internalTransactionAmount) / (internalTransactionAmount + externalTransactionAmount) 
                ELSE 0 END AS internalAmountRatio
    ORDER BY a.communityId, a.riskScore DESC
    """
    
    return conn.query(query)

def apply_lof_within_communities(features_df, n_neighbors=5, contamination=0.1):
    """
    Applies Local Outlier Factor within each community separately.
    """
    print("Applying Local Outlier Factor within communities...")
    
    # Features to use for LOF
    feature_columns = [
        'riskScore', 'isVerified', 'sentTransactions', 'receivedTransactions',
        'totalTransactionAmount', 'avgSentAmount', 'avgReceivedAmount', 'maxSentAmount',
        'fraudulentTransactions', 'totalFraudAmount', 'totalConnections',
        'internalConnectionRatio', 'internalAmountRatio'
    ]
    
    results = []
    communities = features_df['community'].unique()
    
    for community_id in communities:
        community_data = features_df[features_df['community'] == community_id].copy()
        
        # Skip communities with too few members for LOF
        if len(community_data) < max(3, n_neighbors):
            print(f"Skipping community {community_id}: too few members ({len(community_data)})")
            community_data['lof_score'] = 0
            community_data['is_lof_outlier'] = 0
            results.append(community_data)
            continue
        
        # Extract features for this community
        X = community_data[feature_columns].fillna(0)
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Adjust n_neighbors if community is small
        actual_n_neighbors = min(n_neighbors, len(community_data) - 1)
        
        # Apply LOF
        lof = LocalOutlierFactor(n_neighbors=actual_n_neighbors, contamination=contamination)
        outlier_predictions = lof.fit_predict(X_scaled)
        outlier_scores = lof.negative_outlier_factor_
        
        # Add results to community data
        community_data['lof_score'] = outlier_scores
        community_data['is_lof_outlier'] = (outlier_predictions == -1).astype(int)
        
        results.append(community_data)
        
        outlier_count = sum(outlier_predictions == -1)
        print(f"Community {community_id}: {len(community_data)} members, {outlier_count} outliers detected")
    
    return pd.concat(results, ignore_index=True)

def analyze_lof_results(lof_results):
    """
    Analyzes and summarizes the LOF results.
    """
    print("Analyzing LOF results...")
    
    # Overall statistics
    total_accounts = len(lof_results)
    total_outliers = lof_results['is_lof_outlier'].sum()
    communities_with_outliers = lof_results[lof_results['is_lof_outlier'] == 1]['community'].nunique()
    
    print(f"\nLOF Analysis Summary:")
    print(f"Total accounts analyzed: {total_accounts}")
    print(f"Total outliers detected: {total_outliers}")
    print(f"Outlier rate: {(total_outliers/total_accounts)*100:.2f}%")
    print(f"Communities with outliers: {communities_with_outliers}")
    
    # Top outliers by LOF score (most negative scores are most anomalous)
    top_outliers = lof_results.sort_values('lof_score').head(20)
    print(f"\nTop 20 most anomalous accounts (by LOF score):")
    print(top_outliers[['accountId', 'community', 'lof_score', 'riskScore', 'fraudulentTransactions', 'totalTransactionAmount']].to_string(index=False))
    
    # Community-level analysis
    community_summary = lof_results.groupby('community').agg({
        'accountId': 'count',
        'is_lof_outlier': 'sum',
        'lof_score': 'min',  # Most negative score in community
        'riskScore': 'mean',
        'fraudulentTransactions': 'sum',
        'totalTransactionAmount': 'sum'
    }).reset_index()
    
    community_summary.columns = ['community', 'total_members', 'outlier_count', 'min_lof_score', 
                                 'avg_risk_score', 'total_fraud_transactions', 'total_transaction_amount']
    community_summary['outlier_rate'] = community_summary['outlier_count'] / community_summary['total_members']
    
    # Sort by most suspicious communities
    community_summary = community_summary.sort_values(['outlier_rate', 'avg_risk_score'], ascending=[False, False])
    
    print(f"\nCommunity-level outlier analysis:")
    print(community_summary.to_string(index=False))
    
    return top_outliers, community_summary

def update_neo4j_with_lof_results(conn, lof_results):
    """
    Updates Neo4j Account nodes with LOF scores and outlier flags.
    """
    print("Updating Neo4j with LOF results...")
    
    update_query = """
    UNWIND $batch AS row
    MATCH (a:Account {accountId: row.accountId})
    SET a.lof_score = row.lof_score,
        a.is_lof_outlier = row.is_lof_outlier
    """
    
    # Prepare batch data
    batch_data = lof_results[['accountId', 'lof_score', 'is_lof_outlier']].to_dict('records')
    
    # Update in batches of 1000
    batch_size = 1000
    for i in range(0, len(batch_data), batch_size):
        batch = batch_data[i:i+batch_size]
        conn.query(update_query, batch=batch)
    
    print(f"Updated {len(batch_data)} accounts with LOF results in Neo4j.")

def main():
    # Database connection
    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USERNAME")
    password = os.getenv("NEO4J_PASSWORD")
    conn = Neo4jConnection(uri, user, password)

    try:
        # Check if community detection has been run
        community_check = conn.query("MATCH (a:Account) WHERE a.communityId IS NOT NULL RETURN count(a) as count")
        if community_check.iloc[0]['count'] == 0:
            print("Error: No communities found. Please run community_detection.py first.")
            return
        
        # Extract community features
        features_df = get_community_features(conn)
        
        if features_df.empty:
            print("Error: No features extracted. Check your data.")
            return
        
        # Apply LOF within communities
        lof_results = apply_lof_within_communities(features_df)
        
        # Analyze results
        top_outliers, community_summary = analyze_lof_results(lof_results)
        
        # Save results
        lof_results.to_csv("lof_analysis.csv", index=False)
        community_summary.to_csv("lof_community_summary.csv", index=False)
        print(f"\nLOF analysis saved to lof_analysis.csv and lof_community_summary.csv")
        
        # Update Neo4j with results
        update_neo4j_with_lof_results(conn, lof_results)
        
        # Save models for future use
        print("Local Outlier Factor analysis complete!")
        
    except Exception as e:
        print(f"Error during LOF analysis: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Close the connection
        conn.close()

if __name__ == "__main__":
    main()
