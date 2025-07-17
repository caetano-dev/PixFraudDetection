import os
import pandas as pd
from neo4j import GraphDatabase
from config import config

class Neo4jConnection:
    def __init__(self, uri, user, password):
        self._driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self._driver.close()

    def query(self, query, db=None):
        with self._driver.session(database=db) as session:
            result = session.run(query)
            return pd.DataFrame([r.data() for r in result])

def get_graph_features(conn):
    """
    Calculates comprehensive graph-based features for each account.
    - Transaction degree (in + out)
    - Account degree via MONEY_FLOW relationships
    - In-degree and out-degree separately
    """
    print("Calculating graph features...")
    query = """
    MATCH (a:Account)
    
    // Transaction-based degree (via SENT and RECEIVED_BY relationships)
    OPTIONAL MATCH (a)-[:SENT]->(t:Transaction)
    WITH a, count(t) AS sentTransactions
    
    OPTIONAL MATCH (t2:Transaction)-[:RECEIVED_BY]->(a)
    WITH a, sentTransactions, count(t2) AS receivedTransactions
    
    // Account-to-account degree (via MONEY_FLOW relationships)
    OPTIONAL MATCH (a)-[:MONEY_FLOW]->(other:Account)
    WITH a, sentTransactions, receivedTransactions, count(other) AS outFlowDegree
    
    OPTIONAL MATCH (other2:Account)-[:MONEY_FLOW]->(a)
    WITH a, sentTransactions, receivedTransactions, outFlowDegree, count(other2) AS inFlowDegree
    
    RETURN a.accountId AS accountId,
           sentTransactions AS outDegree,
           receivedTransactions AS inDegree,
           sentTransactions + receivedTransactions AS totalDegree,
           outFlowDegree,
           inFlowDegree,
           outFlowDegree + inFlowDegree AS totalFlowDegree
    """
    return conn.query(query)

def get_transactional_features(conn):
    """
    Calculates comprehensive transaction-based features for each account.
    NOTE: No fraud flags used - only observable behavioral patterns
    """
    print("Calculating transactional features...")
    query = """
    MATCH (a:Account)
    
    // Sent transactions features
    OPTIONAL MATCH (a)-[:SENT]->(t:Transaction)
    WITH a, 
         count(t) AS sentCount,
         COALESCE(sum(t.amount), 0) AS totalSentAmount,
         COALESCE(avg(t.amount), 0) AS avgSentAmount,
         COALESCE(max(t.amount), 0) AS maxSentAmount,
         COALESCE(min(t.amount), 0) AS minSentAmount,
         COALESCE(stdev(t.amount), 0) AS amountVariance,
         collect(t.deviceId) AS sentDevices,
         collect(t.channel) AS sentChannels,
         count(CASE WHEN t.is_weekend = true THEN 1 END) AS weekendSentTransactions,
         count(CASE WHEN t.hour_of_day >= 22 OR t.hour_of_day <= 5 THEN 1 END) AS nightSentTransactions
    
    // Received transactions features
    OPTIONAL MATCH (t2:Transaction)-[:RECEIVED_BY]->(a)
    WITH a, sentCount, totalSentAmount, avgSentAmount, maxSentAmount, minSentAmount, amountVariance,
         sentDevices, sentChannels, weekendSentTransactions, nightSentTransactions,
         count(t2) AS receivedCount,
         COALESCE(sum(t2.amount), 0) AS totalReceivedAmount,
         COALESCE(avg(t2.amount), 0) AS avgReceivedAmount,
         COALESCE(max(t2.amount), 0) AS maxReceivedAmount,
         COALESCE(min(t2.amount), 0) AS minReceivedAmount,
         COALESCE(stdev(t2.amount), 0) AS receivedAmountVariance,
         collect(t2.deviceId) AS receivedDevices,
         collect(t2.channel) AS receivedChannels,
         count(CASE WHEN t2.is_weekend = true THEN 1 END) AS weekendReceivedTransactions,
         count(CASE WHEN t2.hour_of_day >= 22 OR t2.hour_of_day <= 5 THEN 1 END) AS nightReceivedTransactions
    
    // Count unique counterparties
    OPTIONAL MATCH (a)-[:SENT]->(t3:Transaction)-[:RECEIVED_BY]->(counterparty:Account)
    WITH a, sentCount, totalSentAmount, avgSentAmount, maxSentAmount, minSentAmount, amountVariance,
         sentDevices, sentChannels, weekendSentTransactions, nightSentTransactions,
         receivedCount, totalReceivedAmount, avgReceivedAmount, maxReceivedAmount, minReceivedAmount, receivedAmountVariance,
         receivedDevices, receivedChannels, weekendReceivedTransactions, nightReceivedTransactions,
         count(DISTINCT counterparty) AS uniqueCounterparties
    
    RETURN a.accountId AS accountId,
           sentCount,
           receivedCount,
           sentCount + receivedCount AS totalTransactions,
           uniqueCounterparties,
           totalSentAmount,
           totalReceivedAmount,
           totalSentAmount + totalReceivedAmount AS totalAmount,
           avgSentAmount,
           avgReceivedAmount,
           (totalSentAmount + totalReceivedAmount) / CASE WHEN (sentCount + receivedCount) > 0 THEN (sentCount + receivedCount) ELSE 1 END AS avgTransactionAmount,
           maxSentAmount,
           maxReceivedAmount,
           CASE WHEN maxSentAmount > maxReceivedAmount THEN maxSentAmount ELSE maxReceivedAmount END AS maxTransactionAmount,
           minSentAmount,
           minReceivedAmount,
           CASE WHEN minSentAmount < minReceivedAmount AND minSentAmount > 0 THEN minSentAmount ELSE minReceivedAmount END AS minTransactionAmount,
           CASE WHEN receivedCount > 0 THEN toFloat(sentCount) / receivedCount ELSE 0 END AS sentToReceivedRatio,
           amountVariance,
           
           // Device and channel diversity (simplified - no APOC needed)
           size([x IN sentDevices + receivedDevices WHERE x IS NOT NULL]) AS uniqueDevices,
           size([x IN sentChannels + receivedChannels WHERE x IS NOT NULL]) AS uniqueChannels,
           CASE WHEN (sentCount + receivedCount) > 0 
                THEN toFloat(size([x IN sentDevices + receivedDevices WHERE x IS NOT NULL])) / (sentCount + receivedCount) 
                ELSE 0 END AS deviceDiversityRatio,
           CASE WHEN (sentCount + receivedCount) > 0 
                THEN toFloat(size([x IN sentChannels + receivedChannels WHERE x IS NOT NULL])) / (sentCount + receivedCount) 
                ELSE 0 END AS channelDiversityRatio,
           
           // Temporal patterns (fraud indicators)
           weekendSentTransactions + weekendReceivedTransactions AS weekendTransactions,
           nightSentTransactions + nightReceivedTransactions AS nightTransactions,
           CASE WHEN (sentCount + receivedCount) > 0 
                THEN toFloat(weekendSentTransactions + weekendReceivedTransactions) / (sentCount + receivedCount) 
                ELSE 0 END AS weekendTransactionRatio,
           CASE WHEN (sentCount + receivedCount) > 0 
                THEN toFloat(nightSentTransactions + nightReceivedTransactions) / (sentCount + receivedCount) 
                ELSE 0 END AS nightTransactionRatio
    """
    return conn.query(query)

def get_account_risk_features(conn):
    """
    Retrieves account-level features including risk scores and verification status.
    """
    print("Retrieving account features...")
    query = """
    MATCH (a:Account)
    RETURN a.accountId AS accountId, 
           COALESCE(a.risk_score, 0) AS riskScore,
           CASE WHEN a.is_verified = true THEN 1 ELSE 0 END AS isVerified
    """
    return conn.query(query)

def get_money_flow_features(conn):
    """
    Calculates money flow features between accounts (observable patterns).
    No fraud flags used - focuses on network behavior patterns.
    """
    print("Calculating money flow features...")
    query = """
    MATCH (a:Account)
    
    // Outgoing money flows
    OPTIONAL MATCH (a)-[flow_out:MONEY_FLOW]->(other:Account)
    WITH a, 
         count(flow_out) AS outFlowCount,
         COALESCE(sum(flow_out.totalAmount), 0) AS totalOutFlowAmount,
         COALESCE(avg(flow_out.totalAmount), 0) AS avgOutFlowAmount,
         COALESCE(max(flow_out.totalAmount), 0) AS maxOutFlowAmount,
         count(DISTINCT other) AS uniqueOutFlowPartners
    
    // Incoming money flows  
    OPTIONAL MATCH (other2:Account)-[flow_in:MONEY_FLOW]->(a)
    WITH a, outFlowCount, totalOutFlowAmount, avgOutFlowAmount, maxOutFlowAmount, uniqueOutFlowPartners,
         count(flow_in) AS inFlowCount,
         COALESCE(sum(flow_in.totalAmount), 0) AS totalInFlowAmount,
         COALESCE(avg(flow_in.totalAmount), 0) AS avgInFlowAmount,
         COALESCE(max(flow_in.totalAmount), 0) AS maxInFlowAmount,
         count(DISTINCT other2) AS uniqueInFlowPartners
    
    // Circular flow detection (potential money laundering indicator)
    OPTIONAL MATCH (a)-[flow1:MONEY_FLOW]->(intermediate:Account)-[flow2:MONEY_FLOW]->(a)
    WITH a, outFlowCount, totalOutFlowAmount, avgOutFlowAmount, maxOutFlowAmount, uniqueOutFlowPartners,
         inFlowCount, totalInFlowAmount, avgInFlowAmount, maxInFlowAmount, uniqueInFlowPartners,
         count(DISTINCT intermediate) AS circularFlowPartners
    
    RETURN a.accountId AS accountId,
           outFlowCount,
           inFlowCount,
           outFlowCount + inFlowCount AS totalFlowCount,
           totalOutFlowAmount,
           totalInFlowAmount,
           totalOutFlowAmount + totalInFlowAmount AS totalFlowAmount,
           avgOutFlowAmount,
           avgInFlowAmount,
           maxOutFlowAmount,
           maxInFlowAmount,
           uniqueOutFlowPartners,
           uniqueInFlowPartners,
           uniqueOutFlowPartners + uniqueInFlowPartners AS totalUniquePartners,
           circularFlowPartners,
           
           // Flow behavior ratios (fraud indicators)
           CASE WHEN (outFlowCount + inFlowCount) > 0 
                THEN toFloat(inFlowCount) / (outFlowCount + inFlowCount) 
                ELSE 0 END AS inFlowRatio,
           CASE WHEN totalInFlowAmount > 0 
                THEN totalOutFlowAmount / totalInFlowAmount 
                ELSE 0 END AS outToInFlowAmountRatio,
           CASE WHEN (uniqueOutFlowPartners + uniqueInFlowPartners) > 0 
                THEN toFloat(circularFlowPartners) / (uniqueOutFlowPartners + uniqueInFlowPartners) 
                ELSE 0 END AS circularityRatio
    """
    return conn.query(query)

def get_velocity_features(conn):
    """
    Calculates transaction velocity features (time-based patterns).
    High velocity can indicate automated fraud or money laundering.
    """
    print("Calculating velocity features...")
    query = """
    MATCH (a:Account)

// Collect all transaction details first
OPTIONAL MATCH (a)-[:SENT]->(t:Transaction)
WITH a, collect(t.timestamp) AS sentTimestamps, collect(t.hour_of_day) AS sentHours, collect(t.day_of_week) AS sentDays

OPTIONAL MATCH (t2:Transaction)-[:RECEIVED_BY]->(a)
WITH a, sentTimestamps, sentHours, sentDays,
     collect(t2.timestamp) AS receivedTimestamps, collect(t2.hour_of_day) AS receivedHours, collect(t2.day_of_week) AS receivedDays

// Combine and create base collections
WITH a,
     sentTimestamps + receivedTimestamps AS allTimestamps,
     sentHours + receivedHours AS allHours,
     sentDays + receivedDays AS allDays
WHERE size(allTimestamps) > 1

// Pre-calculate time differences and counts for entropy
WITH a, allTimestamps, allHours, allDays,
     [i IN range(0, size(allTimestamps)-2) | duration.between(allTimestamps[i], allTimestamps[i+1]).seconds] AS timeDifferences,
     [h IN allHours | {hour: h, count: size([x IN allHours WHERE x = h])}] AS hourCounts,
     [d IN allDays | {day: d, count: size([x IN allDays WHERE x = d])}] AS dayCounts

WITH a, allTimestamps, timeDifferences, hourCounts, dayCounts,
     size(allTimestamps) AS totalVelocityTransactions,
     CASE WHEN size(timeDifferences) > 0 THEN reduce(s = 0, diff IN timeDifferences | s + diff) / toFloat(size(timeDifferences)) ELSE 0 END AS avgTimeBetweenTransactions,
     CASE WHEN size(timeDifferences) > 0 THEN reduce(m = timeDifferences[0], diff IN timeDifferences | CASE WHEN diff < m THEN diff ELSE m END) ELSE 0 END AS minTimeBetweenTransactions,
     // Compute earliest and latest timestamps for spread
     reduce(minTS = allTimestamps[0], ts IN allTimestamps | CASE WHEN ts < minTS THEN ts ELSE minTS END) AS earliestTimestamp,
     reduce(maxTS = allTimestamps[0], ts IN allTimestamps | CASE WHEN ts > maxTS THEN ts ELSE maxTS END) AS latestTimestamp,
     size([diff IN timeDifferences WHERE diff < 60]) AS transactionsWithin1Min,
     size([diff IN timeDifferences WHERE diff < 300]) AS transactionsWithin5Min,
     reduce(entropy = 0.0, item IN hourCounts | entropy - (toFloat(item.count) / size(allHours)) * (log(toFloat(item.count) / size(allHours)) / log(2))) AS hourOfDayEntropy,
     reduce(entropy = 0.0, item IN dayCounts | entropy - (toFloat(item.count) / size(allDays)) * (log(toFloat(item.count) / size(allDays)) / log(2))) AS dayOfWeekEntropy

RETURN a.accountId AS accountId,
       totalVelocityTransactions,
       avgTimeBetweenTransactions,
       minTimeBetweenTransactions,
       // Duration between earliest and latest transactions
       duration.between(earliestTimestamp, latestTimestamp).seconds AS transactionTimeSpread,
       transactionsWithin1Min,
       transactionsWithin5Min,
       CASE WHEN size(timeDifferences) > 0 THEN toFloat(transactionsWithin1Min) / size(timeDifferences) ELSE 0 END AS rapidTransactionRatio,
       hourOfDayEntropy,
       dayOfWeekEntropy
"""
    return conn.query(query)

def create_evaluation_dataset(conn, features_df):
    """
    Creates a separate dataset with fraud labels for evaluation purposes only.
    In a real system, this would be used to measure model performance against known fraud cases.
    """
    print("Creating evaluation dataset with ground truth labels...")
    
    query = """
    MATCH (a:Account)
    
    // Count actual fraud transactions (for evaluation only)
    OPTIONAL MATCH (a)-[:SENT]->(t:Transaction)
    WITH a, count(CASE WHEN t.fraudFlag STARTS WITH 'SMURFING' OR t.fraudFlag = 'CIRCULAR_PAYMENT' THEN 1 END) AS sentFraudCount
    
    OPTIONAL MATCH (t2:Transaction)-[:RECEIVED_BY]->(a)
    WITH a, sentFraudCount, count(CASE WHEN t2.fraudFlag STARTS WITH 'SMURFING' OR t2.fraudFlag = 'CIRCULAR_PAYMENT' THEN 1 END) AS receivedFraudCount
    
    RETURN a.accountId AS accountId,
           sentFraudCount,
           receivedFraudCount,
           sentFraudCount + receivedFraudCount AS totalFraudCount,
           CASE WHEN sentFraudCount + receivedFraudCount > 0 THEN 1 ELSE 0 END AS isFraudulent
    """
    
    fraud_labels = conn.query(query)
    fraud_df = pd.DataFrame(fraud_labels)
    
    # Merge with features for evaluation
    evaluation_df = features_df.merge(fraud_df, on='accountId', how='left')
    evaluation_df = evaluation_df.fillna(0)

    evaluation_df.to_csv("./data/evaluation_dataset.csv", index=False)
    print(f"Evaluation dataset saved to evaluation_dataset.csv")
    
    # Print fraud statistics for reference
    fraud_accounts = evaluation_df[evaluation_df['isFraudulent'] == 1]
    print(f"Ground truth: {len(fraud_accounts)} fraudulent accounts out of {len(evaluation_df)} total accounts")
    
    return evaluation_df


def main():
    # Database connection from config
    neo4j_config = config['neo4j']
    conn = Neo4jConnection(
        uri=neo4j_config['uri'], 
        user=neo4j_config['user'], 
        password=neo4j_config['password']
    )
    
    print("Starting production-ready feature engineering...")
    print("(No fraud flags used - features based on observable behavioral patterns only)")
    
    try:
        # Extract behavioral features (production-ready)
        transactional_features = get_transactional_features(conn)
        money_flow_features = get_money_flow_features(conn)
        velocity_features = get_velocity_features(conn)
        
        # Convert to DataFrames
        transactional_df = pd.DataFrame(transactional_features)
        money_flow_df = pd.DataFrame(money_flow_features)
        velocity_df = pd.DataFrame(velocity_features)
        
        print(f"Extracted features for {len(transactional_df)} accounts")
        
        # Merge all features on accountId
        features_df = transactional_df.merge(money_flow_df, on='accountId', how='outer')
        features_df = features_df.merge(velocity_df, on='accountId', how='outer')
        
        # Fill NaN values with 0 (accounts with no activity)
        features_df = features_df.fillna(0)
        
        # Save production features (no fraud labels)
        features_df.to_csv("./data/account_features.csv", index=False)
        print(f"Production features saved to account_features.csv")
        
        # Create separate evaluation dataset with ground truth
        create_evaluation_dataset(conn, features_df)
        
        print("\nFeature Summary:")
        print(f"- Total accounts: {len(features_df)}")
        print(f"- Total features: {len(features_df.columns) - 1}")  # Excluding accountId
        print("- Features focus on observable behavioral patterns")
        print("- No fraud flags used (production-ready)")
        print("- Evaluation dataset created separately for model validation")
        
        # Display basic feature statistics
        print("\nKey Feature Ranges:")
        numeric_cols = features_df.select_dtypes(include=['number']).columns
        for col in numeric_cols[:10]:  # Show first 10 numeric features
            if col != 'accountId':
                print(f"  {col}: {features_df[col].min():.2f} - {features_df[col].max():.2f}")
        
    except Exception as e:
        print(f"Error during feature extraction: {e}")
        import traceback
        traceback.print_exc()
    finally:
        conn.close()

if __name__ == "__main__":
    main()
