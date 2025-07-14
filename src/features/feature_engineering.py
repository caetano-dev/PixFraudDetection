import os
import pandas as pd
from neo4j import GraphDatabase
from src.config.config import NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD

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
    Calculates graph-based features for each account.
    - Total degree (in + out)
    - In-degree
    - Out-degree
    """
    print("Calculating graph features (degree)...")
    query = """
    MATCH (a:Account)
    RETURN a.accountId AS accountId,
           COUNT { (a)<--(:Transaction) } AS inDegree,
           COUNT { (a)-->(:Transaction) } AS outDegree,
           COUNT { (a)--(:Transaction) } AS totalDegree
    """
    return conn.query(query)

def get_transactional_features(conn):
    """
    Calculates transaction-based features for each account.
    - Total transaction amount
    - Average transaction amount
    - Maximum transaction amount
    - Total number of transactions
    - Number of fraudulent transactions linked to the account
    """
    print("Calculating transactional features...")
    query = """
    MATCH (a:Account)-[:SENT]->(t:Transaction)
    RETURN a.accountId AS accountId,
           sum(t.amount) AS totalAmount,
           avg(t.amount) AS avgAmount,
           max(t.amount) AS maxAmount,
           count(t) AS transactionCount,
           sum(CASE WHEN t.fraudFlag STARTS WITH 'SMURFING' OR t.fraudFlag = 'CIRCULAR_PAYMENT' THEN 1 ELSE 0 END) as fraudTransactionCount
    """
    return conn.query(query)

def get_account_risk_features(conn):
    """
    Retrieves the pre-calculated risk score from the account node.
    """
    print("Retrieving account risk scores...")
    query = """
    MATCH (a:Account)
    RETURN a.accountId AS accountId, a.risk_score as riskScore
    """
    return conn.query(query)


def main():
    # Database connection
    conn = Neo4jConnection(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD)

    # Feature extraction
    graph_features = get_graph_features(conn)
    transactional_features = get_transactional_features(conn)
    risk_features = get_account_risk_features(conn)

    # Merge features into a single DataFrame
    print("Merging features...")
    features_df = pd.merge(graph_features, transactional_features, on="accountId", how="outer")
    features_df = pd.merge(features_df, risk_features, on="accountId", how="outer")

    # Fill NaN values for accounts that might not have transactions
    features_df.fillna(0, inplace=True)

    # Save to CSV
    output_path = "src/data/account_features.csv"
    features_df.to_csv(output_path, index=False)
    print(f"Feature engineering complete. Data saved to {output_path}")

    # Close the connection
    conn.close()

if __name__ == "__main__":
    main()
