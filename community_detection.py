import os
import pandas as pd
import networkx as nx
import community as community_louvain
from neo4j import GraphDatabase
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Neo4jConnection:
    def __init__(self, uri, user, password):
        self._driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self._driver.close()

    def query(self, query, db=None):
        with self._driver.session(database=db) as session:
            result = session.run(query)
            return [r.data() for r in result]

def fetch_graph_data(conn):
    """
    Fetch all MONEY_FLOW relationships from Neo4j to build the graph.
    """
    print("Fetching graph data from Neo4j...")
    
    query = """
    MATCH (a1:Account)-[mf:MONEY_FLOW]->(a2:Account)
    RETURN a1.accountId AS source, 
           a2.accountId AS target, 
           mf.totalAmount AS weight,
           mf.transactionCount AS transaction_count
    """
    
    relationships = conn.query(query)
    print(f"Fetched {len(relationships)} MONEY_FLOW relationships")
    return relationships

def build_networkx_graph(relationships):
    """
    Build a NetworkX graph from the MONEY_FLOW relationships.
    """
    print("Building NetworkX graph...")
    
    # Create a weighted undirected graph
    G = nx.Graph()
    
    for rel in relationships:
        source = rel['source']
        target = rel['target']
        weight = float(rel['weight']) if rel['weight'] else 1.0
        transaction_count = int(rel['transaction_count']) if rel['transaction_count'] else 1
        
        # Add edge with weight (total amount) and transaction count
        if G.has_edge(source, target):
            # If edge already exists, sum the weights and transaction counts
            G[source][target]['weight'] += weight
            G[source][target]['transaction_count'] += transaction_count
        else:
            G.add_edge(source, target, weight=weight, transaction_count=transaction_count)
    
    print(f"Graph built with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    return G

def detect_communities_louvain(G):
    """
    Apply the Louvain algorithm for community detection.
    """
    print("Running Louvain community detection...")
    
    # Apply Louvain algorithm with weights
    partition = community_louvain.best_partition(G, weight='weight', random_state=42)
    
    # Calculate modularity
    modularity = community_louvain.modularity(partition, G, weight='weight')
    
    print(f"Community detection completed:")
    print(f"- Number of communities: {len(set(partition.values()))}")
    print(f"- Modularity score: {modularity:.4f}")
    
    return partition, modularity

def write_communities_to_neo4j(conn, partition):
    """
    Write the detected communities back to Neo4j.
    """
    print("Writing community assignments to Neo4j...")
    
    # Update each account with its community ID
    update_query = """
    UNWIND $community_data AS cd
    MATCH (a:Account {accountId: cd.accountId})
    SET a.communityId = cd.communityId
    """
    
    # Prepare community data
    community_data = [
        {"accountId": account_id, "communityId": int(community_id)}
        for account_id, community_id in partition.items()
    ]
    
    # Execute the update in batches
    batch_size = 1000
    for i in range(0, len(community_data), batch_size):
        batch = community_data[i:i + batch_size]
        with conn._driver.session() as session:
            session.run(update_query, community_data=batch)
    
    print(f"Updated {len(community_data)} accounts with community assignments")

def analyze_communities(conn):
    """
    Analyze the detected communities and generate statistics.
    """
    print("Analyzing communities...")
    
    query = """
    MATCH (a:Account)
    WHERE a.communityId IS NOT NULL
    
    // Get community statistics
    WITH a.communityId AS communityId, 
         collect(a) AS accounts,
         count(a) AS communitySize
    
    // Calculate community metrics
    UNWIND accounts AS account
    OPTIONAL MATCH (account)-[:SENT]->(t:Transaction)
    WITH communityId, communitySize, account,
         count(t) AS sentTransactions,
         COALESCE(sum(t.amount), 0) AS totalSentAmount,
         count(CASE WHEN t.fraudFlag STARTS WITH 'SMURFING' OR t.fraudFlag = 'CIRCULAR_PAYMENT' THEN 1 END) AS fraudTransactions
    
    OPTIONAL MATCH (t2:Transaction)-[:RECEIVED_BY]->(account)
    WITH communityId, communitySize, account, sentTransactions, totalSentAmount, fraudTransactions,
         count(t2) AS receivedTransactions,
         COALESCE(sum(t2.amount), 0) AS totalReceivedAmount,
         count(CASE WHEN t2.fraudFlag STARTS WITH 'SMURFING' OR t2.fraudFlag = 'CIRCULAR_PAYMENT' THEN 1 END) AS receivedFraudTransactions
    
    WITH communityId, communitySize,
         collect({
             accountId: account.accountId,
             riskScore: account.risk_score,
             totalTransactions: sentTransactions + receivedTransactions,
             totalAmount: totalSentAmount + totalReceivedAmount,
             fraudTransactions: fraudTransactions + receivedFraudTransactions
         }) AS accountStats
    
    // Aggregate community statistics
    WITH communityId, communitySize, accountStats,
         reduce(totalAmount = 0, acc IN accountStats | totalAmount + acc.totalAmount) AS communityTotalAmount,
         reduce(totalFraud = 0, acc IN accountStats | totalFraud + acc.fraudTransactions) AS communityFraudTransactions,
         reduce(totalTrans = 0, acc IN accountStats | totalTrans + acc.totalTransactions) AS communityTotalTransactions,
         reduce(totalRisk = 0.0, acc IN accountStats | totalRisk + acc.riskScore) AS totalRiskScore
    
    RETURN communityId,
           communitySize,
           communityTotalAmount AS totalAmount,
           communityFraudTransactions AS totalFraudTransactions,
           communityTotalTransactions AS totalTransactions,
           totalRiskScore / communitySize AS avgRiskScore,
           CASE WHEN communityTotalTransactions > 0 
                THEN toFloat(communityFraudTransactions) / communityTotalTransactions 
                ELSE 0 END AS fraudRate
    ORDER BY fraudRate DESC, totalFraudTransactions DESC
    """
    
    results = conn.query(query)
    
    if results:
        df = pd.DataFrame(results)
        df.to_csv("./data/community_analysis.csv", index=False)
        print(f"Community analysis saved to community_analysis.csv")
        
        # Print summary
        print(f"\nCommunity Analysis Summary:")
        print(f"- Total communities detected: {len(df)}")
        print(f"- Average community size: {df['communitySize'].mean():.1f}")
        print(f"- Communities with fraud: {len(df[df['totalFraudTransactions'] > 0])}")
        print(f"- Highest fraud rate: {df['fraudRate'].max():.2%}")
        
        # Show top suspicious communities
        print(f"\nTop 5 Most Suspicious Communities:")
        top_communities = df.head(5)
        for _, community in top_communities.iterrows():
            print(f"  Community {community['communityId']}: "
                  f"{community['communitySize']} accounts, "
                  f"{community['fraudRate']:.2%} fraud rate, "
                  f"R$ {community['totalAmount']:,.2f} total volume")
        
        return df
    else:
        print("No community data found")
        return None

def main():
    # Database connection
    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USERNAME")
    password = os.getenv("NEO4J_PASSWORD")
    
    print(f"Connecting to Neo4j at {uri}...")
    conn = Neo4jConnection(uri, user, password)

    try:
        # Step 1: Fetch graph data from Neo4j
        relationships = fetch_graph_data(conn)
        
        if not relationships:
            print("No MONEY_FLOW relationships found. Please ensure data is loaded in Neo4j.")
            return
        
        # Step 2: Build NetworkX graph
        G = build_networkx_graph(relationships)
        
        if G.number_of_nodes() == 0:
            print("No nodes in the graph. Cannot perform community detection.")
            return
        
        # Step 3: Detect communities using Louvain algorithm
        partition, modularity = detect_communities_louvain(G)
        
        # Step 4: Write communities back to Neo4j
        write_communities_to_neo4j(conn, partition)
        
        # Step 5: Analyze communities and generate reports
        community_df = analyze_communities(conn)
        
        print(f"\nCommunity detection completed successfully!")
        print(f"- Modularity score: {modularity:.4f}")
        print(f"- Results saved to community_analysis.csv")
        
    except Exception as e:
        print(f"Error during community detection: {e}")
        import traceback
        traceback.print_exc()
    finally:
        conn.close()

if __name__ == "__main__":
    main()
