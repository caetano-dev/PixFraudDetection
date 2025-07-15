import os
import pandas as pd
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
            return pd.DataFrame([r.data() for r in result])

def setup_gds_environment(conn):
    """
    Sets up the Graph Data Science environment and creates the graph projection.
    """
    print("Setting up GDS environment...")
    
    # Drop existing graph projection if it exists
    drop_query = """
    CALL gds.graph.exists('fraud-detection-graph')
    YIELD exists
    CALL apoc.do.when(exists,
        'CALL gds.graph.drop("fraud-detection-graph") YIELD graphName RETURN graphName',
        'RETURN "Graph does not exist" as result',
        {}) YIELD value
    RETURN value
    """
    
    try:
        conn.query(drop_query)
        print("Dropped existing graph projection if it existed.")
    except Exception as e:
        print(f"Note: {e}")
    
    # Create graph projection with Account nodes and MONEY_FLOW relationships
    projection_query = """
    CALL gds.graph.project(
        'fraud-detection-graph',
        'Account',
        {
            MONEY_FLOW: {
                type: 'MONEY_FLOW',
                orientation: 'NATURAL',
                properties: ['totalAmount', 'transactionCount', 'fraudTransactionCount']
            }
        },
        {
            nodeProperties: ['risk_score', 'is_verified']
        }
    )
    YIELD graphName, nodeCount, relationshipCount
    RETURN graphName, nodeCount, relationshipCount
    """
    
    result = conn.query(projection_query)
    print(f"Created graph projection: {result.iloc[0]['graphName']} with {result.iloc[0]['nodeCount']} nodes and {result.iloc[0]['relationshipCount']} relationships")
    return result

def run_louvain_algorithm(conn):
    """
    Runs the Louvain community detection algorithm and writes results back to Account nodes.
    """
    print("Running Louvain community detection algorithm...")
    
    # Run Louvain algorithm
    louvain_query = """
    CALL gds.louvain.write(
        'fraud-detection-graph',
        {
            writeProperty: 'community',
            relationshipWeightProperty: 'totalAmount'
        }
    )
    YIELD communityCount, modularity, modularities, ranLevels, communityDistribution
    RETURN communityCount, modularity, modularities, ranLevels, communityDistribution
    """
    
    result = conn.query(louvain_query)
    print(f"Louvain algorithm completed. Found {result.iloc[0]['communityCount']} communities with modularity {result.iloc[0]['modularity']:.4f}")
    return result

def analyze_communities(conn):
    """
    Analyzes the detected communities for suspicious patterns.
    """
    print("Analyzing communities for suspicious patterns...")
    
    # Get community statistics
    community_stats_query = """
    MATCH (a:Account)
    WHERE a.community IS NOT NULL
    WITH a.community AS communityId, 
         collect(a) AS accounts,
         avg(a.risk_score) AS avgRiskScore,
         count(a) AS communitySize
    WITH communityId, accounts, avgRiskScore, communitySize,
         [a IN accounts WHERE a.risk_score > 0.5 | a] AS highRiskAccounts
    RETURN communityId,
           communitySize,
           avgRiskScore,
           size(highRiskAccounts) AS highRiskCount,
           toFloat(size(highRiskAccounts)) / communitySize AS highRiskRatio
    ORDER BY avgRiskScore DESC, highRiskRatio DESC
    """
    
    community_stats = conn.query(community_stats_query)
    
    # Get fraudulent transaction statistics per community
    fraud_stats_query = """
    MATCH (a:Account)-[r:MONEY_FLOW]->(b:Account)
    WHERE a.community IS NOT NULL AND b.community IS NOT NULL
    WITH a.community AS communityId,
         sum(r.fraudTransactionCount) AS totalFraudTransactions,
         sum(r.transactionCount) AS totalTransactions,
         sum(r.totalAmount) AS totalAmount
    WHERE totalTransactions > 0
    RETURN communityId,
           totalFraudTransactions,
           totalTransactions,
           totalAmount,
           toFloat(totalFraudTransactions) / totalTransactions AS fraudRate
    ORDER BY fraudRate DESC, totalFraudTransactions DESC
    """
    
    fraud_stats = conn.query(fraud_stats_query)
    
    # Merge the statistics
    if not community_stats.empty and not fraud_stats.empty:
        community_analysis = pd.merge(community_stats, fraud_stats, on='communityId', how='outer')
        community_analysis.fillna(0, inplace=True)
    elif not community_stats.empty:
        community_analysis = community_stats
        community_analysis['totalFraudTransactions'] = 0
        community_analysis['totalTransactions'] = 0
        community_analysis['totalAmount'] = 0
        community_analysis['fraudRate'] = 0
    else:
        print("No communities found with valid statistics.")
        return pd.DataFrame()
    
    # Save community analysis results
    output_path = "community_analysis.csv"
    community_analysis.to_csv(output_path, index=False)
    print(f"Community analysis saved to {output_path}")
    
    # Print top suspicious communities
    print("\nTop 10 most suspicious communities:")
    suspicious_communities = community_analysis.sort_values(['fraudRate', 'avgRiskScore'], ascending=[False, False]).head(10)
    print(suspicious_communities.to_string(index=False))
    
    return community_analysis

def get_community_members(conn, community_id):
    """
    Gets detailed information about members of a specific community.
    """
    query = """
    MATCH (a:Account)
    WHERE a.community = $community_id
    RETURN a.accountId AS accountId,
           a.risk_score AS riskScore,
           a.is_verified AS isVerified,
           a.state AS state
    ORDER BY a.risk_score DESC
    """
    
    return conn.query(query, community_id)

def main():
    # Database connection
    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USERNAME")
    password = os.getenv("NEO4J_PASSWORD")
    conn = Neo4jConnection(uri, user, password)

    try:
        # Setup GDS environment and create graph projection
        setup_gds_environment(conn)
        
        # Run Louvain algorithm
        louvain_result = run_louvain_algorithm(conn)
        
        # Analyze communities
        community_analysis = analyze_communities(conn)
        
        print(f"\nCommunity detection complete!")
        print(f"Run 'python local_outlier_factor.py' next to apply LOF within communities.")
        
    except Exception as e:
        print(f"Error during community detection: {e}")
    finally:
        # Close the connection
        conn.close()

if __name__ == "__main__":
    main()
