import redis
from neo4j import GraphDatabase
import json
import os
from dotenv import load_dotenv

load_dotenv()

REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))
REDIS_DB = int(os.getenv('REDIS_DB', 0))

NEO4J_URI = os.getenv('NEO4J_URI', 'neo4j://localhost:7687')
NEO4J_USERNAME = os.getenv('NEO4J_USERNAME', 'neo4j')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD', 'password')

r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

def ingest_transaction(driver, tx_data):
    """
    Ingests a single transaction into Neo4j, enriching nodes with all available data.
    """
    with driver.session() as session:
        # Ingests based on the 'decision' from the fraud detector
        # We only ingest transactions that were approved
        if tx_data.get('decision') == 'APPROVED':
            session.write_transaction(create_transaction_cypher, tx_data)
            print(f"  -> Ingested approved transaction {tx_data['transaction_id']} into Neo4j.")
        else:
            print(f"  -> Skipped ingestion for denied transaction {tx_data['transaction_id']}.")

def create_transaction_cypher(tx, tx_data):
    """
    A Cypher query to create a transaction and connect it to accounts.
    This version is updated for the real-time workflow.
    """
    # Cypher query to merge nodes and create the transaction relationship
    # It uses sender_id and receiver_id from the real-time stream
    query = (
        "MERGE (sender:Account {accountId: $sender_id}) "
        "ON CREATE SET sender.risk_score = $sender_risk_score, sender.verified = $sender_verified "
        "ON MATCH SET sender.risk_score = $sender_risk_score, sender.verified = $sender_verified "
        "MERGE (receiver:Account {accountId: $receiver_id}) "
        "ON CREATE SET receiver.risk_score = $receiver_risk_score, receiver.verified = $receiver_verified "
        "ON MATCH SET receiver.risk_score = $receiver_risk_score, receiver.verified = $receiver_verified "
        "MERGE (device:Device {deviceId: $device_id}) "
        "MERGE (ip:IPAddress {ip: $ip_address}) "
        "CREATE (sender)-[:SENT]->(t:Transaction { "
        "  transaction_id: $transaction_id, "
        "  amount: $amount, "
        "  timestamp: datetime($timestamp), "
        "  transaction_type: $transaction_type, "
        "  fraud_flag: $fraud_flag, "
        "  fraud_probability: $fraud_probability, "
        "  decision: $decision, "
        "  merchant_category: $merchant_category, "
        "  hour: $hour, "
        "  day_of_week: $day_of_week "
        "})-[:TO]->(receiver), "
        "(t)-[:USING_DEVICE]->(device), "
        "(t)-[:FROM_IP]->(ip)"
    )
    tx.run(query, **tx_data)

with driver.session() as session:
    session.run("MATCH (n) DETACH DELETE n")
    print("Cleared existing data in Neo4j.")

# Subscribe to the 'approved_transactions' channel instead of the raw feed
p = r.pubsub()
p.subscribe('approved_transactions')

print("Subscribed to 'approved_transactions'. Listening for messages to ingest...")

try:
    # Create constraints one at a time
    driver.execute_query("CREATE CONSTRAINT accountId_unique IF NOT EXISTS FOR (a:Account) REQUIRE a.accountId IS UNIQUE;")
    driver.execute_query("CREATE CONSTRAINT transactionId_unique IF NOT EXISTS FOR (t:Transaction) REQUIRE t.transactionId IS UNIQUE;")
    driver.execute_query("CREATE CONSTRAINT deviceId_unique IF NOT EXISTS FOR (d:Device) REQUIRE d.deviceId IS UNIQUE;")
    driver.execute_query("CREATE CONSTRAINT ipAddress_unique IF NOT EXISTS FOR (ip:IPAddress) REQUIRE ip.ip IS UNIQUE;")
    for message in p.listen():
        if message['type'] == 'message':
            tx_data = json.loads(message['data'])
            ingest_transaction(driver, tx_data)
            
except KeyboardInterrupt:
    print("\nShutting down ingestion engine.")
finally:
    p.close()
    driver.close()
    print("Connections closed.")