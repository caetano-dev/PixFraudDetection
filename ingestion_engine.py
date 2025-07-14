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
    query = """
    // Find or create the source account and set its properties
    MERGE (src:Account {accountId: $sender_id})
    ON CREATE SET
        src.creation_date = datetime($timestamp), // Placeholder, ideally from account data
        src.is_verified = $sender_verified,
        src.state = $sender_state,
        src.risk_score = toFloat($sender_risk_score)
    ON MATCH SET
        src.is_verified = $sender_verified,
        src.state = $sender_state,
        src.risk_score = toFloat($sender_risk_score)

    // Find or create the destination account and set its properties
    MERGE (dest:Account {accountId: $receiver_id})
    ON CREATE SET
        dest.creation_date = datetime($timestamp), // Placeholder
        dest.is_verified = $receiver_verified,
        dest.state = $receiver_state,
        dest.risk_score = toFloat($receiver_risk_score)
    ON MATCH SET
        dest.is_verified = $receiver_verified,
        dest.state = $receiver_state,
        dest.risk_score = toFloat($receiver_risk_score)

    // MERGE the device and IP address nodes
    MERGE (dev:Device {deviceId: $device_id})
    MERGE (ip:IPAddress {ip: $ip_address})

    // MERGE the transaction node itself, setting all properties
    MERGE (tx:Transaction {transactionId: $transaction_id})
    ON CREATE SET
        tx.amount = toFloat($amount),
        tx.timestamp = datetime($timestamp),
        tx.fraudFlag = $fraud_flag,
        tx.transaction_type = $transaction_type,
        tx.channel = $channel,
        tx.merchant_category = $merchant_category,
        tx.hour_of_day = toInteger($hour_of_day),
        tx.day_of_week = toInteger($day_of_week),
        tx.is_weekend = $is_weekend,
        tx.same_state = $same_state

    // MERGE the relationships connecting all entities
    MERGE (src)-[:SENT]->(tx)
    MERGE (tx)-[:RECEIVED_BY]->(dest)
    MERGE (tx)-[:USED_DEVICE]->(dev)
    MERGE (tx)-[:FROM_IP]->(ip)
    
    // Update the direct account-to-account relationship for flow analysis
    MERGE (src)-[flow:MONEY_FLOW]->(dest)
    ON CREATE SET 
        flow.firstTransaction = datetime($timestamp), 
        flow.totalAmount = toFloat($amount), 
        flow.transactionCount = 1,
        flow.fraudTransactionCount = CASE WHEN $fraud_flag STARTS WITH 'SMURFING' OR $fraud_flag = 'CIRCULAR_PAYMENT' THEN 1 ELSE 0 END
    ON MATCH SET 
        flow.lastTransaction = datetime($timestamp), 
        flow.totalAmount = flow.totalAmount + toFloat($amount), 
        flow.transactionCount = flow.transactionCount + 1,
        flow.fraudTransactionCount = flow.fraudTransactionCount + CASE WHEN $fraud_flag STARTS WITH 'SMURFING' OR $fraud_flag = 'CIRCULAR_PAYMENT' THEN 1 ELSE 0 END
    """
    
    # We need to get the state for sender and receiver to pass to the query
    # This is a simplification; in a real system, this data would be joined
    # before hitting the ingestion engine. For now, we'll assume it's in the tx_data.
    
    driver.execute_query(query, **tx_data)
    print(f"Ingested transaction: {tx_data['transaction_id']}")

with driver.session() as session:
    session.run("MATCH (n) DETACH DELETE n")
    print("Cleared existing data in Neo4j.")


p = r.pubsub()
p.subscribe('pix_transactions')

print("Subscribed to 'pix_transactions'. Listening for messages...")

try:
    # Create constraints one at a time
    driver.execute_query("CREATE CONSTRAINT accountId_unique IF NOT EXISTS FOR (a:Account) REQUIRE a.accountId IS UNIQUE;")
    driver.execute_query("CREATE CONSTRAINT transactionId_unique IF NOT EXISTS FOR (t:Transaction) REQUIRE t.transactionId IS UNIQUE;")
    driver.execute_query("CREATE CONSTRAINT deviceId_unique IF NOT EXISTS FOR (d:Device) REQUIRE d.deviceId IS UNIQUE;")
    driver.execute_query("CREATE CONSTRAINT ipAddress_unique IF NOT EXISTS FOR (ip:IPAddress) REQUIRE ip.ip IS UNIQUE;")
    for message in p.listen():
        if message['type'] == 'message':
            transaction_data = json.loads(message['data'].decode('utf-8'))
            print(f"Received: {transaction_data}")
            
            ingest_transaction(driver, transaction_data)
            
except KeyboardInterrupt:
    print("Stopping ingestion engine...")
finally:
    p.close()
    driver.close()
    print("Connections closed.")