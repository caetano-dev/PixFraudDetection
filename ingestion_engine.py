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
    Ingests a single transaction into Neo4j using a hybrid approach for maximum fraud detection flexibility.
    """
    query = """
    // Use MERGE to find or create the source and destination accounts
    MERGE (src:Account {accountId: $sender_id})
    MERGE (dest:Account {accountId: $receiver_id})

    // MERGE the device and IP address nodes
    MERGE (dev:Device {deviceId: $device_id})
    MERGE (ip:IPAddress {ip: $ip_address})

    // MERGE the transaction node itself, setting properties ON CREATE
    MERGE (tx:Transaction {transactionId: $transaction_id})
    ON CREATE SET
        tx.amount = toFloat($amount),
        tx.timestamp = datetime($timestamp),
        tx.fraudFlag = $fraud_flag

    // MERGE the relationships connecting all entities
    MERGE (src)-[:SENT]->(tx)
    MERGE (tx)-[:RECEIVED_BY]->(dest)
    MERGE (tx)-[:USED_DEVICE]->(dev)
    MERGE (tx)-[:FROM_IP]->(ip)
    
    // ADD: Direct account-to-account relationship for flow visualization and aggregation
    MERGE (src)-[flow:MONEY_FLOW]->(dest)
    ON CREATE SET 
        flow.firstTransaction = datetime($timestamp), 
        flow.totalAmount = toFloat($amount), 
        flow.transactionCount = 1,
        flow.fraudTransactionCount = CASE WHEN $fraud_flag <> 'NORMAL' THEN 1 ELSE 0 END
    ON MATCH SET 
        flow.lastTransaction = datetime($timestamp), 
        flow.totalAmount = flow.totalAmount + toFloat($amount), 
        flow.transactionCount = flow.transactionCount + 1,
        flow.fraudTransactionCount = flow.fraudTransactionCount + CASE WHEN $fraud_flag <> 'NORMAL' THEN 1 ELSE 0 END
    """
    
    driver.execute_query(query, **tx_data)
    print(f"Ingested transaction: {tx_data['transaction_id']}")

with driver.session() as session:
    session.run("MATCH (n) DETACH DELETE n")
    print("Cleared existing data in Neo4j.")

r.flushdb()
print("Cleared existing data in Redis.")

p = r.pubsub()
p.subscribe('pix_transactions')

print("Subscribed to 'pix_transactions'. Listening for messages...")

try:
    for message in p.listen():
        if message['type'] == 'message':
            transaction_data = json.loads(message['data'])
            print(f"Received: {transaction_data}")
            
            ingest_transaction(driver, transaction_data)
            
except KeyboardInterrupt:
    print("Stopping ingestion engine...")
finally:
    p.close()
    driver.close()
    print("Connections closed.")