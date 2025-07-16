import redis
from neo4j import GraphDatabase
import json
import os
import logging
from config import config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load configuration
redis_config = config['redis']
neo4j_config = config['neo4j']
app_config = config['app']

def init_redis() -> redis.Redis:
    """Initialize and return Redis client."""
    try:
        client = redis.Redis(
            host=redis_config['host'],
            port=redis_config['port'],
            db=redis_config['db']
        )
        client.ping()
        logger.info(f"Connected to Redis at {redis_config['host']}:{redis_config['port']}/{redis_config['db']}")
        return client
    except Exception as e:
        logger.exception("Failed to connect to Redis")
        raise

def init_neo4j() -> GraphDatabase:
    """Initialize and return Neo4j driver."""
    try:
        driver = GraphDatabase.driver(
            neo4j_config['uri'],
            auth=(neo4j_config['user'], neo4j_config['password'])
        )
        with driver.session() as session:
            session.run("RETURN 1")
        logger.info(f"Connected to Neo4j at {neo4j_config['uri']}")
        return driver
    except Exception as e:
        logger.exception("Failed to connect to Neo4j")
        raise

def ingest_transaction(driver, tx_data: dict) -> None:
    """Ingest a single transaction into Neo4j with optimized Cypher."""
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
    try:
        driver.execute_query(query, **tx_data)
        logger.debug(f"Ingested transaction: {tx_data['transaction_id']}")
    except Exception:
        logger.exception(f"Error ingesting transaction {tx_data.get('transaction_id')}")

def main() -> None:
    """Main loop: initialize connections, optionally clear DB, subscribe to Redis, and ingest messages."""
    redis_client = init_redis()
    driver = init_neo4j()

    # Optionally clear DB on startup
    if app_config.get('clear_on_start', False):
        with driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        logger.info("Cleared existing data in Neo4j.")

    # Apply constraints
    constraints = [
        "CREATE CONSTRAINT IF NOT EXISTS FOR (a:Account) REQUIRE a.accountId IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (t:Transaction) REQUIRE t.transactionId IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (d:Device) REQUIRE d.deviceId IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (ip:IPAddress) REQUIRE ip.ip IS UNIQUE"
    ]
    for c in constraints:
        try:
            driver.execute_query(c)
            logger.info(f"Applied constraint: {c}")
        except Exception:
            logger.exception(f"Failed to apply constraint: {c}")

    pubsub = redis_client.pubsub()
    pubsub.subscribe('pix_transactions')
    logger.info("Subscribed to 'pix_transactions'. Listening for messages...")

    try:
        for message in pubsub.listen():
            if message['type'] == 'message':
                data = json.loads(message['data'])
                logger.info(f"Received transaction {data.get('transaction_id')}")
                ingest_transaction(driver, data)
    except KeyboardInterrupt:
        logger.info("Stopping ingestion engine...")
    finally:
        pubsub.close()
        driver.close()
        logger.info("Connections closed.")

if __name__ == '__main__':
    main()