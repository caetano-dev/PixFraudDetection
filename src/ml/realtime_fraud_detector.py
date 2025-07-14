import redis
import json
import joblib
import pandas as pd
from neo4j import GraphDatabase
from src.config.config import (
    REDIS_HOST, REDIS_PORT, REDIS_DB,
    NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD
)

# --- Configuration ---
TRANSACTION_CHANNEL = 'pix_transactions'
APPROVED_CHANNEL = 'approved_transactions'
ALERTS_CHANNEL = 'fraud_alerts'

# Model Config
MODEL_PATH = 'src/models/realtime_fraud_model.joblib'
SCALER_PATH = 'src/models/realtime_scaler.joblib'
FEATURES_PATH = 'src/models/realtime_features.joblib'
PREDICTION_THRESHOLD = 0.75 # Probability score to flag as fraud

# --- Load Models and Connect to Services ---
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_list = joblib.load(FEATURES_PATH)
    print("Successfully loaded fraud detection model, scaler, and feature list.")
except FileNotFoundError:
    print(f"Error: Model, scaler or features file not found. Please run 'model_training.py' first.")
    exit()

try:
    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=True)
    neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    print("Successfully connected to Redis and Neo4j.")
except Exception as e:
    print(f"Error connecting to services: {e}")
    exit()


def get_realtime_features(session, account_id):
    """
    Fetches real-time features for a given account from Neo4j.
    This simulates a fast lookup during a live transaction.
    """
    query = """
    MATCH (a:Account {accountId: $account_id})
    OPTIONAL MATCH (a)-[:SENT]->(t:Transaction)
    RETURN
        a.risk_score AS riskScore,
        COUNT { (a)<--(:Transaction) } AS inDegree,
        COUNT { (a)-->(:Transaction) } AS outDegree,
        COUNT { (a)--(:Transaction) } AS totalDegree,
        sum(t.amount) AS totalAmount,
        avg(t.amount) AS avgAmount,
        max(t.amount) AS maxAmount,
        count(t) AS transactionCount
    """
    result = session.run(query, account_id=account_id)
    features = result.single()
    
    if features:
        # Neo4j returns None for aggregations on no rows, so we default to 0
        return {
            'sender_riskScore': features['riskScore'] or 0,
            'sender_inDegree': features['inDegree'] or 0,
            'sender_outDegree': features['outDegree'] or 0,
            'sender_totalDegree': features['totalDegree'] or 0,
            'sender_totalAmount': features['totalAmount'] or 0,
            'sender_avgAmount': features['avgAmount'] or 0,
            'sender_maxAmount': features['maxAmount'] or 0,
            'sender_transactionCount': features['transactionCount'] or 0,
        }
    return {key: 0 for key in ['sender_riskScore', 'sender_inDegree', 'sender_outDegree', 'sender_totalDegree', 'sender_totalAmount', 'sender_avgAmount', 'sender_maxAmount', 'sender_transactionCount']}


def predict_fraud(transaction_data, sender_features):
    """
    Combines transaction data with real-time features and predicts fraud probability.
    """
    # Create a single-row DataFrame for prediction
    features = {
        'amount': transaction_data['amount'],
        'hour': transaction_data['hour'],
        'day': transaction_data['day'],
        'month': transaction_data['month'],
        'day_of_week': transaction_data['day_of_week'],
        **sender_features
    }
    
    feature_df = pd.DataFrame([features])
    
    # Ensure column order matches the training order
    feature_df = feature_df[feature_list]

    # Scale features and predict
    scaled_features = scaler.transform(feature_df)
    probability = model.predict_proba(scaled_features)[:, 1][0]
    
    return probability


def main():
    """
    Main loop to listen for transactions, predict fraud, and take action.
    """
    pubsub = r.pubsub()
    pubsub.subscribe(TRANSACTION_CHANNEL)
    print(f"Subscribed to '{TRANSACTION_CHANNEL}'. Listening for transactions...")

    with neo4j_driver.session() as session:
        try:
            for message in pubsub.listen():
                if message['type'] == 'message':
                    tx_data = json.loads(message['data'])
                    sender_id = tx_data['sender_id']

                    # --- Real-time Feature Engineering ---
                    # Create datetime features from the timestamp string
                    dt_object = pd.to_datetime(tx_data['timestamp'], format='ISO8601')
                    tx_data['hour'] = dt_object.hour
                    tx_data['day'] = dt_object.day
                    tx_data['month'] = dt_object.month
                    tx_data['day_of_week'] = dt_object.dayofweek
                    # ---

                    # 1. Enrich with real-time features
                    sender_features = get_realtime_features(session, sender_id)

                    # 2. Predict fraud probability
                    fraud_prob = predict_fraud(tx_data, sender_features)
                    tx_data['fraud_probability'] = fraud_prob

                    # 3. Take Action
                    if fraud_prob >= PREDICTION_THRESHOLD:
                        # High risk: Block and alert
                        tx_data['decision'] = 'DENIED'
                        alert_message = json.dumps(tx_data)
                        r.publish(ALERTS_CHANNEL, alert_message)
                        print(f"🚨 DENIED transaction {tx_data['transaction_id']} from {sender_id}. Probability: {fraud_prob:.2%}")
                    else:
                        # Low risk: Approve
                        tx_data['decision'] = 'APPROVED'
                        approved_message = json.dumps(tx_data)
                        r.publish(APPROVED_CHANNEL, approved_message)
                        print(f"✅ APPROVED transaction {tx_data['transaction_id']} from {sender_id}. Probability: {fraud_prob:.2%}")

        except KeyboardInterrupt:
            print("\nShutting down fraud detection service.")
        finally:
            pubsub.close()
            neo4j_driver.close()
            print("Connections closed.")

if __name__ == "__main__":
    main()
