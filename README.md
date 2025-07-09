# PIX Fraud Detection System

A real-time fraud detection system for Brazilian PIX transactions using Neo4j graph database and Redis streaming. This system generates synthetic transaction data with realistic fraud patterns and provides graph-based fraud detection capabilities.

## Features

- **Synthetic Data Generation**: Creates realistic Brazilian PIX accounts (CPF/CNPJ) and transactions
- **Fraud Pattern Detection**: Identifies smurfing rings, circular payments, and device/IP sharing
- **Real-time Streaming**: Uses Redis for transaction streaming simulation
- **Graph Database**: Neo4j for relationship-based fraud analysis

## Prerequisites

- Python 3.8+
- Redis Server
- Neo4j Database

## Quick Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Start Services
```bash
# Start Redis
brew services start redis

# Start Neo4j
brew services start neo4j
```

### 3. Configure Environment
Copy and update the environment file:
```bash
cp example.env .env
```

Default configuration:
```
NEO4J_URI=neo4j://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=password
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
```

## Running the System

### Step 1: Generate Data
```bash
python data_generator.py
```
Creates 5,000 accounts and 100,000 transactions with fraud patterns.

### Step 2: Start Ingestion Engine
```bash
python ingestion_engine.py
```
Connects to Redis and Neo4j, ready to process transactions.

### Step 3: Stream Transactions (new terminal)
```bash
python stream_simulator.py
```
Simulates real-time transaction flow.

## Project Structure

- `data_generator.py` - Creates synthetic PIX data with fraud patterns
- `ingestion_engine.py` - Processes Redis streams into Neo4j graph
- `stream_simulator.py` - Simulates real-time transaction streaming
- `test_setup.py` - Verifies system connectivity
- `requirements.txt` - Python dependencies
- `.env` - Database configuration

## Fraud Patterns Detected

- **Smurfing**: Large amounts split through multiple accounts
- **Circular Payments**: Money laundering through transaction cycles  
- **Device/IP Sharing**: Multiple accounts using same devices/IPs
- **Velocity Fraud**: High-frequency suspicious transactions

## Troubleshooting

**Redis Connection Issues:**
```bash
redis-server
```

**Neo4j Password Reset:**
```bash
cypher-shell -u neo4j -p neo4j "ALTER USER neo4j SET PASSWORD 'password'"
```

**Check Connections:**
```bash
python test_setup.py
```