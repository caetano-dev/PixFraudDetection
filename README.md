# PIX Fraud Detection System - Work in progress

Proof of concept for a real-time fraud detection system for Brazilian PIX transactions using Neo4j graph database and Redis streaming. This system generates synthetic transaction data with realistic fraud patterns and provides graph-based fraud detection capabilities.

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
## Project Roadmap & Status

This project is being developed in phases. Here is a summary of what has been implemented and what is planned for the future.

### **Phase 1: Data Pipeline & Graph Foundation (In Progress)**

The core infrastructure for generating, streaming, and storing transaction data is complete.

-   **Synthetic Data Generation** (`data_generator.py`):
    -   [x] Generation of 5,000 realistic PIX accounts, including valid CPF and CNPJ numbers.
    -   [x] Creation of 100,000 transactions with varied amounts and timestamps.
    -   [x] Embedding of specific fraud patterns into the dataset:
        -   **Smurfing/Structuring**: A central account distributes funds to a ring of mule accounts.
        -   **Circular Payments**: Money is moved in a loop between 3-5 accounts to obscure its origin.
        -   **Shared Identifiers**: Fraudulent accounts in a ring share a common device and IP address.
    -   [ ] Make sure data is good enough for the fraud detection algorithms.

-   **Real-time Streaming Simulation**:
    -   [x] A publisher script (`stream_simulator.py`) reads the synthetic data and streams it to a Redis Pub/Sub channel in real-time.

-   **Graph Ingestion Engine** (`ingestion_engine.py`):
    -   [x] A subscriber script connects to Redis and Neo4j, listening for incoming transactions.
    -   [x] Implemented an idempotent ingestion process using `MERGE` to prevent data duplication.
    -   [x] Built a flexible graph schema as planned in the documentation, creating `:Account`, `:Transaction`, `:Device`, and `:IPAddress` nodes and their relationships for rich, contextual queries.

### **Phase 2: Graph Analytics & Fraud Detection (To-Do)**

This phase focuses on building the analytical layer on top of the graph to detect and flag suspicious activity.

-   **Rule-Based Detection (Cypher Queries)**:
    -   [ ] **Shared Device/IP**: Write a Cypher query to find multiple accounts making transactions from the same device or IP address.
    -   [ ] **Circular Payments**: Implement a Cypher path-matching query to detect 3-step (or more) payment loops.
    -   [ ] **Money Mule Profiling**: Develop a query to identify accounts with high "fan-in" (many incoming payments) and low "fan-out" (few outgoing payments).

-   **Community Detection (Graph Data Science)**:
    -   [ ] Set up a GDS graph projection to analyze relationships between accounts.
    -   [ ] Implement the **Louvain** algorithm to find densely connected communities that could represent undiscovered fraud rings.
    -   [ ] Develop analysis queries to profile the detected communities based on size, density, and age.

-   **Anomaly Detection (Machine Learning)**:
    -   [ ] Engineer graph-based features for each account (e.g., degree, PageRank, transaction frequency, average amount).
    -   [ ] Export features to a Pandas DataFrame.
    -   [ ] Apply the **Local Outlier Factor (LOF)** algorithm with `scikit-learn` to identify accounts with anomalous behavior compared to their peers.

### **Phase 3: Investigation & Visualization (To-Do)**

The final phase is to build an interactive dashboard for investigators to explore alerts and analyze fraud patterns visually.

-   **Investigator's Dashboard (Streamlit)**:
    -   [ ] Set up a Streamlit application (`dashboard.py`) and connect it to the Neo4j database.
    -   [ ] Create a main overview page to display high-level metrics and fraud alerts from the detection algorithms.
    -   [ ] Build an interactive "Investigation" tab where an analyst can select a suspicious account or community.

-   **Interactive Graph Visualization**:
    -   [ ] Integrate a graph visualization library (e.g., `neo4j-viz`) into the Streamlit app.
    -   [ ] On user selection, dynamically query the 2-hop neighborhood of a suspicious account and render it as an interactive graph.
    -   [ ] Implement a details pane that displays the properties of a node or relationship when it is clicked in the graph.