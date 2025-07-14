# PIX Fraud Detection System - Work in progress

Proof of concept for a real-time fraud detection system for Brazilian PIX transactions using Neo4j graph database and Redis streaming. This system generates synthetic transaction data with realistic fraud patterns and provides graph-based fraud detection capabilities.

## Features

- **Synthetic Data Generation**: Creates realistic Brazilian PIX accounts (CPF/CNPJ) with calculated risk profiles.
- **Behavioral Simulation**: Generates transactions based on time-of-day, day-of-week, and events like salary payments.
- **Advanced Fraud Patterns**: Identifies smurfing rings and circular payments using targeted high-risk accounts.
- **Real-time Streaming**: Uses Redis for transaction streaming simulation.
- **Graph Database**: Neo4j for relationship-based fraud analysis.

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
Creates 10,000 accounts and 300,000 transactions with realistic behavioral patterns.

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

## Transaction Types Generated

The data generator creates a mix of legitimate and fraudulent transaction types, each with a unique `fraud_flag`.

- **Legitimate Transactions**:
  - `NORMAL`: Standard peer-to-peer or consumer transactions.
  - `NORMAL_SALARY`: Bulk salary payments from businesses to individuals on specific days of the month.
  - `NORMAL_MICRO`: Small, frequent payments for everyday items like coffee or transport.
  - `NORMAL_B2B`: High-value transfers between business accounts during business hours.
- **Fraudulent Transactions**:
  - `SMURFING_VICTIM_TRANSFER`: The initial large transfer from a victim to a mule account.
  - `SMURFING_DISTRIBUTION`: The subsequent distribution of stolen funds from a central mule to other mules in the ring.
  - `CIRCULAR_PAYMENT`: Transactions that form a closed loop between high-risk accounts to launder money.

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

### **Phase 1: Data Pipeline & Graph Foundation (Completed)**

The core infrastructure for generating, streaming, and storing transaction data is complete.

-   **Synthetic Data Generation** (`data_generator.py`):
    -   [x] Generation of 10,000 realistic PIX accounts with valid CPF/CNPJ.
    -   [x] **Enhanced Realism**: Accounts now have a calculated `risk_score` based on age, verification status, and other factors.
    -   [x] Creation of 300,000 transactions with varied amounts and timestamps.
    -   [x] **Behavioral Simulation**: Transaction generation is now weighted based on time of day (business hours, late night), day of week, and special events (salary days), making the data distribution more lifelike.
    -   [x] Embedding of specific fraud patterns into the dataset:
        -   **Smurfing/Structuring**: Fraudulent rings are now formed by targeting high-risk accounts as mules.
        -   **Circular Payments**: Money is moved in a loop between high-risk accounts to obscure its origin.
        -   **Shared Identifiers**: Fraudulent accounts in a ring share a common device and IP address.
    -   [x] Make sure data is good enough for the fraud detection algorithms.

-   **Real-time Streaming Simulation**:
    -   [x] A publisher script (`stream_simulator.py`) reads the synthetic data and streams it to a Redis Pub/Sub channel in real-time.

-   **Graph Ingestion Engine** (`ingestion_engine.py`):
    -   [x] A subscriber script connects to Redis and Neo4j, listening for incoming transactions.
    -   [x] Implemented an idempotent ingestion process using `MERGE` to prevent data duplication.
    -   [x] Built a flexible graph schema that captures rich, contextual properties for all nodes and relationships.

### **Phase 2: Graph Analytics & Fraud Detection (To-Do)**

This phase focuses on building the analytical layer on top of the graph to detect and flag suspicious activity.

-   **Rule-Based Detection (Cypher Queries)**:
    -   [ ] **Shared Infrastructure**:
        -   [ ] Query for devices linked to > N accounts.
        -   [ ] Query for IP addresses linked to > N accounts.
        -   [ ] Refine queries to filter for high-risk or fraudulent transactions from shared infrastructure.
    -   [ ] **Circular Payments**:
        -   [ ] Implement a Cypher query for 3-hop circular paths `(a)->(b)->(c)->(a)`.
        -   [ ] Generalize the query for variable-length paths `(a)-[*3..5]->(a)`.
        -   [ ] Add filters to only show cycles involving high-risk accounts.
    -   [ ] **Money Mule Profiling**:
        -   [ ] Query for accounts with high fan-in (many distinct senders).
        -   [ ] Query for accounts with high fan-out (many distinct receivers).
        -   [ ] Combine fan-in/fan-out logic to find accounts that receive from many and send to few.
        -   [ ] Add time-based constraints (e.g., high velocity fan-in/out within 24 hours).

-   **Community Detection (Graph Data Science)**:
    -   [ ] **GDS Setup**: Install the GDS plugin and create an in-memory graph projection of `(:Account)` nodes and `[:MONEY_FLOW]` relationships.
    -   [ ] **Louvain Algorithm**: Run `gds.louvain.stream` to detect communities and write the results back to the `Account` nodes.
    -   [ ] **Community Analysis**:
        -   [ ] Query for communities with a high average `risk_score`.
        -   [ ] Query for communities with a high number of fraudulent transactions.
        -   [ ] Analyze the size and density of suspicious communities.

-   **Anomaly Detection (Machine Learning)**:
    -   [ ] **Feature Engineering**:
        -   [ ] Create a new script `feature_engineering.py`.
        -   [ ] Write Cypher queries to calculate graph features (degree, PageRank) and transactional features (frequency, avg/max amount) for each account.
        -   [ ] Export all features to a CSV file (`account_features.csv`).
    -   [ ] **Model Training**:
        -   [ ] Create a new script `anomaly_detection.py`.
        -   [ ] Load `account_features.csv` and use `scikit-learn` to apply the Local Outlier Factor (LOF) algorithm.
        -   [ ] Identify outliers and write the results (account ID + anomaly score) back to Neo4j or a new CSV.

### **Phase 3: Investigation & Visualization (To-Do)**

The final phase is to build an interactive dashboard for investigators to explore alerts and analyze fraud patterns visually.

-   **Dashboard Setup (Streamlit)**:
    -   [ ] Create `dashboard.py` with a basic Streamlit layout.
    -   [ ] Add a Neo4j connection utility to the dashboard.
    -   [ ] Add `streamlit` to `requirements.txt`.
-   **High-Level Metrics Page**:
    -   [ ] Create a "Dashboard" tab to display KPIs: Total Transactions, Total Fraud Amount, Fraud Rate %.
    -   [ ] Add a bar chart showing fraud counts by type (`fraud_flag`).
-   **Alerts & Triage View**:
    -   [ ] Create an "Alerts" tab to display a sortable table of high-risk accounts identified by the ML model or Cypher rules.
    -   [ ] Allow users to select an account from the table for investigation.
-   **Interactive Graph Investigation**:
    -   [ ] Add a graph visualization component (e.g., `streamlit-agraph`) to the investigation view.
    -   [ ] When an account is selected, query its 1-hop and 2-hop neighborhood and render the subgraph visually.
    -   [ ] Display properties of a selected node or relationship from the graph.