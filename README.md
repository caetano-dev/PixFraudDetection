# PIX Fraud Detection System

Production-ready fraud detection system for Brazilian PIX transactions using Neo4j graph database, advanced ML algorithms, and real-time streaming. Features realistic fraud pattern generation and state-of-the-art detection algorithms including Louvain community detection and Local Outlier Factor analysis.

## Architecture Overview

This system implements a comprehensive fraud detection pipeline with:
- **Realistic Data Generation**: Sophisticated synthetic PIX transaction patterns including smurfing rings and circular payments
- **Production-Ready Feature Engineering**: Observable behavioral patterns without fraud flag dependencies
- **Advanced Detection Algorithms**: Louvain community detection + Local Outlier Factor within communities
- **Graph-Based Analysis**: Neo4j for relationship modeling and community detection
- **Real-time Capabilities**: Redis streaming with time-series fraud pattern detection
- **Interactive Investigation**: Streamlit dashboard for fraud investigators

## Key Features

### 🔍 **Advanced Fraud Detection**
- **Community Detection**: Louvain algorithm to identify suspicious account clusters
- **Local Outlier Factor**: Targeted anomaly detection within communities
- **Isolation Forest**: Global anomaly detection for baseline comparison
- **Production-Ready Features**: No fraud flag dependencies - only observable behavioral patterns

### 🎯 **Realistic Fraud Simulation**
- **Smurfing Rings**: Multi-account money laundering with cash-out phases
- **Circular Payments**: Complex money flow loops with warm-up transactions
- **Behavioral Realism**: Time-based patterns, device diversity, velocity analysis
- **Brazilian CPF/CNPJ**: Authentic document validation and risk profiling

### 📊 **Interactive Investigation**
- **Community Analysis**: Visual exploration of suspicious account clusters
- **Alert Triage**: Prioritized fraud alerts with contextual information
- **Graph Visualization**: Interactive network analysis of transaction flows
- **Real-time Monitoring**: Live transaction stream analysis

## Prerequisites

- Python 3.8+
- Neo4j Database (with Graph Data Science plugin for Louvain algorithm)
- Redis Server
- Streamlit (for dashboard)

## Quick Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Start Services
```bash
# Start Redis
brew services start redis

# Start Neo4j with GDS plugin
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
Creates accounts and transactions with realistic behavioral patterns.

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

### Step 4: Feature Engineering
This script connects to Neo4j, extracts graph and transactional features for each account, and saves them to a CSV file.

```bash
python feature_engineering.py
```

### Step 5: Production-Ready Feature Engineering
Extract behavioral features without using fraud flags (production deployment ready):

```bash
python feature_engineering.py
```

This creates:
- `account_features.csv` - Production features (39 behavioral patterns)
- `evaluation_dataset.csv` - Evaluation dataset with ground truth labels

### Step 6: Anomaly Detection
Apply machine learning algorithms to detect fraud patterns:

```bash
# Local Outlier Factor (recommended for fraud detection)
python anomaly_detection.py --algorithm lof --contamination 0.05

# Isolation Forest (for baseline comparison)
python anomaly_detection.py --algorithm isolation_forest
```

### Step 7: Community Detection
Identify suspicious account clusters using Louvain algorithm:

```bash
python community_detection.py
```

### Step 8: Local Outlier Factor within Communities
Apply targeted anomaly detection within each community:

```bash
python local_outlier_factor.py
```

### Step 9: Launch the Investigation Dashboard
Interactive fraud investigation interface:

```bash
streamlit run dashboard.py
```

## Detection Algorithms

### 🎯 **Louvain Community Detection**
- **Purpose**: Identify clusters of accounts with dense transaction relationships
- **Implementation**: `community_detection.py`
- **Key Features**:
  - Detects money laundering rings and organized fraud networks
  - Uses Neo4j Graph Data Science plugin
  - Analyzes community structure and fraud concentration
  - Identifies bridging accounts between communities

### 🔍 **Local Outlier Factor (LOF)**
- **Purpose**: Find accounts that behave anomalously within their local neighborhood
- **Implementation**: `local_outlier_factor.py` and `anomaly_detection.py`
- **Key Features**:
  - Community-aware anomaly detection
  - Identifies accounts with unusual patterns relative to peers
  - Excellent for detecting sophisticated fraud that mimics normal behavior
  - Handles varying density of transaction patterns

### 🌲 **Isolation Forest**
- **Purpose**: Global anomaly detection using isolation principles
- **Implementation**: `anomaly_detection.py`
- **Key Features**:
  - Fast, scalable anomaly detection
  - Identifies globally anomalous transaction patterns
  - Good baseline for comparison with LOF results
  - Effective for high-volume fraud detection

## Production-Ready Features

Our feature engineering focuses on **observable behavioral patterns** that work in production environments:

### 📊 **Transaction Patterns**
- Volume metrics (count, amounts, ratios)
- Temporal patterns (timing, velocity, frequency)
- Counterparty diversity and relationship patterns
- Amount distribution and variance analysis

### 🌐 **Network Behavior**
- Money flow relationships and circularity detection
- Community membership and bridging behavior
- Partner diversity and concentration metrics
- Suspicious flow pattern identification

### ⚡ **Velocity Analysis**
- Transaction timing patterns and burst detection
- Rapid transaction sequence identification
- Time-based behavioral anomaly detection
- Velocity ratio analysis

### 🛡️ **Key Design Principles**
- **No Fraud Flag Dependencies**: All features work without knowing fraud labels
- **Real-time Compatible**: Features can be calculated in streaming environments
- **Interpretable**: Clear business meaning for fraud investigators
- **Scalable**: Efficient computation for large transaction volumes

## Project Structure

- `data_generator.py` - Creates sophisticated synthetic PIX data with realistic fraud patterns
- `ingestion_engine.py` - Processes Redis streams into Neo4j graph database
- `stream_simulator.py` - Simulates real-time transaction streaming
- `feature_engineering.py` - **Production-ready behavioral feature extraction** 
- `anomaly_detection.py` - **LOF and Isolation Forest fraud detection**
- `community_detection.py` - **Louvain algorithm for fraud ring detection**
- `local_outlier_factor.py` - **Community-aware anomaly detection**
- `dashboard.py` - Interactive Streamlit fraud investigation dashboard
- `test_setup.py` - System connectivity verification
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
  - `SMURFING_CASH_OUT`: The final transfer from a mule to a high-risk service (e.g., crypto exchange) to withdraw the funds.
  - `CIRCULAR_PAYMENT`: Transactions that form a closed loop between high-risk accounts to launder money.

## Fraud Patterns Detected

## Real-World Applications

### 🏦 **Financial Institution Integration**
- **Real-time Monitoring**: Features designed for streaming fraud detection
- **Investigation Tools**: Interactive dashboard for fraud analysts
- **Compliance Reporting**: Comprehensive audit trails and detection metrics
- **Risk Scoring**: Multi-algorithm approach for account risk assessment

### 🎯 **Fraud Pattern Detection**
- **Smurfing Rings**: Multi-layered detection using community analysis + LOF
- **Circular Payments**: Money laundering loop detection with warm-up period analysis
- **Velocity Fraud**: High-frequency transaction pattern identification
- **Behavioral Anomalies**: Subtle fraud pattern detection within normal communities

### 📈 **Performance Metrics**
- **Detection Rate**: ~5% anomaly detection rate with LOF (297/5,929 accounts)
- **False Positive Management**: Community-aware detection reduces false positives
- **Scalability**: Designed for large-scale transaction processing
- **Real-time Capable**: Feature engineering optimized for streaming environments

## Technical Achievements

### ✅ **Completed Features**
- **Realistic Data Generation**: Sophisticated fraud pattern simulation
- **Production-Ready Features**: 49 behavioral features without fraud flag dependencies
- **Advanced Algorithms**: Louvain + LOF community-aware fraud detection
- **Interactive Dashboard**: Full fraud investigation interface
- **Model Persistence**: Saved models for production deployment
- **Evaluation Framework**: Separate datasets for training and evaluation

### 🔬 **Algorithm Performance**
- **Local Outlier Factor**: 5.01% outlier rate, excellent for sophisticated fraud
- **Isolation Forest**: 10.90% outlier rate, good for high-volume anomalies
- **Community Detection**: Identifies organized fraud rings and money laundering networks
- **Feature Importance**: Transaction amounts and flow patterns most predictive

### 📊 **Data Quality**
- **Account Coverage**: 5,929 synthetic accounts with realistic patterns
- **Fraud Simulation**: 265 fraudulent accounts (4.47% realistic fraud rate)
- **Feature Diversity**: 49 distinct behavioral patterns for comprehensive analysis
- **Production Readiness**: No fraud flag dependencies in feature engineering

## Next Steps & Enhancements

### 🚀 **Production Deployment**
- [ ] Real-time streaming integration with Apache Kafka
- [ ] Model monitoring and drift detection
- [ ] A/B testing framework for algorithm comparison
- [ ] Integration with existing banking fraud systems

### 🧠 **Advanced Analytics**
- [ ] Deep learning models for sequence pattern detection
- [ ] Graph neural networks for relationship modeling
- [ ] Explainable AI for fraud investigation support
- [ ] Multi-modal detection combining transaction and device signals

### 🔧 **System Enhancements**
- [ ] Automated model retraining pipeline
- [ ] Advanced visualization for complex fraud networks
- [ ] Integration with external data sources (sanctions lists, etc.)
- [ ] Performance optimization for high-throughput environments

## Troubleshooting

**Neo4j Connection Issues:**
```bash
# Check Neo4j status
brew services list | grep neo4j

# Restart if needed
brew services restart neo4j
```

**Redis Connection Issues:**
```bash
# Start Redis server
redis-server

# Check Redis connectivity
redis-cli ping
```

**Missing Features Warning:**
Some features may show as missing if the transaction data doesn't include all temporal or device fields. This is expected and doesn't affect core functionality.

**Performance Issues:**
For large datasets, consider:
- Increasing Neo4j memory allocation
- Using database indexes for frequently queried properties
- Implementing batch processing for feature engineering

**Neo4j Password Reset:**
```bash
cypher-shell -u neo4j -p neo4j "ALTER USER neo4j SET PASSWORD 'password'"
```

## Project Roadmap & Status

This project is being developed in phases. Here is a summary of what has been implemented and what is planned for the future.

### **Phase 1: Data Pipeline & Graph Foundation (Completed)**

The core infrastructure for generating, streaming, and storing transaction data is complete.

-   **Synthetic Data Generation** (`data_generator.py`):
    -   [x] Generation of realistic PIX accounts with valid CPF/CNPJ
    -   [x] Enhanced fraud patterns including sleeper accounts, rapid cash-out, and warm-up transactions
    -   [x] Behavioral simulation based on time patterns and fraud psychology
    -   [x] Realistic transaction amounts and fraud thresholds

-   **Real-time Streaming Simulation**:
    -   [x] Redis Pub/Sub streaming of transaction data (`stream_simulator.py`)
    -   [x] Configurable streaming rates and batch processing

-   **Graph Ingestion Engine** (`ingestion_engine.py`):
    -   [x] Neo4j integration with comprehensive schema
    -   [x] Idempotent ingestion using MERGE operations
    -   [x] Rich node and relationship properties for analysis

### **Phase 2: Graph Analytics & Fraud Detection (Completed)**

This phase focuses on building the analytical layer on top of the graph to detect and flag suspicious activity.

-   **Feature Engineering** (`feature_engineering.py`):
    -   [x] Graph-based features (degree centrality, connectivity patterns)
    -   [x] Transactional features (amounts, frequency, fraud ratios)
    -   [x] Account risk scores and verification status
    -   [x] Comprehensive feature extraction for ML models

-   **Anomaly Detection** (`anomaly_detection.py`):
    -   [x] **Local Outlier Factor (LOF)**: Primary algorithm for detecting anomalies
    -   [x] **Isolation Forest**: Alternative algorithm with configurable parameters
    -   [x] Standardized feature scaling and model persistence
    -   [x] Command-line interface for algorithm selection

-   **Community Detection** (`community_detection.py`):
    -   [x] **Louvain Algorithm**: Detects transaction communities using GDS
    -   [x] **Weighted Community Detection**: Uses transaction amounts as edge weights
    -   [x] **Community Analysis**: Statistical analysis of detected communities
    -   [x] **Suspicious Community Identification**: Finds high-risk communities

-   **Local Outlier Factor within Communities** (`local_outlier_factor.py`):
    -   [x] **Community-based LOF**: Applies LOF within each detected community
    -   [x] **Enhanced Feature Engineering**: Community-specific features
    -   [x] **Local Anomaly Detection**: Identifies outliers that blend in globally
    -   [x] **Neo4j Integration**: Updates account nodes with LOF scores

### **Phase 3: Investigation & Visualization (Completed)**

The final phase is to build an interactive dashboard for investigators to explore alerts and analyze fraud patterns visually.

-   **Dashboard Setup (Streamlit)** (`dashboard.py`):
    -   [x] Multi-tab interface for different analysis views
    -   [x] Real-time data loading and caching
    -   [x] Integration with Neo4j for live queries

-   **High-Level Metrics Page**:
    -   [x] Platform-wide KPIs (transaction volume, fraud rates)
    -   [x] Interactive charts showing fraud breakdown by type
    -   [x] Real-time fraud statistics

-   **Alerts & Investigation Page**:
    -   [x] Sortable table of anomalous accounts
    -   [x] Account neighborhood visualization
    -   [x] Interactive graph exploration with 2-hop analysis

-   **Community Analysis Page**:
    -   [x] Community overview with Louvain results
    -   [x] LOF outlier analysis within communities
    -   [x] Community fraud rate visualization
    -   [x] Comprehensive community statistics

Ideas:

 - [ ] Add path finding algorithm to the dashboard to find shortest path between accounts
 - [ ] Unit tests

 Bugs: