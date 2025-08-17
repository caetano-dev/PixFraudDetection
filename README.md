# PIX Fraud Detection System

A modular, production-ready pipeline for synthetic data generation, streaming ingestion, graph-based feature engineering, and ML-driven fraud detection on Brazilian PIX transactions.

[screenshot1](./.github/assets/screenshot1.png)
[screenshot2](./.github/assets/screenshot2.png)
[screenshot3](./.github/assets/screenshot3.png)

## Table of Contents

- [Features](#features)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Pipeline Stages](#pipeline-stages)
- [Configuration](#configuration)
- [Data and Models](#data-and-models)
- [Testing](#testing)
- [Roadmap](#roadmap)
- [Troubleshooting](#troubleshooting)

---

## Features

- **Data Generation**  
  • Realistic synthetic PIX accounts (CPF/CNPJ, device/IP history, risk scores)  
  • Multiple transaction patterns:  
    - Normal peer-to-peer (`NORMAL`)  
    - Salary payouts (`NORMAL_SALARY`)  
    - Micro-transactions (`NORMAL_MICRO`)  
    - B2B transfers (`NORMAL_B2B`)  
    - Smurfing rings (`SMURFING_VICTIM_TRANSFER`, `SMURFING_DISTRIBUTION`, `SMURFING_CASH_OUT`)  
    - Circular payment loops (`CIRCULAR_PAYMENT`)

- **Stream Simulation & Ingestion**  
  • Publish synthetic transactions to Redis Pub/Sub (`stream_simulator.py`)  
  • Ingest messages into a Neo4j graph (`ingestion_engine.py`)

- **Feature Engineering**  
  • Graph-based metrics: degrees, flow counts, circularity  
  • Transactional statistics: counts, sums, ratios, entropy  
  • Temporal patterns: weekend/night ratios, time-spread, velocity  
  • Device & channel diversity features  
  • Produces:
    - `./data/account_features.csv` (ML inputs)
    - `./data/evaluation_dataset.csv` (features + ground truth fraud flags)

- **Anomaly Detection**  
  • Global models (`anomaly_detection.py`):
    - Local Outlier Factor
    - Isolation Forest  
  • Community-aware LOF (`local_outlier_factor.py`):
    - Extract community features from Neo4j
    - Per-community LOF scoring
  • Community detection (`community_detection.py`):
    - Louvain algorithm on Neo4j MONEY_FLOW graph
    - Write `communityId` back to Neo4j
    - Save `./data/community_analysis.csv`

- **Evaluation & Dashboard**  
  • Model evaluation scripts (`model_evaluation.py`)  
  • Interactive Streamlit dashboard (`dashboard.py`)

---

## Prerequisites

- Python 3.8+  
- Redis Server  
- Neo4j 4.x (with Graph Data Science plugin)  
- `pip install -r requirements.txt`

---

## Quick Start

1. **Configure**  
   ```bash
   cp example_config.yaml config.yaml
   # Edit credentials in config.yaml
   ```

2. **Generate synthetic data**  
   ```bash
   python data_generator.py
   ```

3. **Start ingestion**  
   ```bash
   python ingestion_engine.py --csv ./data/pix_transactions.csv
   ```

   Ensure Neo4j is running and the GDS plugin is installed. The script will create the necessary graph structure and ingest the data.

4. **Stream transactions** (in a new shell)  
YOU CAN SKIP THIS IF YOU ARE NOT USING STREAMING
   ```bash
   python stream_simulator.py 
   ```

5. **Feature engineering**  
   ```bash
   python feature_engineering.py \
     --output-features ./data/account_features.csv \
     --output-eval    ./data/evaluation_dataset.csv
   ```

6. **Detect anomalies**  
   ```bash
   # Global LOF
   python anomaly_detection.py --algorithm lof --contamination 0.05

   # Global Isolation Forest
   python anomaly_detection.py --algorithm isolation_forest
   ```

7. **Community detection & LOF within communities**  
   ```bash
   python community_detection.py
   python local_outlier_factor.py
   ```

8. **Evaluate & review**  
   ```bash
   python model_evaluation.py
   streamlit run dashboard.py
   ```

---

## Pipeline Stages

1. **data_generator.py**  
   Build synthetic account and transaction CSV.

2. **stream_simulator.py**  
   Publish transactions to Redis channel.

3. **ingestion_engine.py**  
   Consume Redis stream into Neo4j graph.

4. **feature_engineering.py**  
   Query Neo4j, compute features, save CSVs.

5. **anomaly_detection.py**  
   Apply global LOF / Isolation Forest on features.

6. **community_detection.py**  
   Build NetworkX graph from Neo4j MONEY_FLOW, run Louvain, save results.

7. **local_outlier_factor.py**  
   Extract per-community features from Neo4j, apply LOF, save CSV.

8. **model_evaluation.py**  
   Load `community_analysis.csv`, `lof_analysis.csv`, `anomaly_scores.csv`, `evaluation_dataset.csv` and print performance metrics.

9. **dashboard.py**  
   Streamlit app for interactive visualization.

---

## Configuration

All credentials and parameters are managed in `config.yaml`. See `config.py` for how the settings are loaded into each script.

---

## Data and Models

- **Data directory**: `./data/`  
  - `transactions.csv`  
  - `account_features.csv`  
  - `evaluation_dataset.csv`  
  - `community_analysis.csv`  
  - `lof_analysis.csv`  
  - `anomaly_scores.csv`

- **Models directory**: `./models/`  
  - `isolation_forest_model.joblib`  
  - `scaler.joblib`  
  - `lof_scaler.joblib`

---

## Testing

Unit tests cover each stage in `tests/`. Mock external dependencies (Redis, Neo4j, filesystem). Run:

```bash
python -m unittest discover tests
```

---

## Roadmap

- Real-time Kafka integration  
- Automated retraining & model monitoring  
- Graph neural network detectors  
- Explainable AI / SHAP integration  
- Path-finding analysis in dashboard  

---

## Troubleshooting

- **Missing `./models/` directory**: create it before running anomaly detection.  
- **Neo4j connection**: check `config.yaml` URI/user/password and GDS plugin.  
- **Redis issues**: verify `redis-cli ping` returns `PONG`.  
- **Missing CSVs**: ensure feature engineering and community scripts ran successfully.
