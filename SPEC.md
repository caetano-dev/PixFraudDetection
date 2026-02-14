# SPEC.md: Temporal Graph Feature Extraction for AML Detection

## 1. Project Overview
This project processes financial transaction data to extract temporal graph features for Anti-Money Laundering (AML) detection. The pipeline ingests the AMLworld synthetic dataset (`HI_Small` and `HI_Large`). It uses a sliding window approach to build daily directed graphs, extracts node-level topological features, and saves these features for downstream supervised classification.

**Strict Architectural Constraint:** This project relies exclusively on classical graph metrics (Centrality, HITS, Leiden) evaluated over temporal snapshots. Do not suggest, implement, or write code for K-Core decomposition, unsupervised Graph Neural Networks (GNNs), or deep learning embeddings. 

## 2. Tech Stack
* **Language:** Python 3.10+
* **Graph Processing:** `networkx` (Graph construction, PageRank, HITS)
* **Community Detection:** `cdlib` (Leiden algorithm)
* **Data Manipulation:** `pandas`, `numpy`
* **Evaluation & ML:** `scikit-learn` (ROC-AUC, Average Precision, Random Forest)
* **I/O:** `parquet` engine (fggor fast, typed storage of feature sets)

## 3. Core Graph Metrics & Theoretical Basis
The pipeline computes the following features for every node $i$ at time $t$:

* **Transactional Baselines:** `vol_sent`, `vol_recv`, `tx_count`.
* **PageRank Centrality:** Measures the global importance of a node in the flow of funds.
* **HITS (Hubs and Authorities):** Authorities represent targets where money pools (highly predictive for AML), while Hubs represent sources.
* **Leiden Communities:** Identifies dense subgraphs (potential fraud rings) while guaranteeing well-connected communities.
* **Rank Stability (Temporal Anomaly):** Tracks the derivative of a node's centrality between $t_{i}$ and $t_{i-1}$. Drastic rank shifts flag anomalous financial behavior.

## 4. Data Pipeline & Schema Transformations

The data undergoes three distinct state changes: Raw Ingestion, Graph Aggregation, and Feature Matrix Generation. 

### State 1: Raw Ingestion & Entity Mapping
The pipeline loads pre-filtered synthetic data from Parquet files. 
* **Input Files:** `1_filtered_normal_transactions.parquet`, `2_filtered_laundering_transactions.parquet`, `3_filtered_accounts.parquet`.
* **Transformation (`load_data`):** The data represents transfers between *bank accounts*, but AML detection must happen at the *entity* (person/company) level. 
* **Crucial Mapping:** The pipeline maps `from_account` and `to_account` to `source_entity` and `target_entity` using the accounts lookup table. Transactions with unmapped accounts are dropped.
* **Raw Transaction Schema Requirements:** * `timestamp` (datetime)
  * `source_entity` (string/int)
  * `target_entity` (string/int)
  * `amount_sent_c` (float)
  * `is_laundering` (int: 0 or 1)

### State 2: Temporal Graph Aggregation
Transactions are sliced into temporal windows of `WINDOW_DAYS` (e.g., 3 or 7 days), advancing by `STEP_SIZE` (1 day).
* **Nodes:** Represent unique entities (`source_entity` or `target_entity`) active within the window.
* **Edges (Directed):** A single edge is created between Node A and Node B if *any* transactions occurred between them in the window.
* **Edge Attributes (`build_daily_graph`):** Edges are currently aggregated with two attributes:
  * `weight`: The `sum` of `amount_sent_c` between the two entities.
  * `count`: The `count` of transactions between the two entities.

### State 3: The Output Feature Matrix
The algorithms (PageRank, HITS, Leiden) process the daily graph and output a flattened, tabular feature matrix saved as `sliding_window_features.parquet`. This is the dataset the final supervised classifier will train on.

**Target Schema for Downstream Classification (`results_df`):**
* `date`: The end date of the sliding window (Temporal index)
* `entity_id`: The node identifier (Primary key alongside date)
* **Graph Features:** * `pagerank` (float)
  * `hits_hub`, `hits_auth` (float)
  * `degree`, `in_degree`, `out_degree` (int)
  * `leiden_id` (int), `leiden_size` (int)
* **Transactional Features:** * `vol_sent`, `vol_recv` (float)
  * `tx_count` (int)
* **Temporal/Anomaly Features:**
  * `pagerank_rank_change` (int: change in rank from previous day)
  * `is_rank_anomaly` (int: 0 or 1, based on percentile shift)
* **Label:**
  * `is_fraud` (int: 0 or 1, strictly evaluated using `get_bad_actors_up_to_date` to prevent future data leakage).

## 5. Pending Implementation Tasks (Directives for AI Agent)
The current codebase has three critical flaws that must be addressed in the next iterations:

### Task A: Fix "Smurfing" Detection via Edge Weighting
Currently, `build_daily_graph` aggregates edges by summing the transaction amounts `weight=('amount_sent_c', 'sum')`. This masks "smurfing" (splitting large transactions into smaller ones). 
* **Action:** Update `build_daily_graph` to incorporate edge frequency (`count`) and variance into a composite edge weight, inspired by Oddball ego-network anomaly detection. 
* **Formula Target:** Implement a composite weight $W_{edge}$ that rewards high transaction counts alongside total volume, which must then be passed to `nx.pagerank`.

### Task B: Debug Leiden Community Drop-off
The Leiden algorithm currently returns community assignments for a statistically insignificant portion of the graph (e.g., 32 nodes out of thousands).
* **Action:** Debug `compute_leiden_features`. Ensure the `cdlib` conversion or the `G.to_undirected()` method assigns a `leiden_id` to *every* node in the graph, even if it is an isolated component of size 1.

### Task C: Implement Supervised Classifier Pipeline
The current pipeline terminates by saving `sliding_window_features.parquet`. A final classification step is required to output fraud probabilities.
* **Action:** Create a new script (`train_model.py`) that loads the parquet file, handles temporal train/test splits (e.g., train on days 1-70, test on days 71-97), and trains a supervised model (Random Forest or XGBoost) using the extracted graph features to predict `is_fraud`.