### Phase 1: Temporal Data Architecture

The pipeline will use a **Discrete Sliding Window** to process the AMLworld dataset.

1. **Data Ingestion:** Load the transactions from `1_filtered_normal_transactions.parquet` and `2_filtered_laundering_transactions.parquet` exactly as they are. Maintain all transaction types and currencies to preserve the global graph topology.
2. **Window Slicing:** Group transactions into temporal windows defined by `WINDOW_DAYS` (e.g., 3 days or 7 days), advancing the window chronologically by `STEP_SIZE` (1 day).
3. **Graph Construction:** Build a directed graph (`nx.DiGraph`) for each window.
4. **Edge Aggregation:** For any two nodes $A$ and $B$, aggregate their transactions within the window into two distinct edge attributes:
* `weight`: The sum of transaction amounts.
* `count`: The total number of transactions.



### Phase 2: Feature Extraction & Parameterization

For every node in each temporal window, compute the following features:

1. **PageRank (Volume Signal):** Execute `nx.pagerank` using `weight='weight'` and `alpha=0.85`. This identifies the global "sinks" where large volumes of money accumulate.
2. **PageRank (Frequency Signal):** Execute a second `nx.pagerank` using `weight='count'` and `alpha=0.85`. This specifically targets "Smurfing" (Structuring) behavior where illicit actors use high-frequency, low-value transfers.
3. **HITS (Hubs & Authorities):** Execute `nx.hits` with `max_iter=100`. Extract both the Hub score (money distributors) and Authority score (money receivers).
4. **Leiden Community Detection:** Convert the daily graph to an undirected format and execute `algorithms.leiden`. Run this with `resolution=1.0` and `resolution=2.0`. The higher resolution forces the algorithm to identify the tight, micro-clusters characteristic of localized laundering rings. Record the `leiden_id` and the `leiden_size`.
5. **Flow Topology:** Calculate basic node characteristics:
* `in_degree` and `out_degree`.
* `vol_sent` and `vol_recv`.
* Compute the ratio: `vol_sent / (vol_recv + 1)`.


6. **Rank Stability (Temporal Derivative):** Calculate the $\Delta$ of the PageRank Volume score between Day $T$ and Day $T-1$. Flag nodes that exhibit a rank change in the 95th percentile as rank anomalies.

Store all extracted features in a flattened tabular format (`sliding_window_features.parquet`).

### Phase 3: Machine Learning Classification

Create a separate script (`train_model.py`) to ingest the extracted feature matrix and classify the nodes.

1. **Chronological Split:** Sort the dataset by the `date` column. Assign the first 70% of the timeline to the training set and the final 30% to the testing set.
2. **Algorithm Selection:** Train an **XGBoost** classifier. Financial graph features mix non-linear distributions, extreme outliers (e.g., massive PageRank spikes), and categorical data (Community IDs), making gradient-boosted trees the optimal mathematical fit.
3. **Class Imbalance Handling:** Do not synthesize topological data. Set the XGBoost `scale_pos_weight` parameter to the exact ratio of legitimate nodes to laundering nodes in the training set (e.g., 99.0 if fraud represents 1%).
4. **Performance Metrics:** Evaluate the model strictly on the test set using:
* **Precision@K:** Calculate the precision for the top 10, 50, and 100 highest-probability alerts.
* **Area Under the Precision-Recall Curve (AUPRC).**
* **ROC-AUC.**



### Phase 4: Comparative Evaluation (The Defense)

To fulfill the objective of comparing algorithm performance, extract the **SHAP (SHapley Additive exPlanations)** values from the trained XGBoost model.

1. **Compute SHAP Values:** Pass the test set through the SHAP TreeExplainer.
2. **Feature Importance Ranking:** Generate a SHAP summary plot. This provides absolute mathematical proof of which graph algorithm contributed the highest predictive power to the model.
3. **Conclusion Structuring:** Formulate the final thesis conclusions based directly on the SHAP hierarchy (e.g., quantifying the exact informational gain of the `pagerank_count` metric versus the standard `degree` metric).
