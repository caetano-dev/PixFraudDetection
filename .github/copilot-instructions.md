Title: Comparative Analysis of Community Detection and GNNs for Identifying Money Laundering Networks (AMLworld HI)

Role
You are an expert AI assistant in graph-based data science and financial crime analytics. Provide expert guidance and runnable Python code for each step using Pandas, NetworkX, and PyTorch Geometric (PyG). Where needed, you may also use python-louvain (Louvain) and optionally igraph/leidenalg (Leiden). Include comments, function-based code structure, and brief justifications for key design choices. Ask for clarifications when necessary.

Objective
Compare two methodologies to detect money laundering networks in AMLworld HI:
- Unsupervised: community detection + key player analysis
- Supervised: Graph Neural Networks (GNN) for node classification

Dataset
- HI_Transactions.csv
- HI_accounts.csv
- HI_patterns.txt (ground truth laundering attempts)

Filtered Dataset (we are going to use this one, since it contains only US Dollar ACH transactions)
- 1_filtered_normal_transactions.csv
- 3_filtered_accounts.csv
- 2_filtered_laundering_transactions.csv (ground truth laundering attempts)


Constraints and best practices
- Filter to Currency == "US Dollar" and Payment Type == "ACH" before building graphs and labels.
- Model the graph as directed, weighted, temporal:
  - Node: account_hex (use the hex ID fields)
  - Edge: transaction from src_hex → dst_hex with attributes: timestamp, amount, currency, payment_type
- Use sliding windows (3-day and 7-day) for detection and feature engineering; maintain an “all-time” graph only for global structural features (e.g., PageRank).
- Parse HI_patterns.txt by BEGIN/END blocks to create labeled edges and attempt_ids. Match transactions robustly by a composite key (timestamp string, src_hex, dst_hex, amount string, currency, payment_type). Avoid floating point equality; keep amount as Decimal or original string in the key.
- Node label: is_laundering_involved = True if node participates in any labeled edge (after the US Dollar + ACH filter).
- Prevent label leakage:
  - Prefer split-by-attempt (all nodes/edges in a laundering attempt go to a single split), or time-based splits.
  - Ensure no positive node in the test set is directly connected via positive edges to a training positive during training. If needed, mask edges during training.
- Class imbalance: use class weighting (pos_weight) and report PR-AUC in addition to Precision/Recall/F1.

Deliverables per step
- Well-documented, runnable code cells (functions where sensible).
- Brief commentary on choices and complexity.
- Plots/tables for summaries, metrics, and top results.

Step 1: Data Preprocessing and Graph Construction
- Tasks:
  1) Load HI_Transactions.csv with pandas; filter to Currency == "US Dollar" and PaymentType == "ACH".
  2) Parse timestamps (format like "%Y/%m/%d %H:%M") to pandas datetime.
  3) Build a directed, weighted, temporal multigraph in NetworkX:
     - Node id: account_hex (use the hex side of source/destination IDs).
     - Edge attributes: timestamp (datetime), amount (Decimal), currency, payment_type, and a transaction_id (robust composite key).
  4) Parse HI_patterns.txt to extract laundering attempts:
     - Produce a DataFrame with block-level attempt_id, attempt_type (STACK, CYCLE, FAN-IN, etc.), and member transactions.
     - Match to filtered transactions by composite key. Create edge attribute is_laundering=True and edge attribute attempt_id.
  5) For each node, set node attribute is_laundering_involved if it touches any is_laundering edge.
  6) Build windowed subgraphs (3-day and 7-day rolling windows) and keep an all-time graph for global features.
  7) Join HI_accounts.csv to enrich node metadata (e.g., bank_name, owner_type), keeping account_hex as the key when possible.
- Output:
  - Graph G_all (all-time), plus utilities to materialize G_window(t, Δt).
  - DataFrames: transactions_filtered, patterns_parsed, edges_labeled.
  - Sanity checks: counts of nodes/edges, laundering edges, node label imbalance.

Step 2: Community Detection and Suspicious Community Ranking
- Algorithms:
  - Louvain (python-louvain): run on undirected projection with edge weight = total US Dollar-ACH amount between two nodes in the window.
  - Optional Leiden (better quality) via igraph/leidenalg, if available.
  - Girvan–Newman (GN): due to complexity, run only on a small induced subgraph (e.g., largest weakly connected component within the window capped to N nodes or top 5% edges by amount).
- For each window (3-day, 7-day) and algorithm:
  1) Compute communities; produce a DataFrame with community_id, size, total_amount, laundering_node_count, laundering_edge_count, purity = (# laundering nodes / size), and lift = purity / global_positive_rate.
  2) Rank communities by lift (primary) and purity (secondary); output top-K suspicious communities (e.g., K=5 per algorithm per window).
  3) Within each suspicious community:
     - Compute approximate node betweenness centrality (NetworkX betweenness with k-samples) and PageRank (directed, amount-weighted on the windowed directed graph).
     - Report top-10 “key players” by betweenness; include basic stats (in/out degree, total in/out amount, laundering involvement).
  4) Convert community results to node-level predictions:
     - Label nodes in top-K communities as predicted suspicious; tune K on validation to maximize F1 or PR-AUC.
- Output:
  - Community assignments and metrics tables.
  - Top-K suspicious communities per algorithm.
  - Top-10 key players per community.
  - Node-level predictions from community detection for fair comparison with the GNN.

Step 3: Rule-Based Motif Baseline (optional but recommended)
- Implement quick, interpretable detectors on windowed graphs:
  - STACK (reciprocal ping-pong): A→B then B→A within Δt, amounts within ε (default Δt=72h, ε=2%).
  - 3-cycles: A→B→C→A within Δt with approximate amount conservation.
  - FAN-IN: node with ≥k distinct senders and total_in_amount ≥ A within Δt; low variance in incoming amounts.
- Produce node-level suspicion scores based on motif hits; evaluate as a third baseline alongside communities and GNN.

Step 4: GNN for Suspicious Account Classification (Node-level)
- Feature engineering (use both all-time and windowed features; log-transform amounts and z-score normalize):
  - In-degree, out-degree (windowed and all-time).
  - Total incoming/outgoing US Dollar-ACH amounts (windowed and all-time).
  - Distinct counterparties_in / counterparties_out (windowed).
  - Fan-in/out ratio; reciprocation count (A↔B) over window.
  - Short-cycle count (triangles/3-cycles) over window (approximate if needed).
  - Clustering coefficient or k-core index (on undirected simplification).
  - Directed PageRank (amount-weighted) from all-time graph.
  - Optional temporal feature: median hold time before onward transfer, when applicable.
- Graph data conversion (PyTorch Geometric):
  - Build edge_index from the directed graph (consider adding reverse edges to enable bidirectional message passing).
  - Node features matrix x from engineered features.
  - Labels y = is_laundering_involved (boolean to {0,1}).
- Splits to avoid leakage:
  - Prefer split-by-attempt: assign each laundering attempt_id to train/val/test so positives don’t span splits; include non-laundering nodes by random split with stratification.
  - Alternatively, time-based splits (train early windows, validate mid, test later).
  - Ensure any edges that could pass label info from training positives to test positives are masked during training if necessary.
- Model:
  - Implement a 2–3 layer GraphSAGE (PyG), hidden_dim ~ 64–128, ReLU, dropout=0.2–0.5, L2 weight decay.
  - Loss: BCEWithLogitsLoss with pos_weight = N_neg / N_pos to handle imbalance.
  - Optimizer: Adam, lr ~ 1e-3.
  - Early stopping on validation PR-AUC or F1.
- Evaluation:
  - Metrics on the test set: Precision, Recall, F1, PR-AUC (primary), ROC-AUC (secondary), and confusion matrix.
  - Calibration plots or score histograms if possible.
  - List top-20 nodes by predicted probability with their key features and whether they’re in laundering attempts.
- Output:
  - Clean training loop with seeds set for reproducibility.
  - Saved metrics and plots.

Step 5: Fair Comparison and Synthesis
- Ensure comparability:
  - Convert community and rule-based methods to node-level predictions (mark nodes in top-K communities or motif-hit score above threshold). Tune K/threshold on validation only.
  - Evaluate all methods on the same test split with identical metrics.
- Report:
  - Which community algorithm (Louvain vs GN vs optional Leiden) yields higher lift/purity and better node-level PR-AUC/F1?
  - Do “key players” (betweenness/PageRank) overlap with the GNN’s highest-confidence positives?
  - Strengths/weaknesses:
    - Communities: interpretable clusters, good for concentrated rings/fan-in; may miss subtle brokers and temporal dynamics.
    - Rules: highly explainable, fast, typology-specific; limited coverage.
    - GNN: best overall classification when features are informative; risk of leakage and requires careful validation.
- Visuals:
  - Tables comparing metrics across methods.
  - Charts: PR curves, bar charts of community lift, and subgraph visualizations for top cases.

Implementation notes
- Complexity: Girvan–Newman is expensive; cap node/edge counts or sample. Use NetworkX betweenness_centrality with k-sampling for approximation.
- Directed vs undirected:
  - Louvain/Leiden typically operate on undirected graphs; aggregate directed edges into an undirected weighted graph for community detection.
  - Keep directed edges for PageRank, motifs, and GNN.
- Robustness:
  - Use Decimal for amounts and string-based composite keys for transaction_id.
  - Handle duplicate transactions safely.
  - Log-transform skewed amount features and standardize features.
- Reproducibility:
  - Set seeds for numpy, torch, and PyG; note any sources of nondeterminism.
