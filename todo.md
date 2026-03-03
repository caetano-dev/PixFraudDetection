# To-dos
- [ ] maybe add more features to stability score, like hits, centrality.
- [ ] Run the large dataset and analyze results.
- [ ] for leiden: see how much fraud it finds by comparing the percentage of fraud in the communities with the percentage of fraud in the entire graph

## Flaws found

Critical (Requires Immediate Action)

Epistemic Test Leakage (Lack of Validation Set)
Valid. If you modify the XGBoost parameters (max_depth, n_estimators, or scale_pos_weight) to improve the AUPRC on the test set, you are committing data leakage. You are manually fitting the model to the unseen data, which mathematically invalidates your 87% Precision@100 claim.
Action: Refactor 03_train_model.py to enforce a strictly chronological Train/Validation/Test split (e.g., 60/20/20). Train on the first block, tune parameters on the middle block, and report final metrics exclusively on the final block.

Instantaneous Label Maturity Assumption
Highly Valid. In applied financial forensics, confirmed fraud labels are never instantly available. They require manual investigator review, creating a lag of 30 to 90 days. Splitting the data so that the training set ends on Day 100 and the test set begins on Day 101 assumes instantaneous omnipotence.
Action: Introduce a "label delay buffer" (e.g., 7 or 14 days) between your train and test sets in 03_train_model.py where data is simply dropped. Presenting this in your defense proves deep domain expertise.

### To check: Does it make sense to add Go?

Your current Python structure (separated into `src/` modules and `scripts/`) is clean for an academic project, but the entire data ingestion and graph-building pipeline (`src/data/` and `src/graph/`) is an egregious bottleneck. Python and `pandas` are computationally expensive for iterating over sliding windows and aggregating edge weights row-by-row.

If you want a verifiable, high-performance portfolio piece that proves you can build backend infrastructure for fintech, you must rip out the data engineering layer from Python and rewrite it in Go.

Here is the mathematically correct polyglot architecture that guarantees your graduation while producing a specialized Go asset for your resume.

### The Boundary: Go Data Engine vs. Python ML Engine

You must split the system into two isolated executables communicating via a highly optimized serialized format (Apache Parquet or Protobuf).

#### Phase 1: The Go Data Engine (The High-Performance Portfolio Piece)

Go’s concurrency model (goroutines and channels) is mathematically superior for data ingestion and transformation. You will replace `scripts/01_filter_raw_data.py`, `src/data/loader.py`, `src/data/window_generator.py`, and `src/graph/builder.py` with a pure Go pipeline.

**The Go Responsibilities:**

1. **Concurrent Ingestion:** Read the raw `transactions.csv` and `accounts.csv`. Go handles I/O operations significantly faster than Python.
2. **Fractional Ownership Distribution:** Perform the relational merge and enforce the "Conservation of Mass" logic natively in Go.
3. **Sliding Window Generation:** Use Go channels to fan-out the sliding window calculations across multiple CPU cores. Instead of calculating one 3-day window at a time sequentially, Go can process multiple windows in parallel.
4. **Composite Edge Weight Calculation:** Replicate the Oddball-inspired weight formula natively. Calculate the standard deviation, mean, and frequency of transactions between nodes to output the final edge weights.
5. **Basic Node Topology:** Calculate the `in_degree`, `out_degree`, `vol_sent`, and `vol_recv`.

**The Output:** For each sliding window, Go serializes and exports two clean datasets (e.g., `window_2022-09-01_edges.parquet` and `window_2022-09-01_node_stats.parquet`).

#### Phase 2: The Python ML Engine (The Academic Requirement)

Go is structurally incapable of running the complex algorithms found in `src/features/`. You will keep `scripts/02_extract_features.py`, `centrality.py`, `community.py`, `stability.py`, and your future Graph Neural Networks in Python.

**The Python Responsibilities:**

1. **Graph Instantiation:** Python simply loads the pre-aggregated `.parquet` edge lists directly into `NetworkX` or `igraph`. Bypassing the raw aggregation in Python saves immense memory overhead.
2. **Advanced Algorithmic Execution:** Run PageRank, HITS, and Leiden community detection.
3. **Rank Stability & Evaluation:** Compute the Jaccard similarities, precision/recall metrics, and handle the final ML evaluation.

### The Execution Protocol

To implement this without jeopardizing your TCC deadline, sequence the work as follows:

1. **Do Not Touch Python Yet:** Keep your existing Python code functional as your baseline fallback.
2. **Build the Go Ingestor:** Write a Go application using a library like `github.com/xitongsys/parquet-go` to read the raw data and output the fractional ownership DataFrame.
3. **Implement the Go Windowing Engine:** Write the sliding window logic using goroutines. The goal is to output a directory full of pre-computed daily edge lists that exactly match the output of your current `src/graph/builder.py`.
4. **The Integration Test:** Modify your Python `scripts/02_extract_features.py` to skip `build_daily_graph()` and instead ingest the Go-generated Parquet files. If the evaluation metrics match your Python baseline, the integration is successful.


## oddball paper
The trimmed-down set of features that are very successful in spotting patterns, are the following:
1. Ni: number of neighbors (degree) of ego i,
2. Ei: number of edges in egonet i,
3. Wi: total weight of egonet i,
4. λw,i: principal eigenvalue of the weighted adjacency matrix of egonet i

```
..                                  SMALL           MEDIUM           LARGE
..                                  HI     LI        HI      LI       HI       LI
.. Date Range HI + LI (2022)         Sep 1-10         Sep 1-16        Aug 1 - Nov 5
.. # of Days Spanned                 10     10        16      16       97       97
.. # of Bank Accounts               515K   705K     2077K   2028K    2116K    2064K
.. # of Transactions                  5M     7M       32M     31M      180M    176M
.. # of Laundering Transactions     5.1K   4.0K       35K     16K      223K    100K
.. Laundering Rate (1 per N Trans)  981   1942       905    1948       807     1750
..                                  SMALL           MEDIUM           LARGE
```
