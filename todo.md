- [ ] optimize sliding window to add edges instead of recomputing everything
The Issue: On Day 100, you are re-loading and re-aggregating Day 1 to Day 99 again. window_df grows larger every iteration. On a large dataset, Day 90+ will take exponentially longer to process.
- [ ] add more features to stability score, like hits, centrality.
- [ ] Run the large dataset and analyze results.
- [ ] polish code
- [ ] solve edge aggregation problem (read down in the readme)
- [ ] for leiden: see how much fraud it finds by comparing the percentage of fraud in the communities with the percentage of fraud in the entire graph

The "Pro" Solution (Optional): A truly optimized cumulative graph would update G incrementally (add today's edges to yesterday's graph) rather than rebuilding from scratch.

(oddball paper)
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
Longest laundering chain found: 7 days (filtered HI large)


## Problems found 

Edge Feature Aggregation: In build_daily_graph, you sum the weights (amount_sent_c).

    Flaw: This masks "Structuring" or "Smurfing" (breaking one large transaction into many small ones). A single $10k transfer looks the same as ten $1k transfers in your current graph.

    Fix: In build_daily_graph, add an edge attribute for transaction_count (which you already have) and potentially variance or max_amount. High count + low variance is a strong signal of smurfing.
  
feedback:

It is **partially solved**, but in a way that is arguably still incomplete for your actual goal.

Here is the breakdown of what the code is currently doing versus what the README asked for:

### 1. The "Masking" Issue (Solved ✅)

The README correctly identified that summing weights hides "Smurfing" (splitting $10k into ten $1k transfers).
In `utils.py`, the `build_daily_graph` function **has been updated** to fix this. It now calculates `count` alongside the `weight` (sum) and adds it to the edge attributes.

```python
# From utils.py
edge_aggregation = window_df.groupby(['source_entity', 'target_entity']).agg(
    weight=('amount_sent_c', 'sum'),
    count=('amount_sent_c', 'count') # <--- This was added!
).reset_index()

```

This means your graph `G` now technically *contains* the data to distinguish a single large transfer from many small ones.

### 2. The "Utilization" Issue (Unsolved ❌)

While the data is now in the graph, your algorithms in `main.py` are **ignoring it**.

* **PageRank:** is called with `weight='weight'`. It is looking at the total volume ($10k) and ignoring that it came from 50 tiny transactions.
* **Leiden:** is also called with `weights='weight'`.
* **Missing Stats:** The README also suggested adding `variance` or `max_amount` to the edges, which has **not** been implemented in `utils.py` yet.

**Recommendation:**
To actually *solve* this, you would need to change how you call PageRank in `main.py`, perhaps by creating a synthetic weight that combines volume and count, or by running a second pass of PageRank using `weight='count'`.

---
---
---

### 1. Fix "Smurfing" Detection (Edge Aggregation)

* **The Issue:** `main.py` currently calculates PageRank using `weight='weight'` (total volume). This ignores the `count` attribute added in `utils.py`.
* **The Fix:**
* Create a "Smurfing Score" edge weight that rewards high frequency + low variance.
* Example formula for new weight: `edge_weight = count * (1 / (variance + epsilon))`.
* Pass this new weight key to the `nx.pagerank` function call in `main.py`.



### 2. Debug Leiden Community "Drop-off"

* **The Issue:** The logs show `Total nodes with community assignment: 32` on the final day, but the graph has thousands of nodes.
* **Why it matters:** The community detection is currently failing for 99.9% of the graph, which makes the "Fraud Ring" analysis statistically irrelevant.
* **Likely Cause:** The `cdlib` implementation might be failing on disconnected components or singleton nodes, or the `G.to_undirected()` conversion in `utils.py` is behaving unexpectedly.
* **The Fix:** Ensure every node in `G` gets a `leiden_id`, even if it is a singleton community of size 1.

### 3. Optimize HITS (or Drop "Hubs")

* **The Issue:** HITS **Authority** is performing great (Lift: ~7.16x), identifying accounts that *receive* money. However, HITS **Hub** is performing worse than random guessing (Lift: ~0.2x).
* **Why:** In money laundering, "Hubs" (senders) look like normal people paying bills. "Authorities" (receivers) look like launderers collecting funds.
* **The Fix:**
* Either drop `HITS_HUB` from the metrics to save compute time.
* Or invert the logic: treats high Hub scores as "safe" nodes to reduce false positives.



### 4. Sliding Window Performance (for LARGE dataset)

* **The Issue:** Currently, `main.py` filters the entire dataframe (`all_transactions.loc[...]`) inside the `while` loop.
* **Why it matters:** On the LARGE dataset (180M rows), doing this filter operation 100+ times will cause the script to hang or crash (O(N*Days) complexity).
* **The Fix:** Implement an "Incremental Graph" approach.
* *Add* today's edges to `G`.
* *Remove* edges older than `WINDOW_DAYS`.
* Avoid rebuilding the dataframe/graph from scratch every step.



### 5. "Supervised" Feature Store

* **The Issue:** The current pipeline saves raw features (`pagerank`, `degree`).
* **The Fix:** To technically "finish" the TCC, you usually need a final step that trains a classifier (like Random Forest or XGBoost) on these features to output a final `fraud_probability`.
* **Next Step:** Create `train_model.py` that loads `sliding_window_features.parquet` and runs a simple Random Forest.

---

