# PIX Fraud Detector
To-do:

- [ ] optimize sliding window to add edges instead of recomputing everything
The Issue: On Day 100, you are re-loading and re-aggregating Day 1 to Day 99 again. window_df grows larger every iteration. On a large dataset, Day 90+ will take exponentially longer to process.
- [ ] add more features to stability score, like hits, centrality.
- [ ] Run the large dataset and analyze results.
- [ ] polish code

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


## Quick Start

### Setup

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Unix/macOS
# or: source venv/bin/activate.fish  # For fish shell
# or: venv\Scripts\activate  # On Windows

# Install dependencies
pip install -r requirements.txt
```

Put the HI_Large.csv and LI_Small.csv files in the `data/HI_Large` and `data/HI_Small` directories.

Run clean_dataset.py to remove the laundering transactions we don't need.


Edge Feature Aggregation: In build_daily_graph, you sum the weights (amount_sent_c).

    Flaw: This masks "Structuring" or "Smurfing" (breaking one large transaction into many small ones). A single $10k transfer looks the same as ten $1k transfers in your current graph.

    Fix: In build_daily_graph, add an edge attribute for transaction_count (which you already have) and potentially variance or max_amount. High count + low variance is a strong signal of smurfing.
