# To-dos
- [ ] maybe add more features to stability score, like hits, centrality.
- [ ] Run the large dataset and analyze results.

## oddball paper
The trimmed-down set of features that are very successful in spotting patterns, are the following:
1. Ni: number of neighbors (degree) of ego i,
2. Ei: number of edges in egonet i,
3. Wi: total weight of egonet i,
4. λw,i: principal eigenvalue of the weighted adjacency matrix of egonet i

## Unreferenced Decisions (Empirical / Heuristic)

The following parameters and strategies found in the code have no explicit reference in the provided literature and are identified as empirical or heuristic choices:

    PageRank Adjusted Alpha: The specific damping factors of 0.95 for deep walks and 0.75 for shallow walks.

    Betweenness Pivot Nodes (k): The exact sampling sizes of k=50, 80, 500 used to approximate betweenness centrality for computational scalability.

    HITS Iteration Limits: Setting max_iter to 100, 500, 1000.

    Leiden Resolution Parameters: The exact values of 1.0/2.0 for macro and 2.0/5.0 for micro resolutions.

    Rank Stability Thresholds: The decision to cap the tracking set at exactly RANK_STABILITY_TOP_K=100 and the use of the 95th percentile (RANK_ANOMALY_PERCENTILE=95.0) to define rank anomalies.

    XGBoost Imbalance Weighting: The mathematical choice to use the square root of the imbalance ratio (np.sqrt(imbalance_ratio)) for scale_pos_weight rather than the raw ratio.

    SHAP TreeExplainer: While explainable AI is broadly supported, the exact deployment of shap.TreeExplainer on a capped 10,000-row sample is a system-specific optimization.

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

Longest laundering chain per dataset:

8, 12 and 69 days
