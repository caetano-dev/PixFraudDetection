- [ ] maybe add more features to stability score, like hits, centrality.
- [ ] Run the large dataset and analyze results.
- [ ] for leiden: see how much fraud it finds by comparing the percentage of fraud in the communities with the percentage of fraud in the entire graph

Flaws found

Critical (Requires Immediate Action)

Epistemic Test Leakage (Lack of Validation Set)
Valid. If you modify the XGBoost parameters (max_depth, n_estimators, or scale_pos_weight) to improve the AUPRC on the test set, you are committing data leakage. You are manually fitting the model to the unseen data, which mathematically invalidates your 87% Precision@100 claim.
Action: Refactor 03_train_model.py to enforce a strictly chronological Train/Validation/Test split (e.g., 60/20/20). Train on the first block, tune parameters on the middle block, and report final metrics exclusively on the final block.

Instantaneous Label Maturity Assumption
Highly Valid. In applied financial forensics, confirmed fraud labels are never instantly available. They require manual investigator review, creating a lag of 30 to 90 days. Splitting the data so that the training set ends on Day 100 and the test set begins on Day 101 assumes instantaneous omnipotence.
Action: Introduce a "label delay buffer" (e.g., 7 or 14 days) between your train and test sets in 03_train_model.py where data is simply dropped. Presenting this in your defense proves deep domain expertise.

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
