- [ ] maybe add more features to stability score, like hits, centrality.
- [ ] Run the large dataset and analyze results.
- [ ] for leiden: see how much fraud it finds by comparing the percentage of fraud in the communities with the percentage of fraud in the entire graph

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