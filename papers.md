### 1. The "Get" List (The Foundations)

You must acquire and cite these to defend the fundamental algorithms you are using. You do not need to read them cover-to-cover; cite them when introducing the mathematical formulas.

* **The PageRank Paper:** *The Anatomy of a Large-Scale Hypertextual Web Search Engine* (Brin & Page, 1998).
* **Why:** You are computing `nx.pagerank`. This is the source material.


* **The HITS Paper:** *Authoritative Sources in a Hyperlinked Environment* (Kleinberg, 1999).
* **Why:** You are computing `nx.hits`. This is the source material.


* **A Standard Graph Theory Textbook:** e.g., *Networks* (Newman, 2018).
* **Why:** To define basic concepts like nodes, edges, directed graphs, and degree centrality, which you compute in `compute_node_stats`.



### 2. The Core Keepers (Your Defense Shields)

These papers directly validate the specific architecture choices and datasets in your codebase.

* **`realistic-synthetic-financial-transactions-for-anti-money-laundering-models-Paper-Datasets_and_Benchmarks.pdf`**
* **Why:** This is the AMLSim paper. It generated the `HI_Small` and `HI_Large` datasets you defined in `config.py`. It explicitly defines the laundering topologies (cycles, bipartite, etc.) your pipeline is trying to detect.


* **`From Louvain to Leiden guaranteeing well-connected communities.pdf`**
* **Why:** Defends your use of `algorithms.leiden` in `utils.py` instead of older methods. It proves Leiden prevents internally disconnected communities.


* **`OddBall_Spotting_Anomalies_in_Weighted_Graphs.pdf`**
* **Why:** Contains the exact mathematical framework to solve the "Smurfing" problem listed in your `README.md`. It provides the formula to balance edge counts vs. edge weights in ego-networks.


* **`Weirdnodes centrality based anomaly detection on temporal networks for the antifinancial crime domai.pdf`**
* **Why:** Validates your entire sliding window architecture. It proves that calculating centrality metrics across temporal network snapshots is a valid method for detecting financial crime, directly defending your `STEP_SIZE` and `WINDOW_DAYS` constants.



### 3. The Reinforcements (Domain Context & Future Work)

Keep these to flesh out your introduction, problem statement, and conclusion.

* **`Detecting_Suspicious_Customers_in_Money_Laundering_Activities_Using_Weighted_HITS_Algorithm.pdf`**
* **Why:** Your logs show HITS Authority gives a 7.16x lift. This paper explains *why* HITS works for AML (authorities = money receivers).


* **`USING GRAPH CENTRALITY METRICS FOR DETECTION OF SUSPICIOUS TRANSACTIONS.pdf`**
* **Why:** General backup reinforcing that centrality metrics (PageRank, Degree) flag suspicious accounts.


* **`Anomaly_Detection_in_Graphs_of_Bank_Transactions_for_Anti_Money_Laundering_Applications.pdf`** & **`Complex networks-based anomaly detection for financial transactions in.pdf`**
* **Why:** Use these in your introduction to explain why rule-based AML fails and why graph analysis is required.


* **`WIREs Computational Stats - 2025 - Deprez - Advances in Continual Graph Learning for Anti‐Money Laundering Systems A.pdf`**
* **Why:** Your `README.md` notes a scaling problem with large datasets. Cite this 2025 paper in your "Future Work" section to prove you know the industry solution is Continual Graph Learning, even if you couldn't build it for this TCC.


* **`Comparative Analysis Using Supervised Learning Methods for AntiMoney Laundering in Bitcoin.pdf`**
* **Why:** You will eventually need to feed your `sliding_window_features.parquet` into a Random Forest/XGBoost model. This bridges graph features to supervised classification.



### 4. The Discards (Delete Immediately)

These will cause scope creep, distract you, or do not map to your codebase.

* **`Are k-cores meaningful for temporal graph analysis?.pdf`** (You are not computing k-cores.)
* **`CoreScope_Graph_Mining_Using_k-Core_Analysis__Patterns_Anomalies_and_Algorithms.pdf`** (You are not computing k-cores.)
* **`Patterns and anomalies in k-cores of real-world graphs.pdf`** (You are not computing k-cores.)
* **`Evaluation of Community Detection Methods.pdf`** (Redundant. The Leiden paper is sufficient.)
* **`The_Effectiveness_of_Edge_Centrality_Measures_for_Anomaly_Detection.pdf`** (You are computing *node* centrality, not *edge* centrality.)
* **`NeurIPS-2022-bond-benchmarking-unsupervised-outlier-node-detection-on-static-attributed-graphs...`** (Your graph is temporal and supervised, not static and unsupervised.)
* **`MACHINE LEARNING-BASED UNSUPERVISED ENSEMBLE APPROACH FOR DETECTING NEW MONEY LAUNDERING TYPOLOGIES IN TRANSACTION GRAPHS.pdf`** (You are building a supervised pipeline. Distraction.)
* **`SPACEGNN MULTI-SPACE GRAPH NEURAL NETWORK FOR NODE ANOMALY DETECTION.pdf`** (Too complex. Distraction.)
* **`Scalable Graph Learning for AML.pdf`** & **`Scalable_Semi-Supervised_Graph_Learning_Techniques_for_Anti_Money_Laundering.pdf`** & **`Graph neural networks for financial fraud detection.pdf`** (You already have the 2025 WIREs paper for your GNN/Future Work citation. You don't need three more.)

---

Would you like me to translate the Oddball paper's formula into Python so you can check off the "Smurfing" fix in your `README.md`?