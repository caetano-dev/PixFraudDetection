"""
Utility functions for the Sliding Window Graph Pipeline.

This module provides data loading, graph construction, and evaluation utilities
for the money laundering detection feature generation pipeline.
"""

from __future__ import annotations

import warnings
import pandas as pd
import numpy as np
import networkx as nx
from sklearn.metrics import roc_auc_score, average_precision_score
from typing import Optional
from cdlib import algorithms
from datetime import datetime

from config import (
    DATA_PATH,
    NORMAL_TRANSACTIONS_FILE,
    LAUNDERING_TRANSACTIONS_FILE,
    ACCOUNTS_FILE,
    LEIDEN_RESOLUTION,
)


def load_data() -> tuple[pd.DataFrame, set]:
    """
    Load and preprocess all transaction data from the configured DATA_PATH.
    
    This function:
    1. Loads the three parquet files (normal, laundering, accounts)
    2. Merges transactions and maps accounts to entities
    3. Identifies all bad actors (entities involved in laundering)
    
    Returns:
        tuple containing:
            - all_transactions (pd.DataFrame): Merged transaction data with
              source_entity and target_entity columns
            - bad_actors (set): Set of entity IDs involved in any laundering
              transaction (global ground truth labels)
    """
    # Build file paths
    normal_path = DATA_PATH / NORMAL_TRANSACTIONS_FILE
    laundering_path = DATA_PATH / LAUNDERING_TRANSACTIONS_FILE
    accounts_path = DATA_PATH / ACCOUNTS_FILE
    
    # Load parquet files
    print(f"Loading data from {DATA_PATH}...")
    normal_transactions = pd.read_parquet(normal_path)
    laundering_transactions = pd.read_parquet(laundering_path)
    accounts = pd.read_parquet(accounts_path)
    
    # Concatenate all transactions
    all_transactions = pd.concat(
        [normal_transactions, laundering_transactions], 
        ignore_index=True
    )
    
    # Convert timestamp to datetime
    all_transactions['timestamp'] = pd.to_datetime(all_transactions['timestamp'])
    
    # Create account to entity mapping
    account_to_entity = dict(accounts.set_index('Account Number')['Entity ID'])
    
    # Map accounts to entities using .loc to avoid type issues
    all_transactions = all_transactions.copy()
    all_transactions.loc[:, 'source_entity'] = all_transactions['from_account'].map(account_to_entity)
    all_transactions.loc[:, 'target_entity'] = all_transactions['to_account'].map(account_to_entity)
    
    # Drop transactions with unmapped accounts
    initial_count = len(all_transactions)
    all_transactions = all_transactions.dropna(subset=['source_entity', 'target_entity'])
    dropped_count = initial_count - len(all_transactions)
    
    if dropped_count > 0:
        print(f"Dropped {dropped_count:,} transactions with unmapped accounts")
    
    # Identify bad actors (global ground truth for final evaluation only)
    # NOTE: For time-sliced evaluation without future leakage, use get_bad_actors_up_to_date()
    laundering_txns = all_transactions[all_transactions['is_laundering'] == 1]
    bad_actors_global = set(laundering_txns['source_entity']).union(
        set(laundering_txns['target_entity'])
    )
    
    print(f"Loaded {len(all_transactions):,} transactions")
    print(f"Identified {len(bad_actors_global):,} bad actors globally (entities in laundering transactions)")
    
    return all_transactions, bad_actors_global


def get_bad_actors_up_to_date(
    all_transactions: pd.DataFrame, 
    current_date: datetime
) -> set:
    """
    Get bad actors known up to a specific date (prevents future information leakage).
    
    This function returns only entities that have been involved in laundering
    transactions UP TO (and including) the current_date. This is critical for
    proper temporal evaluation - when evaluating at time t, we should only
    know about fraudulent activity that has occurred by time t.
    
    Args:
        all_transactions: DataFrame with all transactions (must have 'timestamp',
                         'is_laundering', 'source_entity', 'target_entity' columns)
        current_date: The cutoff date (inclusive) for identifying bad actors
        
    Returns:
        Set of entity IDs involved in laundering transactions up to current_date
    """
    # Filter transactions up to (and including) current_date
    mask = all_transactions['timestamp'] <= current_date
    transactions_up_to_date = all_transactions.loc[mask]
    
    # Identify bad actors from only these transactions
    laundering_txns = transactions_up_to_date[transactions_up_to_date['is_laundering'] == 1]
    bad_actors = set(laundering_txns['source_entity']).union(
        set(laundering_txns['target_entity'])
    )
    
    return bad_actors


def build_daily_graph(window_df: pd.DataFrame) -> nx.DiGraph:
    """
    Build a directed graph from a filtered DataFrame of transactions.
    
    Aggregates edges by (source, target) pairs and computes a composite
    edge weight inspired by Oddball ego-network anomaly detection. The
    composite weight rewards high transaction frequency alongside total
    volume and incorporates amount variance to surface "smurfing" patterns
    (many small, uniform transactions used to evade detection thresholds).
    
    Composite weight formula:
        W_edge = volume * log2(1 + count) * (1 + 1 / (1 + CV))
    
    where:
        - volume  = sum of amount_sent_c on the edge
        - count   = number of transactions on the edge
        - CV      = coefficient of variation (std / mean) of amounts
    
    Behaviour:
        - High count amplifies weight (log2 term), penalising smurfing.
        - Low CV (uniform amounts, classic smurfing) further amplifies
          weight via the variance factor approaching 2.
        - For a single transaction (count=1, std=0): factor = 1 * 2 = 2,
          acting as a neutral baseline since PageRank uses relative weights.
    
    Args:
        window_df: DataFrame containing transactions for the current window.
                   Must have columns: source_entity, target_entity, amount_sent_c
    
    Returns:
        nx.DiGraph with edge attributes:
            - 'weight': composite Oddball-inspired weight (used by PageRank)
            - 'volume': raw sum of transaction amounts
            - 'count':  number of transactions on the edge
            - 'amount_std': standard deviation of transaction amounts
    """
    G = nx.DiGraph()
    
    # Handle empty DataFrame
    if window_df.empty:
        return G
    
    # Aggregate edges: group by (source, target) and compute volume, count, std
    edge_aggregation = window_df.groupby(
        ['source_entity', 'target_entity']
    ).agg(
        volume=('amount_sent_c', 'sum'),
        count=('amount_sent_c', 'count'),
        amount_std=('amount_sent_c', 'std'),
    ).reset_index()
    
    # std returns NaN for groups with a single observation; fill with 0
    edge_aggregation['amount_std'] = edge_aggregation['amount_std'].fillna(0.0)
    
    # Compute composite weight
    # mean amount per edge (avoid division by zero; count >= 1 guaranteed)
    mean_amount = edge_aggregation['volume'] / edge_aggregation['count']
    
    # Coefficient of variation (std / mean); use epsilon to avoid 0/0
    epsilon = 1e-9
    cv = edge_aggregation['amount_std'] / (mean_amount + epsilon)
    
    # Composite weight: volume * log2(1+count) * (1 + 1/(1+CV))
    #   - log2(1+count) rewards high frequency (smurfing amplifier)
    #   - (1 + 1/(1+CV)) ranges from ~1 (high variance) to 2 (uniform amounts)
    edge_aggregation['weight'] = (
        edge_aggregation['volume']
        * np.log2(1 + edge_aggregation['count'])
        * (1 + 1.0 / (1.0 + cv))
    )
    
    # Add edges to graph
    for _, row in edge_aggregation.iterrows():
        G.add_edge(
            row['source_entity'],
            row['target_entity'],
            weight=row['weight'],
            volume=row['volume'],
            count=row['count'],
            amount_std=row['amount_std'],
        )
    
    return G


def compute_node_stats(window_df: pd.DataFrame) -> dict:
    """
    Compute basic transactional statistics for each node in the window.
    
    Calculates volume sent, volume received, and transaction counts
    for each entity based on the transactions in the current window.
    
    Args:
        window_df: DataFrame containing transactions for the current window.
                   Must have columns: source_entity, target_entity, amount_sent_c
    
    Returns:
        Dictionary mapping entity_id to a dict with:
            - 'vol_sent': (float) Total amount sent by this entity
            - 'vol_recv': (float) Total amount received by this entity
            - 'tx_count': (int) Total number of transactions (sent + received)
    """
    if window_df.empty:
        return {}
    
    # Calculate sent statistics (entity as source)
    sent_stats = window_df.groupby('source_entity').agg(vol_sent=('amount_sent_c', 'sum'), count_sent=('amount_sent_c', 'count'))
    
    # Calculate received statistics (entity as target)
    recv_stats = window_df.groupby('target_entity').agg(vol_recv=('amount_sent_c', 'sum'), count_recv=('amount_sent_c', 'count'))
    
    # Get all unique entities
    all_entities = set(sent_stats.index) | set(recv_stats.index)
    
    # Build the result dictionary
    node_stats = {}
    for entity in all_entities:
        vol_sent = float(sent_stats.loc[entity, 'vol_sent']) if entity in sent_stats.index else 0.0
        vol_recv = float(recv_stats.loc[entity, 'vol_recv']) if entity in recv_stats.index else 0.0
        count_sent = int(sent_stats.loc[entity, 'count_sent']) if entity in sent_stats.index else 0
        count_recv = int(recv_stats.loc[entity, 'count_recv']) if entity in recv_stats.index else 0
        
        node_stats[entity] = {
            'vol_sent': vol_sent,
            'vol_recv': vol_recv,
            'tx_count': count_sent + count_recv,
        }
    
    return node_stats


def _run_leiden_on_graph(G_undirected: nx.Graph) -> list[list]:
    """
    Run the Leiden algorithm on an undirected graph and return the raw
    community partition as a list of node-lists.

    This helper isolates the cdlib call so that fallback strategies can
    retry on individual connected components when the full-graph call
    fails.

    Args:
        G_undirected: An undirected NetworkX graph (no self-loops expected).

    Returns:
        A list of communities, where each community is a list of node ids.

    Raises:
        Exception: Propagates any error from cdlib / leidenalg so the
                   caller can decide how to handle it.
    """
    # NOTE: cdlib's leiden() wrapper does not expose leidenalg's
    # resolution_parameter; LEIDEN_RESOLUTION from config is reserved
    # for a future direct leidenalg integration.
    communities = algorithms.leiden(
        G_undirected,
        weights='weight',
    )
    return communities.communities


def compute_leiden_features(G: nx.DiGraph) -> dict:
    """
    Compute Leiden community detection features for every node in *G*.

    Guarantees that **every** node in *G* receives a ``leiden_id``,
    including nodes that belong to isolated (size-1) connected
    components.  The function applies three layers of resilience:

    1. Run Leiden on the full undirected projection of *G*.
    2. If (1) fails, fall back to running Leiden on each connected
       component independently.
    3. Any nodes still unassigned after (1) or (2) — whether because
       they were dropped during the igraph conversion, belong to
       trivial components, or survived an internal error — are each
       placed into their own singleton community.

    Self-loops are removed before community detection because they
    carry no information about inter-node community structure and can
    cause numerical issues in leidenalg.

    Args:
        G: A directed NetworkX graph produced by ``build_daily_graph``.

    Returns:
        Dictionary mapping **every** node in *G* to a dict with:
            - ``'leiden_id'``  (int): Community ID the node belongs to.
            - ``'leiden_size'`` (int): Number of nodes in that community.
    """
    if len(G) == 0:
        return {}

    all_nodes = set(G.nodes())

    # --- 1. Prepare a clean undirected copy ---------------------------------
    G_undirected = G.to_undirected()
    G_undirected.remove_edges_from(nx.selfloop_edges(G_undirected))

    # --- 2. Try Leiden on the full graph ------------------------------------
    node_features: dict = {}
    next_community_id = 0

    try:
        raw_communities = _run_leiden_on_graph(G_undirected)

        for community_members in raw_communities:
            community_size = len(community_members)
            for node in community_members:
                node_features[node] = {
                    'leiden_id': next_community_id,
                    'leiden_size': community_size,
                }
            next_community_id += 1

    except Exception as exc:
        warnings.warn(
            f"Leiden failed on full graph ({len(G)} nodes): {exc}. "
            "Falling back to per-component community detection.",
            RuntimeWarning,
            stacklevel=2,
        )

        # --- 3. Fallback: run Leiden per connected component ----------------
        for component in nx.connected_components(G_undirected):
            if len(component) < 2:
                # Isolated nodes are handled in the cleanup step below
                continue

            subgraph = G_undirected.subgraph(component).copy()
            try:
                sub_communities = _run_leiden_on_graph(subgraph)
                for community_members in sub_communities:
                    community_size = len(community_members)
                    for node in community_members:
                        node_features[node] = {
                            'leiden_id': next_community_id,
                            'leiden_size': community_size,
                        }
                    next_community_id += 1
            except Exception:
                # Last resort: treat the whole component as one community
                community_size = len(component)
                for node in component:
                    node_features[node] = {
                        'leiden_id': next_community_id,
                        'leiden_size': community_size,
                    }
                next_community_id += 1

    # --- 4. Guarantee 100 % coverage ----------------------------------------
    missing_nodes = all_nodes - set(node_features.keys())
    if missing_nodes:
        for node in missing_nodes:
            node_features[node] = {
                'leiden_id': next_community_id,
                'leiden_size': 1,
            }
            next_community_id += 1

    return node_features


def rank_nodes_by_score(scores: dict, descending: bool = True) -> list:
    """
    Rank nodes by their scores (PageRank, HITS, etc.).
    
    Args:
        scores: Dictionary mapping node IDs to scores
        descending: If True, rank highest scores first
        
    Returns:
        List of (node, score) tuples sorted by score
    """
    return sorted(scores.items(), key=lambda x: x[1], reverse=descending)


def evaluate_ranking_effectiveness(
    ranked_nodes: list,
    labels: dict,
    k_values: Optional[list] = None
) -> dict:
    """
    Evaluate how well a ranking (e.g., PageRank) identifies fraudulent nodes.
    
    Computes various metrics including Precision@K, Recall@K, Lift@K,
    ROC-AUC, and Average Precision.
    
    Args:
        ranked_nodes: List of (node, score) tuples sorted by score (descending)
        labels: Dictionary mapping node IDs to labels (1=fraud, 0=normal)
        k_values: List of K values for precision/recall@K metrics.
                  If None, uses [10, 50, 100, 500, 1000]
    
    Returns:
        Dictionary containing all computed metrics
    """
    if k_values is None:
        k_values = [10, 50, 100, 500, 1000]
    
    metrics = {}
    
    # Total counts
    total_nodes = len(ranked_nodes)
    total_fraud = sum(labels.values())
    total_normal = total_nodes - total_fraud
    
    metrics['total_nodes'] = total_nodes
    metrics['total_fraud'] = total_fraud
    metrics['total_normal'] = total_normal
    metrics['fraud_rate'] = total_fraud / total_nodes if total_nodes > 0 else 0
    
    # Extract ordered arrays
    nodes_ordered = [node for node, _ in ranked_nodes]
    scores_ordered = np.array([score for _, score in ranked_nodes])
    labels_ordered = np.array([labels.get(node, 0) for node in nodes_ordered])
    
    # Precision@K, Recall@K, Fraud found@K
    metrics['precision_at_k'] = {}
    metrics['recall_at_k'] = {}
    metrics['fraud_found_at_k'] = {}
    
    for k in k_values:
        if k <= total_nodes:
            top_k_labels = labels_ordered[:k]
            fraud_in_top_k = np.sum(top_k_labels)
            
            precision_at_k = fraud_in_top_k / k
            recall_at_k = fraud_in_top_k / total_fraud if total_fraud > 0 else 0
            
            metrics['precision_at_k'][k] = precision_at_k
            metrics['recall_at_k'][k] = recall_at_k
            metrics['fraud_found_at_k'][k] = int(fraud_in_top_k)
    
    # ROC-AUC
    if total_fraud > 0 and total_normal > 0:
        try:
            metrics['roc_auc'] = roc_auc_score(labels_ordered, scores_ordered)
        except ValueError:
            metrics['roc_auc'] = None
    else:
        metrics['roc_auc'] = None
    
    # Average Precision (PR-AUC)
    if total_fraud > 0:
        try:
            metrics['average_precision'] = average_precision_score(labels_ordered, scores_ordered)
        except ValueError:
            metrics['average_precision'] = None
    else:
        metrics['average_precision'] = None
    
    # Lift@K
    metrics['lift_at_k'] = {}
    baseline_rate = total_fraud / total_nodes if total_nodes > 0 else 0
    
    for k in k_values:
        if k <= total_nodes and baseline_rate > 0:
            precision_k = metrics['precision_at_k'].get(k, 0)
            lift = precision_k / baseline_rate
            metrics['lift_at_k'][k] = lift
    
    return metrics


def print_evaluation_report(metrics: dict, metric_name: str = "PageRank"):
    """
    Print a formatted evaluation report for ranking metrics.
    
    Args:
        metrics: Dictionary of metrics from evaluate_ranking_effectiveness()
        metric_name: Name of the metric being evaluated (for display)
    """
    print(f"\n{'=' * 60}")
    print(f"{metric_name} Evaluation Report")
    print('=' * 60)
    
    print("\n[Dataset Statistics]")
    print(f"  Total entities: {metrics['total_nodes']:,}")
    print(f"  Fraudulent entities: {metrics['total_fraud']:,}")
    print(f"  Normal entities: {metrics['total_normal']:,}")
    print(f"  Baseline fraud rate: {metrics['fraud_rate']:.4%}")
    
    print("\n[Precision@K] (Fraud rate in top-K ranked entities)")
    print("-" * 50)
    for k, precision in sorted(metrics['precision_at_k'].items()):
        fraud_found = metrics['fraud_found_at_k'].get(k, 0)
        lift = metrics['lift_at_k'].get(k, 0)
        print(f"  Top {k:>5}: {precision:.4%} precision | {fraud_found:>5} fraud found | {lift:.2f}x lift")
    
    print("\n[Recall@K] (Fraction of total fraud found in top-K)")
    print("-" * 50)
    for k, recall in sorted(metrics['recall_at_k'].items()):
        print(f"  Top {k:>5}: {recall:.4%} recall")
    
    print("\n[Global Metrics]")
    print("-" * 50)
    if metrics['roc_auc'] is not None:
        print(f"  ROC-AUC Score: {metrics['roc_auc']:.4f}")
    else:
        print("  ROC-AUC Score: N/A (insufficient class diversity)")
    
    if metrics['average_precision'] is not None:
        print(f"  Average Precision (PR-AUC): {metrics['average_precision']:.4f}")
    else:
        print("  Average Precision: N/A")
    
    # Effectiveness assessment
    print("\n[Effectiveness Assessment]")
    print("-" * 50)
    
    baseline = metrics['fraud_rate']
    best_precision_k = max(metrics['precision_at_k'].values()) if metrics['precision_at_k'] else 0
    
    if baseline > 0:
        if best_precision_k > baseline * 1.5:
            print(f"  {metric_name} shows STRONG improvement over random selection")
            print(f"  Best precision@K ({best_precision_k:.4%}) is {best_precision_k/baseline:.2f}x better than baseline ({baseline:.4%})")
        elif best_precision_k > baseline:
            print(f"  {metric_name} shows MODERATE improvement over random selection")
            print(f"  Best precision@K ({best_precision_k:.4%}) is {best_precision_k/baseline:.2f}x better than baseline ({baseline:.4%})")
        else:
            print(f"  {metric_name} shows NO improvement over random selection")
            print("  Consider using alternative methods or feature engineering")
    
    if metrics['roc_auc'] is not None:
        if metrics['roc_auc'] > 0.7:
            print(f"  ROC-AUC of {metrics['roc_auc']:.4f} indicates GOOD discriminative ability")
        elif metrics['roc_auc'] > 0.5:
            print(f"  ROC-AUC of {metrics['roc_auc']:.4f} indicates WEAK discriminative ability")
        else:
            print(f"  ROC-AUC of {metrics['roc_auc']:.4f} indicates NO discriminative ability")


def compute_daily_evaluation_metrics(
    pagerank_scores: dict,
    hits_hubs: dict,
    hits_auths: dict,
    bad_actors: set,
    k_values: Optional[list] = None
) -> dict:
    """
    Compute evaluation metrics for a single day's graph algorithms.
    
    Args:
        pagerank_scores: PageRank scores for nodes
        hits_hubs: HITS hub scores for nodes
        hits_auths: HITS authority scores for nodes
        bad_actors: Set of known fraudulent entity IDs (should be from get_bad_actors_up_to_date
                   to prevent future leakage)
        k_values: List of K values for precision/recall@K
        
    Returns:
        Dictionary with metrics for each algorithm
    """
    # Create labels dict from bad_actors
    all_nodes = set(pagerank_scores.keys()) | set(hits_hubs.keys()) | set(hits_auths.keys())
    labels = {node: 1 if node in bad_actors else 0 for node in all_nodes}
    
    daily_metrics = {}
    
    # Evaluate PageRank
    if pagerank_scores:
        ranked_pr = rank_nodes_by_score(pagerank_scores, descending=True)
        daily_metrics['pagerank'] = evaluate_ranking_effectiveness(ranked_pr, labels, k_values)
    
    # Evaluate HITS Hub
    if hits_hubs:
        ranked_hubs = rank_nodes_by_score(hits_hubs, descending=True)
        daily_metrics['hits_hub'] = evaluate_ranking_effectiveness(ranked_hubs, labels, k_values)
    
    # Evaluate HITS Authority
    if hits_auths:
        ranked_auths = rank_nodes_by_score(hits_auths, descending=True)
        daily_metrics['hits_auth'] = evaluate_ranking_effectiveness(ranked_auths, labels, k_values)
    
    return daily_metrics


def compute_rank_stability(
    prev_scores: dict,
    curr_scores: dict,
    top_k: int = 100
) -> dict:
    """
    Compute rank stability metrics between two consecutive time windows.
    
    This implements the "Rank Stability Analysis" methodology for detecting
    anomalies when a node's role shifts drastically between temporal snapshots.
    Large rank changes can indicate suspicious activity.
    
    Args:
        prev_scores: Scores (e.g., PageRank) from previous time window
        curr_scores: Scores from current time window
        top_k: Number of top nodes to consider for stability analysis
        
    Returns:
        Dictionary containing:
            - 'rank_changes': dict mapping node_id to rank change (positive = moved up)
            - 'new_entrants': set of nodes that appeared in top-k (weren't before)
            - 'dropouts': set of nodes that left top-k
            - 'biggest_risers': list of (node, change) tuples for largest positive changes
            - 'biggest_fallers': list of (node, change) tuples for largest negative changes
            - 'stability_score': float in [0,1] where 1 = perfectly stable
    """
    if not prev_scores or not curr_scores:
        return {
            'rank_changes': {},
            'new_entrants': set(),
            'dropouts': set(),
            'biggest_risers': [],
            'biggest_fallers': [],
            'stability_score': None,
        }
    
    # Compute rankings (1 = highest score)
    prev_ranked = rank_nodes_by_score(prev_scores, descending=True)
    curr_ranked = rank_nodes_by_score(curr_scores, descending=True)
    
    # Create rank dictionaries
    prev_ranks = {node: rank + 1 for rank, (node, _) in enumerate(prev_ranked)}
    curr_ranks = {node: rank + 1 for rank, (node, _) in enumerate(curr_ranked)}
    
    # Get top-k sets
    prev_top_k = set(node for node, _ in prev_ranked[:top_k])
    curr_top_k = set(node for node, _ in curr_ranked[:top_k])
    
    # Compute rank changes for nodes present in both windows
    common_nodes = set(prev_ranks.keys()) & set(curr_ranks.keys())
    rank_changes = {}
    
    for node in common_nodes:
        # Positive change means moved up (lower rank number = higher position)
        rank_change = prev_ranks[node] - curr_ranks[node]
        rank_changes[node] = rank_change
    
    # Identify new entrants and dropouts in top-k
    new_entrants = curr_top_k - prev_top_k
    dropouts = prev_top_k - curr_top_k
    
    # Find biggest movers (only among nodes in both windows)
    sorted_changes = sorted(rank_changes.items(), key=lambda x: x[1], reverse=True)
    biggest_risers = sorted_changes[:10]  # Top 10 risers
    biggest_fallers = sorted_changes[-10:][::-1]  # Top 10 fallers (reversed for biggest drops first)
    
    # Compute stability score: Jaccard similarity of top-k sets
    if prev_top_k or curr_top_k:
        stability_score = len(prev_top_k & curr_top_k) / len(prev_top_k | curr_top_k)
    else:
        stability_score = 1.0
    
    return {
        'rank_changes': rank_changes,
        'new_entrants': new_entrants,
        'dropouts': dropouts,
        'biggest_risers': biggest_risers,
        'biggest_fallers': biggest_fallers,
        'stability_score': stability_score,
    }


def detect_rank_anomalies(
    rank_changes: dict,
    threshold_percentile: float = 95
) -> set:
    """
    Detect nodes with anomalous rank changes between time windows.
    
    Nodes with rank changes beyond the threshold percentile (either direction)
    are flagged as potential anomalies, as drastic role shifts can indicate
    suspicious financial activity.
    
    Args:
        rank_changes: Dictionary mapping node_id to rank change value
        threshold_percentile: Percentile threshold for anomaly detection (default 95)
        
    Returns:
        Set of node IDs with anomalous rank changes
    """
    if not rank_changes:
        return set()
    
    changes = np.array(list(rank_changes.values()))
    abs_changes = np.abs(changes)
    
    # Compute threshold based on percentile of absolute changes
    threshold = np.percentile(abs_changes, threshold_percentile)
    
    # Flag nodes exceeding threshold
    anomalous_nodes = {
        node for node, change in rank_changes.items()
        if abs(change) >= threshold
    }
    
    return anomalous_nodes