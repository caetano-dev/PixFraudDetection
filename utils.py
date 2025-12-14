"""
Utility functions for the Sliding Window Graph Pipeline.

This module provides data loading, graph construction, and evaluation utilities
for the money laundering detection feature generation pipeline.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
import networkx as nx
from sklearn.metrics import roc_auc_score, average_precision_score
from typing import Optional

from config import (
    DATA_PATH,
    NORMAL_TRANSACTIONS_FILE,
    LAUNDERING_TRANSACTIONS_FILE,
    ACCOUNTS_FILE,
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
    
    # Identify bad actors (global ground truth)
    # These are entities involved in ANY laundering transaction across the entire dataset
    laundering_txns = all_transactions[all_transactions['is_laundering'] == 1]
    bad_actors = set(laundering_txns['source_entity']).union(
        set(laundering_txns['target_entity'])
    )
    
    print(f"Loaded {len(all_transactions):,} transactions")
    print(f"Identified {len(bad_actors):,} bad actors (entities in laundering transactions)")
    
    return all_transactions, bad_actors


def build_daily_graph(window_df: pd.DataFrame) -> nx.DiGraph:
    """
    Build a directed graph from a filtered DataFrame of transactions.
    
    This function aggregates edges by (source, target) pairs and sums
    the transaction amounts to create weighted edges. It focuses purely
    on graph topology for PageRank/HITS computation.
    
    Args:
        window_df: DataFrame containing transactions for the current window.
                   Must have columns: source_entity, target_entity, amount_sent_c
    
    Returns:
            - Edges have 'weight' (total amount) and 'count' (number of transactions)
    """
    G = nx.DiGraph()
    
    # Handle empty DataFrame
    if window_df.empty:
        return G
    
    # Aggregate edges: group by (source, target) and compute weight/count
    edge_aggregation = window_df.groupby(
        ['source_entity', 'target_entity']
    ).agg(
        weight=('amount_sent_c', 'sum'),
        count=('amount_sent_c', 'count')
    ).reset_index()
    
    # Add edges to graph
    for _, row in edge_aggregation.iterrows():
        G.add_edge(
            row['source_entity'],
            row['target_entity'],
            weight=row['weight'],
            count=row['count']
        )
    
    return G


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
        bad_actors: Set of known fraudulent entity IDs
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