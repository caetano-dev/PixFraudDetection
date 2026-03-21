from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

def _rank_nodes_by_score(scores: dict, descending: bool = True) -> list[tuple]:
    return sorted(scores.items(), key=lambda x: x[1], reverse=descending)

def evaluate_ranking_effectiveness(
    ranked_nodes: list[tuple],
    labels: dict,
    k_values: Optional[list[int]] = None,
) -> dict:
    if k_values is None:
        k_values = [10, 50, 100, 500, 1000]

    metrics: dict = {}

    total_nodes = len(ranked_nodes)
    total_fraud = sum(labels.values())
    total_normal = total_nodes - total_fraud

    metrics["total_nodes"] = total_nodes
    metrics["total_fraud"] = total_fraud
    metrics["total_normal"] = total_normal
    metrics["fraud_rate"] = total_fraud / total_nodes if total_nodes > 0 else 0

    nodes_ordered = [node for node, _ in ranked_nodes]
    scores_ordered = np.array([score for _, score in ranked_nodes])
    labels_ordered = np.array([labels.get(node, 0) for node in nodes_ordered])

    metrics["precision_at_k"] = {}
    metrics["recall_at_k"] = {}
    metrics["fraud_found_at_k"] = {}

    for k in k_values:
        if k <= total_nodes:
            top_k_labels = labels_ordered[:k]
            fraud_in_top_k = int(np.sum(top_k_labels))

            precision_at_k = fraud_in_top_k / k
            recall_at_k = fraud_in_top_k / total_fraud if total_fraud > 0 else 0

            metrics["precision_at_k"][k] = precision_at_k
            metrics["recall_at_k"][k] = recall_at_k
            metrics["fraud_found_at_k"][k] = fraud_in_top_k

    # ROC-AUC                                                                 #
    if total_fraud > 0 and total_normal > 0:
        try:
            metrics["roc_auc"] = roc_auc_score(labels_ordered, scores_ordered)
        except ValueError:
            metrics["roc_auc"] = None
    else:
        metrics["roc_auc"] = None

    # Average Precision (PR-AUC)                                              #
    if total_fraud > 0:
        try:
            metrics["average_precision"] = average_precision_score(
                labels_ordered, scores_ordered
            )
        except ValueError:
            metrics["average_precision"] = None
    else:
        metrics["average_precision"] = None

    # Lift@K                                                                  #
    metrics["lift_at_k"] = {}
    baseline_rate = metrics["fraud_rate"]

    for k in k_values:
        if k <= total_nodes and baseline_rate > 0:
            precision_k = metrics["precision_at_k"].get(k, 0)
            metrics["lift_at_k"][k] = precision_k / baseline_rate

    return metrics



def compute_daily_evaluation_metrics(
    bad_actors: set,
    k_values: list[int] | None = None,
    **score_dicts: dict[object, float | int],
) -> dict:

    if not score_dicts:
        return {}

    all_nodes = set()
    for d in score_dicts.values():
        all_nodes.update(d.keys())
        
    labels = {node: (1 if node in bad_actors else 0) for node in all_nodes}
    daily_metrics: dict = {}

    for algo_name, scores in score_dicts.items():
        if scores:
            ranked_nodes = _rank_nodes_by_score(scores, descending=True)
            daily_metrics[algo_name] = evaluate_ranking_effectiveness(
                ranked_nodes, labels, k_values
            )

    return daily_metrics
