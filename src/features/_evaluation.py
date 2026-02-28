"""
Internal evaluation helpers for the PixFraudDetection pipeline.

This module migrates ``rank_nodes_by_score``, ``evaluate_ranking_effectiveness``,
and ``compute_daily_evaluation_metrics`` from the legacy ``utils.py``.

These functions are considered **internal** to the features package (hence the
leading underscore in the module name).  They are consumed exclusively by the
orchestrator in ``scripts/02_extract_features.py`` and are not part of the
public ``src.features`` API.

Mathematical invariants preserved from the original implementation
------------------------------------------------------------------
* Precision@K  = (fraud nodes in top-K) / K
* Recall@K     = (fraud nodes in top-K) / (total fraud nodes)
* Lift@K       = Precision@K / baseline_fraud_rate
* ROC-AUC      — computed via ``sklearn.metrics.roc_auc_score``
* Average Precision (PR-AUC) — computed via
  ``sklearn.metrics.average_precision_score``

All formulas, edge-case guards (zero fraud, zero normal, insufficient nodes),
and the three-algorithm structure (pagerank / hits_hub / hits_auth) are
preserved verbatim from ``utils.py``.  No mathematical logic has been altered.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

# ---------------------------------------------------------------------------
# Internal ranking helper
# ---------------------------------------------------------------------------


def _rank_nodes_by_score(scores: dict, descending: bool = True) -> list[tuple]:
    """
    Sort a ``{node: score}`` mapping into a ranked list of ``(node, score)``
    tuples.

    Mirrors ``utils.rank_nodes_by_score`` exactly.

    Parameters
    ----------
    scores : dict
        Mapping of node ID to numeric score.
    descending : bool
        When ``True`` (default) the highest-scoring node is ranked first.

    Returns
    -------
    list[tuple]
        List of ``(node_id, score)`` tuples sorted by score.
    """
    return sorted(scores.items(), key=lambda x: x[1], reverse=descending)


# ---------------------------------------------------------------------------
# Core evaluation function
# ---------------------------------------------------------------------------


def evaluate_ranking_effectiveness(
    ranked_nodes: list[tuple],
    labels: dict,
    k_values: Optional[list[int]] = None,
) -> dict:
    """
    Evaluate how well a ranking (e.g. PageRank) surfaces fraudulent nodes.

    Computes Precision@K, Recall@K, Fraud-found@K, Lift@K, ROC-AUC, and
    Average Precision (PR-AUC) for the supplied ranked node list.

    Parameters
    ----------
    ranked_nodes : list[tuple]
        ``[(node_id, score), ...]`` sorted highest-score-first, as produced
        by :func:`_rank_nodes_by_score`.
    labels : dict
        ``{node_id: 1}`` for fraudulent nodes, ``{node_id: 0}`` for benign
        nodes.  Nodes absent from *labels* are treated as benign (score 0).
    k_values : list[int], optional
        K values for which to compute @K metrics.
        Defaults to ``[10, 50, 100, 500, 1000]``.

    Returns
    -------
    dict
        Dictionary with the following keys:

        * ``total_nodes``      — total number of nodes in the ranked list
        * ``total_fraud``      — number of fraud nodes in the ranked list
        * ``total_normal``     — number of non-fraud nodes
        * ``fraud_rate``       — baseline fraud rate (total_fraud / total_nodes)
        * ``precision_at_k``   — ``{k: float}`` Precision@K for each k
        * ``recall_at_k``      — ``{k: float}`` Recall@K for each k
        * ``fraud_found_at_k`` — ``{k: int}`` raw fraud count in top-K
        * ``lift_at_k``        — ``{k: float}`` Lift@K over baseline
        * ``roc_auc``          — ROC-AUC score (``float`` or ``None``)
        * ``average_precision``— Average Precision / PR-AUC (``float`` or ``None``)
    """
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

    # Build ordered arrays aligned to the ranking
    nodes_ordered = [node for node, _ in ranked_nodes]
    scores_ordered = np.array([score for _, score in ranked_nodes])
    labels_ordered = np.array([labels.get(node, 0) for node in nodes_ordered])

    # ---------------------------------------------------------------------- #
    # Precision@K, Recall@K, Fraud-found@K                                   #
    # ---------------------------------------------------------------------- #
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

    # ---------------------------------------------------------------------- #
    # ROC-AUC                                                                 #
    # ---------------------------------------------------------------------- #
    if total_fraud > 0 and total_normal > 0:
        try:
            metrics["roc_auc"] = roc_auc_score(labels_ordered, scores_ordered)
        except ValueError:
            metrics["roc_auc"] = None
    else:
        metrics["roc_auc"] = None

    # ---------------------------------------------------------------------- #
    # Average Precision (PR-AUC)                                              #
    # ---------------------------------------------------------------------- #
    if total_fraud > 0:
        try:
            metrics["average_precision"] = average_precision_score(
                labels_ordered, scores_ordered
            )
        except ValueError:
            metrics["average_precision"] = None
    else:
        metrics["average_precision"] = None

    # ---------------------------------------------------------------------- #
    # Lift@K                                                                  #
    # ---------------------------------------------------------------------- #
    metrics["lift_at_k"] = {}
    baseline_rate = metrics["fraud_rate"]

    for k in k_values:
        if k <= total_nodes and baseline_rate > 0:
            precision_k = metrics["precision_at_k"].get(k, 0)
            metrics["lift_at_k"][k] = precision_k / baseline_rate

    return metrics


# ---------------------------------------------------------------------------
# Per-day multi-algorithm wrapper
# ---------------------------------------------------------------------------


def compute_daily_evaluation_metrics(
    pagerank_scores: dict,
    hits_hubs: dict,
    hits_auths: dict,
    bad_actors: set,
    k_values: Optional[list[int]] = None,
) -> dict:
    """
    Compute evaluation metrics for all three algorithms for a single day.

    Wraps :func:`evaluate_ranking_effectiveness` for each of PageRank,
    HITS-Hub, and HITS-Authority, building a shared ``labels`` dict from
    the union of all nodes present in any score dictionary.

    Parameters
    ----------
    pagerank_scores : dict
        ``{node_id: pagerank_score}`` for the current window.
    hits_hubs : dict
        ``{node_id: hub_score}`` for the current window.
    hits_auths : dict
        ``{node_id: authority_score}`` for the current window.
    bad_actors : set
        Set of entity IDs identified as bad actors **up to** the current date
        (time-aware labels — no future leakage).  Produced by
        :func:`src.data.loader.get_bad_actors_up_to_date`.
    k_values : list[int], optional
        K values forwarded to :func:`evaluate_ranking_effectiveness`.

    Returns
    -------
    dict
        ``{"pagerank": {...}, "hits_hub": {...}, "hits_auth": {...}}``
        where each value is the full metrics dict returned by
        :func:`evaluate_ranking_effectiveness`.  An algorithm key is omitted
        when its score dictionary is empty.
    """
    # Build a shared label dict from all nodes visible across all algorithms.
    all_nodes = (
        set(pagerank_scores.keys()) | set(hits_hubs.keys()) | set(hits_auths.keys())
    )
    labels = {node: (1 if node in bad_actors else 0) for node in all_nodes}

    daily_metrics: dict = {}

    # --- PageRank ---
    if pagerank_scores:
        ranked_pr = _rank_nodes_by_score(pagerank_scores, descending=True)
        daily_metrics["pagerank"] = evaluate_ranking_effectiveness(
            ranked_pr, labels, k_values
        )

    # --- HITS Hub ---
    if hits_hubs:
        ranked_hubs = _rank_nodes_by_score(hits_hubs, descending=True)
        daily_metrics["hits_hub"] = evaluate_ranking_effectiveness(
            ranked_hubs, labels, k_values
        )

    # --- HITS Authority ---
    if hits_auths:
        ranked_auths = _rank_nodes_by_score(hits_auths, descending=True)
        daily_metrics["hits_auth"] = evaluate_ranking_effectiveness(
            ranked_auths, labels, k_values
        )

    return daily_metrics
