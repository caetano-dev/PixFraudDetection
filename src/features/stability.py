"""
Rank stability tracker for the PixFraudDetection pipeline.

This module provides :class:`RankStabilityTracker`, a *stateful* feature
extractor that implements the WeirdNodes algorithm as defined by Vilella et 
al. (2025). It processes multiple centrality metrics simultaneously and uses
ordinal stability validation (Spearman's ρ and Kendall's τ) to ensure only 
concordant signals are included in the ensemble.

The algorithm computes:
1. Global Ordinal Stability Check - validates each metric using correlation
2. Multi-Centrality Ensemble - aggregates normalized residuals across valid metrics
3. Top-K Prioritization - ranks nodes by aggregated residual magnitude
4. Directional Labeling - distinguishes "Risers" from "Fallers"

Design notes
------------
* :class:`RankStabilityTracker` intentionally does **not** subclass
  :class:`~src.features.base.FeatureExtractor` directly because its
  ``compute`` method signature differs from the stateless ``extract(G)``
  contract: it takes pre-computed score dictionaries rather than a raw
  graph.  This keeps the base interface clean while still fitting naturally
  into the orchestrator loop.
* State is minimal: only ``self.previous_window_metrics`` is retained between
  calls.  Everything else is recomputed fresh each window.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr, kendalltau


class RankStabilityTracker:
    """
    Stateful tracker implementing the WeirdNodes algorithm (Vilella et al., 2025).

    On each call to :meth:`compute` the tracker:

    1. Validates each centrality metric using Spearman's ρ and Kendall's τ
       (Validity Threshold: ρ>0.4 or τ>0.3).
    2. For each valid metric, computes normalized rank residuals δ=r_x−r_y.
    3. Aggregates residuals across all valid metrics (Summation Strategy).
    4. Ranks nodes by the magnitude of the aggregated residual |δ|.
    5. Flags top-K nodes as rank anomalies based on threshold percentile.
    6. Separates "Risers" (positive residuals) from "Fallers" (negative).

    Parameters
    ----------
    top_k : int
        Number of top-ranked nodes for reporting. Defaults to ``100``.
    threshold_percentile : float
        Percentile of the absolute ensemble residual above which a node 
        is flagged as a rank anomaly.  Defaults to ``95.0`` (top 5 %).
    spearman_threshold : float
        Minimum Spearman's ρ for a metric to be considered valid. Defaults to 0.4.
    kendall_threshold : float
        Minimum Kendall's τ for a metric to be considered valid. Defaults to 0.3.

    Attributes
    ----------
    previous_window_metrics : dict
        Nested dict ``{metric_name: {node_id: score}}`` from the most recently 
        processed window. Empty on initialisation.

    Examples
    --------
    >>> tracker = RankStabilityTracker(top_k=100, threshold_percentile=95.0)
    >>> daily_metrics = {
    ...     "pr_vol_deep": {node: {"pagerank": score}, ...},
    ...     "hits": {node: {"hits_hub": hub_score, "hits_auth": auth_score}, ...},
    ...     ...
    ... }
    >>> result = tracker.compute(daily_metrics)
    """

    def __init__(
        self,
        top_k: int = 100,
        threshold_percentile: float = 95.0,
        spearman_threshold: float = 0.4,
        kendall_threshold: float = 0.3,
    ) -> None:
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}.")
        if not (0.0 <= threshold_percentile <= 100.0):
            raise ValueError(
                f"threshold_percentile must be in [0, 100], got {threshold_percentile}."
            )

        self._top_k = top_k
        self._threshold_percentile = threshold_percentile
        self._spearman_threshold = spearman_threshold
        self._kendall_threshold = kendall_threshold
        self.previous_window_metrics: dict = {}

    @staticmethod
    def _flatten_metrics(daily_metrics: dict) -> dict[str, dict]:
        """
        Flatten nested daily_metrics structure into separate metric lists.
        
        Transforms:
            {"pr_vol_deep": {node: {"pagerank": 0.5}}, "hits": {node: {"hits_hub": 0.3, "hits_auth": 0.2}}}
        Into:
            {"pagerank": {node: 0.5}, "hits_hub": {node: 0.3}, "hits_auth": {node: 0.2}}
        """
        flattened = {}
        
        for extractor_key, node_metrics in daily_metrics.items():
            if not node_metrics:
                continue
                
            # Get a sample node to inspect the metric structure
            sample_node = next(iter(node_metrics.keys()))
            sample_metrics = node_metrics[sample_node]
            
            if isinstance(sample_metrics, dict):
                # Multiple metrics from one extractor (e.g., HITS)
                for metric_name, _ in sample_metrics.items():
                    flattened[metric_name] = {
                        node: metrics.get(metric_name, 0.0)
                        for node, metrics in node_metrics.items()
                    }
            else:
                # Single metric (shouldn't happen with current structure, but handle it)
                flattened[extractor_key] = {
                    node: score for node, score in node_metrics.items()
                }
        
        return flattened

    @staticmethod
    def _compute_rank_vector(scores: dict) -> tuple[list, dict]:
        """
        Convert scores to rank vector (1-indexed, 1 = highest).
        
        Returns
        -------
        ranked_nodes : list
            Sorted list of (node, score) tuples
        rank_map : dict
            {node: rank} mapping
        """
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        rank_map = {node: rank + 1 for rank, (node, _) in enumerate(ranked)}
        return ranked, rank_map

    def _validate_metric_stability(
        self, 
        prev_ranks: dict, 
        curr_ranks: dict,
        common_nodes: set,
    ) -> tuple[bool, float, float]:
        """
        Apply Ordinal Stability Check using Spearman's ρ and Kendall's τ.
        
        A metric is "Valid" if it exhibits moderate concordance:
        ρ > spearman_threshold OR τ > kendall_threshold
        
        Returns
        -------
        is_valid : bool
        spearman_rho : float
        kendall_tau : float
        """
        if not common_nodes or len(common_nodes) < 2:
            return False, 0.0, 0.0
        
        # Build aligned rank vectors for common nodes
        common_list = sorted(common_nodes)
        prev_rank_vec = [prev_ranks[node] for node in common_list]
        curr_rank_vec = [curr_ranks[node] for node in common_list]
        
        # Compute correlation coefficients
        try:
            rho, _ = spearmanr(prev_rank_vec, curr_rank_vec)
            tau, _ = kendalltau(prev_rank_vec, curr_rank_vec)
            
            # Handle NaN (can occur with constant vectors)
            if np.isnan(rho):
                rho = 0.0
            if np.isnan(tau):
                tau = 0.0
                
        except Exception:
            return False, 0.0, 0.0
        
        # Validity check
        is_valid = (rho > self._spearman_threshold) or (tau > self._kendall_threshold)
        
        return is_valid, float(rho), float(tau)

    def _compute_normalized_residuals(
        self,
        prev_ranks: dict,
        curr_ranks: dict,
        total_nodes: int,
    ) -> dict:
        """
        Compute normalized signed residuals for each common node.
        
        δ = (prev_rank - curr_rank) / total_nodes
        Positive δ = node moved up (lower rank number)
        Negative δ = node moved down (higher rank number)
        """
        common_nodes = set(prev_ranks.keys()) & set(curr_ranks.keys())
        
        residuals = {}
        for node in common_nodes:
            # Normalized signed residual
            delta = (prev_ranks[node] - curr_ranks[node]) / total_nodes
            residuals[node] = delta
        
        return residuals

    def _aggregate_ensemble_residuals(
        self,
        valid_metric_residuals: dict[str, dict],
    ) -> dict:
        """
        Sum normalized residuals across all valid metrics (Summation Strategy).
        
        Parameters
        ----------
        valid_metric_residuals : dict
            {metric_name: {node: normalized_residual}}
        
        Returns
        -------
        dict
            {node: aggregated_residual}
        """
        ensemble_residuals = {}
        
        # Collect all nodes that appear in any valid metric
        all_nodes = set()
        for residuals in valid_metric_residuals.values():
            all_nodes.update(residuals.keys())
        
        # Sum residuals for each node
        for node in all_nodes:
            total_residual = sum(
                residuals.get(node, 0.0) 
                for residuals in valid_metric_residuals.values()
            )
            ensemble_residuals[node] = total_residual
        
        return ensemble_residuals

    def _detect_rank_anomalies(self, ensemble_residuals: dict) -> dict[object, float]:
        """
        Instead of a binary set, return the absolute magnitude of the 
        ensemble residual for every node. This follows the WeirdNodes 
        'Ranked Information Retrieval' approach.
        
        Returns
        -------
        dict
            {node: weirdness_score} where weirdness_score is |ensemble_residual|
        """
        if not ensemble_residuals:
            return {}
        
        # WeirdNodes focuses on the magnitude of the shift |δ|
        return {node: abs(res) for node, res in ensemble_residuals.items()}

    def compute(self, current_metrics: dict) -> dict | None:
        """
        Compute WeirdNodes ensemble rank-stability features.

        Compares *current_metrics* against the internally stored
        ``previous_window_metrics``, updates the internal state, and returns
        aggregated results.

        Parameters
        ----------
        current_metrics : dict
            Nested dictionary ``{extractor_name: {node: {metric: score}}}``
            containing all extracted centrality metrics for the current window.

        Returns
        -------
        dict or None
            ``None`` on the first call (no previous window).

            Otherwise a dictionary with the following keys:

            * ``weirdness_scores``    – ``{node: |ensemble_residual|}`` - continuous
              feature representing the magnitude of rank instability (WeirdNodes signal).
            * ``ensemble_residuals``  – ``{node: aggregated_residual}`` across 
              all valid metrics (signed, for directional analysis).
            * ``valid_metrics``       – ``list`` of metric names that passed
              the stability validation.
            * ``validation_summary``  – ``dict`` with validation stats for each
              metric: ``{metric: {"valid": bool, "spearman": float, "kendall": float}}``.
        """
        # First window: no previous data — store and return None.
        if not self.previous_window_metrics:
            self.previous_window_metrics = current_metrics
            return None

        # Flatten nested structure
        prev_flat = self._flatten_metrics(self.previous_window_metrics)
        curr_flat = self._flatten_metrics(current_metrics)

        # Track validation results and residuals per metric
        validation_summary = {}
        valid_metric_residuals = {}

        # Process each metric independently
        for metric_name in set(prev_flat.keys()) & set(curr_flat.keys()):
            prev_scores = prev_flat[metric_name]
            curr_scores = curr_flat[metric_name]

            if not prev_scores or not curr_scores:
                continue

            # Compute rank vectors
            _, prev_ranks = self._compute_rank_vector(prev_scores)
            _, curr_ranks = self._compute_rank_vector(curr_scores)

            # Get common nodes
            common_nodes = set(prev_ranks.keys()) & set(curr_ranks.keys())
            
            if not common_nodes:
                continue

            # Validate metric stability
            is_valid, rho, tau = self._validate_metric_stability(
                prev_ranks, curr_ranks, common_nodes
            )

            validation_summary[metric_name] = {
                "valid": is_valid,
                "spearman": rho,
                "kendall": tau,
            }

            # Only include valid metrics in ensemble
            if is_valid:
                total_nodes = max(len(prev_scores), len(curr_scores))
                residuals = self._compute_normalized_residuals(
                    prev_ranks, curr_ranks, total_nodes
                )
                valid_metric_residuals[metric_name] = residuals

        # Aggregate residuals across all valid metrics (Summation Strategy)
        ensemble_residuals = self._aggregate_ensemble_residuals(valid_metric_residuals)

        # Instead of binary flagging, we provide the continuous 'weirdness' score
        # which is the absolute magnitude of the ensemble residual.
        weirdness_scores = self._detect_rank_anomalies(ensemble_residuals)

        # Advance internal state
        self.previous_window_metrics = current_metrics

        return {
            "weirdness_scores": weirdness_scores,  # Continuous feature for ML
            "ensemble_residuals": ensemble_residuals,
            "valid_metrics": list(valid_metric_residuals.keys()),
            "validation_summary": validation_summary,
        }

    def reset(self) -> None:
        """
        Clear internal state, resetting the tracker to its initial condition.

        Useful when re-running the pipeline on a different dataset or date
        range without re-instantiating the object.
        """
        self.previous_window_metrics = {}

    def __repr__(self) -> str:
        state = "initialised" if not self.previous_window_metrics else "tracking"
        return (
            f"RankStabilityTracker("
            f"top_k={self._top_k}, "
            f"threshold_percentile={self._threshold_percentile}, "
            f"spearman_threshold={self._spearman_threshold}, "
            f"kendall_threshold={self._kendall_threshold}, "
            f"state={state!r})"
        )
