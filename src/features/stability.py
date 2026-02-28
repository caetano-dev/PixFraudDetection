"""
Rank stability tracker for the PixFraudDetection pipeline.

This module provides :class:`RankStabilityTracker`, a *stateful* feature
extractor that computes how much each node's PageRank rank has shifted
between consecutive temporal windows.  Sudden, large rank changes are a
validated signal for suspicious financial activity — a node that jumps from
obscurity into the top-100 overnight warrants investigation.

Design notes
------------
* :class:`RankStabilityTracker` intentionally does **not** subclass
  :class:`~src.features.base.FeatureExtractor` directly because its
  ``compute`` method signature differs from the stateless ``extract(G)``
  contract: it takes a pre-computed score dictionary rather than a raw
  graph.  This keeps the base interface clean while still fitting naturally
  into the orchestrator loop.
* All mathematics — Jaccard stability score, 95th-percentile anomaly
  threshold, rank-change arithmetic — are preserved verbatim from
  ``utils.compute_rank_stability`` and ``utils.detect_rank_anomalies``.
  No mathematical logic has been altered.
* State is minimal: only ``self.previous_scores`` is retained between
  calls.  Everything else is recomputed fresh each window.
"""

from __future__ import annotations

import numpy as np


class RankStabilityTracker:
    """
    Stateful tracker that measures PageRank rank-change between windows.

    On each call to :meth:`compute` the tracker:

    1. Ranks nodes in both the previous and current score dictionaries.
    2. Computes a signed rank-change for every node present in both windows
       (positive = moved *up* in ranking, negative = moved *down*).
    3. Identifies new entrants and dropouts in the top-K set.
    4. Computes a Jaccard-based stability score in ``[0, 1]`` where
       ``1.0`` means the top-K set is identical between windows.
    5. Flags nodes whose *absolute* rank change exceeds the
       ``threshold_percentile``-th percentile as anomalies.
    6. Stores the current scores as ``previous_scores`` for the next call.

    Parameters
    ----------
    top_k : int
        Number of top-ranked nodes used when computing the Jaccard stability
        score and identifying new entrants / dropouts.  Defaults to ``100``.
    threshold_percentile : float
        Percentile of the absolute rank-change distribution above which a
        node is flagged as a rank anomaly.  Defaults to ``95.0`` (top 5 %).

    Attributes
    ----------
    previous_scores : dict
        The score dictionary from the most recently processed window.
        Empty on initialisation; populated after the first :meth:`compute`
        call.

    Examples
    --------
    >>> tracker = RankStabilityTracker(top_k=100, threshold_percentile=95.0)
    >>> for current_date, window_df in generator:
    ...     G = build_daily_graph(window_df)
    ...     pr_scores = PageRankVolumeExtractor().extract(G)
    ...     # Flatten to {node: score} for the tracker
    ...     flat_scores = {n: v["pagerank"] for n, v in pr_scores.items()}
    ...     result = tracker.compute(flat_scores)
    ...     # result is None for the very first window (no previous scores)
    """

    def __init__(
        self,
        top_k: int = 100,
        threshold_percentile: float = 95.0,
    ) -> None:
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}.")
        if not (0.0 <= threshold_percentile <= 100.0):
            raise ValueError(
                f"threshold_percentile must be in [0, 100], got {threshold_percentile}."
            )

        self._top_k = top_k
        self._threshold_percentile = threshold_percentile

        # Internal state: scores from the previous window.
        # Empty dict signals "first window — no stability data yet".
        self.previous_scores: dict = {}

    # ------------------------------------------------------------------
    # Private helpers (mirror the original utils functions 1-to-1)
    # ------------------------------------------------------------------

    @staticmethod
    def _rank_nodes_by_score(scores: dict, descending: bool = True) -> list[tuple]:
        """
        Sort a ``{node: score}`` dict into a ranked list of ``(node, score)``
        tuples.

        Mirrors ``utils.rank_nodes_by_score`` exactly.
        """
        return sorted(scores.items(), key=lambda x: x[1], reverse=descending)

    def _compute_rank_stability(
        self,
        prev_scores: dict,
        curr_scores: dict,
    ) -> dict:
        """
        Compute rank-change metrics between two consecutive score dicts.

        Mirrors ``utils.compute_rank_stability`` exactly, including the
        Jaccard stability score formula and the signed rank-change
        arithmetic (positive = moved up, lower rank number = higher position).

        Parameters
        ----------
        prev_scores : dict
            ``{node_id: score}`` mapping for the previous window.
        curr_scores : dict
            ``{node_id: score}`` mapping for the current window.

        Returns
        -------
        dict
            Keys:

            * ``rank_changes``    – ``{node: signed_rank_change}`` for nodes
              present in **both** windows.
            * ``new_entrants``    – set of nodes that entered the top-K.
            * ``dropouts``        – set of nodes that left the top-K.
            * ``biggest_risers``  – top-10 positive movers as
              ``[(node, change), ...]``.
            * ``biggest_fallers`` – top-10 negative movers as
              ``[(node, change), ...]``.
            * ``stability_score`` – Jaccard similarity of the top-K sets,
              or ``None`` if both sets are empty.
        """
        if not prev_scores or not curr_scores:
            return {
                "rank_changes": {},
                "new_entrants": set(),
                "dropouts": set(),
                "biggest_risers": [],
                "biggest_fallers": [],
                "stability_score": None,
            }

        # Compute full rankings (1-indexed, 1 = highest score)
        prev_ranked = self._rank_nodes_by_score(prev_scores, descending=True)
        curr_ranked = self._rank_nodes_by_score(curr_scores, descending=True)

        prev_ranks = {node: rank + 1 for rank, (node, _) in enumerate(prev_ranked)}
        curr_ranks = {node: rank + 1 for rank, (node, _) in enumerate(curr_ranked)}

        # Top-K sets for Jaccard / entrant / dropout analysis
        prev_top_k: set = {node for node, _ in prev_ranked[: self._top_k]}
        curr_top_k: set = {node for node, _ in curr_ranked[: self._top_k]}

        # Signed rank change for nodes present in *both* windows:
        # positive value  → node moved up   (lower rank number now)
        # negative value  → node moved down (higher rank number now)
        common_nodes = set(prev_ranks.keys()) & set(curr_ranks.keys())
        rank_changes: dict = {
            node: prev_ranks[node] - curr_ranks[node] for node in common_nodes
        }

        # New entrants / dropouts relative to the top-K
        new_entrants: set = curr_top_k - prev_top_k
        dropouts: set = prev_top_k - curr_top_k

        # Biggest movers (top 10 in each direction)
        sorted_changes = sorted(rank_changes.items(), key=lambda x: x[1], reverse=True)
        biggest_risers: list = sorted_changes[:10]
        biggest_fallers: list = sorted_changes[-10:][::-1]

        # Jaccard stability score of the top-K sets
        if prev_top_k or curr_top_k:
            stability_score: float | None = len(prev_top_k & curr_top_k) / len(
                prev_top_k | curr_top_k
            )
        else:
            stability_score = 1.0

        return {
            "rank_changes": rank_changes,
            "new_entrants": new_entrants,
            "dropouts": dropouts,
            "biggest_risers": biggest_risers,
            "biggest_fallers": biggest_fallers,
            "stability_score": stability_score,
        }

    def _detect_rank_anomalies(self, rank_changes: dict) -> set:
        """
        Flag nodes whose absolute rank change exceeds the configured percentile.

        Mirrors ``utils.detect_rank_anomalies`` exactly.

        Parameters
        ----------
        rank_changes : dict
            ``{node_id: signed_rank_change}`` as returned by
            :meth:`_compute_rank_stability`.

        Returns
        -------
        set
            Node IDs whose ``|rank_change|`` is at or above the
            ``threshold_percentile``-th percentile of all absolute changes.
            Returns an empty set when *rank_changes* is empty.
        """
        if not rank_changes:
            return set()

        abs_changes = np.abs(np.array(list(rank_changes.values())))
        threshold: float = float(np.percentile(abs_changes, self._threshold_percentile))

        return {
            node for node, change in rank_changes.items() if abs(change) >= threshold
        }

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def compute(self, current_scores: dict) -> dict | None:
        """
        Compute rank-stability features for the current window.

        Compares *current_scores* against the internally stored
        ``previous_scores``, updates the internal state, and returns a
        result bundle.

        If ``previous_scores`` is empty (i.e. this is the first window),
        the method stores *current_scores* and returns ``None`` — there is
        no previous window to compare against so no stability features can
        be produced.

        Parameters
        ----------
        current_scores : dict
            ``{node_id: pagerank_score}`` mapping for the current window.
            Typically the flattened output of
            :class:`~src.features.centrality.PageRankVolumeExtractor`.

        Returns
        -------
        dict or None
            ``None`` on the first call (no previous window).

            Otherwise a dictionary with the following keys:

            * ``rank_changes``    – ``{node_id: signed_rank_change}``
              for nodes present in both windows.  Use this to add a
              per-node ``pagerank_rank_change`` column to the feature table.
            * ``anomalies``       – ``set`` of node IDs flagged as rank
              anomalies (absolute change above the percentile threshold).
              Use this to populate ``is_rank_anomaly`` in the feature table.
            * ``stability_score`` – Jaccard similarity of the top-K sets
              (``float`` in ``[0, 1]``).
            * ``num_new_entrants``– number of nodes that entered the top-K.
            * ``num_dropouts``    – number of nodes that left the top-K.
            * ``num_anomalies``   – number of flagged anomaly nodes.
        """
        # First window: no previous data — store and return None.
        if not self.previous_scores:
            self.previous_scores = current_scores
            return None

        # Compute stability metrics between the two windows.
        stability = self._compute_rank_stability(self.previous_scores, current_scores)
        anomalies = self._detect_rank_anomalies(stability["rank_changes"])

        # Advance internal state *after* computing so that the returned
        # result always reflects the (previous → current) transition.
        self.previous_scores = current_scores

        return {
            "rank_changes": stability["rank_changes"],
            "anomalies": anomalies,
            "stability_score": stability["stability_score"],
            "num_new_entrants": len(stability["new_entrants"]),
            "num_dropouts": len(stability["dropouts"]),
            "num_anomalies": len(anomalies),
        }

    def reset(self) -> None:
        """
        Clear internal state, resetting the tracker to its initial condition.

        Useful when re-running the pipeline on a different dataset or date
        range without re-instantiating the object.
        """
        self.previous_scores = {}

    def __repr__(self) -> str:
        state = "initialised" if not self.previous_scores else "tracking"
        return (
            f"RankStabilityTracker("
            f"top_k={self._top_k}, "
            f"threshold_percentile={self._threshold_percentile}, "
            f"state={state!r})"
        )
