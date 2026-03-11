"""
Centrality-based feature extractors for the PixFraudDetection pipeline.

This module provides four concrete :class:`~src.features.base.FeatureExtractor`
strategies that wrap NetworkX centrality algorithms:

* :class:`PageRankVolumeExtractor`   — PageRank weighted by transaction *volume*
  (``weight='weight'``).  Surfaces "heavy-hitter" nodes moving large sums.
* :class:`PageRankFrequencyExtractor` — PageRank weighted by transaction
  *frequency* (``weight='count'``).  Surfaces "smurfing" nodes making many
  small transactions.
* :class:`HITSExtractor`             — HITS algorithm, returning both hub and
  authority scores simultaneously.
* :class:`BetweennessExtractor`      — Approximate betweenness centrality using
  a random sample of ``k`` pivot nodes for scalability.

All classes preserve the **exact** NetworkX calls, hyper-parameters, and
error-handling behaviour from the original ``main.py``.  No mathematical logic
has been altered.
"""

from __future__ import annotations

import networkx as nx

from src.features.base import FeatureExtractor


class PageRankVolumeExtractor(FeatureExtractor):
    """
    PageRank weighted by aggregated transaction *volume*.
    """
    def __init__(self, alpha: float, max_iter: int) -> None:
        self._alpha = alpha
        self._max_iter = max_iter

    @property
    def name(self) -> str:
        return f"PageRankVolumeExtractor(alpha={self._alpha})"

    def extract(self, G: nx.DiGraph) -> dict[object, dict[str, float]]:
        """
        Run PageRank (volume-weighted) on *G*.

        Parameters
        ----------
        G : nx.DiGraph
            Directed transaction graph for the current window.

        Returns
        -------
        dict
            ``{node_id: {"pagerank": score}}`` for every node in *G*, or
            an empty dict if *G* has no nodes or PageRank fails to converge.
        """
        if len(G) == 0:
            return {}

        try:
            scores: dict = nx.pagerank(G, weight="weight", alpha=self._alpha, max_iter=self._max_iter)
        except (nx.NetworkXError, nx.PowerIterationFailedConvergence):
            return {}

        return {node: {"pagerank": score} for node, score in scores.items()}


class PageRankFrequencyExtractor(FeatureExtractor):
    """
    PageRank weighted by transaction *frequency* (count).
    """
    def __init__(self, alpha: float, max_iter: int) -> None:
        self._alpha = alpha
        self._max_iter = max_iter

    @property
    def name(self) -> str:
        return f"PageRankFrequencyExtractor(alpha={self._alpha})"

    def extract(self, G: nx.DiGraph) -> dict[object, dict[str, float]]:
        """
        Run PageRank (frequency-weighted) on *G*.

        Parameters
        ----------
        G : nx.DiGraph
            Directed transaction graph for the current window.

        Returns
        -------
        dict
            ``{node_id: {"pagerank_count": score}}`` for every node in *G*,
            or an empty dict if *G* has no nodes or PageRank fails to
            converge.
        """
        if len(G) == 0:
            return {}

        try:
            scores: dict = nx.pagerank(G, weight="count", alpha=self._alpha, max_iter=self._max_iter)
        except (nx.NetworkXError, nx.PowerIterationFailedConvergence):
            return {}

        return {node: {"pagerank_count": score} for node, score in scores.items()}


class HITSExtractor(FeatureExtractor):
    """
    Hyperlink-Induced Topic Search (HITS) algorithm.

    Computes both *hub* and *authority* scores for every node in the graph
    in a single pass.  In the AML context:

    * **Hub score** — nodes that send money to many high-authority nodes
      (potential money-mule coordinators).
    * **Authority score** — nodes that receive money from many high-hub nodes
      (potential collection accounts).

    Parameters
    ----------
    max_iter : int
        Maximum number of power-iteration steps before declaring
        non-convergence.  Defaults to :data:`src.config.HITS_MAX_ITER` (100).

    Returns (via ``extract``)
    -------------------------
    ``{"hits_hub": float, "hits_auth": float}`` per node, or ``{}`` on
    algorithm failure.
    """

    def __init__(self, max_iter: int = 500) -> None:
        self._max_iter = max_iter

    @property
    def name(self) -> str:
        return f"HITSExtractor(max_iter={self._max_iter})"

    def extract(self, G: nx.DiGraph) -> dict[object, dict[str, float]]:
        """
        Run HITS on *G* and return combined hub + authority features.

        Parameters
        ----------
        G : nx.DiGraph
            Directed transaction graph for the current window.

        Returns
        -------
        dict
            ``{node_id: {"hits_hub": hub_score, "hits_auth": auth_score}}``
            for every node in *G*, or an empty dict if *G* has no nodes or
            HITS fails to converge.
        """
        if len(G) == 0:
            return {}

        try:
            hubs, auths = nx.hits(G, max_iter=self._max_iter)
        except (nx.NetworkXError, nx.PowerIterationFailedConvergence):
            return {}

        return {
            node: {"hits_hub": hubs.get(node, 0.0), "hits_auth": auths.get(node, 0.0)}
            for node in G.nodes()
        }


class BetweennessExtractor(FeatureExtractor):
    """
    Approximate betweenness centrality extractor.

    Computes betweenness centrality using a random sample of ``k`` pivot
    nodes rather than the exact all-pairs shortest-path algorithm.  This
    makes it practical on the large daily transaction graphs seen in the
    PixFraudDetection pipeline, where the exact O(VE) algorithm would be
    prohibitively slow.

    A high betweenness score identifies *broker* nodes that sit on many
    shortest paths between other nodes — a structural signature of money-
    mule accounts that act as intermediaries in layering schemes.

    Parameters
    ----------
    k : int
        Number of pivot nodes sampled for the approximation.
        Larger values → more accurate estimate, higher runtime cost.
        Defaults to ``50``.
    seed : int
        Random seed forwarded to NetworkX for reproducibility across runs.
        Defaults to ``42``.

    Returns (via ``extract``)
    -------------------------
    ``{"betweenness": float}`` per node, or ``{}`` on algorithm failure.
    """

    def __init__(self, k: int = 50, seed: int = 42) -> None:
        self._k = k
        self._seed = seed if seed is not None else 42

    @property
    def name(self) -> str:
        return f"BetweennessExtractor(k={self._k}, seed={self._seed})"

    def extract(self, G: nx.DiGraph) -> dict[object, dict[str, float]]:
        """
        Run approximate betweenness centrality on *G*.

        Parameters
        ----------
        G : nx.DiGraph
            Directed transaction graph for the current window.

        Returns
        -------
        dict
            ``{node_id: {"betweenness": score}}`` for every node in *G*,
            or an empty dict if *G* has no nodes or the algorithm raises a
            recoverable NetworkX error.
        """
        if len(G) == 0:
            return {}

        try:
            b_scores: dict = nx.betweenness_centrality(G, k=self._k, seed=self._seed)
        except nx.NetworkXError:
            return {}

        return {node: {"betweenness": score} for node, score in b_scores.items()}
