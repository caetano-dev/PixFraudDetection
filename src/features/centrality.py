"""
Centrality-based feature extractors for the PixFraudDetection pipeline.

This module provides three concrete :class:`~src.features.base.FeatureExtractor`
strategies that wrap NetworkX centrality algorithms:

* :class:`PageRankVolumeExtractor`   — PageRank weighted by transaction *volume*
  (``weight='weight'``).  Surfaces "heavy-hitter" nodes moving large sums.
* :class:`PageRankFrequencyExtractor` — PageRank weighted by transaction
  *frequency* (``weight='count'``).  Surfaces "smurfing" nodes making many
  small transactions.
* :class:`HITSExtractor`             — HITS algorithm, returning both hub and
  authority scores simultaneously.

All three classes preserve the **exact** NetworkX calls, hyper-parameters, and
error-handling behaviour from the original ``main.py``.  No mathematical logic
has been altered.
"""

from __future__ import annotations

import networkx as nx

from src.config import HITS_MAX_ITER, PAGERANK_ALPHA
from src.features.base import FeatureExtractor


class PageRankVolumeExtractor(FeatureExtractor):
    """
    PageRank weighted by aggregated transaction *volume*.

    Uses ``weight='weight'``, where the ``weight`` edge attribute is the
    Oddball-inspired composite weight computed by
    :func:`src.graph.builder.build_daily_graph`:

        W_edge = volume * log2(1 + count) * (1 + 1 / (1 + CV))

    A high score identifies "heavy-hitter" nodes that are central to large
    flows of money — a primary signal for layering in AML detection.

    Parameters
    ----------
    alpha : float
        PageRank damping factor.  Defaults to :data:`src.config.PAGERANK_ALPHA`
        (0.85 — the standard NetworkX / Google value).

    Returns (via ``extract``)
    -------------------------
    ``{"pagerank": float}`` per node, or ``{}`` on algorithm failure.
    """

    def __init__(self, alpha: float = PAGERANK_ALPHA) -> None:
        self._alpha = alpha

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
            scores: dict = nx.pagerank(G, weight="weight", alpha=self._alpha)
        except (nx.NetworkXError, nx.PowerIterationFailedConvergence):
            return {}

        return {node: {"pagerank": score} for node, score in scores.items()}


class PageRankFrequencyExtractor(FeatureExtractor):
    """
    PageRank weighted by transaction *frequency* (count).

    Uses ``weight='count'``, where the ``count`` edge attribute is the raw
    number of individual transactions that were aggregated onto each directed
    edge by :func:`src.graph.builder.build_daily_graph`.

    A high score identifies "smurfing" nodes involved in a high number of
    transactions regardless of individual transaction size — a key pattern
    in structuring / smurfing AML typologies.

    Parameters
    ----------
    alpha : float
        PageRank damping factor.  Defaults to :data:`src.config.PAGERANK_ALPHA`.

    Returns (via ``extract``)
    -------------------------
    ``{"pagerank_count": float}`` per node, or ``{}`` on algorithm failure.
    """

    def __init__(self, alpha: float = PAGERANK_ALPHA) -> None:
        self._alpha = alpha

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
            scores: dict = nx.pagerank(G, weight="count", alpha=self._alpha)
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

    def __init__(self, max_iter: int = HITS_MAX_ITER) -> None:
        self._max_iter = max_iter

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
