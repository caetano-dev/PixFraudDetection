from __future__ import annotations

import networkx as nx

from src.features.base import FeatureExtractor


class PageRankVolumeExtractor(FeatureExtractor):
    def __init__(self, alpha: float, max_iter: int) -> None:
        self._alpha = alpha
        self._max_iter = max_iter

    @property
    def name(self) -> str:
        return f"PageRankVolumeExtractor(alpha={self._alpha})"

    def extract(self, G: nx.DiGraph) -> dict[object, dict[str, float]]:
        if len(G) == 0:
            return {}

        try:
            scores: dict = nx.pagerank(G, weight="volume", alpha=self._alpha, max_iter=self._max_iter)
        except (nx.NetworkXError, nx.PowerIterationFailedConvergence):
            return {}

        return {node: {"pagerank": score} for node, score in scores.items()}


class PageRankFrequencyExtractor(FeatureExtractor):
    def __init__(self, alpha: float, max_iter: int) -> None:
        self._alpha = alpha
        self._max_iter = max_iter

    @property
    def name(self) -> str:
        return f"PageRankFrequencyExtractor(alpha={self._alpha})"

    def extract(self, G: nx.DiGraph) -> dict[object, dict[str, float]]:
        if len(G) == 0:
            return {}

        try:
            scores: dict = nx.pagerank(G, weight="count", alpha=self._alpha, max_iter=self._max_iter)
        except (nx.NetworkXError, nx.PowerIterationFailedConvergence):
            return {}

        return {node: {"pagerank_count": score} for node, score in scores.items()}


class HITSExtractor(FeatureExtractor):
    def __init__(self, max_iter: int = 500) -> None:
        self._max_iter = max_iter

    @property
    def name(self) -> str:
        return f"HITSExtractor(max_iter={self._max_iter})"

    def extract(self, G: nx.DiGraph) -> dict[object, dict[str, float]]:
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
    def __init__(self, k: int = 50, seed: int = 42) -> None:
        self._k = k
        self._seed = seed if seed is not None else 42

    @property
    def name(self) -> str:
        return f"BetweennessExtractor(k={self._k}, seed={self._seed})"

    def extract(self, G: nx.DiGraph) -> dict[object, dict[str, float]]:
        if len(G) == 0:
            return {}

        try:
            b_scores: dict = nx.betweenness_centrality(G, k=self._k, seed=self._seed)
        except nx.NetworkXError:
            return {}

        return {node: {"betweenness": score} for node, score in b_scores.items()}
