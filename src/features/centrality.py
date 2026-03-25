from __future__ import annotations
import igraph as ig
from src.features.base import FeatureExtractor

class PageRankVolumeExtractor(FeatureExtractor):
    def __init__(self, alpha: float, max_iter: int) -> None:
        self._alpha = alpha
        self._max_iter = max_iter # Kept for pipeline compatibility

    @property
    def name(self) -> str:
        return f"PageRankVolumeExtractor(alpha={self._alpha})"

    def extract(self, G: ig.Graph) -> dict[object, dict[str, float]]:
        if G.vcount() == 0:
            return {}

        try:
            scores = G.pagerank(
                vertices=None, directed=True, damping=self._alpha, 
                weights="volume"
            )
        except ig.InternalError:
            return {}

        return {name: {"pagerank": score} for name, score in zip(G.vs["name"], scores)}

class PageRankFrequencyExtractor(FeatureExtractor):
    def __init__(self, alpha: float, max_iter: int) -> None:
        self._alpha = alpha
        self._max_iter = max_iter # Kept for pipeline compatibility

    @property
    def name(self) -> str:
        return f"PageRankFrequencyExtractor(alpha={self._alpha})"

    def extract(self, G: ig.Graph) -> dict[object, dict[str, float]]:
        if G.vcount() == 0:
            return {}

        try:
            scores = G.pagerank(
                vertices=None, directed=True, damping=self._alpha, 
                weights="count"
            )
        except ig.InternalError:
            return {}

        return {name: {"pagerank_count": score} for name, score in zip(G.vs["name"], scores)}

class HITSExtractor(FeatureExtractor):
    def __init__(self, max_iter: int = 500) -> None:
        self._max_iter = max_iter

    @property
    def name(self) -> str:
        return f"HITSExtractor(max_iter={self._max_iter})"

    def extract(self, G: ig.Graph) -> dict[object, dict[str, float]]:
        if G.vcount() == 0:
            return {}

        try:
            # hub_score and authority_score do not accept max_iter in python-igraph
            hubs = G.hub_score()
            auths = G.authority_score()
        except ig.InternalError:
            return {}

        return {
            name: {"hits_hub": h, "hits_auth": a} 
            for name, h, a in zip(G.vs["name"], hubs, auths)
        }

class BetweennessExtractor(FeatureExtractor):
    def __init__(self, k: int = 50, seed: int = 42) -> None:
        self._k = k
        self._seed = seed if seed is not None else 42
        self._cutoff = 6 # Bounds the BFS search depth

    @property
    def name(self) -> str:
        return f"BetweennessExtractor(cutoff={self._cutoff})"

    def extract(self, G: ig.Graph) -> dict[object, dict[str, float]]:
        if G.vcount() == 0:
            return {}

        try:
            # The cutoff parameter prevents the O(|V||E|) explosion
            b_scores = G.betweenness(directed=True, cutoff=self._cutoff)
        except ig.InternalError:
            return {}

        return {name: {"betweenness": score} for name, score in zip(G.vs["name"], b_scores)}
