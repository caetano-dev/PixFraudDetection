from __future__ import annotations
import warnings
import igraph as ig
import leidenalg
from src.features.base import FeatureExtractor

class LeidenCommunityExtractor(FeatureExtractor):
    def __init__(self, resolution: float = 1.0) -> None:
        if resolution <= 0:
            raise ValueError(f"resolution must be > 0, got {resolution}.")
        self._resolution = resolution

    @property
    def name(self) -> str:
        return f"LeidenCommunityExtractor(resolution={self._resolution})"

    def extract(self, G: ig.Graph) -> dict[object, dict[str, float | int]]:
        if G.vcount() == 0:
            return {}

        node_features = {}
        
        try:
            G_un = G.as_undirected(mode="collapse", combine_edges={"count": "sum", "volume": "sum"})
            G_un.simplify(multiple=True, loops=True, combine_edges={"count": "sum", "volume": "sum"})
            
            weight_attr = "count" if "count" in G_un.edge_attributes() else None

            partition = leidenalg.find_partition(
                G_un,
                leidenalg.RBConfigurationVertexPartition,
                weights=weight_attr,
                resolution_parameter=self._resolution,
            )

            mod_score = G_un.modularity(partition.membership)

            for comm_idx, cluster in enumerate(partition):
                community_size = len(cluster)
                for v_idx in cluster:
                    node_name = G_un.vs[v_idx]["name"]
                    node_features[node_name] = {
                        "leiden_id": comm_idx,
                        "leiden_size": community_size,
                        "leiden_modularity": mod_score,
                    }
                    
        except Exception as exc:
            warnings.warn(f"Leiden failed: {exc}", RuntimeWarning, stacklevel=2)
            for name in G.vs["name"]:
                node_features[name] = {
                    "leiden_id": 0,
                    "leiden_size": 1,
                    "leiden_modularity": 0.0,
                }

        return node_features

class KCoreExtractor(FeatureExtractor):
    @property
    def name(self) -> str:
        return "KCoreExtractor()"

    def extract(self, G: ig.Graph) -> dict[object, dict[str, float | int]]:
        if G.vcount() == 0:
            return {}

        try:
            G_un = G.as_undirected(mode="collapse")
            G_un.simplify(multiple=True, loops=True)
            core_scores = G_un.coreness(mode="all")
        except ig.InternalError:
            return {}

        return {name: {"k_core": score} for name, score in zip(G_un.vs["name"], core_scores)}
