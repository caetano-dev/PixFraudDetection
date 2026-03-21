from __future__ import annotations

import warnings

import igraph as ig
import leidenalg
import networkx as nx

from src.features.base import FeatureExtractor

# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------


def _run_leiden_on_graph(
    G_undirected: nx.Graph,
    resolution: float = 1.0,
) -> list[list]:
    ig_graph = ig.Graph.from_networkx(G_undirected)

    weight_attr = "count" if "count" in ig_graph.edge_attributes() else None

    partition = leidenalg.find_partition(
        ig_graph,
        leidenalg.RBConfigurationVertexPartition,
        weights=weight_attr,
        resolution_parameter=resolution,
    )

    nx_names: list = ig_graph.vs["_nx_name"]
    communities: list[list] = [[nx_names[v] for v in cluster] for cluster in partition]
    return communities

class LeidenCommunityExtractor(FeatureExtractor):
    def __init__(self, resolution: float = 1.0) -> None:
        if resolution <= 0:
            raise ValueError(f"resolution must be > 0, got {resolution}.")
        self._resolution = resolution

    @property
    def name(self) -> str:
        return f"LeidenCommunityExtractor(resolution={self._resolution})"

    def extract(self, G: nx.DiGraph) -> dict[object, dict[str, float | int]]:
        if len(G) == 0:
            return {}

        all_nodes: set = set(G.nodes())

        G_undirected = nx.Graph()
        for u, v, data in G.edges(data=True):
            if G_undirected.has_edge(u, v):
                G_undirected[u][v]["weight"] += data.get("weight", 0.0)
            else:
                G_undirected.add_edge(u, v, weight=data.get("weight", 0.0))

        G_undirected.remove_edges_from(nx.selfloop_edges(G_undirected))

        node_features: dict = {}
        next_community_id: int = 0

        try:
            raw_communities = _run_leiden_on_graph(
                G_undirected, resolution=self._resolution
            )

            _ig_for_mod = ig.Graph.from_networkx(G_undirected)
            _nx_names_mod: list = _ig_for_mod.vs["_nx_name"]
            _name_to_idx: dict = {name: idx for idx, name in enumerate(_nx_names_mod)}

            _membership: list[int] = [0] * _ig_for_mod.vcount()
            for comm_idx, community_members in enumerate(raw_communities):
                for node in community_members:
                    if node in _name_to_idx:
                        _membership[_name_to_idx[node]] = comm_idx

            mod_score: float = _ig_for_mod.modularity(_membership)

            for comm_idx, community_members in enumerate(raw_communities):
                community_size = len(community_members)
                for node in community_members:
                    node_features[node] = {
                        "leiden_id": next_community_id,
                        "leiden_size": community_size,
                        "leiden_modularity": mod_score,
                    }
                next_community_id += 1

        except Exception as exc:
            warnings.warn(
                f"Leiden failed on full graph ({len(G)} nodes, "
                f"resolution={self._resolution}): {exc}. "
                "Falling back to per-component community detection.",
                RuntimeWarning,
                stacklevel=2,
            )

            for component in nx.connected_components(G_undirected):
                if len(component) < 2:
                    continue

                subgraph = G_undirected.subgraph(component).copy()
                try:
                    sub_communities = _run_leiden_on_graph(
                        subgraph, resolution=self._resolution
                    )

                    _ig_sub = ig.Graph.from_networkx(subgraph)
                    _sub_names: list = _ig_sub.vs["_nx_name"]
                    _sub_name_to_idx: dict = {
                        name: idx for idx, name in enumerate(_sub_names)
                    }
                    _sub_membership: list[int] = [0] * _ig_sub.vcount()
                    for comm_idx, community_members in enumerate(sub_communities):
                        for node in community_members:
                            if node in _sub_name_to_idx:
                                _sub_membership[_sub_name_to_idx[node]] = comm_idx
                    sub_mod_score: float = _ig_sub.modularity(_sub_membership)

                    for community_members in sub_communities:
                        community_size = len(community_members)
                        for node in community_members:
                            node_features[node] = {
                                "leiden_id": next_community_id,
                                "leiden_size": community_size,
                                "leiden_modularity": sub_mod_score,
                            }
                        next_community_id += 1
                except Exception:
                    community_size = len(component)
                    for node in component:
                        node_features[node] = {
                            "leiden_id": next_community_id,
                            "leiden_size": community_size,
                            "leiden_modularity": 0.0,
                        }
                    next_community_id += 1

        missing_nodes = all_nodes - set(node_features.keys())
        if missing_nodes:
            for node in missing_nodes:
                node_features[node] = {
                    "leiden_id": next_community_id,
                    "leiden_size": 1,
                    "leiden_modularity": 0.0,
                }
                next_community_id += 1

        return node_features


class KCoreExtractor(FeatureExtractor):
    @property
    def name(self) -> str:
        return "KCoreExtractor()"

    def extract(self, G: nx.DiGraph) -> dict[object, dict[str, float | int]]:
        if len(G) == 0:
            return {}

        try:
            G_un = nx.Graph(G)
            G_un.remove_edges_from(nx.selfloop_edges(G_un))
            core_scores: dict = nx.core_number(G_un)
        except nx.NetworkXError:
            return {}

        return {node: {"k_core": score} for node, score in core_scores.items()}
