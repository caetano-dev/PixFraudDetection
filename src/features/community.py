"""
Community-detection feature extractor for the PixFraudDetection pipeline.

This module provides two concrete :class:`~src.features.base.FeatureExtractor`
strategies that wrap community-structure algorithms:

* :class:`LeidenCommunityExtractor` — assigns every node in the graph a
  community ID and community size.  Isolated nodes and any nodes that fall
  through the leidenalg conversion are guaranteed a singleton community so
  that downstream feature tables have no ``NaN`` values.
* :class:`KCoreExtractor` — computes the k-core number of every node via
  k-core decomposition on an undirected projection of the graph.  A node's
  core number is the largest *k* such that the node belongs to the k-core
  (the maximal subgraph in which every node has degree >= k).

The ``resolution`` parameter controls community granularity:
    * ``resolution=1.0``  (macro) — larger, coarser communities
    * ``resolution=2.0``  (micro) — smaller, more granular communities

Resolution is implemented via ``leidenalg.RBConfigurationVertexPartition``,
which exposes ``resolution_parameter`` directly.  This replaces the previous
``cdlib.algorithms.leiden()`` call, which did not expose this parameter.

All fallback strategies and self-loop removal logic are preserved verbatim
from ``utils.compute_leiden_features``.  No mathematical logic has been altered.
"""

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
    """
    Run the Leiden algorithm on an undirected NetworkX graph and return the
    raw community partition as a list of node-lists.

    Uses ``leidenalg.RBConfigurationVertexPartition`` directly so that the
    ``resolution_parameter`` can be forwarded without restriction.

    Parameters
    ----------
    G_undirected : nx.Graph
        An undirected NetworkX graph.  Self-loops must be removed by the
        caller before this function is invoked.
    resolution : float
        Resolution parameter forwarded to
        ``leidenalg.RBConfigurationVertexPartition``.
        Higher values → smaller, more granular communities.
        Lower values  → larger, coarser communities.
        Defaults to ``1.0``.

    Returns
    -------
    list[list]
        A list of communities, where each community is a list of node IDs
        drawn from the original NetworkX graph (not igraph integer IDs).

    Raises
    ------
    Exception
        Propagates any error from leidenalg / igraph so that the caller can
        decide how to handle it (retry per component, fall back to treating
        the whole component as one community, etc.).
    """
    # Convert NetworkX graph → igraph, preserving node identity.
    # ig.Graph.from_networkx() stores the original node label in the
    # "_nx_name" vertex attribute.
    ig_graph = ig.Graph.from_networkx(G_undirected)

    # Use edge weights when available.
    weight_attr = "count" if "count" in ig_graph.edge_attributes() else None

    partition = leidenalg.find_partition(
        ig_graph,
        leidenalg.RBConfigurationVertexPartition,
        weights=weight_attr,
        resolution_parameter=resolution,
    )

    # Map igraph integer vertex IDs back to original NetworkX node labels.
    nx_names: list = ig_graph.vs["_nx_name"]
    communities: list[list] = [[nx_names[v] for v in cluster] for cluster in partition]
    return communities


# ---------------------------------------------------------------------------
# Concrete strategy
# ---------------------------------------------------------------------------


class LeidenCommunityExtractor(FeatureExtractor):
    """
    Leiden community detection feature extractor.

    Assigns every node in the directed graph *G* a ``leiden_id`` and
    ``leiden_size`` by running the Leiden algorithm on an undirected
    projection of the graph.

    Three layers of resilience guarantee **100 % node coverage**:

    1. Run Leiden on the full undirected projection of *G*.
    2. If (1) fails, fall back to running Leiden on each connected component
       independently.
    3. Any nodes still unassigned after (1) or (2) — isolated nodes, nodes
       dropped during the igraph conversion, or survivors of internal errors
       — are each placed into their own singleton community.

    Self-loops are removed from the undirected projection before community
    detection because they carry no inter-node structure information and can
    cause numerical issues in leidenalg.

    For reciprocal directed edges ``(u → v)`` and ``(v → u)`` the undirected
    projection aggregates their weights by addition, preventing arbitrary
    data loss.

    Parameters
    ----------
    resolution : float
        Resolution parameter forwarded to
        ``leidenalg.RBConfigurationVertexPartition``.
        Higher values → smaller, more granular communities (``micro``).
        Lower values  → larger, coarser communities (``macro``).
        Defaults to ``1.0``.

    Returns (via ``extract``)
    -------------------------
    ``{"leiden_id": int, "leiden_size": int}`` per node, or ``{}`` if *G*
    has no nodes.

    Examples
    --------
    >>> macro = LeidenCommunityExtractor(resolution=1.0)
    >>> micro = LeidenCommunityExtractor(resolution=2.0)
    """

    def __init__(self, resolution: float = 1.0) -> None:
        if resolution <= 0:
            raise ValueError(f"resolution must be > 0, got {resolution}.")
        self._resolution = resolution

    @property
    def name(self) -> str:
        return f"LeidenCommunityExtractor(resolution={self._resolution})"

    def extract(self, G: nx.DiGraph) -> dict[object, dict[str, float | int]]:
        """
        Run Leiden community detection on *G*.

        Parameters
        ----------
        G : nx.DiGraph
            Directed transaction graph for the current window.

        Returns
        -------
        dict
            ``{node_id: {"leiden_id": community_id, "leiden_size": size}}``
            for **every** node in *G*, or an empty dict if *G* has no nodes.
        """
        if len(G) == 0:
            return {}

        all_nodes: set = set(G.nodes())

        # ------------------------------------------------------------------ #
        # 1. Build a clean undirected projection                               #
        # ------------------------------------------------------------------ #
        # Manually aggregate weights to avoid losing data on reciprocal edges.
        G_undirected = nx.Graph()
        for u, v, data in G.edges(data=True):
            if G_undirected.has_edge(u, v):
                G_undirected[u][v]["weight"] += data.get("weight", 0.0)
            else:
                G_undirected.add_edge(u, v, weight=data.get("weight", 0.0))

        # Remove self-loops — they carry no community-structure information
        # and can trigger numerical issues inside leidenalg.
        G_undirected.remove_edges_from(nx.selfloop_edges(G_undirected))

        # ------------------------------------------------------------------ #
        # 2. Try Leiden on the full graph                                      #
        # ------------------------------------------------------------------ #
        node_features: dict = {}
        next_community_id: int = 0

        try:
            raw_communities = _run_leiden_on_graph(
                G_undirected, resolution=self._resolution
            )

            # Compute modularity using igraph's native method so we can
            # mathematically defend the clustering quality at thesis defence.
            # We need a fresh igraph conversion here because _run_leiden_on_graph
            # returns plain Python lists; igraph.modularity() requires the
            # membership vector aligned to its own vertex ordering.
            _ig_for_mod = ig.Graph.from_networkx(G_undirected)
            _nx_names_mod: list = _ig_for_mod.vs["_nx_name"]
            _name_to_idx: dict = {name: idx for idx, name in enumerate(_nx_names_mod)}

            # Build a membership vector in igraph vertex order from the
            # community lists returned by _run_leiden_on_graph.
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

            # -------------------------------------------------------------- #
            # 3. Fallback: run Leiden on each connected component separately   #
            # -------------------------------------------------------------- #
            for component in nx.connected_components(G_undirected):
                if len(component) < 2:
                    # Isolated / trivial components are handled in step 4.
                    continue

                subgraph = G_undirected.subgraph(component).copy()
                try:
                    sub_communities = _run_leiden_on_graph(
                        subgraph, resolution=self._resolution
                    )

                    # Compute per-component modularity for the fallback path.
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
                    # Last resort: treat the entire component as one community.
                    # Modularity of a single-community partition is 0 by definition.
                    community_size = len(component)
                    for node in component:
                        node_features[node] = {
                            "leiden_id": next_community_id,
                            "leiden_size": community_size,
                            "leiden_modularity": 0.0,
                        }
                    next_community_id += 1

        # ------------------------------------------------------------------ #
        # 4. Guarantee 100 % coverage — assign singleton communities to any   #
        #    node not yet assigned (isolated nodes, igraph conversion drops,   #
        #    etc.)                                                             #
        # ------------------------------------------------------------------ #
        missing_nodes = all_nodes - set(node_features.keys())
        if missing_nodes:
            for node in missing_nodes:
                # Singleton communities have no cross-community edges, so their
                # contribution to modularity is 0.
                node_features[node] = {
                    "leiden_id": next_community_id,
                    "leiden_size": 1,
                    "leiden_modularity": 0.0,
                }
                next_community_id += 1

        return node_features


class KCoreExtractor(FeatureExtractor):
    """
    Computes the k-core number for every node.

    The graph is first cast to a standard undirected nx.Graph. 
    Self-loops are explicitly removed, as k-core decomposition is 
    mathematically undefined for self-looping topologies in NetworkX.

    Returns (via ``extract``)
    -------------------------
    ``{"k_core": int}`` per node, or ``{}`` on algorithm failure.
    """

    @property
    def name(self) -> str:
        return "KCoreExtractor()"

    def extract(self, G: nx.DiGraph) -> dict[object, dict[str, float | int]]:
        """
        Run k-core decomposition on an undirected projection of *G*.

        Parameters
        ----------
        G : nx.DiGraph
            Directed transaction graph for the current window.

        Returns
        -------
        dict
            ``{node_id: {"k_core": core_number}}`` for every node in *G*,
            or an empty dict if *G* has no nodes or the algorithm raises a
            recoverable NetworkX error.
        """
        if len(G) == 0:
            return {}

        try:
            G_un = nx.Graph(G)
            G_un.remove_edges_from(nx.selfloop_edges(G_un))
            core_scores: dict = nx.core_number(G_un)
        except nx.NetworkXError:
            return {}

        return {node: {"k_core": score} for node, score in core_scores.items()}
