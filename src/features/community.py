"""
Community-detection feature extractor for the PixFraudDetection pipeline.

This module provides a single concrete :class:`~src.features.base.FeatureExtractor`
strategy that wraps the Leiden community-detection algorithm via ``cdlib``:

* :class:`LeidenCommunityExtractor` — assigns every node in the graph a
  community ID and community size.  Isolated nodes and any nodes that fall
  through the cdlib / leidenalg conversion are guaranteed a singleton
  community so that downstream feature tables have no ``NaN`` values.

All cdlib calls, fallback strategies, and self-loop removal logic are
preserved verbatim from ``utils.compute_leiden_features`` and
``utils._run_leiden_on_graph``.  No mathematical logic has been altered.
"""

from __future__ import annotations

import warnings

import networkx as nx
from cdlib import algorithms

from src.features.base import FeatureExtractor

# ---------------------------------------------------------------------------
# Internal helper — mirrors utils._run_leiden_on_graph exactly
# ---------------------------------------------------------------------------


def _run_leiden_on_graph(G_undirected: nx.Graph) -> list[list]:
    """
    Run the Leiden algorithm on an undirected graph and return the raw
    community partition as a list of node-lists.

    This helper isolates the cdlib call so that fallback strategies can
    retry on individual connected components when the full-graph call fails.

    Parameters
    ----------
    G_undirected : nx.Graph
        An undirected NetworkX graph.  Self-loops should be removed by the
        caller before passing the graph here.

    Returns
    -------
    list[list]
        A list of communities, where each community is a list of node IDs.

    Raises
    ------
    Exception
        Propagates any error from cdlib / leidenalg so that the caller can
        decide how to handle it (retry per component, fall back to treating
        the whole component as one community, etc.).

    Notes
    -----
    cdlib's ``leiden()`` wrapper does not expose leidenalg's
    ``resolution_parameter``; ``LEIDEN_RESOLUTION`` from config is reserved
    for a future direct leidenalg integration.
    """
    communities = algorithms.leiden(
        G_undirected,
        weights="weight",
    )
    return communities.communities


# ---------------------------------------------------------------------------
# Concrete strategy
# ---------------------------------------------------------------------------


class LeidenCommunityExtractor(FeatureExtractor):
    """
    Leiden community detection feature extractor.

    Assigns every node in the directed graph *G* a ``leiden_id`` and
    ``leiden_size`` by running the Leiden algorithm (via ``cdlib``) on an
    undirected projection of the graph.

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

    Returns (via ``extract``)
    -------------------------
    ``{"leiden_id": int, "leiden_size": int}`` per node, or ``{}`` if *G*
    has no nodes.
    """

    def extract(self, G: nx.DiGraph) -> dict[object, dict[str, float | int]]:
        """
        Run Leiden community detection on *G*.

        Parameters
        ----------
        G : nx.DiGraph
            Directed transaction graph for the current window, as produced by
            :func:`src.graph.builder.build_daily_graph`.

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
            raw_communities = _run_leiden_on_graph(G_undirected)

            for community_members in raw_communities:
                community_size = len(community_members)
                for node in community_members:
                    node_features[node] = {
                        "leiden_id": next_community_id,
                        "leiden_size": community_size,
                    }
                next_community_id += 1

        except Exception as exc:
            warnings.warn(
                f"Leiden failed on full graph ({len(G)} nodes): {exc}. "
                "Falling back to per-component community detection.",
                RuntimeWarning,
                stacklevel=2,
            )

            # -------------------------------------------------------------- #
            # 3. Fallback: run Leiden on each connected component separately   #
            # -------------------------------------------------------------- #
            for component in nx.connected_components(G_undirected):
                if len(component) < 2:
                    # Isolated / trivial components are handled in step 4
                    # below alongside any other unassigned nodes.
                    continue

                subgraph = G_undirected.subgraph(component).copy()
                try:
                    sub_communities = _run_leiden_on_graph(subgraph)
                    for community_members in sub_communities:
                        community_size = len(community_members)
                        for node in community_members:
                            node_features[node] = {
                                "leiden_id": next_community_id,
                                "leiden_size": community_size,
                            }
                        next_community_id += 1
                except Exception:
                    # Last resort: treat the entire component as one community.
                    community_size = len(component)
                    for node in component:
                        node_features[node] = {
                            "leiden_id": next_community_id,
                            "leiden_size": community_size,
                        }
                    next_community_id += 1

        # ------------------------------------------------------------------ #
        # 4. Guarantee 100 % coverage — assign singleton communities to any   #
        #    node not yet assigned (isolated nodes, cdlib conversion drops,    #
        #    etc.)                                                             #
        # ------------------------------------------------------------------ #
        missing_nodes = all_nodes - set(node_features.keys())
        if missing_nodes:
            for node in missing_nodes:
                node_features[node] = {
                    "leiden_id": next_community_id,
                    "leiden_size": 1,
                }
                next_community_id += 1

        return node_features
