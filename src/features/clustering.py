from __future__ import annotations

import networkx as nx

from .base import FeatureExtractor


class ClusteringExtractor(FeatureExtractor):
    """
    Extracts clustering-based features for each node.
    
    Features:
        - local_clustering_coefficient: Weighted clustering coefficient using edge 'volume'
        - triangle_count: Number of triangles the node participates in (undirected projection)
    
    Handles isolated nodes and zero-degree nodes gracefully.
    """
    
    def extract(self, G: nx.DiGraph) -> dict[object, dict[str, float | int]]:
        """
        Extract clustering features for all nodes in the graph.
        
        Args:
            G: Directed graph with 'volume' edge attribute
            
        Returns:
            Dictionary mapping node -> feature_dict with clustering metrics
        """
        if len(G) == 0:
            return {}
        
        result = {}
        
        # Compute weighted clustering coefficient on the directed graph
        # nx.clustering handles weighted graphs and returns 0.0 for isolated nodes
        try:
            clustering_coeffs = nx.clustering(G, weight='volume')
        except Exception:
            # Fallback: unweighted clustering or all zeros
            try:
                clustering_coeffs = nx.clustering(G)
            except Exception:
                clustering_coeffs = {node: 0.0 for node in G.nodes()}
        
        # For triangle count, convert to undirected and remove self-loops
        # Triangles are only meaningful in undirected graphs
        try:
            # Create an undirected copy without self-loops
            G_undirected = G.to_undirected()
            G_undirected.remove_edges_from(nx.selfloop_edges(G_undirected))
            
            # Compute triangles for each node
            # nx.triangles returns a dict {node: triangle_count}
            triangles = nx.triangles(G_undirected)
        except Exception:
            # Fallback: all nodes have zero triangles
            triangles = {node: 0 for node in G.nodes()}
        
        # Combine results
        for node in G.nodes():
            result[node] = {
                'local_clustering_coefficient': clustering_coeffs.get(node, 0.0),
                'triangle_count': triangles.get(node, 0)
            }
        
        return result
    
    @property
    def name(self) -> str:
        return "ClusteringExtractor"
