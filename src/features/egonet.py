from __future__ import annotations

import networkx as nx

from .base import FeatureExtractor


class EgonetExtractor(FeatureExtractor):
    """
    Extracts 1-hop neighborhood (egonet) features for each node.
    
    Features:
        - egonet_node_count: Number of nodes in the 1-hop neighborhood (including ego)
        - egonet_edge_count: Number of edges within the egonet
        - egonet_density: Density of the egonet subgraph
        - egonet_total_weight: Sum of edge volumes strictly within the egonet
    
    Handles isolated nodes gracefully by returning zeros/fallback values.
    """
    
    def __init__(self, radius: int = 1):
        """
        Args:
            radius: Radius of the ego network (default=1 for 1-hop neighborhood)
        """
        self.radius = radius
    
    def extract(self, G: nx.DiGraph) -> dict[object, dict[str, float | int]]:
        """
        Extract egonet features for all nodes in the graph.
        
        Args:
            G: Directed graph with 'volume' edge attribute
            
        Returns:
            Dictionary mapping node -> feature_dict with egonet metrics
        """
        if len(G) == 0:
            return {}
        
        result = {}
        
        for node in G.nodes():
            try:
                # Extract 1-hop neighborhood subgraph (ego + neighbors)
                egonet = nx.ego_graph(G, node, radius=self.radius, undirected=False)
                
                num_nodes = len(egonet)
                num_edges = egonet.number_of_edges()
                
                # Calculate density (handle isolated nodes and single-node egonets)
                # Density = m / (n * (n - 1)) for directed graphs
                if num_nodes <= 1:
                    density = 0.0
                else:
                    max_possible_edges = num_nodes * (num_nodes - 1)
                    density = num_edges / max_possible_edges if max_possible_edges > 0 else 0.0
                
                # Calculate total weight (sum of edge volumes within the egonet)
                total_weight = 0.0
                for u, v, data in egonet.edges(data=True):
                    total_weight += data.get('volume', 0.0)
                
                result[node] = {
                    'egonet_node_count': num_nodes,
                    'egonet_edge_count': num_edges,
                    'egonet_density': density,
                    'egonet_total_weight': total_weight
                }
                
            except Exception:
                # Fallback for any unexpected errors
                result[node] = {
                    'egonet_node_count': 1,  # At minimum, the node itself
                    'egonet_edge_count': 0,
                    'egonet_density': 0.0,
                    'egonet_total_weight': 0.0
                }
        
        return result
    
    @property
    def name(self) -> str:
        return f"EgonetExtractor(radius={self.radius})"
