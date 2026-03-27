from __future__ import annotations

import networkx as nx

from .base import FeatureExtractor


class NeighborAggregationExtractor(FeatureExtractor):
    """
    Extracts neighbor aggregation features for each node.
    
    Features:
        - average_neighbor_degree: Average degree of neighboring nodes
        - successor_avg_volume: Average transaction volume to immediate successors
        - successor_max_volume: Maximum transaction volume to any immediate successor
    
    Handles nodes without successors gracefully by returning 0.0 fallback values.
    """
    
    def extract(self, G: nx.DiGraph) -> dict[object, dict[str, float]]:
        """
        Extract neighbor aggregation features for all nodes in the graph.
        
        Args:
            G: Directed graph with 'volume' edge attribute
            
        Returns:
            Dictionary mapping node -> feature_dict with neighbor aggregation metrics
        """
        if len(G) == 0:
            return {}
        
        result = {}
        
        # Compute average neighbor degree using NetworkX
        # This computes average degree of all neighbors (predecessors + successors)
        try:
            avg_neighbor_deg = nx.average_neighbor_degree(G)
        except Exception:
            # Fallback: all nodes get 0.0
            avg_neighbor_deg = {node: 0.0 for node in G.nodes()}
        
        # Compute successor volume statistics
        for node in G.nodes():
            # Get successors (outgoing neighbors)
            successors = list(G.successors(node))
            
            if len(successors) == 0:
                # No successors: fallback to 0.0
                successor_avg_vol = 0.0
                successor_max_vol = 0.0
            else:
                # Collect volumes of outgoing edges
                volumes = []
                for succ in successors:
                    # There might be multiple edges between node and successor
                    # Sum them if multigraph, or get single edge data
                    edge_data = G.get_edge_data(node, succ)
                    if edge_data is not None:
                        volumes.append(edge_data.get('volume', 0.0))
                
                if len(volumes) == 0:
                    successor_avg_vol = 0.0
                    successor_max_vol = 0.0
                else:
                    successor_avg_vol = sum(volumes) / len(volumes)
                    successor_max_vol = max(volumes)
            
            result[node] = {
                'average_neighbor_degree': avg_neighbor_deg.get(node, 0.0),
                'successor_avg_volume': successor_avg_vol,
                'successor_max_volume': successor_max_vol
            }
        
        return result
    
    @property
    def name(self) -> str:
        return "NeighborAggregationExtractor"
