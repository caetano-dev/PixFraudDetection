"""
Subgraph motif counting feature extractor based on AMLworld paper (Section 3.2).

This module implements memory-efficient O(V⋅D²) algorithms for laundering topology
detection in large multi-day cumulative graphs. It strictly avoids naive subgraph
isomorphism (O(V³)) to stay within 6GB memory constraints.

Motif patterns:
- Fan-out / Fan-in: High-degree node patterns
- Scatter-Gather / Gather-Scatter: 2-hop multipath structures
- Bounded Simple Cycles: Short cycles (length ≤ 4)
"""

from __future__ import annotations

import networkx as nx

from src.features.base import FeatureExtractor


class SubgraphMotifExtractor(FeatureExtractor):
    """
    Extract subgraph motif counts from transaction graphs.
    
    Uses highly optimized set intersection algorithms for multi-hop patterns
    and bounded generators for cycle detection to avoid combinatorial explosion.
    
    Parameters
    ----------
    fan_threshold : int, optional
        Minimum degree to count as fan-out/fan-in pattern (default: 5)
    cycle_bound : int, optional
        Maximum cycle length to search (default: 4)
    """
    
    def __init__(self, fan_threshold: int = 5, cycle_bound: int = 4):
        self.fan_threshold = fan_threshold
        self.cycle_bound = cycle_bound
    
    def extract(self, G: nx.DiGraph) -> dict[object, dict[str, float | int]]:
        """
        Extract motif features using memory-efficient algorithms.
        
        Parameters
        ----------
        G : nx.DiGraph
            The directed transaction graph
            
        Returns
        -------
        dict
            Nested dictionary mapping node_id -> {feature_name: count}
        """
        if not G.nodes():
            return {}
        
        features = {
            node: {
                "fan_out_count": 0,
                "fan_in_count": 0,
                "scatter_gather_count": 0,
                "gather_scatter_count": 0,
                "cycle_count": 0,
            }
            for node in G.nodes()
        }
        
        # Fan-out / Fan-in: O(V)
        for node in G.nodes():
            out_deg = G.out_degree(node)
            in_deg = G.in_degree(node)
            
            if out_deg > self.fan_threshold:
                features[node]["fan_out_count"] = 1
            if in_deg > self.fan_threshold:
                features[node]["fan_in_count"] = 1
        
        # Scatter-Gather / Gather-Scatter: O(V⋅D²) via set intersections
        for u in G.nodes():
            u_successors = set(G.successors(u))
            
            if len(u_successors) < 2:
                continue
            
            # For each 2nd-hop node V, check if multiple paths exist from U
            second_hop_nodes = set()
            for mid in u_successors:
                second_hop_nodes.update(G.successors(mid))
            
            for v in second_hop_nodes:
                if v == u:
                    continue
                
                # Count how many paths U -> mid -> V exist
                v_predecessors = set(G.predecessors(v))
                shared_nodes = u_successors & v_predecessors
                
                if len(shared_nodes) >= 2:
                    # U has scatter-gather pattern (U -> [mid1, mid2, ...] -> V)
                    features[u]["scatter_gather_count"] += 1
                    # V has gather-scatter pattern ([mid1, mid2, ...] -> V)
                    features[v]["gather_scatter_count"] += 1
        
        # Bounded Simple Cycles: O(V⋅D^k) with strict bound
        try:
            # Use length_bound to prevent infinite generator hanging
            cycles = nx.simple_cycles(G, length_bound=self.cycle_bound)
            
            for cycle in cycles:
                for node in cycle:
                    if node in features:
                        features[node]["cycle_count"] += 1
        except (nx.NetworkXError, nx.NetworkXNoCycle):
            pass
        
        return features
