"""
Subgraph motif counting feature extractor based on AMLworld paper (Section 3.2).
"""

from __future__ import annotations

import networkx as nx

from src.features.base import FeatureExtractor


class SubgraphMotifExtractor(FeatureExtractor):
    def __init__(
        self, 
        fan_threshold: int = 5, 
        cycle_bound: int = 5, 
        max_degree: int = 16, 
        max_cycles: int = 1000
    ):
        self.fan_threshold = fan_threshold
        self.cycle_bound = cycle_bound
        self.max_degree = max_degree
        self.max_cycles = max_cycles
    
    @property
    def name(self) -> str:
        return (
            f"SubgraphMotifExtractor(fan={self.fan_threshold}, "
            f"max_deg={self.max_degree}, cycle_bound={self.cycle_bound}, "
            f"max_cycles_cap={self.max_cycles})"
        )

    def extract(self, G: nx.DiGraph) -> dict[object, dict[str, float | int]]:
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
        
        # 1. Fan-out / Fan-in (Strictly bounded)
        for node in G.nodes():
            out_deg = G.out_degree(node)
            in_deg = G.in_degree(node)
            
            if self.fan_threshold <= out_deg <= self.max_degree:
                features[node]["fan_out_count"] = 1
            if self.fan_threshold <= in_deg <= self.max_degree:
                features[node]["fan_in_count"] = 1
        
        # 2. Scatter-Gather / Gather-Scatter (Pruned to max 16x16 operations)
        for u in G.nodes():
            u_successors = set(G.successors(u))
            
            # Prune: U must fit the specific low-degree laundering profile
            if not (2 <= len(u_successors) <= self.max_degree):
                continue
            
            second_hop_nodes = set()
            for mid in u_successors:
                # Prune: Intermediate routing accounts also obey degree limits
                if G.out_degree(mid) <= self.max_degree:
                    second_hop_nodes.update(G.successors(mid))
            
            for v in second_hop_nodes:
                if v == u:
                    continue
                
                v_predecessors = set(G.predecessors(v))
                
                # Prune: Target V must fit the low-degree profile
                if not (2 <= len(v_predecessors) <= self.max_degree):
                    continue
                    
                shared_nodes = u_successors & v_predecessors
                
                if len(shared_nodes) >= 2:
                    features[u]["scatter_gather_count"] += 1
                    features[v]["gather_scatter_count"] += 1
        
        # 3. Bounded Simple Cycles (Operating on a decimated subgraph)
        # Pruning dead-ends and mega-hubs reduces the graph size by 90%+ 
        valid_cycle_nodes = [
            n for n in G.nodes() 
            if G.in_degree(n) > 0 
            and G.out_degree(n) > 0 
            and G.degree(n) <= self.max_degree
        ]
        
        if valid_cycle_nodes:
            cycle_subgraph = G.subgraph(valid_cycle_nodes)
            try:
                # Evaluate the generator lazily and break at max_cycles
                cycle_generator = nx.simple_cycles(cycle_subgraph, length_bound=self.cycle_bound)
                cycles_found = 0
                
                for cycle in cycle_generator:
                    for node in cycle:
                        features[node]["cycle_count"] += 1
                    
                    cycles_found += 1
                    if cycles_found >= self.max_cycles:
                        break
                        
            except (nx.NetworkXError, nx.NetworkXNoCycle, NotImplementedError):
                pass
        
        return features
