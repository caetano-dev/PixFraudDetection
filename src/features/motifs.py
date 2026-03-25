from __future__ import annotations
import igraph as ig
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

    def extract(self, G: ig.Graph) -> dict[object, dict[str, float | int]]:
        if G.vcount() == 0:
            return {}
        
        names = G.vs["name"]
        features = {
            name: {
                "fan_out_count": 0, "fan_in_count": 0,
                "scatter_gather_count": 0, "gather_scatter_count": 0,
                "cycle_count": 0,
            }
            for name in names
        }
        
        out_degrees = G.degree(mode="out")
        in_degrees = G.degree(mode="in")
        all_degrees = G.degree(mode="all")
        
        # 1. Fan-out / Fan-in
        for v in range(G.vcount()):
            name = names[v]
            if self.fan_threshold <= out_degrees[v] <= self.max_degree:
                features[name]["fan_out_count"] = 1
            if self.fan_threshold <= in_degrees[v] <= self.max_degree:
                features[name]["fan_in_count"] = 1

        # Pre-calculate adjacent nodes for scatter/gather mapping
        succs = [
            set(G.successors(v)) if 2 <= out_degrees[v] <= self.max_degree else set() 
            for v in range(G.vcount())
        ]
        preds = [
            set(G.predecessors(v)) if 2 <= in_degrees[v] <= self.max_degree else set() 
            for v in range(G.vcount())
        ]

        # 2. Scatter-Gather / Gather-Scatter
        for u in range(G.vcount()):
            u_succs = succs[u]
            if not u_succs:
                continue
            
            second_hop_nodes = set()
            for mid in u_succs:
                if out_degrees[mid] <= self.max_degree:
                    second_hop_nodes.update(G.successors(mid))
            
            for v in second_hop_nodes:
                if v == u:
                    continue
                v_preds = preds[v]
                if not v_preds:
                    continue
                    
                shared = u_succs & v_preds
                if len(shared) >= 2:
                    features[names[u]]["scatter_gather_count"] += 1
                    features[names[v]]["gather_scatter_count"] += 1
        
        # 3. Simple Cycles
        valid_cycle_nodes = [
            v for v in range(G.vcount()) 
            if in_degrees[v] > 0 and out_degrees[v] > 0 and all_degrees[v] <= self.max_degree
        ]
        
        if valid_cycle_nodes:
            cycles_found = 0
            for v in valid_cycle_nodes:
                if cycles_found >= self.max_cycles:
                    break
                
                preds_v = set(G.predecessors(v)) & set(valid_cycle_nodes)
                if not preds_v:
                    continue
                
                try:
                    paths = G.get_all_simple_paths(v, to=preds_v, cutoff=self.cycle_bound - 1)
                    for path in paths:
                        for node_idx in path:
                            features[names[node_idx]]["cycle_count"] += 1
                        cycles_found += 1
                        if cycles_found >= self.max_cycles:
                            break
                except AttributeError:
                    pass
        
        return features
