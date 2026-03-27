"""
Temporal Motif Extraction for AML Detection

This module implements temporal motif counting that preserves chronological
patterns in money laundering sequences. Unlike static topological features,
these motifs respect timestamp ordering to capture sequential behavior.

Memory constraint: Designed for 8GB RAM using chunked edge-stream processing.
"""

from __future__ import annotations
from collections import defaultdict
from typing import Dict, Set, Tuple
import gc

import networkx as nx
import numpy as np
import pandas as pd

from src.features.base import FeatureExtractor


class TemporalMotifExtractor(FeatureExtractor):
    """
    Extract temporal motifs from transaction graphs.
    
    Implements:
    1. Temporal Triangles: A→B→C→A where t1 < t2 < t3
    2. Sequential Fan-Out/Fan-In: Node receives funds, splits to N targets,
       which then forward to common sink within time windows
    
    These patterns capture chronological laundering sequences common in
    structured layering and integration phases.
    """
    
    def __init__(
        self,
        delta_t_window: int = 86400,  # 1 day in seconds
        min_fan_size: int = 2,
        max_fan_size: int = 20,
        max_triangle_time: int = 259200  # 3 days in seconds
    ):
        """
        Args:
            delta_t_window: Time window (seconds) for fan-out/fan-in patterns
            min_fan_size: Minimum number of intermediate nodes for fan patterns
            max_fan_size: Maximum fan size (memory constraint)
            max_triangle_time: Maximum time span for temporal triangles
        """
        self.delta_t_window = delta_t_window
        self.min_fan_size = min_fan_size
        self.max_fan_size = max_fan_size
        self.max_triangle_time = max_triangle_time
    
    @property
    def name(self) -> str:
        return f"TemporalMotifExtractor(delta_t={self.delta_t_window}s)"
    
    def extract(
        self, 
        G: nx.DiGraph,
        edge_timestamps: Dict[Tuple[str, str], float] = None
    ) -> Dict[object, Dict[str, int]]:
        """
        Extract temporal motifs from directed graph with timestamps.
        
        Args:
            G: NetworkX DiGraph with edges
            edge_timestamps: Dict mapping (source, target) -> timestamp
                            If None, uses 'timestamp' edge attribute
        
        Returns:
            Dict mapping node_id -> {
                'temporal_triangle_count': int,
                'temporal_fan_out_count': int,
                'temporal_fan_in_count': int,
                'sequential_scatter_gather_count': int
            }
        """
        if not G.nodes():
            return {}
        
        # Initialize features for all nodes
        features = {
            node: {
                'temporal_triangle_count': 0,
                'temporal_fan_out_count': 0,
                'temporal_fan_in_count': 0,
                'sequential_scatter_gather_count': 0
            }
            for node in G.nodes()
        }
        
        # Extract edge timestamps
        if edge_timestamps is None:
            edge_timestamps = {}
            for u, v, data in G.edges(data=True):
                ts = data.get('timestamp', 0)
                if isinstance(ts, pd.Timestamp):
                    ts = ts.timestamp()
                edge_timestamps[(u, v)] = float(ts)
        
        if not edge_timestamps:
            # No temporal data available, return zeros
            return features
        
        # Compute temporal motifs using memory-efficient streaming
        self._count_temporal_triangles(G, edge_timestamps, features)
        self._count_sequential_fan_patterns(G, edge_timestamps, features)
        
        return features
    
    def _count_temporal_triangles(
        self,
        G: nx.DiGraph,
        edge_timestamps: Dict[Tuple[str, str], float],
        features: Dict
    ) -> None:
        """
        Count temporal triangles: A→B→C→A where t(A→B) < t(B→C) < t(C→A).
        
        This pattern captures rapid circular flow of funds, common in
        integration-phase laundering.
        """
        # Build temporal edge list sorted by timestamp
        temporal_edges = [
            (u, v, ts) 
            for (u, v), ts in edge_timestamps.items()
            if G.has_edge(u, v)
        ]
        temporal_edges.sort(key=lambda x: x[2])
        
        # For each edge A→B, look for B→C and check if C→A exists with correct timing
        for i, (A, B, t1) in enumerate(temporal_edges):
            if t1 <= 0:
                continue
            
            # Find edges B→C that occur after A→B
            for j in range(i + 1, len(temporal_edges)):
                B_prime, C, t2 = temporal_edges[j]
                
                if B_prime != B:
                    continue
                
                if t2 - t1 > self.max_triangle_time:
                    break  # Too far in the future
                
                # Check if C→A exists and occurs after B→C
                if G.has_edge(C, A):
                    t3 = edge_timestamps.get((C, A), 0)
                    if t2 < t3 <= t1 + self.max_triangle_time:
                        # Valid temporal triangle found
                        features[A]['temporal_triangle_count'] += 1
                        features[B]['temporal_triangle_count'] += 1
                        features[C]['temporal_triangle_count'] += 1
        
        gc.collect()
    
    def _count_sequential_fan_patterns(
        self,
        G: nx.DiGraph,
        edge_timestamps: Dict[Tuple[str, str], float],
        features: Dict
    ) -> None:
        """
        Count sequential fan-out/fan-in patterns.
        
        Pattern: Node U receives funds, then splits to N nodes {M1, M2, ..., MN}
        within delta_t, and these nodes forward to common sink V within another delta_t.
        
        This captures layering patterns where funds are split and re-merged.
        """
        # Build adjacency lists with timestamps
        out_edges = defaultdict(list)  # node -> [(target, timestamp)]
        in_edges = defaultdict(list)   # node -> [(source, timestamp)]
        
        for (u, v), ts in edge_timestamps.items():
            if G.has_edge(u, v):
                out_edges[u].append((v, ts))
                in_edges[v].append((u, ts))
        
        # Sort by timestamp for efficient range queries
        for node in out_edges:
            out_edges[node].sort(key=lambda x: x[1])
        for node in in_edges:
            in_edges[node].sort(key=lambda x: x[1])
        
        # For each node, check for fan-out patterns
        for source_node in G.nodes():
            out_neighbors = out_edges.get(source_node, [])
            
            if len(out_neighbors) < self.min_fan_size:
                continue
            
            # Group outgoing edges by time windows
            for i, (first_target, t_start) in enumerate(out_neighbors):
                fan_targets = set()
                
                # Collect all targets within delta_t window
                for target, ts in out_neighbors[i:]:
                    if ts - t_start > self.delta_t_window:
                        break
                    fan_targets.add(target)
                    
                    if len(fan_targets) > self.max_fan_size:
                        break
                
                if len(fan_targets) < self.min_fan_size:
                    continue
                
                features[source_node]['temporal_fan_out_count'] += 1
                
                # Check for fan-in (do targets converge to common sinks?)
                self._check_fan_convergence(
                    fan_targets, 
                    out_edges, 
                    t_start + self.delta_t_window,
                    features,
                    source_node
                )
        
        gc.collect()
    
    def _check_fan_convergence(
        self,
        fan_targets: Set[str],
        out_edges: Dict,
        time_threshold: float,
        features: Dict,
        source_node: str
    ) -> None:
        """
        Check if fan-out targets converge to common sinks.
        """
        # Collect all next-hop destinations from fan targets
        next_hops = defaultdict(set)  # sink -> set of fan targets that reach it
        
        for mid_node in fan_targets:
            for sink, ts in out_edges.get(mid_node, []):
                if ts <= time_threshold + self.delta_t_window:
                    next_hops[sink].add(mid_node)
        
        # Count patterns where multiple fan targets converge to same sink
        for sink, converging_nodes in next_hops.items():
            if len(converging_nodes) >= self.min_fan_size:
                # Sequential scatter-gather found
                features[source_node]['sequential_scatter_gather_count'] += 1
                features[sink]['temporal_fan_in_count'] += 1


def extract_temporal_motifs_from_transactions(
    transactions_df: pd.DataFrame,
    window_id: int,
    source_col: str = 'from_account',
    target_col: str = 'to_account',
    timestamp_col: str = 'timestamp',
    delta_t_window: int = 86400
) -> Dict[str, Dict[str, int]]:
    """
    High-level function to extract temporal motifs from transaction DataFrame.
    
    This is a convenience function for use in feature extraction pipelines
    that work directly with transaction data rather than pre-built graphs.
    
    Args:
        transactions_df: DataFrame with transaction records
        window_id: Window identifier for filtering
        source_col: Column name for source account
        target_col: Column name for target account
        timestamp_col: Column name for timestamp
        delta_t_window: Time window for fan patterns (seconds)
    
    Returns:
        Dict mapping account_id -> temporal motif features
    """
    # Filter to window
    if 'window_id' in transactions_df.columns:
        txns = transactions_df[transactions_df['window_id'] == window_id].copy()
    else:
        txns = transactions_df.copy()
    
    if txns.empty:
        return {}
    
    # Build graph
    G = nx.DiGraph()
    edge_timestamps = {}
    
    for _, row in txns.iterrows():
        source = row[source_col]
        target = row[target_col]
        ts = row[timestamp_col]
        
        if isinstance(ts, pd.Timestamp):
            ts = ts.timestamp()
        
        G.add_edge(source, target)
        
        # Keep earliest timestamp for each edge (for multiple transactions)
        key = (source, target)
        if key not in edge_timestamps or ts < edge_timestamps[key]:
            edge_timestamps[key] = float(ts)
    
    # Extract motifs
    extractor = TemporalMotifExtractor(delta_t_window=delta_t_window)
    features = extractor.extract(G, edge_timestamps)
    
    return features
