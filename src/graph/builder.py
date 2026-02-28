"""
Graph construction utilities for the PixFraudDetection pipeline.

This module migrates ``build_daily_graph`` and ``compute_node_stats`` from the
legacy ``utils.py``.  All graph-construction concerns are isolated here so that
the feature extractors and the orchestrator can treat a ``nx.DiGraph`` as an
opaque, fully-formed input.

Mathematical invariants preserved from the original implementation
------------------------------------------------------------------
Composite edge weight formula (Oddball-inspired):

    W_edge = volume * log2(1 + count) * (1 + 1 / (1 + CV))

where:
    * ``volume``  = sum of ``amount_sent_c`` on the directed edge
    * ``count``   = number of individual transactions aggregated onto the edge
    * ``CV``      = coefficient of variation (std / mean) of per-transaction
                    amounts on the edge; an epsilon of 1e-9 prevents 0/0

Behaviour:
    * High ``count`` amplifies weight via the log2 term — penalises smurfing.
    * Low ``CV`` (uniform amounts, classic smurfing) further amplifies weight
      via the variance factor, which approaches 2 as CV → 0.
    * For a single transaction (count=1, std=0): factor = log2(2) * 2 = 2,
      acting as a neutral baseline since PageRank uses relative weights.

Edge attributes stored on the graph:
    * ``weight``     — composite Oddball-inspired weight (used by PageRank volume)
    * ``volume``     — raw sum of transaction amounts on the edge
    * ``count``      — number of transactions aggregated onto the edge
    * ``amount_std`` — standard deviation of per-transaction amounts (0 for
                       single-transaction edges)
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd


def build_daily_graph(window_df: pd.DataFrame) -> nx.DiGraph:
    """
    Build a directed weighted graph from a window of transaction records.

    Aggregates all individual transactions between each (source, target) pair
    into a single directed edge, then computes the composite Oddball-inspired
    weight described in the module docstring.

    Parameters
    ----------
    window_df : pd.DataFrame
        Transactions for the current temporal window, as produced by
        :class:`src.data.window_generator.TemporalWindowGenerator`.
        Required columns:

        * ``source_entity`` — sending entity ID
        * ``target_entity`` — receiving entity ID
        * ``amount_sent_c`` — conservation-of-mass-adjusted sent amount

    Returns
    -------
    nx.DiGraph
        Directed graph whose nodes are entity IDs and whose edges carry the
        attributes ``weight``, ``volume``, ``count``, and ``amount_std``.
        Returns an **empty** ``DiGraph`` when *window_df* is empty.

    Notes
    -----
    The returned graph is always a new object; modifying it does not affect
    *window_df* or any other shared state.
    """
    G = nx.DiGraph()

    if window_df.empty:
        return G

    # ---------------------------------------------------------------------- #
    # 1. Aggregate edges: one row per (source_entity, target_entity) pair     #
    # ---------------------------------------------------------------------- #
    edge_aggregation = (
        window_df.groupby(["source_entity", "target_entity"])
        .agg(
            volume=("amount_sent_c", "sum"),
            count=("amount_sent_c", "count"),
            amount_std=("amount_sent_c", "std"),
        )
        .reset_index()
    )

    # pandas std() returns NaN for single-element groups; treat as 0 variance.
    edge_aggregation["amount_std"] = edge_aggregation["amount_std"].fillna(0.0)

    # ---------------------------------------------------------------------- #
    # 2. Compute the composite Oddball-inspired edge weight                   #
    # ---------------------------------------------------------------------- #
    # Mean amount per edge (count >= 1 is guaranteed by groupby).
    mean_amount = edge_aggregation["volume"] / edge_aggregation["count"]

    # Coefficient of variation: std / mean.
    # Epsilon prevents 0/0 when mean_amount is effectively zero.
    epsilon: float = 1e-9
    cv = edge_aggregation["amount_std"] / (mean_amount + epsilon)

    # W = volume * log2(1 + count) * (1 + 1 / (1 + CV))
    #   log2(1+count): rewards high frequency, penalises smurfing
    #   (1 + 1/(1+CV)): ranges from ~1 (high variance) to 2 (uniform amounts)
    edge_aggregation["weight"] = (
        edge_aggregation["volume"]
        * np.log2(1 + edge_aggregation["count"])
        * (1 + 1.0 / (1.0 + cv))
    )

    # ---------------------------------------------------------------------- #
    # 3. Populate the graph                                                   #
    # ---------------------------------------------------------------------- #
    for _, row in edge_aggregation.iterrows():
        G.add_edge(
            row["source_entity"],
            row["target_entity"],
            weight=row["weight"],
            volume=row["volume"],
            count=row["count"],
            amount_std=row["amount_std"],
        )

    return G


def compute_node_stats(window_df: pd.DataFrame) -> dict:
    """
    Compute basic transactional statistics for each node in the window.

    Calculates the total volume sent, total volume received, and total
    transaction count (sent + received) for every entity that appears in
    *window_df*, using the conservation-of-mass-adjusted ``amount_sent_c``
    column.

    Parameters
    ----------
    window_df : pd.DataFrame
        Transactions for the current temporal window.
        Required columns:

        * ``source_entity`` — sending entity ID
        * ``target_entity`` — receiving entity ID
        * ``amount_sent_c`` — conservation-of-mass-adjusted sent amount

    Returns
    -------
    dict
        ``{entity_id: {"vol_sent": float, "vol_recv": float, "tx_count": int}}``
        for every entity that appears as a source or target in *window_df*.
        Returns an empty dict when *window_df* is empty.

    Notes
    -----
    ``tx_count`` is the total number of transaction records in which the
    entity participated (as sender *plus* as receiver), not the number of
    unique counterparties.
    """
    if window_df.empty:
        return {}

    # ------------------------------------------------------------------ #
    # Sent statistics — entity appears as source_entity                   #
    # ------------------------------------------------------------------ #
    sent_stats = window_df.groupby("source_entity").agg(
        vol_sent=("amount_sent_c", "sum"),
        count_sent=("amount_sent_c", "count"),
    )

    # ------------------------------------------------------------------ #
    # Received statistics — entity appears as target_entity               #
    # ------------------------------------------------------------------ #
    recv_stats = window_df.groupby("target_entity").agg(
        vol_recv=("amount_sent_c", "sum"),
        count_recv=("amount_sent_c", "count"),
    )

    # ------------------------------------------------------------------ #
    # Merge into a single dict covering all unique entities               #
    # ------------------------------------------------------------------ #
    all_entities: set = set(sent_stats.index) | set(recv_stats.index)

    node_stats: dict = {}
    for entity in all_entities:
        vol_sent = (
            float(sent_stats.loc[entity, "vol_sent"])
            if entity in sent_stats.index
            else 0.0
        )
        vol_recv = (
            float(recv_stats.loc[entity, "vol_recv"])
            if entity in recv_stats.index
            else 0.0
        )
        count_sent = (
            int(sent_stats.loc[entity, "count_sent"])
            if entity in sent_stats.index
            else 0
        )
        count_recv = (
            int(recv_stats.loc[entity, "count_recv"])
            if entity in recv_stats.index
            else 0
        )

        node_stats[entity] = {
            "vol_sent": vol_sent,
            "vol_recv": vol_recv,
            "tx_count": count_sent + count_recv,
        }

    return node_stats
