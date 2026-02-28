"""
Feature extraction orchestrator for the PixFraudDetection pipeline.

This script replaces the monolithic ``main.py``.  It is intentionally
declarative: the *what* (which extractors to run, which outputs to write) is
expressed at the top of ``main()``, while the *how* is entirely delegated to
the strategy objects and helper modules in ``src/``.

Pipeline flow
-------------
1.  Load all transaction data once via :func:`src.data.loader.load_data`.
2.  Declare a list of instantiated :class:`~src.features.base.FeatureExtractor`
    strategy objects.
3.  Iterate over :class:`~src.data.window_generator.TemporalWindowGenerator`
    to receive ``(current_date, window_df)`` tuples — one per non-empty day.
4.  Build the directed transaction graph with
    :func:`src.graph.builder.build_daily_graph`.
5.  Run each extractor strategy by calling ``.extract(G)``.
6.  If rank-stability is enabled, run
    :class:`~src.features.stability.RankStabilityTracker` against the
    current PageRank volume scores.
7.  If evaluation is enabled, compute daily ranking metrics.
8.  Compile a flat feature record per node and append to a list.
9.  Save all feature records to a Parquet file.
10. Save evaluation metrics to a second Parquet file (when enabled).
11. Print summary statistics and the Leiden community analysis.

Usage
-----
Run from the project root so that the ``src`` package is on the path::

    python scripts/02_extract_features.py

Or, if your shell is already inside ``PixFraudDetection/``::

    python -m scripts.02_extract_features

Configuration is driven entirely by ``src/config.py`` — edit that file to
switch datasets, toggle algorithms, or adjust hyper-parameters.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Path bootstrap — make ``src`` importable when the script is run directly
# from the project root or from within the ``scripts/`` sub-directory.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
from src.config import (
    DATASET_SIZE,
    EVALUATION_K_VALUES,
    OUTPUT_FEATURES_FILE,
    OUTPUT_METRICS_FILE,
    PAGERANK_ALPHA,
    RANK_ANOMALY_PERCENTILE,
    RANK_STABILITY_TOP_K,
    RUN_EVALUATION,
    RUN_LEIDEN,
    RUN_RANK_STABILITY,
    STEP_SIZE,
    WINDOW_DAYS,
)
from src.data.loader import get_bad_actors_up_to_date, load_data
from src.data.window_generator import TemporalWindowGenerator
from src.features.centrality import (
    HITSExtractor,
    PageRankFrequencyExtractor,
    PageRankVolumeExtractor,
)
from src.features.community import LeidenCommunityExtractor
from src.features.stability import RankStabilityTracker
from src.graph.builder import build_daily_graph, compute_node_stats

# ---------------------------------------------------------------------------
# Post-processing analysis (preserved verbatim from main.py)
# ---------------------------------------------------------------------------


def _analyze_leiden_effectiveness(results_df: pd.DataFrame) -> None:
    """
    Analyze and print how effective Leiden communities are at identifying fraud.

    Key metrics:
    - Distribution of fraud across community sizes
    - Communities with highest fraud concentration
    - Whether small communities are more likely to be fraudulent
    """
    print("\n" + "=" * 60)
    print("Leiden Community Detection Analysis")
    print("=" * 60)

    valid_df = results_df[results_df["leiden_id"] != -1].copy()

    if valid_df.empty:
        print("No valid Leiden communities found.")
        return

    last_date = valid_df["date"].max()
    final_df = valid_df[valid_df["date"] == last_date].copy()

    print(f"\nAnalysis based on final day: {last_date}")
    print(f"Total nodes with community assignment: {len(final_df):,}")

    n_communities = final_df["leiden_id"].nunique()
    print(f"Total communities detected: {n_communities:,}")

    community_stats = (
        final_df.groupby("leiden_id")
        .agg(
            size=("entity_id", "count"),
            fraud_count=("is_fraud", "sum"),
            fraud_rate=("is_fraud", "mean"),
        )
        .reset_index()
    )

    overall_fraud_rate = final_df["is_fraud"].mean()
    print(f"Overall fraud rate: {overall_fraud_rate:.4%}")

    print("\n[Community Size Distribution]")
    print("-" * 50)
    size_bins = [1, 2, 5, 10, 50, 100, 500, float("inf")]
    size_labels = ["1", "2-4", "5-9", "10-49", "50-99", "100-499", "500+"]

    community_stats["size_bin"] = pd.cut(
        community_stats["size"],
        bins=size_bins,
        labels=size_labels,
        right=False,
    )

    size_analysis = (
        community_stats.groupby("size_bin", observed=True)
        .agg(
            n_communities=("leiden_id", "count"),
            total_nodes=("size", "sum"),
            total_fraud=("fraud_count", "sum"),
        )
        .reset_index()
    )

    size_analysis["fraud_rate"] = (
        size_analysis["total_fraud"] / size_analysis["total_nodes"]
    )
    size_analysis["lift"] = size_analysis["fraud_rate"] / overall_fraud_rate

    print(
        f"{'Size':<10} {'Communities':<12} {'Nodes':<10} "
        f"{'Fraud':<8} {'Rate':<10} {'Lift':<8}"
    )
    print("-" * 60)
    for _, row in size_analysis.iterrows():
        print(
            f"{row['size_bin']:<10} {int(row['n_communities']):<12} "
            f"{int(row['total_nodes']):<10} {int(row['total_fraud']):<8} "
            f"{row['fraud_rate']:.4%}   {row['lift']:.2f}x"
        )

    print("\n[Top 10 Communities by Fraud Rate (min 3 members)]")
    print("-" * 70)

    significant_communities = community_stats[community_stats["size"] >= 3].copy()
    top_fraud_communities = significant_communities.nlargest(10, "fraud_rate")

    print(f"{'Comm ID':<10} {'Size':<8} {'Fraud':<8} {'Rate':<12} {'Lift':<8}")
    print("-" * 50)
    for _, row in top_fraud_communities.iterrows():
        lift = row["fraud_rate"] / overall_fraud_rate if overall_fraud_rate > 0 else 0
        print(
            f"{int(row['leiden_id']):<10} {int(row['size']):<8} "
            f"{int(row['fraud_count']):<8} {row['fraud_rate']:.4%}      {lift:.2f}x"
        )

    pure_fraud_communities = community_stats[
        (community_stats["fraud_rate"] == 1.0) & (community_stats["size"] >= 2)
    ]

    if not pure_fraud_communities.empty:
        print("\n[Potential Fraud Rings: 100% Fraud Communities (size >= 2)]")
        print("-" * 50)
        print(
            f"Found {len(pure_fraud_communities)} communities that are 100% fraudulent"
        )
        print(
            f"Total nodes in these communities: "
            f"{pure_fraud_communities['size'].sum():,}"
        )
        print(
            f"Size distribution: {pure_fraud_communities['size'].describe().to_dict()}"
        )

    print("\n[Leiden Effectiveness Summary]")
    print("-" * 50)

    small_community_threshold = 10
    small_communities = final_df[final_df["leiden_size"] <= small_community_threshold]
    fraud_in_small = small_communities["is_fraud"].sum()
    total_fraud = final_df["is_fraud"].sum()

    if total_fraud > 0:
        pct_fraud_in_small = fraud_in_small / total_fraud
        print(
            f"Fraud in small communities (size <= {small_community_threshold}): "
            f"{fraud_in_small:,} / {total_fraud:,} ({pct_fraud_in_small:.2%})"
        )

    avg_community_fraud_rate = community_stats["fraud_rate"].mean()
    print(f"Average community fraud rate: {avg_community_fraud_rate:.4%}")
    print(f"Overall fraud rate: {overall_fraud_rate:.4%}")

    high_fraud_communities = community_stats[
        community_stats["fraud_rate"] > overall_fraud_rate * 2
    ]
    if not high_fraud_communities.empty:
        fraud_in_high = high_fraud_communities["fraud_count"].sum()
        if total_fraud > 0:
            print(
                f"Fraud in high-concentration communities (>2x avg rate): "
                f"{fraud_in_high:,} / {total_fraud:,} "
                f"({fraud_in_high / total_fraud:.2%})"
            )


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------


def main() -> None:
    """
    Execute the sliding window feature generation pipeline.

    Pipeline steps:
    1.  Load all transaction data once.
    2.  Declare feature extractor strategies.
    3.  Iterate over TemporalWindowGenerator.
    4.  For each window: build graph, run all extractors, compile features.
    5.  Optionally compute rank-stability and evaluation metrics per day.
    6.  Save all features and metrics to Parquet files.
    7.  Print summary statistics.
    """
    print("=" * 60)
    print("Graph Feature Generation Pipeline")
    print(f"Dataset: {DATASET_SIZE}")
    print(f"Window: {WINDOW_DAYS} days | Step: {STEP_SIZE} day(s)")
    print(f"Evaluation:     {'Enabled' if RUN_EVALUATION else 'Disabled'}")
    print(f"Leiden:         {'Enabled' if RUN_LEIDEN else 'Disabled'}")
    print(f"Rank Stability: {'Enabled' if RUN_RANK_STABILITY else 'Disabled'}")
    print("=" * 60)

    # ------------------------------------------------------------------ #
    # 1. Load data                                                         #
    # ------------------------------------------------------------------ #
    all_transactions, _bad_actors_global = load_data()

    # ------------------------------------------------------------------ #
    # 2. Declare extractor strategies                                      #
    #                                                                      #
    # Each entry in `extractors` is a concrete FeatureExtractor whose     #
    # .extract(G) method returns {node_id: {feature_name: value}}.        #
    # To disable an algorithm, simply remove it from this list.           #
    # ------------------------------------------------------------------ #
    extractors = [
        PageRankVolumeExtractor(alpha=PAGERANK_ALPHA),
        PageRankFrequencyExtractor(alpha=PAGERANK_ALPHA),
        HITSExtractor(),
    ]

    if RUN_LEIDEN:
        extractors.append(LeidenCommunityExtractor())

    # Stateful rank-stability tracker (operates on PageRank volume scores).
    # Instantiated unconditionally; only queried when RUN_RANK_STABILITY is True.
    stability_tracker = RankStabilityTracker(
        top_k=RANK_STABILITY_TOP_K,
        threshold_percentile=RANK_ANOMALY_PERCENTILE,
    )

    # ------------------------------------------------------------------ #
    # 3. Build temporal window generator                                   #
    # ------------------------------------------------------------------ #
    generator = TemporalWindowGenerator(
        transactions=all_transactions,
        window_days=WINDOW_DAYS,
        step_size=STEP_SIZE,
    )

    print(f"\nData range: {generator.start_date.date()} to {generator.end_date.date()}")
    print(
        f"Processing {len(generator)} days with {WINDOW_DAYS}-day sliding window...\n"
    )

    # ------------------------------------------------------------------ #
    # 4. Sliding window loop                                               #
    # ------------------------------------------------------------------ #
    all_features: list[dict] = []
    all_daily_metrics: list[dict] = []
    all_rank_stability: list[dict] = []

    k_values = EVALUATION_K_VALUES

    with tqdm(total=len(generator), desc="Processing days", unit="day") as pbar:
        for current_date, window_df in generator:
            # ---------------------------------------------------------- #
            # 4a. Build the transaction graph for this window             #
            # ---------------------------------------------------------- #
            G = build_daily_graph(window_df)

            if len(G) == 0:
                pbar.update(1)
                continue

            # ---------------------------------------------------------- #
            # 4b. Compute raw transactional node stats                    #
            # ---------------------------------------------------------- #
            node_stats = compute_node_stats(window_df)

            # ---------------------------------------------------------- #
            # 4c. Run all registered feature extractor strategies         #
            # ---------------------------------------------------------- #
            # merged_features: {node_id: {feature_name: value, ...}}
            merged_features: dict[object, dict] = {}

            for extractor in extractors:
                result = extractor.extract(G)
                # Merge each node's sub-dict into the running accumulator.
                for node, feats in result.items():
                    if node not in merged_features:
                        merged_features[node] = {}
                    merged_features[node].update(feats)

            # ---------------------------------------------------------- #
            # 4d. Rank stability analysis                                 #
            #                                                             #
            # The tracker operates on a flat {node: score} dict, so we   #
            # extract that from the PageRankVolumeExtractor's output      #
            # (keyed as "pagerank" in merged_features).                   #
            # ---------------------------------------------------------- #
            pr_rank_changes: dict = {}
            pr_anomalies: set = set()

            if RUN_RANK_STABILITY:
                # Build flat {node: pagerank_score} from merged results.
                flat_pr_scores: dict = {
                    node: feats.get("pagerank", 0.0)
                    for node, feats in merged_features.items()
                }

                stability_result = stability_tracker.compute(flat_pr_scores)

                if stability_result is not None:
                    pr_rank_changes = stability_result["rank_changes"]
                    pr_anomalies = stability_result["anomalies"]

                    all_rank_stability.append(
                        {
                            "date": current_date.date(),
                            "algorithm": "pagerank",
                            "stability_score": stability_result["stability_score"],
                            "num_new_entrants": stability_result["num_new_entrants"],
                            "num_dropouts": stability_result["num_dropouts"],
                            "num_anomalies": stability_result["num_anomalies"],
                        }
                    )

            # ---------------------------------------------------------- #
            # 4e. Time-aware fraud labels (no future leakage)             #
            # ---------------------------------------------------------- #
            bad_actors_current = get_bad_actors_up_to_date(
                all_transactions, current_date
            )

            # ---------------------------------------------------------- #
            # 4f. Compile one flat feature record per node                #
            # ---------------------------------------------------------- #
            for node in G.nodes():
                node_feats = merged_features.get(node, {})
                stats = node_stats.get(node, {})

                record: dict = {
                    "date": current_date.date(),
                    "entity_id": node,
                    # --- Centrality features ---
                    "pagerank": node_feats.get("pagerank", 0.0),
                    "pagerank_count": node_feats.get("pagerank_count", 0.0),
                    "hits_hub": node_feats.get("hits_hub", 0.0),
                    "hits_auth": node_feats.get("hits_auth", 0.0),
                    # --- Degree features ---
                    "degree": G.degree(node),
                    "in_degree": G.in_degree(node),
                    "out_degree": G.out_degree(node),
                    # --- Community features ---
                    "leiden_id": node_feats.get("leiden_id", -1),
                    "leiden_size": node_feats.get("leiden_size", 0),
                    # --- Transactional stats ---
                    "vol_sent": stats.get("vol_sent", 0.0),
                    "vol_recv": stats.get("vol_recv", 0.0),
                    "tx_count": stats.get("tx_count", 0),
                    # --- Ground truth label (time-aware) ---
                    "is_fraud": 1 if node in bad_actors_current else 0,
                    # --- Rank stability features ---
                    "pagerank_rank_change": (
                        pr_rank_changes.get(node, 0) if RUN_RANK_STABILITY else 0
                    ),
                    "is_rank_anomaly": (
                        (1 if node in pr_anomalies else 0) if RUN_RANK_STABILITY else 0
                    ),
                }
                all_features.append(record)

            # ---------------------------------------------------------- #
            # 4g. Daily evaluation metrics (optional)                     #
            # ---------------------------------------------------------- #
            if RUN_EVALUATION:
                pagerank_scores: dict = {
                    node: merged_features.get(node, {}).get("pagerank", 0.0)
                    for node in G.nodes()
                }
                hits_hubs: dict = {
                    node: merged_features.get(node, {}).get("hits_hub", 0.0)
                    for node in G.nodes()
                }
                hits_auths: dict = {
                    node: merged_features.get(node, {}).get("hits_auth", 0.0)
                    for node in G.nodes()
                }

                if pagerank_scores:
                    valid_k_values = [k for k in k_values if k <= len(G)]

                    if valid_k_values:
                        # Import here to avoid a circular dependency at the
                        # module level; these evaluation helpers are kept in
                        # the legacy utils for now and called directly.
                        from src.features._evaluation import (
                            compute_daily_evaluation_metrics,
                        )

                        daily_metrics = compute_daily_evaluation_metrics(
                            pagerank_scores=pagerank_scores,
                            hits_hubs=hits_hubs,
                            hits_auths=hits_auths,
                            bad_actors=bad_actors_current,
                            k_values=valid_k_values,
                        )

                        for algo_name, algo_metrics in daily_metrics.items():
                            metric_record: dict = {
                                "date": current_date.date(),
                                "algorithm": algo_name,
                                "total_nodes": algo_metrics["total_nodes"],
                                "total_fraud": algo_metrics["total_fraud"],
                                "fraud_rate": algo_metrics["fraud_rate"],
                                "roc_auc": algo_metrics.get("roc_auc"),
                                "average_precision": algo_metrics.get(
                                    "average_precision"
                                ),
                            }

                            for k in valid_k_values:
                                metric_record[f"precision_at_{k}"] = algo_metrics[
                                    "precision_at_k"
                                ].get(k)
                                metric_record[f"recall_at_{k}"] = algo_metrics[
                                    "recall_at_k"
                                ].get(k)
                                metric_record[f"lift_at_{k}"] = algo_metrics[
                                    "lift_at_k"
                                ].get(k)
                                metric_record[f"fraud_found_at_{k}"] = algo_metrics[
                                    "fraud_found_at_k"
                                ].get(k)

                            all_daily_metrics.append(metric_record)

            pbar.update(1)

    # ------------------------------------------------------------------ #
    # 5. Compile and save feature results                                  #
    # ------------------------------------------------------------------ #
    print("\nCompiling results...")
    results_df = pd.DataFrame(all_features)

    if results_df.empty:
        print("Warning: No features were generated!")
        return

    print(f"Generated {len(results_df):,} feature records")
    print(f"Unique entities: {results_df['entity_id'].nunique():,}")
    print(f"Unique dates:    {results_df['date'].nunique():,}")
    print(
        f"Fraud records:   {results_df['is_fraud'].sum():,} "
        f"({100 * results_df['is_fraud'].mean():.2f}%)"
    )

    # Rank stability summary
    if RUN_RANK_STABILITY and all_rank_stability:
        stability_df = pd.DataFrame(all_rank_stability)
        avg_stability = stability_df["stability_score"].mean()
        total_anomalies = stability_df["num_anomalies"].sum()
        print("\nRank Stability Analysis:")
        print(f"  Average stability score:       {avg_stability:.4f}")
        print(f"  Total rank anomalies detected: {total_anomalies:,}")
        print(
            f"  Entities flagged as rank anomalies: "
            f"{results_df['is_rank_anomaly'].sum():,}"
        )

    output_path = _PROJECT_ROOT / OUTPUT_FEATURES_FILE
    print(f"\nSaving features to {output_path}...")
    results_df.to_parquet(output_path, index=False)

    # ------------------------------------------------------------------ #
    # 6. Save and summarize evaluation metrics                             #
    # ------------------------------------------------------------------ #
    if RUN_EVALUATION and all_daily_metrics:
        metrics_df = pd.DataFrame(all_daily_metrics)
        metrics_path = _PROJECT_ROOT / OUTPUT_METRICS_FILE
        print(f"Saving metrics to {metrics_path}...")
        metrics_df.to_parquet(metrics_path, index=False)

        print("\n" + "=" * 60)
        print("Aggregate Evaluation Summary (Mean Across All Days)")
        print("=" * 60)

        for algo in ["pagerank", "hits_hub", "hits_auth"]:
            algo_df = metrics_df[metrics_df["algorithm"] == algo]
            if algo_df.empty:
                continue

            print(f"\n[{algo.upper()}]")
            print("-" * 40)

            mean_roc = float(algo_df["roc_auc"].mean())
            mean_ap = float(algo_df["average_precision"].mean())

            if not math.isnan(mean_roc):
                print(f"  Mean ROC-AUC:            {mean_roc:.4f}")
            if not math.isnan(mean_ap):
                print(f"  Mean Average Precision:  {mean_ap:.4f}")

            print("\n  Mean Precision@K:")
            for k in k_values:
                col = f"precision_at_{k}"
                if col in algo_df.columns:
                    mean_prec = float(algo_df[col].mean())
                    if not math.isnan(mean_prec):
                        lift_col = f"lift_at_{k}"
                        mean_lift = (
                            float(algo_df[lift_col].mean())
                            if lift_col in algo_df.columns
                            else 0.0
                        )
                        print(f"    @{k:>5}: {mean_prec:.4%} (lift: {mean_lift:.2f}x)")

    # ------------------------------------------------------------------ #
    # 7. Post-pipeline analyses                                            #
    # ------------------------------------------------------------------ #
    if RUN_LEIDEN:
        _analyze_leiden_effectiveness(results_df)

    print("\n" + "=" * 60)
    print("Feature Statistics Summary")
    print("=" * 60)
    print(
        results_df[
            [
                "pagerank",
                "hits_hub",
                "hits_auth",
                "degree",
                "leiden_size",
                "vol_sent",
                "vol_recv",
                "tx_count",
                "pagerank_rank_change",
            ]
        ].describe()
    )

    if RUN_RANK_STABILITY and results_df["is_rank_anomaly"].sum() > 0:
        print("\n" + "=" * 60)
        print("Rank Anomaly Analysis")
        print("=" * 60)
        anomaly_df = results_df[results_df["is_rank_anomaly"] == 1]
        fraud_in_anomalies = anomaly_df["is_fraud"].sum()
        total_anomaly_flags = len(anomaly_df)
        overall_fraud_rate = results_df["is_fraud"].mean()
        anomaly_fraud_rate = (
            anomaly_df["is_fraud"].mean() if total_anomaly_flags > 0 else 0
        )

        print(f"Total rank anomaly flags: {total_anomaly_flags:,}")
        print(
            f"Fraud among rank anomalies: {fraud_in_anomalies:,} "
            f"({anomaly_fraud_rate:.2%})"
        )
        print(f"Overall fraud rate: {overall_fraud_rate:.2%}")
        if overall_fraud_rate > 0:
            print(
                f"Lift from rank anomaly detection: "
                f"{anomaly_fraud_rate / overall_fraud_rate:.2f}x"
            )

    print("\nDone!")


if __name__ == "__main__":
    main()
