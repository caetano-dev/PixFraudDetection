"""
Standalone summary reporting script for the PixFraudDetection pipeline.

This script reads the materialized Parquet files (features.parquet and metrics.parquet)
and prints comprehensive terminal summary statistics that were previously embedded
in 03_extract_features.py.

All analysis is performed via DuckDB SQL queries or Pandas aggregations over the
saved Parquet files. No graph state is recomputed.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import duckdb
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import (
    DATA_PATH,
    DATASET_SIZE,
    EVALUATION_K_VALUES,
    OUTPUT_FEATURES_FILE,
    OUTPUT_METRICS_FILE,
    RUN_EVALUATION,
    RUN_LEIDEN,
    RUN_RANK_STABILITY,
)


def print_aggregate_evaluation_summary(con: duckdb.DuckDBPyConnection, metrics_path: Path) -> None:
    """Print aggregate evaluation summary statistics across all days."""
    if not RUN_EVALUATION:
        print("\n" + "=" * 60)
        print("Aggregate Evaluation Summary")
        print("=" * 60)
        print("SKIPPED: RUN_EVALUATION is disabled in src.config")
        return
    
    if not metrics_path.exists():
        print("\n" + "=" * 60)
        print("Aggregate Evaluation Summary")
        print("=" * 60)
        print(f"SKIPPED: Metrics file not found at {metrics_path}")
        print("The feature extraction may have been run with RUN_EVALUATION=False,")
        print("or the metrics were not saved. Please re-run 03_extract_features.py")
        print("with RUN_EVALUATION=True to generate evaluation metrics.")
        return

    print("\n" + "=" * 60)
    print("Aggregate Evaluation Summary (Mean & Median Across All Days)")
    print("=" * 60)

    metrics_df = con.execute(f"SELECT * FROM read_parquet('{metrics_path}')").df()

    algorithms = [
        "pr_vol_deep", "pr_vol_shallow", "pr_count", "hits_hub", "hits_auth",
        "betweenness", "k_core", "vol_sent", "vol_recv", "in_degree",
        "out_degree", "tx_count", "time_variance"
    ]

    for algo in algorithms:
        algo_df = metrics_df[metrics_df["algorithm"] == algo]
        if algo_df.empty:
            continue

        print(f"\n[{algo.upper()}]")
        print("-" * 40)

        mean_roc = float(algo_df["roc_auc"].mean())
        median_roc = float(algo_df["roc_auc"].median())
        mean_ap = float(algo_df["average_precision"].mean())
        median_ap = float(algo_df["average_precision"].median())

        if not math.isnan(mean_roc):
            print(f"  Mean ROC-AUC:            {mean_roc:.4f}")
        if not math.isnan(median_roc):
            print(f"  Median ROC-AUC:          {median_roc:.4f}")
        if not math.isnan(mean_ap):
            print(f"  Mean Average Precision:  {mean_ap:.4f}")
        if not math.isnan(median_ap):
            print(f"  Median Average Precision:{median_ap:.4f}")

        print("\n  Mean Precision@K:")
        for k in EVALUATION_K_VALUES:
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

        print("\n  Median Precision@K:")
        for k in EVALUATION_K_VALUES:
            col = f"precision_at_{k}"
            if col in algo_df.columns:
                median_prec = float(algo_df[col].median())
                if not math.isnan(median_prec):
                    lift_col = f"lift_at_{k}"
                    median_lift = (
                        float(algo_df[lift_col].median())
                        if lift_col in algo_df.columns
                        else 0.0
                    )
                    print(f"    @{k:>5}: {median_prec:.4%} (lift: {median_lift:.2f}x)")


def print_leiden_community_analysis(con: duckdb.DuckDBPyConnection, features_path: Path) -> None:
    """Analyze Leiden community detection effectiveness."""
    if not RUN_LEIDEN or not features_path.exists():
        return

    # Check if required Leiden columns exist
    sample_df = con.execute(f"SELECT * FROM read_parquet('{features_path}') LIMIT 1").df()
    required_leiden_cols = {
        "date", "entity_id", "leiden_macro_id", "leiden_macro_size",
        "leiden_macro_modularity", "is_fraud"
    }
    if not required_leiden_cols.issubset(set(sample_df.columns)):
        return

    print("\n" + "=" * 60)
    print("Leiden Community Detection Analysis")
    print("=" * 60)

    # Read Leiden data
    leiden_df = con.execute(
        f"""
        SELECT
            date,
            entity_id,
            leiden_macro_id AS leiden_id,
            leiden_macro_size AS leiden_size,
            leiden_macro_modularity AS leiden_modularity,
            is_fraud
        FROM read_parquet('{features_path}')
        WHERE leiden_macro_id != -1
        """
    ).df()

    if leiden_df.empty:
        print("No valid Leiden communities found.")
        return

    # Filter to the last date
    last_date = leiden_df["date"].max()
    final_df = leiden_df[leiden_df["date"] == last_date].copy()

    print(f"\nAnalysis based on final day: {last_date}")
    print(f"Total nodes with community assignment: {len(final_df):,}")

    n_communities = final_df["leiden_id"].nunique()
    print(f"Total communities detected: {n_communities:,}")

    # Compute community statistics
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

    # Community Size Distribution
    print("\n[Community Size Distribution]")
    print("-" * 50)
    size_bins = [1, 2, 5, 10, 50, 100, 500, float("inf")]
    size_labels = ["1", "2-4", "5-9", "10-49", "50-99", "100-499", "500+"]

    community_stats["size_bin"] = pd.cut(
        community_stats["size"], bins=size_bins, labels=size_labels, right=False
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

    size_analysis["fraud_rate"] = size_analysis["total_fraud"] / size_analysis["total_nodes"]
    size_analysis["lift"] = size_analysis["fraud_rate"] / overall_fraud_rate

    print(f"{'Size':<10} {'Communities':<12} {'Nodes':<10} {'Fraud':<8} {'Rate':<10} {'Lift':<8}")
    print("-" * 60)
    for _, row in size_analysis.iterrows():
        print(
            f"{row['size_bin']:<10} {int(row['n_communities']):<12} "
            f"{int(row['total_nodes']):<10} {int(row['total_fraud']):<8} "
            f"{row['fraud_rate']:.4%}   {row['lift']:.2f}x"
        )

    # Top 10 Communities by Fraud Rate
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

    # Potential Fraud Rings
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

    # Leiden Effectiveness Summary
    print("\n[Leiden Effectiveness Summary]")
    print("-" * 50)

    if "leiden_modularity" in leiden_df.columns:
        mean_modularity = leiden_df["leiden_modularity"].mean()
        print(f"Mean Leiden Modularity (Q): {mean_modularity:.4f}")

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


def print_feature_statistics_summary(con: duckdb.DuckDBPyConnection, features_path: Path) -> None:
    """Print pandas describe() summary for continuous features."""
    if not features_path.exists():
        return

    print("\n" + "=" * 60)
    print("Feature Statistics Summary")
    print("=" * 60)

    summary_cols = [
        "pr_vol_deep",
        "pr_vol_shallow",
        "pr_count",
        "hits_hub",
        "hits_auth",
        "betweenness",
        "k_core",
        "degree",
        "vol_sent",
        "vol_recv",
        "tx_count",
        "time_variance",
        "flow_ratio",
        "pagerank_rank_change",
        "fan_out_count",
        "fan_in_count",
        "scatter_gather_count",
        "gather_scatter_count",
        "cycle_count",
        "distinct_currencies_sent",
        "distinct_currencies_recv",
        "wire_count_sent",
        "wire_count_recv",
        "cash_count_sent",
        "cash_count_recv",
        "bitcoin_count_sent",
        "bitcoin_count_recv",
        "cheque_count_sent",
        "cheque_count_recv",
        "credit_card_count_sent",
        "credit_card_count_recv",
        "ach_count_sent",
        "ach_count_recv",
        "reinvestment_count_sent",
        "reinvestment_count_recv",
    ]

    if RUN_LEIDEN:
        summary_cols += [
            "leiden_macro_size",
            "leiden_micro_size",
            "leiden_macro_modularity",
            "leiden_micro_modularity",
        ]

    # Filter to columns that actually exist
    sample_df = con.execute(f"SELECT * FROM read_parquet('{features_path}') LIMIT 1").df()
    output_columns = set(sample_df.columns)
    selected_summary_cols = [c for c in summary_cols if c in output_columns]

    if selected_summary_cols:
        # Quote identifiers to handle special characters
        quoted_cols = [f'"{c}"' for c in selected_summary_cols]
        summary_df = con.execute(
            f"""
            SELECT {", ".join(quoted_cols)}
            FROM read_parquet('{features_path}')
            """
        ).df()
        print(summary_df.describe())


def print_rank_anomaly_analysis(con: duckdb.DuckDBPyConnection, features_path: Path) -> None:
    """
    Analyze rank anomalies using the weirdnodes_magnitude column.
    
    NOTE: This is a corrected version that uses weirdnodes_magnitude instead of
    the non-existent is_rank_anomaly column from the old script.
    """
    if not RUN_RANK_STABILITY or not features_path.exists():
        return

    # Check if weirdnodes_magnitude column exists
    sample_df = con.execute(f"SELECT * FROM read_parquet('{features_path}') LIMIT 1").df()
    if "weirdnodes_magnitude" not in sample_df.columns or "is_fraud" not in sample_df.columns:
        return

    print("\n" + "=" * 60)
    print("Rank Anomaly Analysis (reference: WeirdNodes ensemble)")
    print("=" * 60)

    # Calculate anomaly threshold (95th percentile of weirdnodes_magnitude where > 0)
    threshold_result = con.execute(
        f"""
        SELECT PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY weirdnodes_magnitude) AS threshold
        FROM read_parquet('{features_path}')
        WHERE weirdnodes_magnitude > 0
        """
    ).df()
    
    if threshold_result.empty or threshold_result.iloc[0]["threshold"] is None:
        print("No rank anomalies detected (all weirdnodes_magnitude <= 0).")
        return

    anomaly_threshold = float(threshold_result.iloc[0]["threshold"])
    print(f"Anomaly threshold (95th percentile): {anomaly_threshold:.6f}")

    # Compute anomaly statistics
    anomaly_stats = con.execute(
        f"""
        SELECT
            SUM(CASE WHEN weirdnodes_magnitude >= {anomaly_threshold} THEN 1 ELSE 0 END) AS total_anomaly_flags,
            SUM(CASE WHEN weirdnodes_magnitude >= {anomaly_threshold} THEN CAST(is_fraud AS DOUBLE) ELSE 0 END) AS fraud_in_anomalies,
            AVG(CAST(is_fraud AS DOUBLE)) AS overall_fraud_rate,
            AVG(CASE WHEN weirdnodes_magnitude >= {anomaly_threshold} THEN CAST(is_fraud AS DOUBLE) END) AS anomaly_fraud_rate
        FROM read_parquet('{features_path}')
        """
    ).df().iloc[0]

    total_anomaly_flags = int(anomaly_stats["total_anomaly_flags"] or 0)
    fraud_in_anomalies = float(anomaly_stats["fraud_in_anomalies"] or 0.0)
    overall_fraud_rate = float(anomaly_stats["overall_fraud_rate"] or 0.0)
    anomaly_fraud_rate = float(anomaly_stats["anomaly_fraud_rate"] or 0.0)

    if total_anomaly_flags > 0:
        print(f"Total rank anomaly flags: {total_anomaly_flags:,}")
        print(
            f"Fraud among rank anomalies: {fraud_in_anomalies:,.0f} "
            f"({anomaly_fraud_rate:.2%})"
        )
        print(f"Overall fraud rate: {overall_fraud_rate:.2%}")
        if overall_fraud_rate > 0:
            lift = anomaly_fraud_rate / overall_fraud_rate
            print(f"Lift from rank anomaly detection: {lift:.2f}x")
    else:
        print("No anomalies detected above threshold.")


def main() -> None:
    """Main entry point for summary reporting."""
    print("=" * 60)
    print("PixFraudDetection Pipeline Summary")
    print(f"Dataset: {DATASET_SIZE}")
    print("=" * 60)

    features_path = DATA_PATH / OUTPUT_FEATURES_FILE
    metrics_path = DATA_PATH / OUTPUT_METRICS_FILE

    # Check if files exist
    if not features_path.exists():
        print(f"\nError: Features file not found at {features_path}")
        print("Please run 03_extract_features.py first.")
        return

    con = duckdb.connect(":memory:")

    # Print all summary sections
    print_aggregate_evaluation_summary(con, metrics_path)
    print_leiden_community_analysis(con, features_path)
    print_feature_statistics_summary(con, features_path)
    print_rank_anomaly_analysis(con, features_path)

    con.close()

    print("\n" + "=" * 60)
    print("Summary Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
