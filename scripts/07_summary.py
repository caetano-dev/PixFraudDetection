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
        "out_degree", "tx_count", "time_variance",
        "wire_count_sent", "wire_count_recv",
        "cash_count_sent", "cash_count_recv",
        "bitcoin_count_sent", "bitcoin_count_recv",
        "cheque_count_sent", "cheque_count_recv",
        "credit_card_count_sent", "credit_card_count_recv",
        "ach_count_sent", "ach_count_recv",
        "reinvestment_count_sent", "reinvestment_count_recv",
        "distinct_currencies_sent", "distinct_currencies_recv"
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


def print_motif_analysis(con: duckdb.DuckDBPyConnection, features_path: Path) -> None:
    """Analyze subgraph motif patterns in the transaction network."""
    if not features_path.exists():
        return
    
    # Check if required motif columns exist
    sample_df = con.execute(f"SELECT * FROM read_parquet('{features_path}') LIMIT 1").df()
    motif_cols = {"fan_out_count", "fan_in_count", "scatter_gather_count", "gather_scatter_count", "cycle_count"}
    if not motif_cols.issubset(set(sample_df.columns)):
        return
    
    print("\n" + "=" * 60)
    print("Subgraph Motif Analysis")
    print("=" * 60)
    
    # Read motif data
    motif_df = con.execute(
        f"""
        SELECT
            entity_id,
            fan_out_count,
            fan_in_count,
            scatter_gather_count,
            gather_scatter_count,
            cycle_count,
            is_fraud
        FROM read_parquet('{features_path}')
        """
    ).df()
    
    if motif_df.empty:
        print("No motif data found.")
        return
    
    # Overall statistics
    total_entities = len(motif_df)
    total_fraud = motif_df["is_fraud"].sum()
    fraud_rate = motif_df["is_fraud"].mean()
    
    print(f"\nTotal entities: {total_entities:,}")
    print(f"Fraudulent entities: {int(total_fraud):,} ({fraud_rate:.2%})")
    
    # Motif prevalence
    print("\n[Motif Pattern Prevalence]")
    print("-" * 60)
    
    motif_patterns = {
        "Fan-out (1→many)": "fan_out_count",
        "Fan-in (many→1)": "fan_in_count", 
        "Scatter-Gather (1→many→1)": "scatter_gather_count",
        "Gather-Scatter (many→1→many)": "gather_scatter_count",
        "Cycles": "cycle_count"
    }
    
    for pattern_name, col_name in motif_patterns.items():
        entities_with_pattern = (motif_df[col_name] > 0).sum()
        pct_entities = entities_with_pattern / total_entities if total_entities > 0 else 0
        total_instances = motif_df[col_name].sum()
        avg_per_entity = motif_df[col_name].mean()
        
        print(f"\n{pattern_name}:")
        print(f"  Entities with pattern: {int(entities_with_pattern):,} ({pct_entities:.2%})")
        print(f"  Total instances: {int(total_instances):,}")
        print(f"  Average per entity: {avg_per_entity:.2f}")
    
    # Fraud correlation
    print("\n[Motif Patterns vs Fraud]")
    print("-" * 60)
    
    fraud_df = motif_df[motif_df["is_fraud"] == 1]
    non_fraud_df = motif_df[motif_df["is_fraud"] == 0]
    
    print(f"{'Motif Pattern':<30} {'Fraud Avg':<12} {'Non-Fraud Avg':<15} {'Ratio':<8}")
    print("-" * 60)
    
    for pattern_name, col_name in motif_patterns.items():
        fraud_avg = fraud_df[col_name].mean() if len(fraud_df) > 0 else 0
        non_fraud_avg = non_fraud_df[col_name].mean() if len(non_fraud_df) > 0 else 0
        ratio = fraud_avg / non_fraud_avg if non_fraud_avg > 0 else float('inf')
        
        print(f"{pattern_name:<30} {fraud_avg:<12.2f} {non_fraud_avg:<15.2f} {ratio:<8.2f}x")
    
    # High-risk entities by motif patterns
    print("\n[High-Risk Entities by Motif Patterns]")
    print("-" * 60)
    
    # Create a combined motif score
    motif_df['total_motif_count'] = (
        motif_df['fan_out_count'] + 
        motif_df['fan_in_count'] + 
        motif_df['scatter_gather_count'] + 
        motif_df['gather_scatter_count'] + 
        motif_df['cycle_count']
    )
    
    # Find entities with high motif activity
    top_motif_entities = motif_df.nlargest(10, 'total_motif_count')
    
    if not top_motif_entities.empty:
        print("\nTop 10 Entities by Total Motif Count:")
        print(f"{'Entity ID':<15} {'Total':<8} {'Cycles':<8} {'Fan-out':<10} {'Fan-in':<10} {'Fraud':<8}")
        print("-" * 70)
        for _, row in top_motif_entities.iterrows():
            fraud_status = "YES" if row['is_fraud'] == 1 else "NO"
            print(
                f"{str(row['entity_id']):<15} "
                f"{int(row['total_motif_count']):<8} "
                f"{int(row['cycle_count']):<8} "
                f"{int(row['fan_out_count']):<10} "
                f"{int(row['fan_in_count']):<10} "
                f"{fraud_status:<8}"
            )
    
    # Fraud detection based on cycle presence
    print("\n[Cycle-Based Fraud Detection]")
    print("-" * 60)
    
    entities_with_cycles = motif_df[motif_df['cycle_count'] > 0]
    if not entities_with_cycles.empty:
        cycle_fraud_rate = entities_with_cycles['is_fraud'].mean()
        cycle_lift = cycle_fraud_rate / fraud_rate if fraud_rate > 0 else 0
        
        print(f"Entities with cycles: {len(entities_with_cycles):,}")
        print(f"Fraud rate in entities with cycles: {cycle_fraud_rate:.2%}")
        print(f"Lift over baseline: {cycle_lift:.2f}x")
        
        if cycle_fraud_rate > fraud_rate * 1.5:
            print("\n✓ Cycles are a strong indicator of fraudulent activity")
        else:
            print("\n⚠ Cycles show weak correlation with fraud")


def print_leiden_community_analysis(con: duckdb.DuckDBPyConnection, features_path: Path) -> None:
    """Analyze Leiden community detection effectiveness for money laundering detection."""
    if not RUN_LEIDEN or not features_path.exists():
        return

    # Check if required Leiden columns exist
    sample_df = con.execute(f"SELECT * FROM read_parquet('{features_path}') LIMIT 1").df()
    required_leiden_cols = {
        "entity_id", "leiden_macro_id", "leiden_macro_size",
        "leiden_macro_modularity", "is_fraud"
    }
    if not required_leiden_cols.issubset(set(sample_df.columns)):
        return

    # Pick an available temporal column for per-window Leiden analysis
    temporal_col = None
    for candidate in ("window_end", "window_id", "date"):
        if candidate in sample_df.columns:
            temporal_col = candidate
            break

    print("\n" + "=" * 60)
    print("Leiden Community Detection Analysis (Money Laundering)")
    print("=" * 60)

    if temporal_col is None:
        print("SKIPPED: No temporal column found for per-window analysis.")
        print("Expected one of: window_end, window_id, date")
        return

    # Read Leiden data (using available temporal column)
    leiden_df = con.execute(
        f"""
        SELECT
            {temporal_col} AS window_key,
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

    # Print Leiden summary for each window
    windows = sorted(leiden_df["window_key"].dropna().unique().tolist())
    for window_key in windows:
        final_df = leiden_df[leiden_df["window_key"] == window_key].copy()
        if final_df.empty:
            continue

        print("\n" + "=" * 60)
        print(f"Leiden Analysis for {temporal_col}: {window_key}")
        print("=" * 60)

        print(f"Total nodes with community assignment: {len(final_df):,}")

        n_communities = final_df["leiden_id"].nunique()
        print(f"Total communities detected: {n_communities:,}")

        # Overall statistics
        total_fraud = final_df["is_fraud"].sum()
        overall_fraud_rate = final_df["is_fraud"].mean()
        print(f"Total fraudulent entities: {int(total_fraud):,}")
        print(f"Overall fraud rate: {overall_fraud_rate:.4%}")

        # Compute community statistics
        community_stats = (
            final_df.groupby("leiden_id", observed=True)
            .agg(
                size=("entity_id", "count"),
                fraud_count=("is_fraud", "sum"),
                fraud_rate=("is_fraud", "mean"),
            )
            .reset_index()
        )

        # ============================================
        # [Effectiveness Metric 1: Fraud Concentration]
        # ============================================
        print("\n[Leiden Effectiveness: Fraud Concentration]")
        print("-" * 60)

        # Communities with fraud vs without fraud
        communities_with_fraud = (community_stats["fraud_count"] > 0).sum()
        fraud_only_communities = (community_stats["fraud_rate"] == 1.0).sum()

        print(f"Communities containing fraud: {communities_with_fraud:,} / {n_communities:,}")
        print(f"Pure fraud communities (100% fraud): {fraud_only_communities:,}")

        # Average fraud rate in communities with fraud
        communities_with_any_fraud = community_stats[community_stats["fraud_count"] > 0]
        if not communities_with_any_fraud.empty:
            avg_fraud_rate_with_fraud = communities_with_any_fraud["fraud_rate"].mean()
            print(f"Avg fraud rate in communities with fraud: {avg_fraud_rate_with_fraud:.4%}")

        # ============================================
        # [Community Size Distribution & Fraud Rate]
        # ============================================
        print("\n[Community Size Distribution]")
        print("-" * 60)

        # Manually categorize for better performance
        size_categories = {
            "1": (1, 2),
            "2-4": (2, 5),
            "5-9": (5, 10),
            "10-49": (10, 50),
            "50-99": (50, 100),
            "100-499": (100, 500),
            "500+": (500, float("inf"))
        }

        print(f"{'Size':<10} {'# Comm':<8} {'# Nodes':<10} {'# Fraud':<8} {'Rate':<10} {'Lift':<8}")
        print("-" * 60)
        for label, (min_size, max_size) in size_categories.items():
            mask = (community_stats["size"] >= min_size) & (community_stats["size"] < max_size)
            if mask.any():
                filtered = community_stats[mask]
                n_comms = len(filtered)
                n_nodes = filtered["size"].sum()
                n_fraud = filtered["fraud_count"].sum()
                fraud_rate = n_fraud / n_nodes if n_nodes > 0 else 0
                lift = fraud_rate / overall_fraud_rate if overall_fraud_rate > 0 else 0
                print(
                    f"{label:<10} {int(n_comms):<8} "
                    f"{int(n_nodes):<10} {int(n_fraud):<8} "
                    f"{fraud_rate:.4%}   {lift:.2f}x"
                )

        # ============================================
        # [Top 10 High-Risk Communities]
        # ============================================
        print("\n[Top 10 Communities by Fraud Risk (min 3 members)]")
        print("-" * 70)

        significant_communities = community_stats[community_stats["size"] >= 3].copy()
        if not significant_communities.empty:
            top_fraud_communities = significant_communities.nlargest(10, "fraud_rate")
            print(f"{'Comm ID':<10} {'Size':<8} {'Fraud':<8} {'Rate':<12} {'Lift':<8}")
            print("-" * 50)
            for _, row in top_fraud_communities.iterrows():
                lift = row["fraud_rate"] / overall_fraud_rate if overall_fraud_rate > 0 else 0
                print(
                    f"{int(row['leiden_id']):<10} {int(row['size']):<8} "
                    f"{int(row['fraud_count']):<8} {row['fraud_rate']:.4%}      {lift:.2f}x"
                )

        # ============================================
        # [Pure Fraud Rings - Highest Confidence]
        # ============================================
        pure_fraud_communities = community_stats[
            (community_stats["fraud_rate"] == 1.0) & (community_stats["size"] >= 2)
        ]

        if not pure_fraud_communities.empty:
            print("\n[Potential Fraud Rings: 100% Fraudulent Communities (size >= 2)]")
            print("-" * 60)
            print(
                f"Found {len(pure_fraud_communities)} suspicious communities that are 100% fraudulent"
            )
            print(
                f"Total fraudulent nodes in these rings: {pure_fraud_communities['size'].sum():,}"
            )
            size_stats = pure_fraud_communities['size'].describe()
            print(
                f"Community sizes (min/mean/max): "
                f"{int(size_stats['min'])}/{int(size_stats['mean'])}/{int(size_stats['max'])}"
            )

        # ============================================
        # [High-Risk Community Detection]
        # ============================================
        print("\n[High-Risk Communities (>2x baseline fraud rate)]")
        print("-" * 60)

        high_fraud_communities = community_stats[
            community_stats["fraud_rate"] > overall_fraud_rate * 2
        ]
        if not high_fraud_communities.empty:
            fraud_in_high = int(high_fraud_communities["fraud_count"].sum())
            pct_fraud_captured = fraud_in_high / total_fraud if total_fraud > 0 else 0
            print(
                f"Communities capturing {pct_fraud_captured:.2%} of fraud: {len(high_fraud_communities):,} communities"
            )
            print(
                f"Fraudulent nodes in high-risk communities: {fraud_in_high:,} / {int(total_fraud):,}"
            )
        else:
            print("No communities with >2x baseline fraud rate detected.")

        # ============================================
        # [Leiden Algorithm Quality]
        # ============================================
        print("\n[Leiden Algorithm Quality Metrics]")
        print("-" * 60)

        # Modularity score (higher is better - indicates strong community structure)
        mean_modularity = final_df["leiden_modularity"].mean()
        print(f"Mean Modularity Score (Q): {mean_modularity:.4f}")
        print("  (Range: 0-1, higher indicates stronger community structure)")

        # Community density analysis
        community_density = (n_communities / len(final_df)) * 100 if len(final_df) > 0 else 0
        print(f"Community Density: {community_density:.2f}% (communities per node)")

        # Entropy / balance of community sizes
        size_std = community_stats["size"].std()
        size_mean = community_stats["size"].mean()
        print(f"Community size distribution (mean/std): {size_mean:.1f} / {size_std:.1f}")

        # ============================================
        # [Overall Effectiveness Summary]
        # ============================================
        print("\n[Leiden Effectiveness for Money Laundering Detection]")
        print("-" * 60)

        # Key effectiveness metrics
        recall_threshold_10_pct = (
            (high_fraud_communities["fraud_count"].sum() / total_fraud) if total_fraud > 0 else 0
        )
        print(f"Fraud detected in 2x-risk communities: {recall_threshold_10_pct:.1%}")

        # Best case: pure fraud communities
        if not pure_fraud_communities.empty:
            pure_fraud_recall = pure_fraud_communities["size"].sum() / total_fraud if total_fraud > 0 else 0
            print(f"Fraud in 100% fraud communities (highest confidence): {pure_fraud_recall:.1%}")

        # Average precision: what % of detected fraud is actually fraud
        if not high_fraud_communities.empty:
            avg_precision_high = high_fraud_communities["fraud_rate"].mean()
            print(f"Precision in 2x-risk communities: {avg_precision_high:.2%} fraud rate")

        # Final verdict
        print("\n[Verdict]")
        print("-" * 60)
        if communities_with_fraud / n_communities > 0.3:
            print("✓ Leiden effectively separates fraud into distinct communities")
            print("  Many communities contain fraudulent entities as distinct groups")
        else:
            print("⚠ Leiden shows mixed results")
            print("  Fraud is spread across many communities")

        if overall_fraud_rate < 0.5 and recall_threshold_10_pct > 0.1:
            print("✓ Strong signal: High-risk communities concentrate fraud")
            print("  Using Leiden for anomaly detection is promising")
        else:
            print("⚠ Leiden alone may not be sufficient for filtering")
            print("  Consider combining with other detection methods")

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
    print_motif_analysis(con, features_path)
    print_leiden_community_analysis(con, features_path)

    con.close()

    print("\n" + "=" * 60)
    print("Summary Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
