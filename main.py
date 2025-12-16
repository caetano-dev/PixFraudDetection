"""
Main pipeline for Sliding Window Graph Feature Generation.

This script generates time-series features (PageRank, HITS, Leiden) for downstream
AI models by iterating through the transaction data day-by-day using slideing windows.
"""

from __future__ import annotations

import math
import pandas as pd
import networkx as nx
from tqdm import tqdm

from config import (
    WINDOW_DAYS,
    STEP_SIZE,
    OUTPUT_FEATURES_FILE,
    OUTPUT_METRICS_FILE,
    HITS_MAX_ITER,
    PAGERANK_ALPHA,
    DATASET_SIZE,
    EVALUATION_K_VALUES,
    RUN_EVALUATION,
    RUN_LEIDEN,
    RUN_RANK_STABILITY,
    RANK_STABILITY_TOP_K,
    RANK_ANOMALY_PERCENTILE,
    USE_TIME_AWARE_BAD_ACTORS,
)
from utils import (
    load_data,
    build_daily_graph,
    compute_node_stats,
    compute_leiden_features,
    compute_daily_evaluation_metrics,
    get_bad_actors_up_to_date,
    compute_rank_stability,
    detect_rank_anomalies,
)


def analyze_leiden_effectiveness(results_df: pd.DataFrame) -> None:
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
    
    # Filter to valid Leiden assignments (leiden_id != -1)
    valid_df = results_df[results_df['leiden_id'] != -1].copy()
    
    if valid_df.empty:
        print("No valid Leiden communities found.")
        return
    
    # Get the last date (full graph) for analysis
    last_date = valid_df['date'].max()
    final_df = valid_df[valid_df['date'] == last_date].copy()
    
    print(f"\nAnalysis based on final day: {last_date}")
    print(f"Total nodes with community assignment: {len(final_df):,}")
    
    # Basic community statistics
    n_communities = final_df['leiden_id'].nunique()
    print(f"Total communities detected: {n_communities:,}")
    
    # Compute community-level fraud statistics
    community_stats = final_df.groupby('leiden_id').agg(
        size=('entity_id', 'count'),
        fraud_count=('is_fraud', 'sum'),
        fraud_rate=('is_fraud', 'mean')
    ).reset_index()
    
    # Overall fraud rate for comparison
    overall_fraud_rate = final_df['is_fraud'].mean()
    print(f"Overall fraud rate: {overall_fraud_rate:.4%}")
    
    # Community size distribution
    print("\n[Community Size Distribution]")
    print("-" * 50)
    size_bins = [1, 2, 5, 10, 50, 100, 500, float('inf')]
    size_labels = ['1', '2-4', '5-9', '10-49', '50-99', '100-499', '500+']
    
    community_stats['size_bin'] = pd.cut(
        community_stats['size'], 
        bins=size_bins, 
        labels=size_labels,
        right=False
    )
    
    size_analysis = community_stats.groupby('size_bin', observed=True).agg(
        n_communities=('leiden_id', 'count'),
        total_nodes=('size', 'sum'),
        total_fraud=('fraud_count', 'sum')
    ).reset_index()
    
    size_analysis['fraud_rate'] = size_analysis['total_fraud'] / size_analysis['total_nodes']
    size_analysis['lift'] = size_analysis['fraud_rate'] / overall_fraud_rate
    
    print(f"{'Size':<10} {'Communities':<12} {'Nodes':<10} {'Fraud':<8} {'Rate':<10} {'Lift':<8}")
    print("-" * 60)
    for _, row in size_analysis.iterrows():
        print(f"{row['size_bin']:<10} {int(row['n_communities']):<12} {int(row['total_nodes']):<10} "
              f"{int(row['total_fraud']):<8} {row['fraud_rate']:.4%}   {row['lift']:.2f}x")
    
    # Top fraud-concentrated communities
    print("\n[Top 10 Communities by Fraud Rate (min 3 members)]")
    print("-" * 70)
    
    # Filter communities with at least 3 members for meaningful rates
    significant_communities = community_stats[community_stats['size'] >= 3].copy()
    top_fraud_communities = significant_communities.nlargest(10, 'fraud_rate')
    
    print(f"{'Comm ID':<10} {'Size':<8} {'Fraud':<8} {'Rate':<12} {'Lift':<8}")
    print("-" * 50)
    for _, row in top_fraud_communities.iterrows():
        lift = row['fraud_rate'] / overall_fraud_rate if overall_fraud_rate > 0 else 0
        print(f"{int(row['leiden_id']):<10} {int(row['size']):<8} {int(row['fraud_count']):<8} "
              f"{row['fraud_rate']:.4%}      {lift:.2f}x")
    
    # Communities that are 100% fraud (potential fraud rings)
    pure_fraud_communities = community_stats[
        (community_stats['fraud_rate'] == 1.0) & (community_stats['size'] >= 2)
    ]
    
    if not pure_fraud_communities.empty:
        print(f"\n[Potential Fraud Rings: 100% Fraud Communities (size >= 2)]")
        print("-" * 50)
        print(f"Found {len(pure_fraud_communities)} communities that are 100% fraudulent")
        print(f"Total nodes in these communities: {pure_fraud_communities['size'].sum():,}")
        print(f"Size distribution: {pure_fraud_communities['size'].describe().to_dict()}")
    
    # Effectiveness summary
    print("\n[Leiden Effectiveness Summary]")
    print("-" * 50)
    
    # Calculate what % of fraud is in small communities
    small_community_threshold = 10
    small_communities = final_df[final_df['leiden_size'] <= small_community_threshold]
    fraud_in_small = small_communities['is_fraud'].sum()
    total_fraud = final_df['is_fraud'].sum()
    
    if total_fraud > 0:
        pct_fraud_in_small = fraud_in_small / total_fraud
        print(f"Fraud in small communities (size <= {small_community_threshold}): "
              f"{fraud_in_small:,} / {total_fraud:,} ({pct_fraud_in_small:.2%})")
    
    # Calculate average community fraud rate vs overall
    avg_community_fraud_rate = community_stats['fraud_rate'].mean()
    print(f"Average community fraud rate: {avg_community_fraud_rate:.4%}")
    print(f"Overall fraud rate: {overall_fraud_rate:.4%}")
    
    # Concentration metric: how much more concentrated is fraud in high-fraud communities?
    high_fraud_communities = community_stats[community_stats['fraud_rate'] > overall_fraud_rate * 2]
    if not high_fraud_communities.empty:
        fraud_in_high = high_fraud_communities['fraud_count'].sum()
        if total_fraud > 0:
            print(f"Fraud in high-concentration communities (>2x avg rate): "
                  f"{fraud_in_high:,} / {total_fraud:,} ({fraud_in_high/total_fraud:.2%})")

def main():
    """
    Execute the sliding window feature generation pipeline.
    
    Pipeline steps:
    1. Load all transaction data once
    2. Determine date range from dataset
    3. Iterate day-by-day with sliding window
    4. For each window: build graph, run algorithms, extract features
    5. Optionally compute evaluation metrics per day
    6. Save all features and metrics to parquet files
    """
    print("=" * 60)
    print("Graph Feature Generation Pipeline")
    print(f"Dataset: {DATASET_SIZE}")
    print(f"Window: {WINDOW_DAYS} days | Step: {STEP_SIZE} day(s)")
    print(f"Evaluation: {'Enabled' if RUN_EVALUATION else 'Disabled'}")
    print(f"Leiden: {'Enabled' if RUN_LEIDEN else 'Disabled'}")
    print(f"Rank Stability: {'Enabled' if RUN_RANK_STABILITY else 'Disabled'}")
    print(f"Time-aware bad actors: {'Yes (no future leakage)' if USE_TIME_AWARE_BAD_ACTORS else 'No (global)'}")
    print("=" * 60)
    
    # 1. Load all data once
    # NOTE: bad_actors_global is for final summary only - for temporal evaluation,
    # we use get_bad_actors_up_to_date() to prevent future information leakage
    all_transactions, bad_actors_global = load_data()
    
    # 2. Determine date range
    start_date = all_transactions['timestamp'].min()
    end_date = all_transactions['timestamp'].max()
    print(f"\nData range: {start_date.date()} to {end_date.date()}")
    
    # Start iteration - start from first day (or after a small warmup)
    # For SLIDING, this means the first few windows will be partial (growing) until they reach WINDOW_DAYS size
    current_date = start_date + pd.Timedelta(days=1)
    
    total_days = (end_date - current_date).days + 1
    
    print(f"Processing {total_days} days with {WINDOW_DAYS}-day sliding window...\n")
    
    # Lists to collect all records
    all_features = []
    all_daily_metrics = []
    all_rank_stability = []
    
    # Track previous window scores for rank stability analysis
    prev_pagerank_scores = {}
    prev_hits_hubs = {}
    prev_hits_auths = {}
    
    # Filter k_values to reasonable sizes (will be further filtered per-day)
    k_values = EVALUATION_K_VALUES
    
    # 3. Sliding window loop
    pbar = tqdm(total=total_days, desc="Processing days", unit="day")
    
    while current_date <= end_date:
        # Calculate window boundaries based on mode
        window_start = current_date - pd.Timedelta(days=WINDOW_DAYS)
        mask = (
            (all_transactions['timestamp'] > window_start) & 
            (all_transactions['timestamp'] <= current_date)
        )
        
        # 4. Filter transactions for current window
        window_df = all_transactions.loc[mask].copy()
        
        # Skip empty windows (weekends/holidays with no transactions)
        if window_df.empty:
            current_date += pd.Timedelta(days=STEP_SIZE)
            pbar.update(1)
            continue
        
        # 5. Build graph for this window
        G = build_daily_graph(window_df)
        
        # 6. Compute node-level transaction statistics
        node_stats = compute_node_stats(window_df)
        
        # 7. Run algorithms and extract features
        if len(G) > 0:
            # PageRank with error handling
            try:
                pagerank_scores = nx.pagerank(
                    G, 
                    weight='weight', 
                    alpha=PAGERANK_ALPHA
                )
            except (nx.NetworkXError, nx.PowerIterationFailedConvergence):
                pagerank_scores = {}
            
            try:
                hits_hubs, hits_auths = nx.hits(G, max_iter=HITS_MAX_ITER)
            except (nx.NetworkXError, nx.PowerIterationFailedConvergence):
                hits_hubs, hits_auths = {}, {}
            
            # 7b. Compute rank stability analysis (comparing with previous window)
            # This detects anomalies when nodes shift drastically between snapshots
            if RUN_RANK_STABILITY and prev_pagerank_scores:
                pr_stability = compute_rank_stability(
                    prev_pagerank_scores, 
                    pagerank_scores, 
                    top_k=RANK_STABILITY_TOP_K
                )
                pr_anomalies = detect_rank_anomalies(
                    pr_stability['rank_changes'], 
                    threshold_percentile=RANK_ANOMALY_PERCENTILE
                )
                
                # Store rank change for each node (to be added to features)
                pr_rank_changes = pr_stability['rank_changes']
                
                # Record stability metrics for this window transition
                stability_record = {
                    'date': current_date.date(),
                    'algorithm': 'pagerank',
                    'stability_score': pr_stability['stability_score'],
                    'num_new_entrants': len(pr_stability['new_entrants']),
                    'num_dropouts': len(pr_stability['dropouts']),
                    'num_anomalies': len(pr_anomalies),
                }
                all_rank_stability.append(stability_record)
            else:
                pr_rank_changes = {}
                pr_anomalies = set()
            
            # Leiden community detection
            if RUN_LEIDEN:
                leiden_features = compute_leiden_features(G)
            else:
                leiden_features = {}
            
            # Get bad actors for evaluation
            # USE_TIME_AWARE_BAD_ACTORS=True: Only use bad actors known UP TO current date
            # (prevents future information leakage - critical for proper temporal evaluation)
            # USE_TIME_AWARE_BAD_ACTORS=False: Use global bad actors (for quick testing only)
            if USE_TIME_AWARE_BAD_ACTORS:
                bad_actors_current = get_bad_actors_up_to_date(all_transactions, current_date)
            else:
                bad_actors_current = bad_actors_global
            
            # 8. Extract features for each node in the graph
            for node in G.nodes():
                # Get Leiden features for this node (if available)
                node_leiden = leiden_features.get(node, {})
                # Get transaction stats for this node
                stats = node_stats.get(node, {})
                
                record = {
                    'date': current_date.date(),
                    'entity_id': node,
                    'pagerank': pagerank_scores.get(node, 0.0),
                    'hits_hub': hits_hubs.get(node, 0.0),
                    'hits_auth': hits_auths.get(node, 0.0),
                    'degree': G.degree(node),
                    'in_degree': G.in_degree(node),
                    'out_degree': G.out_degree(node),
                    'leiden_id': node_leiden.get('leiden_id', -1),
                    'leiden_size': node_leiden.get('leiden_size', 0),
                    'vol_sent': stats.get('vol_sent', 0.0),
                    'vol_recv': stats.get('vol_recv', 0.0),
                    'tx_count': stats.get('tx_count', 0),
                    # Fraud label (time-aware if USE_TIME_AWARE_BAD_ACTORS=True)
                    'is_fraud': 1 if node in bad_actors_current else 0,
                    # Rank stability features (0 if first window, disabled, or node not in previous)
                    'pagerank_rank_change': pr_rank_changes.get(node, 0) if RUN_RANK_STABILITY else 0,
                    'is_rank_anomaly': (1 if node in pr_anomalies else 0) if RUN_RANK_STABILITY else 0,
                }
                all_features.append(record)
            
            # 9. Compute evaluation metrics for this day (optional)
            # Use time-aware bad actors to prevent future information leakage
            if RUN_EVALUATION and pagerank_scores:
                # Filter k_values to not exceed nodes in this window
                valid_k_values = [k for k in k_values if k <= len(G)]
                
                if valid_k_values:
                    daily_metrics = compute_daily_evaluation_metrics(
                        pagerank_scores=pagerank_scores,
                        hits_hubs=hits_hubs,
                        hits_auths=hits_auths,
                        bad_actors=bad_actors_current,  # Time-aware: no future leakage
                        k_values=valid_k_values
                    )
                    
                    # Flatten metrics for storage
                    for algo_name, algo_metrics in daily_metrics.items():
                        metric_record = {
                            'date': current_date.date(),
                            'algorithm': algo_name,
                            'total_nodes': algo_metrics['total_nodes'],
                            'total_fraud': algo_metrics['total_fraud'],
                            'fraud_rate': algo_metrics['fraud_rate'],
                            'roc_auc': algo_metrics.get('roc_auc'),
                            'average_precision': algo_metrics.get('average_precision'),
                        }
                        
                        # Add precision@k, recall@k, lift@k for each k
                        for k in valid_k_values:
                            metric_record[f'precision_at_{k}'] = algo_metrics['precision_at_k'].get(k)
                            metric_record[f'recall_at_{k}'] = algo_metrics['recall_at_k'].get(k)
                            metric_record[f'lift_at_{k}'] = algo_metrics['lift_at_k'].get(k)
                            metric_record[f'fraud_found_at_{k}'] = algo_metrics['fraud_found_at_k'].get(k)
                        
                        all_daily_metrics.append(metric_record)
                    
            # Store current scores for next iteration's rank stability analysis
            prev_pagerank_scores = pagerank_scores.copy()
            prev_hits_hubs = hits_hubs.copy()
            prev_hits_auths = hits_auths.copy()
        
        # Move to next day
        current_date += pd.Timedelta(days=STEP_SIZE)
        pbar.update(1)
    
    pbar.close()
    
    # 9. Save feature results
    print("\nCompiling results...")
    results_df = pd.DataFrame(all_features)
    
    if results_df.empty:
        print("Warning: No features were generated!")
        return
    
    print(f"Generated {len(results_df):,} feature records")
    print(f"Unique entities: {results_df['entity_id'].nunique():,}")
    print(f"Unique dates: {results_df['date'].nunique():,}")
    print(f"Fraud records: {results_df['is_fraud'].sum():,} ({100 * results_df['is_fraud'].mean():.2f}%)")
    
    # Print rank stability summary
    if RUN_RANK_STABILITY and all_rank_stability:
        stability_df = pd.DataFrame(all_rank_stability)
        avg_stability = stability_df['stability_score'].mean()
        total_anomalies = stability_df['num_anomalies'].sum()
        print(f"\nRank Stability Analysis:")
        print(f"  Average stability score: {avg_stability:.4f}")
        print(f"  Total rank anomalies detected: {total_anomalies:,}")
        print(f"  Entities flagged as rank anomalies: {results_df['is_rank_anomaly'].sum():,}")
    
    print(f"\nSaving features to {OUTPUT_FEATURES_FILE}...")
    results_df.to_parquet(OUTPUT_FEATURES_FILE, index=False)
    
    # 10. Save and summarize evaluation metrics
    if RUN_EVALUATION and all_daily_metrics:
        metrics_df = pd.DataFrame(all_daily_metrics)
        print(f"Saving metrics to {OUTPUT_METRICS_FILE}...")
        metrics_df.to_parquet(OUTPUT_METRICS_FILE, index=False)
        
        # Print aggregate evaluation summary
        print("\n" + "=" * 60)
        print("Aggregate Evaluation Summary (Mean Across All Days)")
        print("=" * 60)
        
        for algo in ['pagerank', 'hits_hub', 'hits_auth']:
            algo_df = metrics_df[metrics_df['algorithm'] == algo]
            if algo_df.empty:
                continue
                
            print(f"\n[{algo.upper()}]")
            print("-" * 40)
            
            # Global metrics
            mean_roc = float(algo_df['roc_auc'].mean())
            mean_ap = float(algo_df['average_precision'].mean())
            
            if not math.isnan(mean_roc):
                print(f"  Mean ROC-AUC: {mean_roc:.4f}")
            if not math.isnan(mean_ap):
                print(f"  Mean Average Precision: {mean_ap:.4f}")
            
            # Precision@K summary
            print("\n  Mean Precision@K:")
            for k in k_values:
                col = f'precision_at_{k}'
                if col in algo_df.columns:
                    mean_prec = float(algo_df[col].mean())
                    if not math.isnan(mean_prec):
                        lift_col = f'lift_at_{k}'
                        mean_lift = float(algo_df[lift_col].mean()) if lift_col in algo_df.columns else 0.0
                        print(f"    @{k:>5}: {mean_prec:.4%} (lift: {mean_lift:.2f}x)")
    
    # 11. Leiden community analysis
    if RUN_LEIDEN:
        analyze_leiden_effectiveness(results_df)
    
    # Feature statistics summary
    print("\n" + "=" * 60)
    print("Feature Statistics Summary")
    print("=" * 60)
    print(results_df[['pagerank', 'hits_hub', 'hits_auth', 'degree', 'leiden_size', 'vol_sent', 'vol_recv', 'tx_count', 'pagerank_rank_change']].describe())
    
    # Rank anomaly analysis
    if RUN_RANK_STABILITY and results_df['is_rank_anomaly'].sum() > 0:
        print("\n" + "=" * 60)
        print("Rank Anomaly Analysis")
        print("=" * 60)
        anomaly_df = results_df[results_df['is_rank_anomaly'] == 1]
        fraud_in_anomalies = anomaly_df['is_fraud'].sum()
        total_anomalies = len(anomaly_df)
        overall_fraud_rate = results_df['is_fraud'].mean()
        anomaly_fraud_rate = anomaly_df['is_fraud'].mean() if total_anomalies > 0 else 0
        
        print(f"Total rank anomaly flags: {total_anomalies:,}")
        print(f"Fraud among rank anomalies: {fraud_in_anomalies:,} ({anomaly_fraud_rate:.2%})")
        print(f"Overall fraud rate: {overall_fraud_rate:.2%}")
        if overall_fraud_rate > 0:
            print(f"Lift from rank anomaly detection: {anomaly_fraud_rate / overall_fraud_rate:.2f}x")
    
    print("\nDone!")


if __name__ == "__main__":
    main()