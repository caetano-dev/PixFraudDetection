"""
Main pipeline for Sliding/Cumulative Window Graph Feature Generation.

This script generates time-series features (PageRank, HITS) for downstream
AI models by iterating through the transaction data day-by-day using either:
  - SLIDING window: Fixed-size window (current_date - WINDOW_DAYS, current_date]
  - CUMULATIVE window: Growing window [start_date, current_date]

The cumulative mode preserves full network structure, which is better for
capturing long laundering chains that span multiple days.
"""

from __future__ import annotations

import math
import pandas as pd
import networkx as nx
from tqdm import tqdm

from config import (
    WINDOW_DAYS,
    WINDOW_MODE,
    STEP_SIZE,
    OUTPUT_FEATURES_FILE,
    OUTPUT_METRICS_FILE,
    HITS_MAX_ITER,
    PAGERANK_ALPHA,
    DATASET_SIZE,
    EVALUATION_K_VALUES,
    RUN_EVALUATION,
    VERBOSE_EVALUATION,
)
from utils import (
    load_data,
    build_daily_graph,
    print_evaluation_report,
    compute_daily_evaluation_metrics,
)

# Type alias for clarity
DegreeView = int


def main():
    """
    Execute the sliding/cumulative window feature generation pipeline.
    
    Pipeline steps:
    1. Load all transaction data once
    2. Determine date range from dataset
    3. Iterate day-by-day with sliding or cumulative window
    4. For each window: build graph, run algorithms, extract features
    5. Optionally compute evaluation metrics per day
    6. Save all features and metrics to parquet files
    """
    print("=" * 60)
    print("Graph Feature Generation Pipeline")
    print(f"Dataset: {DATASET_SIZE} | Mode: {WINDOW_MODE}")
    if WINDOW_MODE == "SLIDING":
        print(f"Window: {WINDOW_DAYS} days | Step: {STEP_SIZE} day(s)")
    else:
        print(f"Window: Cumulative (grows from start) | Step: {STEP_SIZE} day(s)")
    print(f"Evaluation: {'Enabled' if RUN_EVALUATION else 'Disabled'}")
    print("=" * 60)
    
    # 1. Load all data once
    all_transactions, bad_actors = load_data()
    
    # 2. Determine date range
    start_date = all_transactions['timestamp'].min()
    end_date = all_transactions['timestamp'].max()
    print(f"\nData range: {start_date.date()} to {end_date.date()}")
    
    # Start iteration - for SLIDING we need a full window first, for CUMULATIVE start immediately
    if WINDOW_MODE == "SLIDING":
        current_date = start_date + pd.Timedelta(days=WINDOW_DAYS)
    else:
        # CUMULATIVE: start from first day (or after a small warmup)
        current_date = start_date + pd.Timedelta(days=1)
    
    total_days = (end_date - current_date).days + 1
    
    if WINDOW_MODE == "SLIDING":
        print(f"Processing {total_days} days with {WINDOW_DAYS}-day sliding window...\n")
    else:
        print(f"Processing {total_days} days with cumulative window (graph grows over time)...\n")
    
    # Lists to collect all records
    all_features = []
    all_daily_metrics = []
    
    # Filter k_values to reasonable sizes (will be further filtered per-day)
    k_values = EVALUATION_K_VALUES
    
    # 3. Sliding window loop
    pbar = tqdm(total=total_days, desc="Processing days", unit="day")
    
    while current_date <= end_date:
        # Calculate window boundaries based on mode
        if WINDOW_MODE == "SLIDING":
            # SLIDING: (current_date - WINDOW_DAYS, current_date]
            window_start = current_date - pd.Timedelta(days=WINDOW_DAYS)
            mask = (
                (all_transactions['timestamp'] > window_start) & 
                (all_transactions['timestamp'] <= current_date)
            )
        else:
            # CUMULATIVE: [start_date, current_date] - graph grows over time
            mask = (all_transactions['timestamp'] <= current_date)
        
        # 4. Filter transactions for current window
        window_df = all_transactions.loc[mask].copy()
        
        # Skip empty windows (weekends/holidays with no transactions)
        if window_df.empty:
            current_date += pd.Timedelta(days=STEP_SIZE)
            pbar.update(1)
            continue
        
        # 5. Build graph for this window
        G = build_daily_graph(window_df)
        
        # 6. Run algorithms and extract features
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
            
            # HITS with error handling
            try:
                hits_hubs, hits_auths = nx.hits(G, max_iter=HITS_MAX_ITER)
            except (nx.NetworkXError, nx.PowerIterationFailedConvergence):
                hits_hubs, hits_auths = {}, {}
            
            # 7. Extract features for each node in the graph
            for node in G.nodes():
                record = {
                    'date': current_date.date(),
                    'entity_id': node,
                    'pagerank': pagerank_scores.get(node, 0.0),
                    'hits_hub': hits_hubs.get(node, 0.0),
                    'hits_auth': hits_auths.get(node, 0.0),
                    'degree': int(G.degree(node)),
                    'in_degree': int(G.in_degree(node)),
                    'out_degree': int(G.out_degree(node)),
                    'is_fraud': 1 if node in bad_actors else 0,
                }
                all_features.append(record)
            
            # 8. Compute evaluation metrics for this day (optional)
            if RUN_EVALUATION and pagerank_scores:
                # Filter k_values to not exceed nodes in this window
                valid_k_values = [k for k in k_values if k <= len(G)]
                
                if valid_k_values:
                    daily_metrics = compute_daily_evaluation_metrics(
                        pagerank_scores=pagerank_scores,
                        hits_hubs=hits_hubs,
                        hits_auths=hits_auths,
                        bad_actors=bad_actors,
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
                    
                    # Print verbose evaluation if enabled
                    if VERBOSE_EVALUATION:
                        print(f"\n--- Day: {current_date.date()} ---")
                        for algo_name, algo_metrics in daily_metrics.items():
                            print_evaluation_report(algo_metrics, algo_name.upper())
        
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
    
    # Feature statistics summary
    print("\n" + "=" * 60)
    print("Feature Statistics Summary")
    print("=" * 60)
    print(results_df[['pagerank', 'hits_hub', 'hits_auth', 'degree']].describe())
    
    print("\nDone!")


if __name__ == "__main__":
    main()