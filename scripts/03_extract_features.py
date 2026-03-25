from __future__ import annotations
import gc
import json
import shutil
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import duckdb
import networkx as nx
import pandas as pd
from tqdm import tqdm
from src.config import DATA_PATH, PR_ALPHA_DEEP, PR_ALPHA_SHALLOW, PR_MAX_ITER, BETWEENNESS_K, HITS_MAX_ITER, LEIDEN_RESOLUTION_MACRO, LEIDEN_RESOLUTION_MICRO
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
from src.config import (
    DATASET_SIZE,
    EVALUATION_K_VALUES,
    OUTPUT_FEATURES_FILE,
    OUTPUT_METRICS_FILE,
    RUN_EVALUATION,
    RUN_LEIDEN,
)
from src.features.base import FeatureExtractor
from src.features.centrality import (
    BetweennessExtractor,
    HITSExtractor,
    PageRankFrequencyExtractor,
    PageRankVolumeExtractor,
)
from src.features.community import KCoreExtractor, LeidenCommunityExtractor
from src.features.motifs import SubgraphMotifExtractor


def process_window(
    window_id: int,
    edges_path: Path,
    nodes_path: Path,
    run_flags: dict,
) -> dict:
    query_edges = (
        f"SELECT * FROM read_parquet('{str(edges_path)}') "
        f"WHERE window_id = {window_id}"
    )
    day_edges = duckdb.query(query_edges).df()

    query_nodes = (
        f"SELECT * FROM read_parquet('{str(nodes_path)}') "
        f"WHERE window_id = {window_id}"
    )
    day_nodes = duckdb.query(query_nodes).df()

    if day_edges.empty or day_nodes.empty:
        del day_edges, day_nodes
        gc.collect()
        return {"window_id": window_id, "window_start": None, "features": [], "metrics": [], "daily_metrics": {}}

    extractors: dict[str, FeatureExtractor] = {
        "pr_vol_deep": PageRankVolumeExtractor(alpha=PR_ALPHA_DEEP, max_iter=PR_MAX_ITER),
        "pr_vol_shallow": PageRankVolumeExtractor(alpha=PR_ALPHA_SHALLOW, max_iter=PR_MAX_ITER),
        "pr_count": PageRankFrequencyExtractor(alpha=PR_ALPHA_DEEP, max_iter=PR_MAX_ITER),
        "hits": HITSExtractor(max_iter=HITS_MAX_ITER),
        "betweenness": BetweennessExtractor(k=BETWEENNESS_K, seed=42),
        "k_core": KCoreExtractor(),
        "motifs": SubgraphMotifExtractor(fan_threshold=6, cycle_bound=6, max_degree=16, max_cycles=2000),
    }

    if run_flags.get("run_leiden", False):
        extractors["leiden_macro"] = LeidenCommunityExtractor(resolution=LEIDEN_RESOLUTION_MACRO)
        extractors["leiden_micro"] = LeidenCommunityExtractor(resolution=LEIDEN_RESOLUTION_MICRO)

    window_start = day_edges["window_start"].iloc[0] if not day_edges.empty else None
    window_end = day_edges["window_end"].iloc[0] if not day_edges.empty else None

    G = nx.from_pandas_edgelist(
        day_edges,
        source="source",
        target="target",
        edge_attr=["volume", "count"],
        create_using=nx.DiGraph,
    )

    node_stat_columns = [
        "vol_sent", "vol_recv", "tx_count", "time_variance",
        "distinct_currencies_sent", "distinct_currencies_recv",
        "wire_count_sent", "wire_count_recv", 
        "cash_count_sent", "cash_count_recv",
        "bitcoin_count_sent", "bitcoin_count_recv",
        "cheque_count_sent", "cheque_count_recv",
        "credit_card_count_sent", "credit_card_count_recv",
        "ach_count_sent", "ach_count_recv",
        "reinvestment_count_sent", "reinvestment_count_recv",
        "in_degree", "out_degree", "degree"
    ]

    if "is_fraud" in day_nodes.columns:
        node_stat_columns.append("is_fraud")
    node_stats = day_nodes.set_index("entity_id")[node_stat_columns].to_dict(orient="index")

    daily_metrics: dict[str, dict] = {}
    for key, extractor in extractors.items():
        daily_metrics[key] = extractor.extract(G)

    bad_actors_up_to_date: set = run_flags.get("bad_actors", set()) # AMLworld, graph based anomaly survey

    features: list[dict] = []
    for node in day_nodes["entity_id"].unique():
        ns = node_stats.get(node, {})
        vol_s = ns.get("vol_sent", 0.0)
        vol_r = ns.get("vol_recv", 0.0)

        record: dict = {
            "window_id": window_id,
            "window_start": window_start,
            "window_end": window_end,
            "entity_id": node,
            "pr_vol_deep": daily_metrics.get("pr_vol_deep", {}).get(node, {}).get("pagerank", 0.0),
            "pr_vol_shallow": daily_metrics.get("pr_vol_shallow", {}).get(node, {}).get("pagerank", 0.0),
            "pr_count": daily_metrics.get("pr_count", {}).get(node, {}).get("pagerank_count", 0.0),
            "hits_hub": daily_metrics.get("hits", {}).get(node, {}).get("hits_hub", 0.0),
            "hits_auth": daily_metrics.get("hits", {}).get(node, {}).get("hits_auth", 0.0),
            "leiden_macro_id": daily_metrics.get("leiden_macro", {}).get(node, {}).get("leiden_id", -1),
            "leiden_macro_size": daily_metrics.get("leiden_macro", {}).get(node, {}).get("leiden_size", 0),
            "leiden_macro_modularity": daily_metrics.get("leiden_macro", {}).get(node, {}).get("leiden_modularity", 0.0),
            "leiden_micro_id": daily_metrics.get("leiden_micro", {}).get(node, {}).get("leiden_id", -1),
            "leiden_micro_size": daily_metrics.get("leiden_micro", {}).get(node, {}).get("leiden_size", 0),
            "leiden_micro_modularity": daily_metrics.get("leiden_micro", {}).get(node, {}).get("leiden_modularity", 0.0),
            "degree": ns.get("degree", 0),
            "in_degree": ns.get("in_degree", 0),
            "out_degree": ns.get("out_degree", 0),
            "vol_sent": vol_s,
            "vol_recv": vol_r,
            "tx_count": ns.get("tx_count", 0),
            "time_variance": ns.get("time_variance", 0.0),
            "is_fraud": int(ns.get("is_fraud", 0) or 0),
            "flow_ratio": vol_s / (vol_r if vol_r > 0 else vol_s),
            "distinct_currencies_sent": ns.get("distinct_currencies_sent", 0),
            "distinct_currencies_recv": ns.get("distinct_currencies_recv", 0),
            "wire_count_sent": ns.get("wire_count_sent", 0),
            "wire_count_recv": ns.get("wire_count_recv", 0),
            "cash_count_sent": ns.get("cash_count_sent", 0),
            "cash_count_recv": ns.get("cash_count_recv", 0),
            "bitcoin_count_sent": ns.get("bitcoin_count_sent", 0),
            "bitcoin_count_recv": ns.get("bitcoin_count_recv", 0),
            "cheque_count_sent": ns.get("cheque_count_sent", 0),
            "cheque_count_recv": ns.get("cheque_count_recv", 0),
            "credit_card_count_sent": ns.get("credit_card_count_sent", 0),
            "credit_card_count_recv": ns.get("credit_card_count_recv", 0),
            "ach_count_sent": ns.get("ach_count_sent", 0),
            "ach_count_recv": ns.get("ach_count_recv", 0),
            "reinvestment_count_sent": ns.get("reinvestment_count_sent", 0),
            "reinvestment_count_recv": ns.get("reinvestment_count_recv", 0),
            "betweenness": daily_metrics.get("betweenness", {}).get(node, {}).get("betweenness", 0.0),
            "k_core": daily_metrics.get("k_core", {}).get(node, {}).get("k_core", 0),
            "fan_out_count": daily_metrics.get("motifs", {}).get(node, {}).get("fan_out_count", 0),
            "fan_in_count": daily_metrics.get("motifs", {}).get(node, {}).get("fan_in_count", 0),
            "scatter_gather_count": daily_metrics.get("motifs", {}).get(node, {}).get("scatter_gather_count", 0),
            "gather_scatter_count": daily_metrics.get("motifs", {}).get(node, {}).get("gather_scatter_count", 0),
            "cycle_count": daily_metrics.get("motifs", {}).get(node, {}).get("cycle_count", 0),
        }
        features.append(record)

    eval_metric_records: list[dict] = []

    if run_flags.get("run_evaluation", False):
        pr_vol_deep_scores = {
            node: daily_metrics.get("pr_vol_deep", {}).get(node, {}).get("pagerank", 0.0)
            for node in G.nodes()
        }
        pr_vol_shallow_scores = {
            node: daily_metrics.get("pr_vol_shallow", {}).get(node, {}).get("pagerank", 0.0)
            for node in G.nodes()
        }
        pr_count_scores = {
            node: daily_metrics.get("pr_count", {}).get(node, {}).get("pagerank_count", 0.0)
            for node in G.nodes()
        }
        hits_hubs = {
            node: daily_metrics.get("hits", {}).get(node, {}).get("hits_hub", 0.0)
            for node in G.nodes()
        }
        hits_auths = {
            node: daily_metrics.get("hits", {}).get(node, {}).get("hits_auth", 0.0)
            for node in G.nodes()
        }
        betweenness_scores = {
            node: daily_metrics.get("betweenness", {}).get(node, {}).get("betweenness", 0.0)
            for node in G.nodes()
        }
        k_core_scores = {
            node: daily_metrics.get("k_core", {}).get(node, {}).get("k_core", 0)
            for node in G.nodes()
        }
        
        # Basic feature scores
        vol_sent_scores = {
            node: node_stats.get(node, {}).get("vol_sent", 0.0)
            for node in G.nodes()
        }
        vol_recv_scores = {
            node: node_stats.get(node, {}).get("vol_recv", 0.0)
            for node in G.nodes()
        }
        in_degree_scores = {
            node: node_stats.get(node, {}).get("in_degree", 0)
            for node in G.nodes()
        }
        out_degree_scores = {
            node: node_stats.get(node, {}).get("out_degree", 0)
            for node in G.nodes()
        }
        tx_count_scores = {
            node: node_stats.get(node, {}).get("tx_count", 0)
            for node in G.nodes()
        }
        time_variance_scores = {
            node: node_stats.get(node, {}).get("time_variance", 0.0)
            for node in G.nodes()
        }
        
        # Baseline heuristic feature scores
        wire_count_sent_scores = {
            node: node_stats.get(node, {}).get("wire_count_sent", 0.0)
            for node in G.nodes()
        }
        wire_count_recv_scores = {
            node: node_stats.get(node, {}).get("wire_count_recv", 0.0)
            for node in G.nodes()
        }
        cash_count_sent_scores = {
            node: node_stats.get(node, {}).get("cash_count_sent", 0.0)
            for node in G.nodes()
        }
        cash_count_recv_scores = {
            node: node_stats.get(node, {}).get("cash_count_recv", 0.0)
            for node in G.nodes()
        }
        bitcoin_count_sent_scores = {
            node: node_stats.get(node, {}).get("bitcoin_count_sent", 0.0)
            for node in G.nodes()
        }
        bitcoin_count_recv_scores = {
            node: node_stats.get(node, {}).get("bitcoin_count_recv", 0.0)
            for node in G.nodes()
        }
        cheque_count_sent_scores = {
            node: node_stats.get(node, {}).get("cheque_count_sent", 0.0)
            for node in G.nodes()
        }
        cheque_count_recv_scores = {
            node: node_stats.get(node, {}).get("cheque_count_recv", 0.0)
            for node in G.nodes()
        }
        credit_card_count_sent_scores = {
            node: node_stats.get(node, {}).get("credit_card_count_sent", 0.0)
            for node in G.nodes()
        }
        credit_card_count_recv_scores = {
            node: node_stats.get(node, {}).get("credit_card_count_recv", 0.0)
            for node in G.nodes()
        }
        ach_count_sent_scores = {
            node: node_stats.get(node, {}).get("ach_count_sent", 0.0)
            for node in G.nodes()
        }
        ach_count_recv_scores = {
            node: node_stats.get(node, {}).get("ach_count_recv", 0.0)
            for node in G.nodes()
        }
        reinvestment_count_sent_scores = {
            node: node_stats.get(node, {}).get("reinvestment_count_sent", 0.0)
            for node in G.nodes()
        }
        reinvestment_count_recv_scores = {
            node: node_stats.get(node, {}).get("reinvestment_count_recv", 0.0)
            for node in G.nodes()
        }
        distinct_currencies_sent_scores = {
            node: node_stats.get(node, {}).get("distinct_currencies_sent", 0.0)
            for node in G.nodes()
        }
        distinct_currencies_recv_scores = {
            node: node_stats.get(node, {}).get("distinct_currencies_recv", 0.0)
            for node in G.nodes()
        }

        if pr_vol_deep_scores:
            evaluation_k_values = run_flags.get("evaluation_k_values", [])
            valid_k_values = [k for k in evaluation_k_values if k <= len(G)]
            if valid_k_values:
                from src.features._evaluation import compute_daily_evaluation_metrics

                eval_metrics = compute_daily_evaluation_metrics(
                    bad_actors=bad_actors_up_to_date,
                    k_values=valid_k_values,
                    pr_vol_deep=pr_vol_deep_scores,
                    pr_vol_shallow=pr_vol_shallow_scores,
                    pr_count=pr_count_scores,
                    hits_hub=hits_hubs,
                    hits_auth=hits_auths,
                    betweenness=betweenness_scores,
                    k_core=k_core_scores,
                    vol_sent=vol_sent_scores,
                    vol_recv=vol_recv_scores,
                    in_degree=in_degree_scores,
                    out_degree=out_degree_scores,
                    tx_count=tx_count_scores,
                    time_variance=time_variance_scores,
                    wire_count_sent=wire_count_sent_scores,
                    wire_count_recv=wire_count_recv_scores,
                    cash_count_sent=cash_count_sent_scores,
                    cash_count_recv=cash_count_recv_scores,
                    bitcoin_count_sent=bitcoin_count_sent_scores,
                    bitcoin_count_recv=bitcoin_count_recv_scores,
                    cheque_count_sent=cheque_count_sent_scores,
                    cheque_count_recv=cheque_count_recv_scores,
                    credit_card_count_sent=credit_card_count_sent_scores,
                    credit_card_count_recv=credit_card_count_recv_scores,
                    ach_count_sent=ach_count_sent_scores,
                    ach_count_recv=ach_count_recv_scores,
                    reinvestment_count_sent=reinvestment_count_sent_scores,
                    reinvestment_count_recv=reinvestment_count_recv_scores,
                    distinct_currencies_sent=distinct_currencies_sent_scores,
                    distinct_currencies_recv=distinct_currencies_recv_scores,
                )
                for algo_name, algo_metrics in eval_metrics.items():
                    metric_record: dict = {
                        "window_id": window_id,
                        "window_start": window_start,
                        "window_end": window_end,
                        "algorithm": algo_name,
                        "total_nodes": algo_metrics["total_nodes"],
                        "total_fraud": algo_metrics["total_fraud"],
                        "fraud_rate": algo_metrics["fraud_rate"],
                        "roc_auc": algo_metrics.get("roc_auc"),
                        "average_precision": algo_metrics.get("average_precision"),
                    }
                    for k in valid_k_values:
                        metric_record[f"precision_at_{k}"] = algo_metrics["precision_at_k"].get(k)
                        metric_record[f"recall_at_{k}"] = algo_metrics["recall_at_k"].get(k)
                        metric_record[f"lift_at_{k}"] = algo_metrics["lift_at_k"].get(k)
                        metric_record[f"fraud_found_at_{k}"] = algo_metrics["fraud_found_at_k"].get(k)
                    eval_metric_records.append(metric_record)

    result = {
        "window_id": window_id,
        "window_start": window_start,
        "window_end": window_end,
        "features": features,
        "metrics": eval_metric_records,
        "daily_metrics": daily_metrics,
    }

    del G, day_edges, day_nodes, node_stats, daily_metrics
    gc.collect()

    return result


def load_checkpoint(checkpoint_path: Path) -> dict:
    """Load checkpoint data from disk."""
    if checkpoint_path.exists():
        with open(checkpoint_path, 'r') as f:
            return json.load(f)
    return {"completed_windows": [], "feature_chunks_dir": None, "metrics_chunks_dir": None}


def save_checkpoint(checkpoint_path: Path, data: dict) -> None:
    """Save checkpoint data to disk."""
    with open(checkpoint_path, 'w') as f:
        json.dump(data, f, indent=2)


def main() -> None:
    print("=" * 60)
    print("Graph Feature Generation Pipeline")
    print(f"Dataset: {DATASET_SIZE}")
    print(f"Evaluation:     {'Enabled' if RUN_EVALUATION else 'Disabled'}")
    print(f"Leiden:         {'Enabled' if RUN_LEIDEN else 'Disabled'}")
    print("=" * 60)

    edges_path = DATA_PATH / "lookback_edges.parquet"
    nodes_path = DATA_PATH / "target_nodes.parquet" 
    features_out = DATA_PATH / OUTPUT_FEATURES_FILE
    
    # Checkpoint file to track progress
    checkpoint_path = DATA_PATH / "feature_extraction_checkpoint.json"
    checkpoint = load_checkpoint(checkpoint_path)
    
    # Reuse existing chunk directories if resuming, otherwise create new ones
    if checkpoint.get("feature_chunks_dir") and Path(checkpoint["feature_chunks_dir"]).exists():
        feature_chunks_dir = Path(checkpoint["feature_chunks_dir"])
        print(f"Resuming from checkpoint: {checkpoint_path}")
        print(f"Using existing chunk directory: {feature_chunks_dir}")
    else:
        feature_chunks_dir = Path(tempfile.mkdtemp(prefix="feature_chunks_"))
        checkpoint["feature_chunks_dir"] = str(feature_chunks_dir)
    
    if RUN_EVALUATION:
        if checkpoint.get("metrics_chunks_dir") and Path(checkpoint["metrics_chunks_dir"]).exists():
            metrics_chunks_dir = Path(checkpoint["metrics_chunks_dir"])
        else:
            metrics_chunks_dir = Path(tempfile.mkdtemp(prefix="metrics_chunks_"))
            checkpoint["metrics_chunks_dir"] = str(metrics_chunks_dir)
    else:
        metrics_chunks_dir = None

    con = duckdb.connect()

    print("Extracting unique temporal windows...")
    unique_windows_df = con.execute(
        f"""
        SELECT DISTINCT window_id, window_start, window_end
        FROM read_parquet('{edges_path}')
        ORDER BY window_id
        """
    ).df()
    unique_window_ids = unique_windows_df["window_id"].tolist()
    del unique_windows_df
    gc.collect()

    # Filter out already completed windows
    completed_windows = set(checkpoint.get("completed_windows", []))
    remaining_window_ids = [wid for wid in unique_window_ids if wid not in completed_windows]
    
    print(f"Found {len(unique_window_ids)} total windows.")
    if completed_windows:
        print(f"Already completed: {len(completed_windows)} windows")
        print(f"Remaining to process: {len(remaining_window_ids)} windows")
    
    if not remaining_window_ids:
        print("All windows already processed. Proceeding to final merge...")
        # Skip to merge step
        unique_window_ids = []
    elif not unique_window_ids:
        print("No windows found. Nothing to process.")
        if checkpoint_path.exists():
            checkpoint_path.unlink()
        shutil.rmtree(feature_chunks_dir, ignore_errors=False)
        con.close()
        return
    else:
        unique_window_ids = remaining_window_ids

    def get_bad_actors_for_window(wid: int) -> set:
        """Fetch fraud labels on-demand to avoid holding all in memory."""
        bad_actor_df = con.execute(
            f"""
            SELECT DISTINCT entity_id
            FROM read_parquet('{nodes_path}')
            WHERE window_id = {wid} AND is_fraud = 1
            """,
        ).df()
        result = set(bad_actor_df["entity_id"].tolist())
        del bad_actor_df
        return result

    run_flags_base: dict = {
        "run_evaluation": RUN_EVALUATION,
        "run_leiden": RUN_LEIDEN,
        "evaluation_k_values": list(EVALUATION_K_VALUES),
    }

    max_workers = 1
    print(f"Launching ProcessPoolExecutor with {max_workers} workers...\n")

    # Count existing chunks
    written_chunk_files = len(list(feature_chunks_dir.glob("*.parquet"))) if feature_chunks_dir.exists() else 0
    written_metrics_files = len(list(metrics_chunks_dir.glob("*.parquet"))) if metrics_chunks_dir and metrics_chunks_dir.exists() else 0
    
    print(f"Existing feature chunks: {written_chunk_files}")
    if RUN_EVALUATION:
        print(f"Existing metrics chunks: {written_metrics_files}")

    # Only run processing if there are windows remaining
    if unique_window_ids:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures: dict = {}
            window_queue = list(unique_window_ids)
            
            # Submit initial batch (max_workers * 2 to keep workers busy)
            initial_batch_size = min(max_workers * 2, len(window_queue))
            for _ in range(initial_batch_size):
                if window_queue:
                    window_id = window_queue.pop(0)
                    bad_actors = get_bad_actors_for_window(window_id) if RUN_EVALUATION else set()
                    run_flags = {
                        **run_flags_base,
                        "bad_actors": bad_actors,
                    }
                    future = executor.submit(
                        process_window,
                        window_id,
                        edges_path,
                        nodes_path,
                        run_flags,
                    )
                    futures[future] = window_id

            # Show progress including already completed windows
            total_windows = len(checkpoint.get("completed_windows", [])) + len(unique_window_ids)
            initial_progress = len(checkpoint.get("completed_windows", []))
            
            with tqdm(total=total_windows, desc="Processing windows", unit="window", initial=initial_progress) as pbar:
                for future in as_completed(futures):
                    completed_window_id = futures[future]
                    try:
                        result = future.result()  # propagates worker exceptions

                        if result["features"]:
                            feature_df = pd.DataFrame(result["features"])
                            chunk_path = feature_chunks_dir / f"features_{result['window_id']}.parquet"
                            feature_df.to_parquet(chunk_path, index=False)
                            written_chunk_files += 1
                            del feature_df

                        if RUN_EVALUATION and result["metrics"] and metrics_chunks_dir:
                            metrics_df = pd.DataFrame(result["metrics"])
                            metrics_chunk_path = metrics_chunks_dir / f"metrics_{result['window_id']}.parquet"
                            metrics_df.to_parquet(metrics_chunk_path, index=False)
                            written_metrics_files += 1
                            del metrics_df
                        
                        # Save checkpoint after each successful window
                        checkpoint["completed_windows"].append(completed_window_id)
                        save_checkpoint(checkpoint_path, checkpoint)
                        
                        # Free memory: daily_metrics is not needed after features are extracted
                        del result["daily_metrics"]
                        del result
                        gc.collect()
                    except Exception as exc:
                        print(f"\nWindow {completed_window_id} generated an exception: {exc}")
                        print("Checkpoint saved. You can restart the script to resume.")
                    
                    # Remove completed future and submit next window
                    del futures[future]
                    if window_queue:
                        window_id = window_queue.pop(0)
                        bad_actors = get_bad_actors_for_window(window_id) if RUN_EVALUATION else set()
                        run_flags = {
                            **run_flags_base,
                            "bad_actors": bad_actors,
                        }
                        new_future = executor.submit(
                            process_window,
                            window_id,
                            edges_path,
                            nodes_path,
                            run_flags,
                        )
                        futures[new_future] = window_id
                    
                    pbar.update(1)

    # Recount chunks in case we resumed from checkpoint
    written_chunk_files = len(list(feature_chunks_dir.glob("*.parquet")))
    if RUN_EVALUATION and metrics_chunks_dir:
        written_metrics_files = len(list(metrics_chunks_dir.glob("*.parquet")))
    
    if written_chunk_files == 0:
        print("Warning: No features were generated!")
        shutil.rmtree(feature_chunks_dir, ignore_errors=False)
        if checkpoint_path.exists():
            checkpoint_path.unlink()
        con.close()
        return

    print("\nUsing pre-aggregated is_fraud labels from aggregated_nodes.parquet...")

    stability_join = ""
    stability_select = (
        "0::DOUBLE AS weirdnodes_magnitude,\n"
        "                0::DOUBLE AS weirdnodes_residual,\n"
        "                0::INTEGER AS is_riser,\n"
        "                0::INTEGER AS is_faller"
    )

    print(f"Saving features to {features_out} via DuckDB out-of-core join...")
    con.execute(
        f"""
        COPY (
            SELECT
                f.*,
                {stability_select}
            FROM read_parquet('{feature_chunks_dir / "*.parquet"}') f
            {stability_join}
        ) TO '{features_out}' (FORMAT PARQUET);
        """
    )

    shutil.rmtree(feature_chunks_dir, ignore_errors=False)

    if RUN_EVALUATION and metrics_chunks_dir and written_metrics_files > 0:
        metrics_path = DATA_PATH / OUTPUT_METRICS_FILE
        print(f"Merging {written_metrics_files} metrics chunks to {metrics_path}...")
        con.execute(
            f"""
            COPY (
                SELECT * FROM read_parquet('{metrics_chunks_dir / "*.parquet"}')
            ) TO '{metrics_path}' (FORMAT PARQUET);
            """
        )
        shutil.rmtree(metrics_chunks_dir, ignore_errors=False)
    elif metrics_chunks_dir:
        shutil.rmtree(metrics_chunks_dir, ignore_errors=False)

    con.close()
    
    # Clean up checkpoint file after successful completion
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        print(f"Checkpoint file removed: {checkpoint_path}")
    
    print("\nFeature extraction complete!")


if __name__ == "__main__":
    main()
