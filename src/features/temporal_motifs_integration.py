"""
Integration Module for Temporal Motifs in Feature Extraction

This module provides integration utilities to add temporal motif features
to the existing feature extraction pipeline in 03_extract_features.py.

Since temporal motifs require transaction-level timestamps (not aggregated edges),
this module provides two integration strategies:

Strategy 1: Add temporal motifs as a separate processing step
Strategy 2: Modify process_window() to load transaction data when available

Usage:
    from src.features.temporal_motifs_integration import add_temporal_motifs_to_pipeline
"""

from pathlib import Path
from typing import Dict
import gc

import duckdb
import pandas as pd

from src.features.temporal_motifs import extract_temporal_motifs_from_transactions


def add_temporal_motifs_to_window_features(
    window_id: int,
    transactions_path: Path,
    existing_features_df: pd.DataFrame,
    source_col: str = 'from_account',
    target_col: str = 'to_account',
    timestamp_col: str = 'timestamp',
    delta_t_window: int = 86400
) -> pd.DataFrame:
    """
    Add temporal motif features to existing window features.
    
    This function loads transaction data for a specific window, computes
    temporal motifs, and merges them with existing features.
    
    Args:
        window_id: Window identifier
        transactions_path: Path to transaction parquet file
        existing_features_df: DataFrame with existing features (must have entity_id)
        source_col: Source account column in transactions
        target_col: Target account column in transactions
        timestamp_col: Timestamp column in transactions
        delta_t_window: Time window for temporal patterns (seconds)
    
    Returns:
        DataFrame with original features plus temporal motif features
    """
    # Load transactions for this window
    con = duckdb.connect()
    
    # Check if window_id exists in transactions
    # Note: Transactions may not have window_id, so we may need to join
    # or filter by date range. For now, assume direct filtering.
    try:
        txns = con.execute(f"""
            SELECT {source_col}, {target_col}, {timestamp_col}
            FROM read_parquet('{transactions_path}')
            WHERE window_id = {window_id}
        """).df()
    except:
        # Fallback: load all and filter by window dates if available
        print(f"Warning: Could not filter transactions by window_id {window_id}")
        return existing_features_df
    
    con.close()
    
    if txns.empty:
        # No transactions, add zero columns
        existing_features_df['temporal_triangle_count'] = 0
        existing_features_df['temporal_fan_out_count'] = 0
        existing_features_df['temporal_fan_in_count'] = 0
        existing_features_df['sequential_scatter_gather_count'] = 0
        return existing_features_df
    
    # Extract temporal motifs
    temporal_features = extract_temporal_motifs_from_transactions(
        transactions_df=txns,
        window_id=window_id,
        source_col=source_col,
        target_col=target_col,
        timestamp_col=timestamp_col,
        delta_t_window=delta_t_window
    )
    
    # Convert to DataFrame for merging
    temporal_df = pd.DataFrame([
        {
            'entity_id': entity_id,
            'temporal_triangle_count': features['temporal_triangle_count'],
            'temporal_fan_out_count': features['temporal_fan_out_count'],
            'temporal_fan_in_count': features['temporal_fan_in_count'],
            'sequential_scatter_gather_count': features['sequential_scatter_gather_count']
        }
        for entity_id, features in temporal_features.items()
    ])
    
    # Merge with existing features
    result = existing_features_df.merge(
        temporal_df,
        on='entity_id',
        how='left'
    )
    
    # Fill missing values (entities not in temporal graph)
    temporal_cols = [
        'temporal_triangle_count',
        'temporal_fan_out_count',
        'temporal_fan_in_count',
        'sequential_scatter_gather_count'
    ]
    for col in temporal_cols:
        if col in result.columns:
            result[col] = result[col].fillna(0).astype(int)
    
    del txns, temporal_features, temporal_df
    gc.collect()
    
    return result


def patch_process_window_for_temporal_motifs(
    original_process_window_func,
    transactions_path: Path
):
    """
    Create a patched version of process_window that includes temporal motifs.
    
    This is a decorator-style function that wraps the existing process_window
    to add temporal motif extraction.
    
    Example usage in 03_extract_features.py:
    
        from src.features.temporal_motifs_integration import patch_process_window_for_temporal_motifs
        
        # After defining process_window, wrap it:
        transactions_path = DATA_PATH / "1_filtered_transactions.parquet"
        process_window = patch_process_window_for_temporal_motifs(
            process_window,
            transactions_path
        )
    
    Args:
        original_process_window_func: The original process_window function
        transactions_path: Path to transaction data
    
    Returns:
        Patched process_window function
    """
    def patched_process_window(*args, **kwargs):
        # Call original
        result = original_process_window_func(*args, **kwargs)
        
        # If successful and has features, add temporal motifs
        if result.get('has_features', False):
            window_id = args[0] if len(args) > 0 else kwargs.get('window_id')
            feature_chunks_dir = args[4] if len(args) > 4 else kwargs.get('feature_chunks_dir')
            
            # Load the feature chunk that was just written
            chunk_path = feature_chunks_dir / f"features_window_{window_id}.parquet"
            
            if chunk_path.exists():
                features_df = pd.read_parquet(chunk_path)
                
                # Add temporal motifs
                features_df = add_temporal_motifs_to_window_features(
                    window_id=window_id,
                    transactions_path=transactions_path,
                    existing_features_df=features_df
                )
                
                # Write back
                features_df.to_parquet(chunk_path, index=False)
                
                del features_df
                gc.collect()
        
        return result
    
    return patched_process_window


# Quick integration snippet for copy-paste into 03_extract_features.py
INTEGRATION_SNIPPET = """
# ============================================================================
# TEMPORAL MOTIF INTEGRATION (Add after imports section)
# ============================================================================
from src.features.temporal_motifs_integration import patch_process_window_for_temporal_motifs

# (In main function, after process_window is defined)
# Check if transaction data is available
transactions_path = DATA_PATH / "1_filtered_transactions.parquet"
if transactions_path.exists():
    print("Temporal motif extraction: ENABLED")
    process_window = patch_process_window_for_temporal_motifs(
        process_window,
        transactions_path
    )
else:
    print("Temporal motif extraction: DISABLED (transaction data not found)")
# ============================================================================
"""
