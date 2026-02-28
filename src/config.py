"""
Pipeline configuration for the PixFraudDetection graph feature pipeline.

This module is the single source of truth for every tunable constant used
across the pipeline.  Import directly from here in all src/ modules and
scripts — never from the legacy root-level config.py.
"""

from pathlib import Path

# ============================================================================
# DATASET SELECTION
# ============================================================================
# Set to "SMALL" or "LARGE" to switch the active dataset.
DATASET_SIZE: str = "SMALL"

# Dataset-specific settings
_DATASET_CONFIG = {
    "SMALL": {
        "data_path": Path("data/HI_Small"),
        "window_days": 3,
    },
    "LARGE": {
        "data_path": Path("data/HI_Large"),
        "window_days": 7,
    },
}

DATA_PATH: Path = _DATASET_CONFIG[DATASET_SIZE]["data_path"]

# ============================================================================
# SLIDING WINDOW SETTINGS
# ============================================================================
# Number of days covered by each temporal window snapshot.
WINDOW_DAYS: int = _DATASET_CONFIG[DATASET_SIZE]["window_days"]

# How many days to advance the window on each iteration.
STEP_SIZE: int = 1

# ============================================================================
# INPUT FILE NAMES
# ============================================================================
NORMAL_TRANSACTIONS_FILE: str = "1_filtered_normal_transactions.parquet"
LAUNDERING_TRANSACTIONS_FILE: str = "2_filtered_laundering_transactions.parquet"
ACCOUNTS_FILE: str = "3_filtered_accounts.parquet"

# ============================================================================
# OUTPUT FILE NAMES
# ============================================================================
OUTPUT_FEATURES_FILE: str = "sliding_window_features.parquet"
OUTPUT_METRICS_FILE: str = "sliding_window_metrics.parquet"

# ============================================================================
# ALGORITHM HYPER-PARAMETERS
# ============================================================================
# Maximum iterations for the HITS algorithm before declaring non-convergence.
HITS_MAX_ITER: int = 100

# PageRank damping factor (standard NetworkX default).
PAGERANK_ALPHA: float = 0.85

# ============================================================================
# EVALUATION SETTINGS
# ============================================================================
# K values used when computing Precision@K / Recall@K / Lift@K.
EVALUATION_K_VALUES: list = [1, 2, 10, 50, 100, 500]

# Toggle the evaluation metrics pass (disable to speed up pure feature runs).
RUN_EVALUATION: bool = True

# ============================================================================
# LEIDEN COMMUNITY DETECTION SETTINGS
# ============================================================================
# Toggle Leiden community detection entirely.
RUN_LEIDEN: bool = True

# Resolution parameter passed to Leiden.
# Higher values → smaller, more granular communities.
# Lower values  → larger, coarser communities.
LEIDEN_RESOLUTION: float = 1.0

# ============================================================================
# RANK STABILITY ANALYSIS SETTINGS
# ============================================================================
# Rank stability analysis detects anomalies when a node's role shifts
# drastically between consecutive temporal snapshots.  Sudden changes in
# centrality metrics are a validated signal for suspicious financial activity.

# Toggle the rank stability pass.
RUN_RANK_STABILITY: bool = True

# Number of top nodes used when computing the Jaccard stability score and
# identifying new entrants / dropouts between windows.
RANK_STABILITY_TOP_K: int = 100

# Absolute rank-change percentile above which a node is flagged as anomalous.
# 95.0 means the top 5 % most extreme movers are flagged each window.
RANK_ANOMALY_PERCENTILE: float = 95.0
