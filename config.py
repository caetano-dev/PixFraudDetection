from pathlib import Path
# Set this to "SMALL" or "LARGE" to switch datasets
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
WINDOW_DAYS: int = _DATASET_CONFIG[DATASET_SIZE]["window_days"]

# Step size for sliding window (in days)
STEP_SIZE: int = 1

NORMAL_TRANSACTIONS_FILE: str = "1_filtered_normal_transactions.parquet"
LAUNDERING_TRANSACTIONS_FILE: str = "2_filtered_laundering_transactions.parquet"
ACCOUNTS_FILE: str = "3_filtered_accounts.parquet"

OUTPUT_FEATURES_FILE: str = "sliding_window_features.parquet"
OUTPUT_METRICS_FILE: str = "sliding_window_metrics.parquet"

# Maximum iterations for HITS algorithm
HITS_MAX_ITER: int = 100

# PageRank damping factor (default NetworkX value)
PAGERANK_ALPHA: float = 0.85

EVALUATION_K_VALUES: list = [1, 2, 10, 50, 100, 500]

# Whether to run evaluation metrics during the pipeline
RUN_EVALUATION: bool = True

# Whether to run Leiden community detection
RUN_LEIDEN: bool = True

# Leiden resolution parameter. Higher values = smaller, granular communities. Lower values = larger, coarser communities
LEIDEN_RESOLUTION: float = 1.0

# ============================================================================
# RANK STABILITY ANALYSIS SETTINGS
# ============================================================================
# Rank stability analysis detects anomalies when a node's role shifts
# drastically between temporal snapshots. This is validated for financial
# crime detection as sudden changes in centrality metrics can indicate
# suspicious activity.

# Whether to compute rank stability features between consecutive windows
RUN_RANK_STABILITY: bool = True

# Number of top nodes to consider for stability analysis (top-k comparison)
RANK_STABILITY_TOP_K: int = 100

# Percentile threshold for anomaly detection (95 = flag nodes with rank changes
# in the top 5% most extreme changes)
RANK_ANOMALY_PERCENTILE: float = 95.0

# ============================================================================
# TEMPORAL EVALUATION SETTINGS
# ============================================================================
# Use time-aware bad actors to prevent future information leakage.
# When True, evaluation at time t only uses bad actors known up to time t.
# When False, uses global bad actors (all-time) - NOT recommended for proper
# temporal evaluation, but useful for quick testing.
USE_TIME_AWARE_BAD_ACTORS: bool = True