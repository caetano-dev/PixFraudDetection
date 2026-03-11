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
        "pr_alpha_deep": 0.85,       # 6.6 hops expected walk
        "pr_alpha_shallow": 0.75,    # 4 hops expected walk
        "pr_max_iter": 100,
        "betweenness_k": 80,          # Monte Carlo pivot nodes
        "hits_max_iter": 500         # Add this
    },
    "LARGE": {
        "data_path": Path("data/HI_Large"),
        "window_days": 7,
        "pr_alpha_deep": 0.95,       # 20 hops expected walk (required for 14-hop cycles)
        "pr_alpha_shallow": 0.75,    # 4 hops expected walk
        "pr_max_iter": 1000,         # Required for alpha=0.95 convergence
        "betweenness_k": 500,        # Scaled up for larger graph
        "hits_max_iter": 1000        # Add this
    },
}

# Expose these variables at the module level
PR_ALPHA_DEEP: float = _DATASET_CONFIG[DATASET_SIZE]["pr_alpha_deep"]
PR_ALPHA_SHALLOW: float = _DATASET_CONFIG[DATASET_SIZE]["pr_alpha_shallow"]
PR_MAX_ITER: int = _DATASET_CONFIG[DATASET_SIZE]["pr_max_iter"]
BETWEENNESS_K: int = _DATASET_CONFIG[DATASET_SIZE]["betweenness_k"]

DATA_PATH: Path = _DATASET_CONFIG[DATASET_SIZE]["data_path"]

# ============================================================================
# SLIDING WINDOW SETTINGS
# ============================================================================
# Number of days covered by each temporal window snapshot.
WINDOW_DAYS: int = _DATASET_CONFIG[DATASET_SIZE]["window_days"]

# ============================================================================
# OUTPUT FILE NAMES
# ============================================================================
OUTPUT_FEATURES_FILE: str = "sliding_window_features.parquet"
OUTPUT_METRICS_FILE: str = "sliding_window_metrics.parquet"

# ============================================================================
# ALGORITHM HYPER-PARAMETERS
# ============================================================================
# Maximum iterations for the HITS algorithm before declaring non-convergence.
HITS_MAX_ITER: int = _DATASET_CONFIG[DATASET_SIZE]["hits_max_iter"]

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
