from pathlib import Path
# Set this to "SMALL" or "LARGE" to switch datasets
DATASET_SIZE: str = "SMALL"
# Window mode: "SLIDING" or "CUMULATIVE"
#   - SLIDING: Window is (current_date - WINDOW_DAYS, current_date]
#              Each window has fixed size, older transactions are dropped
#   - CUMULATIVE: Window is [start_date, current_date]
#                 Graph grows over time, preserving full network structure
#                 Better for capturing long laundering chains
WINDOW_MODE: str = "CUMULATIVE"

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
# Note: WINDOW_DAYS is only used when WINDOW_MODE = "SLIDING"
WINDOW_DAYS: int = _DATASET_CONFIG[DATASET_SIZE]["window_days"]

# Step size for sliding window (in days)
STEP_SIZE: int = 1

# Input file names (within DATA_PATH)
NORMAL_TRANSACTIONS_FILE: str = "1_filtered_normal_transactions.parquet"
LAUNDERING_TRANSACTIONS_FILE: str = "2_filtered_laundering_transactions.parquet"
ACCOUNTS_FILE: str = "3_filtered_accounts.parquet"

# Output file names
OUTPUT_FEATURES_FILE: str = "sliding_window_features.parquet"
OUTPUT_METRICS_FILE: str = "sliding_window_metrics.parquet"

# Maximum iterations for HITS algorithm
HITS_MAX_ITER: int = 100

# PageRank damping factor (default NetworkX value)
PAGERANK_ALPHA: float = 0.85

# K values for Precision@K, Recall@K, and Lift@K metrics
EVALUATION_K_VALUES: list = [10, 50, 100, 500, 1000, 5000]

# Whether to run evaluation metrics during the pipeline
RUN_EVALUATION: bool = True

# Whether to print detailed evaluation reports
VERBOSE_EVALUATION: bool = False