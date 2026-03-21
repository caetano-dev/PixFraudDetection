from pathlib import Path
DATASET_SIZE: str = "SMALL"

_DATASET_CONFIG = {
    "SMALL": {
        "data_path": Path("data/HI_Small"),
        "window_size": 3,                # Sliding window size in days
        "window_stride": 1,              # Sliding window stride in days (non-overlapping)
        "pr_alpha_deep": 0.85,           # 6.6 hops expected walk
        "pr_alpha_shallow": 0.75,        # 4 hops expected walk
        "pr_max_iter": 100,
        "betweenness_k": 80,             # Monte Carlo pivot nodes
        "hits_max_iter": 500,
        "leiden_resolution_macro": 1.0,  # Coarser communities
        "leiden_resolution_micro": 2.0,  # Finer communities
    },
    "MEDIUM": {
        "data_path": Path("data/LI_Medium"),
        "window_size": 7,                # Sliding window size in days
        "window_stride": 1,              # Sliding window stride in days (non-overlapping)
        "pr_alpha_deep": 0.95,           # 20 hops expected walk (required for 14-hop cycles)
        "pr_alpha_shallow": 0.75,        # 4 hops expected walk
        "pr_max_iter": 1000,             # Required for alpha=0.95 convergence
        "betweenness_k": 500,            # Scaled up for larger graph
        "hits_max_iter": 1000,
        "leiden_resolution_macro": 2.0,  # Coarser communities (higher baseline for large graph)
        "leiden_resolution_micro": 5.0,  # Finer communities
    },
    "LARGE": {
        "data_path": Path("data/LI_Large"),
        "window_size": 7,                # Sliding window size in days
        "window_stride": 1,              # Sliding window stride in days (non-overlapping)
        "pr_alpha_deep": 0.95,           # 20 hops expected walk (required for 14-hop cycles)
        "pr_alpha_shallow": 0.75,        # 4 hops expected walk
        "pr_max_iter": 1000,             # Required for alpha=0.95 convergence
        "betweenness_k": 500,            # Scaled up for larger graph
        "hits_max_iter": 1000,
        "leiden_resolution_macro": 2.0,  # Coarser communities (higher baseline for large graph)
        "leiden_resolution_micro": 5.0,  # Finer communities
    },
}

PR_ALPHA_DEEP: float = _DATASET_CONFIG[DATASET_SIZE]["pr_alpha_deep"]
PR_ALPHA_SHALLOW: float = _DATASET_CONFIG[DATASET_SIZE]["pr_alpha_shallow"]
PR_MAX_ITER: int = _DATASET_CONFIG[DATASET_SIZE]["pr_max_iter"]
BETWEENNESS_K: int = _DATASET_CONFIG[DATASET_SIZE]["betweenness_k"]
DATA_PATH: Path = _DATASET_CONFIG[DATASET_SIZE]["data_path"]
WINDOW_SIZE: int = _DATASET_CONFIG[DATASET_SIZE]["window_size"]
WINDOW_STRIDE: int = _DATASET_CONFIG[DATASET_SIZE]["window_stride"]
OUTPUT_FEATURES_FILE: str = "sliding_window_features.parquet"
OUTPUT_METRICS_FILE: str = "sliding_window_metrics.parquet"
HITS_MAX_ITER: int = _DATASET_CONFIG[DATASET_SIZE]["hits_max_iter"]
EVALUATION_K_VALUES: list = [1, 2, 10, 50, 100, 500]
RUN_EVALUATION: bool = True
RUN_LEIDEN: bool = True
LEIDEN_RESOLUTION_MACRO: float = _DATASET_CONFIG[DATASET_SIZE]["leiden_resolution_macro"]
LEIDEN_RESOLUTION_MICRO: float = _DATASET_CONFIG[DATASET_SIZE]["leiden_resolution_micro"]
