**Object-Oriented Pipeline Architecture** using the **Strategy Design Pattern** for feature extraction.

### 1. The Directory Structure

Move away from flat files in a single folder. Separate your concerns into a standard Python package structure:

```text
pix_fraud_detection/
│
├── data/                       # Ignored in git, holds raw/parquet files
├── notebooks/                  # Jupyter notebooks for EDA and SHAP plots
│
├── src/                        # The core package
│   ├── __init__.py
│   ├── config.py               # Centralized dataclass for configurations
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py           # Loads normal/laundering parquets
│   │   └── window_generator.py # Yields daily temporal slices cleanly
│   │
│   ├── features/               # THE STRATEGY PATTERN (Decoupled algorithms)
│   │   ├── __init__.py
│   │   ├── base.py             # Abstract Base Class for Extractors
│   │   ├── centrality.py       # PageRank, HITS classes
│   │   ├── community.py        # Leiden class
│   │   └── stability.py        # Stateful class to track Rank Anomalies
│   │
│   ├── graph/
│   │   ├── __init__.py
│   │   └── builder.py          # Pure function: DataFrame -> nx.DiGraph
│   │
│   └── models/
│       ├── __init__.py
│       ├── train.py            # XGBoost training and temporal splitting
│       └── evaluate.py         # SHAP value extraction and Precision@K
│
├── scripts/
│   ├── 01_preprocess.py        # Your new_filter_raw_data.py
│   ├── 02_extract_features.py  # Replaces your current main.py
│   └── 03_train_model.py       # Executes the ML pipeline
│
├── requirements.txt
└── SPEC.md

```

### 2. The Paradigm Shift: Generators and Strategies

#### A. The Window Generator (Data Layer)

Instead of managing a messy `while current_date <= end_date:` loop inside your main script, abstract this into a Python Generator. It should simply yield the dataframe for the current window.

```python
# src/data/window_generator.py
class TemporalWindowGenerator:
    def __init__(self, df, window_days, step_size):
        self.df = df
        self.window_days = window_days
        self.step_size = step_size
        
    def __iter__(self):
        # Logic to slice self.df by dates
        # yield current_date, window_df

```

#### B. The Feature Extractors (Strategy Pattern)

This is the most critical change. Right now, `nx.pagerank` and `algorithms.leiden` are hardcoded in your main loop.
You need an Abstract Base Class (ABC). Every algorithm becomes its own isolated class with a `.extract(graph)` method.

```python
# src/features/base.py
from abc import ABC, abstractmethod

class FeatureExtractor(ABC):
    @abstractmethod
    def extract(self, G: nx.DiGraph) -> dict:
        """Takes a graph, returns a dict mapping node_id -> feature_value"""
        pass

# src/features/centrality.py
class PageRankVolumeExtractor(FeatureExtractor):
    def __init__(self, alpha=0.85):
        self.alpha = alpha

    def extract(self, G):
        try:
            return nx.pagerank(G, weight='weight', alpha=self.alpha)
        except:
            return {}

```

#### C. The Stateful Tracker (Rank Stability)

Rank stability currently relies on a floating `prev_pagerank_scores` variable at the bottom of your loop. This is dangerous. Create a dedicated class that holds its own state.

```python
# src/features/stability.py
class RankStabilityTracker:
    def __init__(self):
        self.previous_scores = {}
        
    def update_and_calculate_derivative(self, current_scores):
        # Calculate Delta
        # Update self.previous_scores = current_scores
        # Return Delta

```

### 3. The New Orchestrator (`02_extract_features.py`)

If you build the architecture above, your massive 400-line `main.py` shrinks into a beautiful, highly readable, declarative pipeline.

It will look exactly like this:

```python
from src.data.loader import load_data
from src.data.window_generator import TemporalWindowGenerator
from src.graph.builder import build_directed_graph
from src.features.centrality import PageRankVolume, PageRankFrequency, HITSExtractor
from src.features.community import LeidenExtractor
from src.features.stability import RankStabilityTracker

def main():
    # 1. Load Data
    df = load_data('data/HI_Small/')
    
    # 2. Initialize Extractors (Easy to add/remove without breaking code)
    extractors = {
        'pr_vol': PageRankVolume(alpha=0.85),
        'pr_count': PageRankFrequency(alpha=0.85),
        'hits': HITSExtractor(max_iter=100),
        'leiden': LeidenExtractor(resolution=2.0)
    }
    stability_tracker = RankStabilityTracker()
    
    all_features = []

    # 3. The Pipeline
    for target_date, window_df in TemporalWindowGenerator(df, window_days=3):
        
        G = build_directed_graph(window_df)
        
        # Run all algorithms dynamically
        daily_metrics = {name: ext.extract(G) for name, ext in extractors.items()}
        
        # Track stability based on PageRank Volume
        rank_delta = stability_tracker.update_and_calculate(daily_metrics['pr_vol'])
        
        # Flatten and append to all_features list...
        
    # 4. Save to Parquet
    save_features(all_features, 'sliding_window_features.parquet')

if __name__ == "__main__":
    main()

```