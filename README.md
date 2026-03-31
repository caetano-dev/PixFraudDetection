# AML pipeline

A machine learning pipeline for detecting fraudulent *entities* in the AMLworld dataset. This project uses graph-based features, network analysis, and gradient boosting models to identify suspicious transaction patterns and account behaviors.

## Features

- **Data Processing**: Filters and aggregates raw transaction data from the AMLworld dataset
- **Feature Engineering**: Extracts behavioral features from account transaction histories
- **Graph Analysis**: Leverages network patterns using community detection algorithms
- **Model Training**: Implements forward-chaining cross-validation to prevent data leakage
- **Model Explainability**: Provides SHAP values and ablation studies to understand feature importance
- **Support for Multiple Datasets**: Compatible with AMLworld datasets of varying sizes (HI_Small, HI_Large, LI_Small, LI_Large)

## Quick Start

### Setup

Download the dataset https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml/data

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Unix/macOS
# or: source venv/bin/activate.fish  # For fish shell
# or: venv\Scripts\activate  # On Windows

# Install dependencies
pip install -r requirements.txt

# Install duckdb

curl https://install.duckdb.org | sh
```

## Data Preparation

Put the source AMLworld csv files in their `data` directories. Rename them to transactions.csv and accounts.csv as needed.

```bash
# Create and activate virtual environment
mkdir data/HI_Small #change this depending on what dataset you use.
```

## Running the Pipeline

Adjust the configuration in src/config.py to select your dataset. Use the following ordered scripts to run the full pipeline. File/module names below match this repository's scripts and entrypoints.

```
# 0. Generate conversion rates (if needed)
python3 scripts/00_generate_conversion_rates.py

# 1. Filter raw data for the chosen dataset (examples: HI_Small, HI_Large, LI_Small, LI_Large)
python3 scripts/01_filter_raw_data.py --dataset HI_Small

# 2. Aggregate transactions into temporal graph windows
python3 scripts/02_run_aggregation.py   # runs scripts/02_aggregate_graph.sql via DuckDB

# 3. Extract graph & behavioral features (writes features parquet chunks -> merged output)
python3 scripts/03_extract_features.py

# 4. Train models with forward-chaining validation and produce predictions and SHAP
python3 scripts/05_train_models.py

# 5. Ablation study comparing Baseline vs Full model (uses model outputs)
python3 scripts/06_ablation_study.py

# 6. Optional: Feature pruning and topology summaries
python3 scripts/07_feature_pruning.py
python3 scripts/08_topology_summary.py
```

Notes:
- Ensure DATA_PATH and dataset selection are correct in src/config.py before running.
- scripts/02_run_aggregation.py executes the SQL in scripts/02_aggregate_graph.sql and writes lookback_edges.parquet and target_nodes.parquet under the dataset data folder.
- scripts/03_extract_features.py produces a merged features parquet file (check src/config.py: OUTPUT_FEATURES_FILE).
- scripts/05_train_models.py expects the merged features file and will output results to data/results/.
- If you previously removed temporal motif extractors, temporal motif steps are not available unless re-added.

## Pipeline Overview

1. **Conversion Rates**: Generate exchange rate data for transaction conversion
2. **Data Filtering**: Filter and prepare raw transaction data for the selected dataset
3. **Aggregation**: Aggregate transaction features at the account level
4. **Feature Extraction**: Engineer behavioral and network-based features
5. **Model Training**: Train and evaluate models using temporal forward-chaining validation
6. **Interpretability**: Analyze model decisions with SHAP and ablation studies
7. **Summary**: Generate results and evaluation reports

```
..                                  SMALL           MEDIUM           LARGE
..                                  HI     LI        HI      LI       HI       LI
.. Date Range HI + LI (2022)         Sep 1-10         Sep 1-16        Aug 1 - Nov 5
.. # of Days Spanned                 10     10        16      16       97       97
.. # of Bank Accounts               515K   705K     2077K   2028K    2116K    2064K
.. # of Transactions                  5M     7M       32M     31M      180M    176M
.. # of Laundering Transactions     5.1K   4.0K       35K     16K      223K    100K
.. Laundering Rate (1 per N Trans)  981   1942       905    1948       807     1750
..                                  SMALL           MEDIUM           LARGE
```

Longest laundering chain per dataset:

8, 12 and 69 days
