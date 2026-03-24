# AML pipeline

A machine learning pipeline for detecting fraudulent *entities* in the AMLworld dataset. This project uses graph-based features, network analysis, and gradient boosting models to identify suspicious transaction patterns and account behaviors.

## Features

- **Data Processing**: Filters and aggregates raw PIX transaction data from the AMLworld dataset
- **Feature Engineering**: Extracts behavioral features from account transaction histories
- **Graph Analysis**: Leverages network patterns using community detection algorithms
- **Model Training**: Implements forward-chaining cross-validation to prevent data leakage
- **Model Explainability**: Provides SHAP values and ablation studies to understand feature importance
- **Support for Multiple Datasets**: Compatible with AMLworld datasets of varying sizes (HI_Small, HI_Large, LI_Small, LI_Large)

## Quick Start

### Setup

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

Adjust the configuration in the src/config.py file to select your dataset.
```
python3 scripts/00_generate_convertion_rates.py
python3 scripts/01_filter_raw_data.py --dataset HI_Small # or HI_Large, LI_Small, LI_Large
python3 -m scripts.02_run_aggregation
python3 -m scripts.03_extract_features
python3 -m scripts.04_train_model_forward_chaining
python3 -m scripts.05_ablation_study_and_shap
python3 -m scripts.07_summary
```

## Pipeline Overview

1. **Conversion Rates**: Generate exchange rate data for transaction conversion
2. **Data Filtering**: Filter and prepare raw transaction data for the selected dataset
3. **Aggregation**: Aggregate transaction features at the account level
4. **Feature Extraction**: Engineer behavioral and network-based features
5. **Model Training**: Train and evaluate models using temporal forward-chaining validation
6. **Interpretability**: Analyze model decisions with SHAP and ablation studies
7. **Summary**: Generate results and evaluation reports
