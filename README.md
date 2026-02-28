# PIX Fraud Detector

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
```

## Data Preparation

Put the HI_Large.csv and LI_Small.csv files in the `data/HI_Large` and `data/HI_Small` directories.

## Running the Pipeline

```
python3 scripts/01_filter_raw_data.py
python3 scripts/02_extract_features.py
python3 scripts/01_train_mode.py
```