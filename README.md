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

Put the source AMLworld csv and text files in their `data` directories. Rename them to transactions.csv and accounts.csv as needed.

## Running the Pipeline

```
python3 scripts/01_filter_raw_data.py --dataset HI_Small # or HI_Large, LI_Small, LI_Large
python3 -m scripts.02_extract_features
python3 -m scripts.03_train_mode
```