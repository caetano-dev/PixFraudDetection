import pandas as pd
import numpy as np
import argparse
from pathlib import Path

# Config for column mapping
RAW_HEADER = [
    "Timestamp", "From Bank", "Account", "To Bank", "Account", 
    "Amount Received", "Receiving Currency", "Amount Paid", 
    "Payment Currency", "Payment Format", "Is Laundering"
]

CLEAN_HEADER = [
    "timestamp", "from_bank", "from_account", "to_bank", "to_account",
    "amount_received", "receiving_currency", "amount_sent_c", 
    "payment_currency", "payment_format", "is_laundering"
]

# Strict cutoff dates to prevent "future leakage" where only fraud exists
# Based on the AMLSim documentation
CUTOFF_CONFIG = {
    "HI_Small": "2022-09-10",
    "LI_Small": "2022-09-10",
    "HI_Medium": "2022-09-16",
    "LI_Medium": "2022-09-16",
    "HI_Large": "2022-11-05",
    "LI_Large": "2022-11-05"
}

def process_dataset(dataset_name: str, base_dir: str = "data"):
    """
    Process raw AMLworld CSVs:
    1. Load data without topology filtering (keep all types/currencies).
    2. Filter out 'post-simulation' dates where artifacts occur.
    """
    data_path = Path(base_dir) / dataset_name
    raw_tx_file = data_path / "transactions.csv"
    raw_acct_file = data_path / "accounts.csv"
    
    print(f"Processing {dataset_name}...")
    
    # 1. Determine Cutoff Date
    if dataset_name in CUTOFF_CONFIG:
        cutoff_date = pd.Timestamp(CUTOFF_CONFIG[dataset_name]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        # Sets cutoff to 2022-09-10 23:59:59
        print(f"  - Cutoff Date Enforcement: Keeping transactions <= {cutoff_date}")
    else:
        print(f"  - WARNING: No cutoff date found for '{dataset_name}'. Using full dataset (Risk of data leakage!).")
        cutoff_date = None

    if not raw_tx_file.exists():
        raise FileNotFoundError(f"Could not find {raw_tx_file}")

    # 2. Load Transactions
    # Check header length to handle duplicate 'Account' columns
    df = pd.read_csv(raw_tx_file)
    if len(df.columns) == len(CLEAN_HEADER):
        df.columns = CLEAN_HEADER
    else:
        print("  - Warning: Column count mismatch. Attempting to match by position...")
        df = df.iloc[:, :11]
        df.columns = CLEAN_HEADER

    # 3. DateTime Conversion
    print("  - Converting timestamps...")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    initial_count = len(df)
    
    # 4. CRITICAL: Date Filtering
    # Remove transactions that happen after the simulation window (artifacts)
    if cutoff_date:
        df = df[df['timestamp'] <= cutoff_date]
        dropped_count = initial_count - len(df)
        print(f"  - Dropped {dropped_count:,} post-simulation transactions (potential artifacts).")
    

    output_normal = data_path / "1_filtered_normal_transactions.parquet"
    output_laundering = data_path / "2_filtered_laundering_transactions.parquet"

    print(f"  - Saving to {output_normal}...")
    df[df['is_laundering'] == 0].to_parquet(output_normal, index=False)
    
    print(f"  - Saving to {output_laundering}...")
    df[df['is_laundering'] == 1].to_parquet(output_laundering, index=False)

    if raw_acct_file.exists():
        print(f"  - Processing accounts...")
        accts_df = pd.read_csv(raw_acct_file)
        output_accts = data_path / "3_filtered_accounts.parquet"
        accts_df.to_parquet(output_accts, index=False)
    
    print(f"\n[Success] Dataset {dataset_name} cleaned and saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="HI_Small", 
                        help="Name of the folder (e.g., HI_Small, HI_Large)")
    parser.add_argument("--dir", type=str, default="data", 
                        help="Base data directory")
    
    args = parser.parse_args()
    process_dataset(args.dataset, args.dir)
