import pandas as pd
import numpy as np
import argparse
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq

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
    Process raw AMLworld CSVs via memory-safe disk streaming:
    1. Read data in chunks of 1 million rows to prevent OOM errors.
    2. Enforce explicit datetime formatting for extreme parsing speed.
    3. Filter out 'post-simulation' dates where artifacts occur.
    4. Stream directly to Parquet files on disk.
    """
    data_path = Path(base_dir) / dataset_name
    raw_tx_file = data_path / "transactions.csv"
    raw_acct_file = data_path / "accounts.csv"

    if not raw_tx_file.exists():
        print(f"Error: {raw_tx_file} not found.")
        return

    cutoff_date_str = CUTOFF_CONFIG.get(dataset_name)
    cutoff_date = pd.to_datetime(cutoff_date_str) if cutoff_date_str else None

    output_normal = data_path / "1_filtered_normal_transactions.parquet"
    output_laundering = data_path / "2_filtered_laundering_transactions.parquet"

    print(f"Processing {dataset_name} in chunks to prevent OOM errors...")

    # Initialize PyArrow writers for disk streaming
    normal_writer = None
    laundering_writer = None
    
    total_rows = 0
    dropped_count = 0

    # 1. THE CHUNKSIZE FIX: Read 1M rows into memory at a time
    chunk_iter = pd.read_csv(raw_tx_file, chunksize=1_000_000, low_memory=False)

    for i, chunk in enumerate(chunk_iter):
        initial_chunk_size = len(chunk)
        total_rows += initial_chunk_size

        # Clean headers
        chunk = chunk.iloc[:, :11]
        chunk.columns = CLEAN_HEADER

        # 2. THE DATETIME FIX: Explicit format avoids row-by-row inference
        # ISO8601 safely covers standard AMLSim output formats (e.g., '2022-09-01T00:00:00Z')
        chunk['timestamp'] = pd.to_datetime(chunk['timestamp'], format='ISO8601')

        # 3. Date Filtering
        if cutoff_date:
            chunk = chunk[chunk['timestamp'] <= cutoff_date]
            dropped_count += (initial_chunk_size - len(chunk))

        # Split normal and laundering
        normal_chunk = chunk[chunk['is_laundering'] == 0]
        laundering_chunk = chunk[chunk['is_laundering'] == 1]

        # 4. Stream directly to Parquet on disk (Memory clears after this step)
        if not normal_chunk.empty:
            table_n = pa.Table.from_pandas(normal_chunk)
            if normal_writer is None:
                normal_writer = pq.ParquetWriter(output_normal, table_n.schema)
            normal_writer.write_table(table_n)

        if not laundering_chunk.empty:
            table_l = pa.Table.from_pandas(laundering_chunk)
            if laundering_writer is None:
                laundering_writer = pq.ParquetWriter(output_laundering, table_l.schema)
            laundering_writer.write_table(table_l)
            
        print(f"  - Processed chunk {i+1} ({(i+1)*1_000_000:,} rows read so far)")

    # Close the disk streams
    if normal_writer: normal_writer.close()
    if laundering_writer: laundering_writer.close()

    if cutoff_date:
        print(f"  - Dropped {dropped_count:,} post-simulation transactions (artifacts).")
    
    print(f"  - Saved to {output_normal}")
    print(f"  - Saved to {output_laundering}")

    # Accounts file is small enough (usually < 2 million rows) to process safely in memory
    if raw_acct_file.exists():
        print(f"  - Processing accounts...")
        accts_df = pd.read_csv(raw_acct_file)
        output_accts = data_path / "3_filtered_accounts.parquet"
        accts_df.to_parquet(output_accts, index=False)
    
    print(f"\n[Success] Dataset {dataset_name} cleaned and saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="HI_Small", help="Dataset folder name")
    args = parser.parse_args()
    
    process_dataset(args.dataset)
