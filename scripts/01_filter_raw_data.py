import pandas as pd
import numpy as np
import argparse
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq

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

# https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml/data
CUTOFF_CONFIG = {
    "HI_Small": "2022-09-10",
    "LI_Small": "2022-09-10",
    "HI_Medium": "2022-09-16",
    "LI_Medium": "2022-09-16",
    "HI_Large": "2022-11-05",
    "LI_Large": "2022-11-05"
}

def process_dataset(dataset_name: str, base_dir: str = "data"):
    data_path = Path(base_dir) / dataset_name
    raw_tx_file = data_path / "transactions.csv"
    raw_acct_file = data_path / "accounts.csv"

    if not raw_tx_file.exists():
        print(f"Error: {raw_tx_file} not found.")
        return

    cutoff_date_str = CUTOFF_CONFIG.get(dataset_name)
    cutoff_date = pd.to_datetime(cutoff_date_str) if cutoff_date_str else None

    output = data_path / "1_filtered_transactions.parquet"

    normal_writer = None
    
    total_rows = 0

    chunk_iter = pd.read_csv(raw_tx_file, chunksize=1_000_000, low_memory=False)

    for i, chunk in enumerate(chunk_iter):
        initial_chunk_size = len(chunk)
        total_rows += initial_chunk_size

        chunk = chunk.iloc[:, :11]
        chunk.columns = CLEAN_HEADER
        chunk['timestamp'] = pd.to_datetime(chunk['timestamp'], format='ISO8601')

        if cutoff_date:
            chunk = chunk[chunk['timestamp'] <= cutoff_date]

        if not chunk.empty:
            table_n = pa.Table.from_pandas(chunk, preserve_index=False)
            if normal_writer is None:
                normal_writer = pq.ParquetWriter(str(output), table_n.schema)
            normal_writer.write_table(table_n)

        print(f"Chunk {i}, {total_rows:,} rows read")

    if normal_writer: 
      normal_writer.close()
    
    print(f"  - Saved to {output}")

    if raw_acct_file.exists():
        print("Processing accounts...")
        accts_df = pd.read_csv(raw_acct_file)
        output_accts = data_path / "2_filtered_accounts.parquet"
        accts_df.to_parquet(output_accts, index=False)
    
    print(f"\n[Success] Dataset {dataset_name} cleaned and saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="HI_Small", help="Dataset folder name")
    args = parser.parse_args()
    process_dataset(args.dataset)
