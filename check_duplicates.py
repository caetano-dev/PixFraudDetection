import pandas as pd
from pathlib import Path

def audit_account_duplicates(filepath: str | Path):
    """
    Empirically tests account mapping variance to determine safe graph construction methods.
    """
    file_path = Path(filepath)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    print(f"Auditing data: {file_path.name}")
    df = pd.read_parquet(file_path)

    total_rows = len(df)
    unique_accounts = df['Account Number'].nunique()
    duplicate_rows = total_rows - unique_accounts

    print("-" * 50)
    print(f"Total Entries:   {total_rows:,}")
    print(f"Unique Accounts: {unique_accounts:,}")
    print(f"Duplicate Rows:  {duplicate_rows:,}")
    print("-" * 50)

    if duplicate_rows == 0:
        print("Verdict: No duplicates exist. 1:1 mapping is natively guaranteed.")
        return

    # Isolate accounts that appear multiple times
    duplicates = df[df.duplicated(subset=['Account Number'], keep=False)]

    # Variance test: Count unique Entity IDs per Account Number
    variance = duplicates.groupby('Account Number')['Entity ID'].nunique()
    
    exact_duplicates = (variance == 1).sum()
    joint_accounts = (variance > 1).sum()

    print("Variance Test Results:")
    print(f"Scenario A (Maps to 1 Entity):  {exact_duplicates:,} accounts")
    print(f"Scenario B (Maps to >1 Entity): {joint_accounts:,} accounts")
    print("-" * 50)

    if joint_accounts > 0:
        print("CRITICAL ALERT: SCENARIO B DETECTED.")
        print("Do NOT use drop_duplicates(). Graph topology will be corrupted.")
    else:
        print("CLEAR: SCENARIO A DETECTED.")
        print("It is mathematically safe to use drop_duplicates(subset=['Account Number']).")

if __name__ == "__main__":
    # Target the parsed accounts file for the SMALL dataset
    target_file = Path("data/HI_Small/3_filtered_accounts.parquet")
    audit_account_duplicates(target_file)
