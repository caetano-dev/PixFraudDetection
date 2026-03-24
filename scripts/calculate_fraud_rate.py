#!/usr/bin/env python3
"""
Calculate fraud rates for accounts in the Pix Fraud Detection dataset.

This script analyzes transactions to identify fraudulent activity at the account level,
aggregating by Entity ID (which can have multiple accounts).
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

import duckdb
import pandas as pd

# Add project root to path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def calculate_fraud_rates(dataset: str = "HI_Small", output_csv: str | None = None) -> pd.DataFrame:
    """
    Calculate fraud rates by account and entity.
    
    Args:
        dataset: Name of dataset folder (e.g., 'HI_Small', 'HI_Medium', 'LI_Small')
        output_csv: Optional path to save results as CSV
        
    Returns:
        DataFrame with fraud rate statistics
    """
    
    data_path = _PROJECT_ROOT / "data" / dataset
    transactions_path = data_path / "1_filtered_transactions.parquet"
    accounts_path = data_path / "2_filtered_accounts.parquet"
    
    if not transactions_path.exists():
        raise FileNotFoundError(f"Transactions file not found: {transactions_path}")
    if not accounts_path.exists():
        raise FileNotFoundError(f"Accounts file not found: {accounts_path}")
    
    print(f"Analyzing fraud rates for dataset: {dataset}")
    print(f"Transactions: {transactions_path}")
    print(f"Accounts: {accounts_path}")
    print()
    
    # Calculate overall fraud rate at transaction level
    print("=" * 80)
    print("TRANSACTION-LEVEL FRAUD RATE")
    print("=" * 80)
    
    tx_query = f"""
    SELECT 
        COUNT(*) as total_transactions,
        SUM(CAST(is_laundering AS INTEGER)) as fraudulent_transactions,
        ROUND(100.0 * SUM(CAST(is_laundering AS INTEGER)) / COUNT(*), 4) as fraud_rate_pct
    FROM read_parquet('{transactions_path}')
    """
    
    tx_stats = duckdb.query(tx_query).df()
    print(tx_stats.to_string(index=False))
    print()
    
    # Calculate fraud rate by account
    print("=" * 80)
    print("ACCOUNT-LEVEL FRAUD RATE")
    print("=" * 80)
    
    account_query = f"""
    WITH AccountTransactions AS (
        SELECT 
            from_account as account_number,
            CAST(is_laundering AS INTEGER) as is_fraud
        FROM read_parquet('{transactions_path}')
        UNION ALL
        SELECT 
            to_account as account_number,
            CAST(is_laundering AS INTEGER) as is_fraud
        FROM read_parquet('{transactions_path}')
    ),
    AccountFraudStatus AS (
        SELECT 
            account_number,
            COUNT(*) as total_txns,
            SUM(is_fraud) as fraud_txns,
            MAX(is_fraud) as is_fraudulent_account
        FROM AccountTransactions
        GROUP BY account_number
    )
    SELECT 
        COUNT(*) as total_accounts,
        SUM(is_fraudulent_account) as fraudulent_accounts,
        ROUND(100.0 * SUM(is_fraudulent_account) / COUNT(*), 4) as fraud_rate_pct
    FROM AccountFraudStatus
    """
    
    acct_stats = duckdb.query(account_query).df()
    print(acct_stats.to_string(index=False))
    print()
    
    # Calculate fraud rate by entity
    print("=" * 80)
    print("ENTITY-LEVEL FRAUD RATE")
    print("=" * 80)
    
    entity_query = f"""
    WITH AccountMap AS (
        SELECT 
            "Account Number" as account_number,
            "Entity ID" as entity_id,
            "Entity Name" as entity_name
        FROM read_parquet('{accounts_path}')
    ),
    EntityTransactions AS (
        SELECT 
            am.entity_id,
            am.entity_name,
            t.is_laundering
        FROM read_parquet('{transactions_path}') t
        JOIN AccountMap am ON t.from_account = am.account_number
        UNION ALL
        SELECT 
            am.entity_id,
            am.entity_name,
            t.is_laundering
        FROM read_parquet('{transactions_path}') t
        JOIN AccountMap am ON t.to_account = am.account_number
    ),
    EntityFraudStatus AS (
        SELECT 
            entity_id,
            ANY_VALUE(entity_name) as entity_name,
            COUNT(*) as total_txns,
            SUM(CAST(is_laundering AS INTEGER)) as fraud_txns,
            MAX(CAST(is_laundering AS INTEGER)) as is_fraudulent_entity
        FROM EntityTransactions
        GROUP BY entity_id
    )
    SELECT 
        COUNT(*) as total_entities,
        SUM(is_fraudulent_entity) as fraudulent_entities,
        ROUND(100.0 * SUM(is_fraudulent_entity) / COUNT(*), 4) as fraud_rate_pct
    FROM EntityFraudStatus
    """
    
    entity_stats = duckdb.query(entity_query).df()
    print(entity_stats.to_string(index=False))
    print()
    
    # Get detailed entity-level breakdown
    print("=" * 80)
    print("DETAILED ENTITY FRAUD STATISTICS")
    print("=" * 80)
    
    entity_detail_query = f"""
    WITH AccountMap AS (
        SELECT 
            "Account Number" as account_number,
            "Entity ID" as entity_id,
            "Entity Name" as entity_name
        FROM read_parquet('{accounts_path}')
    ),
    EntityTransactions AS (
        SELECT 
            am.entity_id,
            am.entity_name,
            t.is_laundering
        FROM read_parquet('{transactions_path}') t
        JOIN AccountMap am ON t.from_account = am.account_number
        UNION ALL
        SELECT 
            am.entity_id,
            am.entity_name,
            t.is_laundering
        FROM read_parquet('{transactions_path}') t
        JOIN AccountMap am ON t.to_account = am.account_number
    ),
    EntityFraudStatus AS (
        SELECT 
            entity_id,
            ANY_VALUE(entity_name) as entity_name,
            COUNT(*) as total_txns,
            SUM(CAST(is_laundering AS INTEGER)) as fraud_txns,
            ROUND(100.0 * SUM(CAST(is_laundering AS INTEGER)) / COUNT(*), 2) as fraud_txn_pct,
            MAX(CAST(is_laundering AS INTEGER)) as is_fraudulent
        FROM EntityTransactions
        GROUP BY entity_id
    )
    SELECT 
        entity_id,
        entity_name,
        total_txns,
        fraud_txns,
        fraud_txn_pct,
        is_fraudulent
    FROM EntityFraudStatus
    WHERE is_fraudulent = 1
    ORDER BY fraud_txns DESC, total_txns DESC
    """
    
    entity_details = duckdb.query(entity_detail_query).df()
    print(f"Top 20 fraudulent entities (sorted by fraud transaction count):")
    print(entity_details.head(20).to_string(index=False))
    print()
    print(f"Total fraudulent entities: {len(entity_details)}")
    print()
    
    # Combine all results
    results = {
        "dataset": dataset,
        "transaction_stats": tx_stats,
        "account_stats": acct_stats,
        "entity_stats": entity_stats,
        "fraudulent_entities": entity_details
    }
    
    # Save to CSV if requested
    if output_csv:
        output_path = Path(output_csv)
        entity_details.to_csv(output_path, index=False)
        print(f"Detailed entity fraud data saved to: {output_path}")
        
        # Also save summary statistics
        summary_path = output_path.parent / f"{output_path.stem}_summary.csv"
        summary_df = pd.DataFrame({
            "metric": ["Transactions", "Accounts", "Entities"],
            "total": [
                tx_stats["total_transactions"].iloc[0],
                acct_stats["total_accounts"].iloc[0],
                entity_stats["total_entities"].iloc[0]
            ],
            "fraudulent": [
                tx_stats["fraudulent_transactions"].iloc[0],
                acct_stats["fraudulent_accounts"].iloc[0],
                entity_stats["fraudulent_entities"].iloc[0]
            ],
            "fraud_rate_pct": [
                tx_stats["fraud_rate_pct"].iloc[0],
                acct_stats["fraud_rate_pct"].iloc[0],
                entity_stats["fraud_rate_pct"].iloc[0]
            ]
        })
        summary_df.to_csv(summary_path, index=False)
        print(f"Summary statistics saved to: {summary_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Calculate fraud rates for accounts in Pix Fraud Detection dataset"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="HI_Small",
        help="Dataset folder name (default: HI_Small). Options: HI_Small, HI_Medium, HI_Large, LI_Small, LI_Medium, LI_Large"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV file path for detailed entity fraud data (optional)"
    )
    
    args = parser.parse_args()
    
    try:
        calculate_fraud_rates(dataset=args.dataset, output_csv=args.output)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
