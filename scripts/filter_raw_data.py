# extracted from COLAB!!!!!
import os
import re
import duckdb
import pandas as pd
from google.colab import drive

drive.mount('/content/drive')
DRIVE_DIR = '/content/drive/MyDrive/AML'

#DATASET = 'HI_Small'
DATASET = 'HI_Large'
#DATASET = 'LI_Small'
# DATASET = 'LI_Large'

TX_CSV = os.path.join(DRIVE_DIR, f'{DATASET.replace("_", "-")}_Trans.csv')
PATTERNS_TXT = os.path.join(DRIVE_DIR, f'{DATASET.replace("_", "-")}_Patterns.txt')
ACCOUNTS_CSV = os.path.join(DRIVE_DIR, f'{DATASET.replace("_", "-")}_Accounts.csv')
PROCESSED_DIR = os.path.join(DRIVE_DIR, 'processed', DATASET)

os.makedirs(PROCESSED_DIR, exist_ok=True)

def parse_patterns(file_path):
    attempts, current = [], None
    attempt_id = 0

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('BEGIN LAUNDERING ATTEMPT'):
                attempt_id += 1
                attempt_type = re.search(r'BEGIN LAUNDERING ATTEMPT\s*-\s*(.+)$', line)
                current = {'id': attempt_id, 'type': attempt_type.group(1).strip() if attempt_type else 'UNKNOWN', 'txs': []}
            elif line.startswith('END LAUNDERING ATTEMPT'):
                if current:
                    attempts.append(current)
                current = None
            elif current and line:
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 11:
                    current['txs'].append(parts[:11] + [current['id'], current['type']])

    cols = ['timestamp', 'from_bank', 'from_account', 'to_bank', 'to_account',
            'amount_received', 'currency_received', 'amount_sent', 'currency_sent',
            'payment_type', 'is_laundering', 'attempt_id', 'attempt_type']

    return pd.DataFrame([tx for a in attempts for tx in a['txs']], columns=cols)

con = duckdb.connect(':memory:')
con.execute("PRAGMA threads=8")

patterns_df = parse_patterns(PATTERNS_TXT)
con.register('patterns', patterns_df)

CURRENCIES = ["US Dollar", "Euro", "Yuan", "Shekel", "Canadian Dollar", "UK Pound",
              "Ruble", "Australian Dollar", "Swiss Franc", "Yen", "Mexican Peso",
              "Rupee", "Brazil Real", "Saudi Riyal"]

PARSE_TS = """
    CASE
        WHEN length(timestamp) = 16 THEN strptime(timestamp, '%Y/%m/%d %H:%M')
        WHEN length(timestamp) = 19 THEN strptime(timestamp, '%Y/%m/%d %H:%M:%S')
        ELSE NULL
    END::TIMESTAMP
"""

for currency in CURRENCIES:
    out_dir = os.path.join(PROCESSED_DIR, currency.replace(' ', '_'))
    os.makedirs(out_dir, exist_ok=True)

    normal_file = os.path.join(out_dir, '1_filtered_normal_transactions.parquet')
    laundering_file = os.path.join(out_dir, '2_filtered_laundering_transactions.parquet')
    accounts_file = os.path.join(out_dir, '3_filtered_accounts.parquet')

    filter_clause = f"""
        UPPER(currency_sent) = UPPER('{currency}')
        AND UPPER(currency_received) = UPPER('{currency}')
        AND UPPER(payment_type) = 'ACH'
    """

    # Step 1: Normal transactions
    con.execute(f"""
        COPY (
            WITH raw AS (
                SELECT
                    "Timestamp" AS timestamp,
                    "From Bank" AS from_bank,
                    "Account" AS from_account,
                    "To Bank" AS to_bank,
                    "Account_1" AS to_account,
                    "Amount Received" AS amount_received,
                    "Receiving Currency" AS currency_received,
                    "Amount Paid" AS amount_sent,
                    "Payment Currency" AS currency_sent,
                    "Payment Format" AS payment_type,
                    "Is Laundering" AS is_laundering
                FROM read_csv('{TX_CSV}', header=true, all_varchar=true)
            )
            SELECT
                {PARSE_TS} AS timestamp,
                from_bank, from_account, to_bank, to_account,
                CAST(ROUND(TRY_CAST(amount_sent AS DOUBLE) * 100) AS BIGINT) AS amount_sent_c,
                currency_sent,
                CAST(ROUND(TRY_CAST(amount_received AS DOUBLE) * 100) AS BIGINT) AS amount_received_c,
                currency_received,
                payment_type,
                CAST(is_laundering AS INTEGER) AS is_laundering,
                (from_bank = to_bank) AS same_bank
            FROM raw
            WHERE {filter_clause} AND CAST(is_laundering AS INTEGER) = 0
        ) TO '{normal_file}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)

    # Step 2: Laundering transactions
    con.execute(f"""
        COPY (
            WITH
            patterns_typed AS (
                SELECT
                    {PARSE_TS} AS timestamp,
                    from_bank, from_account, to_bank, to_account,
                    TRY_CAST(amount_sent AS DOUBLE) AS amount_sent,
                    currency_sent,
                    TRY_CAST(amount_received AS DOUBLE) AS amount_received,
                    currency_received,
                    payment_type,
                    CAST(is_laundering AS INTEGER) AS is_laundering,
                    CAST(attempt_id AS BIGINT) AS attempt_id,
                    attempt_type
                FROM patterns
                WHERE {filter_clause} AND CAST(is_laundering AS INTEGER) = 1
            ),
            raw_csv AS (
                SELECT
                    "Timestamp" AS timestamp,
                    "From Bank" AS from_bank,
                    "Account" AS from_account,
                    "To Bank" AS to_bank,
                    "Account_1" AS to_account,
                    "Amount Received" AS amount_received,
                    "Receiving Currency" AS currency_received,
                    "Amount Paid" AS amount_sent,
                    "Payment Currency" AS currency_sent,
                    "Payment Format" AS payment_type,
                    "Is Laundering" AS is_laundering
                FROM read_csv('{TX_CSV}', header=true, all_varchar=true)
            ),
            csv_laundering AS (
                SELECT
                    {PARSE_TS} AS timestamp,
                    from_bank, from_account, to_bank, to_account,
                    TRY_CAST(amount_sent AS DOUBLE) AS amount_sent,
                    currency_sent,
                    TRY_CAST(amount_received AS DOUBLE) AS amount_received,
                    currency_received,
                    payment_type,
                    CAST(is_laundering AS INTEGER) AS is_laundering
                FROM raw_csv
                WHERE {filter_clause} AND CAST(is_laundering AS INTEGER) = 1
            ),
            missing AS (
                SELECT csv.*
                FROM csv_laundering csv
                LEFT JOIN patterns_typed p
                    ON csv.timestamp = p.timestamp
                    AND csv.from_account = p.from_account
                    AND csv.to_account = p.to_account
                    AND CAST(ROUND(csv.amount_sent * 100) AS BIGINT) = CAST(ROUND(p.amount_sent * 100) AS BIGINT)
                WHERE p.timestamp IS NULL
            )
            SELECT
                timestamp, from_bank, from_account, to_bank, to_account,
                CAST(ROUND(amount_sent * 100) AS BIGINT) AS amount_sent_c,
                currency_sent,
                CAST(ROUND(amount_received * 100) AS BIGINT) AS amount_received_c,
                currency_received,
                payment_type, is_laundering,
                attempt_id, attempt_type,
                (from_bank = to_bank) AS same_bank
            FROM patterns_typed
            UNION ALL
            SELECT
                timestamp, from_bank, from_account, to_bank, to_account,
                CAST(ROUND(amount_sent * 100) AS BIGINT) AS amount_sent_c,
                currency_sent,
                CAST(ROUND(amount_received * 100) AS BIGINT) AS amount_received_c,
                currency_received,
                payment_type, is_laundering,
                NULL AS attempt_id, 'UNLISTED' AS attempt_type,
                (from_bank = to_bank) AS same_bank
            FROM missing
            ORDER BY timestamp
        ) TO '{laundering_file}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)

    # Step 3: Accounts
    con.execute(f"""
        COPY (
            WITH all_accounts AS (
                SELECT DISTINCT from_bank AS bank, from_account AS account
                FROM read_parquet('{normal_file}')
                UNION
                SELECT DISTINCT to_bank AS bank, to_account AS account
                FROM read_parquet('{normal_file}')
                UNION
                SELECT DISTINCT from_bank AS bank, from_account AS account
                FROM read_parquet('{laundering_file}')
                UNION
                SELECT DISTINCT to_bank AS bank, to_account AS account
                FROM read_parquet('{laundering_file}')
            )
            SELECT DISTINCT a.*
            FROM read_csv('{ACCOUNTS_CSV}', header=true) a
            INNER JOIN all_accounts acc
                ON a."Bank ID" = acc.bank
                AND a."Account Number" = acc.account
        ) TO '{accounts_file}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)
    print(f"✓ {currency}: {con.execute(f'SELECT COUNT(*) FROM read_parquet({normal_file!r})').fetchone()[0]:,} normal, "
          f"{con.execute(f'SELECT COUNT(*) FROM read_parquet({laundering_file!r})').fetchone()[0]:,} laundering, "
          f"{con.execute(f'SELECT COUNT(*) FROM read_parquet({accounts_file!r})').fetchone()[0]:,} accounts")

con.close()