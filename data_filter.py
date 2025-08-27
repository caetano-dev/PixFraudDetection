import os
import re
import pandas as pd

DATA_DIR = 'data'
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')
os.makedirs(PROCESSED_DIR, exist_ok=True)

standard_columns = [
    'timestamp', 'from_bank', 'from_account', 'to_bank', 'to_account',
    'amount_received', 'currency_received', 'amount_sent', 'currency_sent',
    'payment_type', 'is_laundering'
]

def parse_ts(s: pd.Series) -> pd.Series:
    # Strict parser: first try YYYY/MM/DD HH:MM, then fall back to HH:MM:SS
    s = s.astype(str).str.strip()
    dt = pd.to_datetime(s, format='%Y/%m/%d %H:%M', errors='coerce')
    mask = dt.isna()
    if mask.any():
        dt2 = pd.to_datetime(s[mask], format='%Y/%m/%d %H:%M:%S', errors='coerce')
        dt.loc[mask] = dt2
    return dt

def normalize_strings(df):
    for col in ['currency_sent', 'currency_received', 'payment_type']:
        df[col] = df[col].astype(str).str.strip()
    # Uppercase for robust equality checks
    df['currency_sent_u'] = df['currency_sent'].str.upper()
    df['currency_received_u'] = df['currency_received'].str.upper()
    df['payment_type_u'] = df['payment_type'].str.upper()
    return df

def parse_patterns_file(file_path):
    attempts = []
    current_attempt = None
    attempt_counter = 0

    with open(file_path, 'r') as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith('BEGIN LAUNDERING ATTEMPT'):
                attempt_counter += 1
                # Capture everything after the hyphen, including slashes and parentheses
                m = re.search(r'BEGIN LAUNDERING ATTEMPT\s*-\s*(.+)$', line)
                attempt_type = m.group(1).strip() if m else 'UNKNOWN'
                current_attempt = {
                    'attempt_id': attempt_counter,
                    'attempt_type': attempt_type,
                    'transactions': []
                }
            elif line.startswith('END LAUNDERING ATTEMPT'):
                if current_attempt:
                    attempts.append(current_attempt)
                current_attempt = None
            elif current_attempt:
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 11:
                    tx = dict(zip(standard_columns, parts[:11]))
                    tx['attempt_id'] = current_attempt['attempt_id']
                    tx['attempt_type'] = current_attempt['attempt_type']
                    current_attempt['transactions'].append(tx)

    all_transactions = [tx for attempt in attempts for tx in attempt['transactions']]
    df = pd.DataFrame(all_transactions)

    if df.empty:
        return df

    # Dtypes and normalization
    df['timestamp'] = parse_ts(df['timestamp'])
    # Preserve leading zeros, keep IDs as strings
    for col in ['from_bank', 'to_bank', 'from_account', 'to_account']:
        df[col] = df[col].astype(str)
    # Numeric columns
    df['amount_sent'] = pd.to_numeric(df['amount_sent'], errors='coerce')
    df['amount_received'] = pd.to_numeric(df['amount_received'], errors='coerce')
    df['is_laundering'] = pd.to_numeric(df['is_laundering'], errors='coerce').fillna(0).astype('int8')
    df = normalize_strings(df)

    return df

# ------------------------
# Step 1: Filter HI-Small_Trans.csv and Save
# ------------------------
filtered_normal_df = pd.DataFrame()
try:
    transactions_path = os.path.join(DATA_DIR, 'HI-Small_Trans.csv')
    # These files are typically headerless; define names explicitly and preserve IDs as strings
    transactions_df = pd.read_csv(
        transactions_path,
        header=None,
        names=standard_columns,
        dtype={
            'from_bank': 'string',
            'to_bank': 'string',
            'from_account': 'string',
            'to_account': 'string',
            'currency_received': 'string',
            'currency_sent': 'string',
            'payment_type': 'string',
            'is_laundering': 'string'  # convert after
        },
        low_memory=False
    )

    transactions_df['timestamp'] = parse_ts(transactions_df['timestamp'])

    # Coerce numerics
    transactions_df['amount_sent'] = pd.to_numeric(transactions_df['amount_sent'], errors='coerce')
    transactions_df['amount_received'] = pd.to_numeric(transactions_df['amount_received'], errors='coerce')
    transactions_df['is_laundering'] = pd.to_numeric(transactions_df['is_laundering'], errors='coerce').fillna(0).astype('int8')

    transactions_df = normalize_strings(transactions_df)

    mask_ach_usd = (
        (transactions_df['currency_sent_u'] == 'US DOLLAR') &
        (transactions_df['currency_received_u'] == 'US DOLLAR') &
        (transactions_df['payment_type_u'] == 'ACH')
    )
    filtered_transactions_df = transactions_df[mask_ach_usd].copy()

    # Keep only normal transactions here
    filtered_normal_df = filtered_transactions_df[filtered_transactions_df['is_laundering'] == 0].copy()

    if not filtered_normal_df.empty:
        output_path_step1 = os.path.join(PROCESSED_DIR, '1_filtered_normal_transactions.csv')
        filtered_normal_df.drop(columns=['currency_sent_u', 'currency_received_u', 'payment_type_u']).to_csv(output_path_step1, index=False)
        print(f"Step 1: Saved strictly USD ACH normal transactions to '{output_path_step1}' "
              f"(rows={len(filtered_normal_df):,})")
    else:
        print("ℹ️ Step 1: No matching normal transactions found after filtering.")

except Exception as e:
    print(f"Error in Step 1: {e}")

# ------------------------
# Step 2: Parse patterns, filter, save
# ------------------------

filtered_patterns_df = pd.DataFrame()
try:
    # 2a) Parse HI_patterns.txt and filter to ACH + USD
    patterns_path = os.path.join(DATA_DIR, 'HI_patterns.txt')
    patterns_df = parse_patterns_file(patterns_path)  # uses normalize_strings inside
    if not patterns_df.empty:
        patterns_df['timestamp'] = parse_ts(patterns_df['timestamp'])
        mask_pat = (
            (patterns_df['currency_sent_u'] == 'US DOLLAR') &
            (patterns_df['currency_received_u'] == 'US DOLLAR') &
            (patterns_df['payment_type_u'] == 'ACH') &
            (patterns_df['is_laundering'] == 1)
        )
        filtered_patterns_df = patterns_df[mask_pat].copy()
    else:
        filtered_patterns_df = pd.DataFrame()

    # 2b) Read main CSV and find ACH/USD positives not present in patterns subset
    transactions_path = os.path.join(DATA_DIR, 'HI-Small_Trans.csv')
    raw = pd.read_csv(
        transactions_path,
        header=None,
        names=standard_columns,
        dtype={
            'from_bank': 'string', 'to_bank': 'string',
            'from_account': 'string', 'to_account': 'string',
            'currency_received': 'string', 'currency_sent': 'string',
            'payment_type': 'string', 'is_laundering': 'string'
        },
        low_memory=False
    )
    raw['timestamp'] = parse_ts(raw['timestamp'])
    raw['amount_sent'] = pd.to_numeric(raw['amount_sent'], errors='coerce')
    raw['amount_received'] = pd.to_numeric(raw['amount_received'], errors='coerce')
    raw['is_laundering'] = pd.to_numeric(raw['is_laundering'], errors='coerce').fillna(0).astype('int8')
    raw = normalize_strings(raw)

    mask_usd_ach = (
        (raw['currency_sent_u'] == 'US DOLLAR') &
        (raw['currency_received_u'] == 'US DOLLAR') &
        (raw['payment_type_u'] == 'ACH')
    )
    raw_pos = raw[mask_usd_ach & raw['is_laundering'].eq(1)].copy()

    # Robust join keys using integer cents
    def to_cents(s): return pd.to_numeric(s, errors='coerce').mul(100).round().astype('Int64')
    raw_pos['amount_sent_c'] = to_cents(raw_pos['amount_sent'])
    raw_pos['amount_received_c'] = to_cents(raw_pos['amount_received'])

    if not filtered_patterns_df.empty:
        filtered_patterns_df['amount_sent_c'] = to_cents(filtered_patterns_df['amount_sent'])
        filtered_patterns_df['amount_received_c'] = to_cents(filtered_patterns_df['amount_received'])
    else:
        # Ensure columns exist for the join below
        filtered_patterns_df = pd.DataFrame(columns=list(raw_pos.columns) + ['attempt_id','attempt_type'])

    merge_cols = ['timestamp','from_bank','from_account','to_bank','to_account','amount_received_c','amount_sent_c']

    # Find positives in the main CSV that aren't in the patterns subset
    missing = raw_pos.merge(filtered_patterns_df[merge_cols], on=merge_cols, how='left', indicator=True) \
                     .query("_merge=='left_only'") \
                     .drop(columns=['_merge'])

    added = 0
    if not missing.empty:
        # Recover the full rows for the missing keys
        missing_full = raw_pos.merge(missing[merge_cols], on=merge_cols, how='inner')
        missing_full['attempt_id'] = pd.NA
        missing_full['attempt_type'] = 'UNLISTED'

        # Align columns and union
        union_cols = sorted(set(filtered_patterns_df.columns).union(set(missing_full.columns)))
        for col in union_cols:
            if col not in filtered_patterns_df.columns: filtered_patterns_df[col] = pd.NA
            if col not in missing_full.columns: missing_full[col] = pd.NA

        filtered_patterns_df = pd.concat([filtered_patterns_df[union_cols], missing_full[union_cols]],
                                         ignore_index=True).drop_duplicates(subset=merge_cols)
        added = len(missing_full)

    # Drop helper columns before saving
    drop_helpers = ['currency_sent_u','currency_received_u','payment_type_u','amount_sent_c','amount_received_c']
    keep_cols = [c for c in filtered_patterns_df.columns if c not in drop_helpers]
    filtered_patterns_df = filtered_patterns_df[keep_cols]

    if not filtered_patterns_df.empty:
        output_path_step2 = os.path.join(PROCESSED_DIR, '2_filtered_laundering_transactions.csv')
        filtered_patterns_df.to_csv(output_path_step2, index=False)
        base = len(filtered_patterns_df) - added
        print(f"Step 2: Saved USD/ACH laundering transactions (patterns={base:,}, added_from_csv={added:,}, total={len(filtered_patterns_df):,}) to '{output_path_step2}'")
    else:
        print("ℹ️ Step 2: No USD/ACH laundering transactions found.")

except Exception as e:
    print(f"Error in Step 2: {e}")

# ------------------------
# Step 3: Combine Data, Filter Accounts, and Save
# ------------------------
try:
    all_filtered_transactions = pd.concat(
        [df for df in [filtered_normal_df, filtered_patterns_df] if not df.empty],
        ignore_index=True,
        sort=False
    )

    if all_filtered_transactions.empty:
        print("ℹ️ Step 3: No transactions to derive accounts from.")
    else:
        # Optional dedup on core transaction fields
        all_filtered_transactions.drop_duplicates(
            subset=['timestamp','from_bank','from_account','to_bank','to_account',
                    'amount_received','currency_received','amount_sent','currency_sent','payment_type','is_laundering'],
            inplace=True
        )

        source_accounts = set(all_filtered_transactions['from_account'].dropna().astype(str))
        destination_accounts = set(all_filtered_transactions['to_account'].dropna().astype(str))
        involved_accounts_hex = source_accounts | destination_accounts

        accounts_path = os.path.join(DATA_DIR, 'HI-Small_accounts.csv')
        accounts_df = pd.read_csv(
            accounts_path,
            header=None,
            names=['bank_name', 'bank_id', 'account_id_hex', 'entity_id', 'entity_name'],
            dtype={'bank_id':'string', 'account_id_hex':'string', 'entity_id':'string'},
            low_memory=False
        )

        filtered_accounts_df = accounts_df[accounts_df['account_id_hex'].astype(str).isin(involved_accounts_hex)].copy()

        if not filtered_accounts_df.empty:
            output_path_step3 = os.path.join(PROCESSED_DIR, '3_filtered_accounts.csv')
            filtered_accounts_df.to_csv(output_path_step3, index=False)
            print(f"Step 3: Saved filtered account details to '{output_path_step3}' "
                  f"(rows={len(filtered_accounts_df):,})")
        else:
            print("ℹ️ Step 3: No matching accounts found for the filtered transactions.")

except Exception as e:
    print(f"Error in Step 3: {e}")
