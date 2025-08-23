import pandas as pd
import os
import re

# Define a directory for processed data to keep things organized
DATA_DIR = 'data'
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')
os.makedirs(PROCESSED_DIR, exist_ok=True) # Create the directory if it doesn't exist

standard_columns = [
    'timestamp', 'from_bank', 'from_account', 'to_bank', 'to_account',
    'amount_received', 'currency_received', 'amount_sent', 'currency_sent',
    'payment_type', 'is_laundering'
]

# --- Step 1: Filter HI-Small_Trans.csv and Save ---
try:
    transactions_path = os.path.join(DATA_DIR, 'HI-Small_Trans.csv')
    transactions_df = pd.read_csv(transactions_path)
    
    column_mapping = dict(zip(transactions_df.columns, standard_columns))
    transactions_df = transactions_df.rename(columns=column_mapping)
    
    transactions_df['timestamp'] = pd.to_datetime(transactions_df['timestamp'], format='%Y/%m/%d %H:%M')
    
    # --- FIX APPLIED HERE ---
    filtered_transactions_df = transactions_df[
        (transactions_df['currency_sent'] == 'US Dollar') &
        (transactions_df['currency_received'] == 'US Dollar') & # Condition added
        (transactions_df['payment_type'] == 'ACH')
    ]

    if not filtered_transactions_df.empty:
        output_path_step1 = os.path.join(PROCESSED_DIR, '1_filtered_normal_transactions.csv')
        filtered_transactions_df.to_csv(output_path_step1, index=False)
        print(f"✅ Step 1: Saved strictly USD transactions to '{output_path_step1}'")

except Exception as e:
    print(f"Error in Step 1: {e}")


def parse_patterns_file(file_path):
    """Parses the patterns text file to extract laundering attempts and transactions."""
    attempts = []
    current_attempt = None
    attempt_counter = 0
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('BEGIN LAUNDERING ATTEMPT'):
                attempt_counter += 1
                attempt_type_match = re.search(r'-\s*([\w\s-]+)', line)
                attempt_type = attempt_type_match.group(1).strip() if attempt_type_match else 'UNKNOWN'
                current_attempt = {
                    'attempt_id': attempt_counter,
                    'attempt_type': attempt_type,
                    'transactions': []
                }
            elif line.startswith('END LAUNDERING ATTEMPT'):
                if current_attempt:
                    attempts.append(current_attempt)
                current_attempt = None
            elif line and current_attempt:
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 11:
                    transaction = dict(zip(standard_columns, parts))
                    transaction['attempt_id'] = current_attempt['attempt_id']
                    transaction['attempt_type'] = current_attempt['attempt_type']
                    current_attempt['transactions'].append(transaction)

    all_transactions = [tx for attempt in attempts for tx in attempt['transactions']]
    return pd.DataFrame(all_transactions)

try:
    patterns_path = os.path.join(DATA_DIR, 'HI_patterns.txt')
    patterns_df = parse_patterns_file(patterns_path)

    if not patterns_df.empty:
        patterns_df['timestamp'] = pd.to_datetime(patterns_df['timestamp'], format='%Y/%m/%d %H:%M')
        patterns_df['amount_sent'] = pd.to_numeric(patterns_df['amount_sent'], errors='coerce')
        patterns_df['is_laundering'] = pd.to_numeric(patterns_df['is_laundering'], errors='coerce')

        # --- FIX APPLIED HERE ---
        filtered_patterns_df = patterns_df[
            (patterns_df['currency_sent'] == 'US Dollar') &
            (patterns_df['currency_received'] == 'US Dollar') & # Condition added
            (patterns_df['payment_type'] == 'ACH')
        ]
        
        if not filtered_patterns_df.empty:
            output_path_step2 = os.path.join(PROCESSED_DIR, '2_filtered_laundering_transactions.csv')
            filtered_patterns_df.to_csv(output_path_step2, index=False)
            print(f"✅ Step 2: Saved strictly USD laundering transactions to '{output_path_step2}'")
    else:
        filtered_patterns_df = pd.DataFrame()

except Exception as e:
    print(f"Error in Step 2: {e}")


# --- Step 3: Combine Data, Filter Accounts, and Save ---
# (No changes needed here, it will now work correctly with the fixed inputs)
try:
    all_filtered_transactions = pd.concat(
        [filtered_transactions_df, filtered_patterns_df], ignore_index=True
    )

    source_accounts = set(all_filtered_transactions['from_account'].dropna().astype(str))
    destination_accounts = set(all_filtered_transactions['to_account'].dropna().astype(str))
    involved_accounts_hex = source_accounts | destination_accounts

    accounts_path = os.path.join(DATA_DIR, 'HI-Small_accounts.csv')
    accounts_df = pd.read_csv(accounts_path, header=None, names=['bank_name', 'bank_id', 'account_id_hex', 'entity_id', 'entity_name'])
    
    filtered_accounts_df = accounts_df[accounts_df['account_id_hex'].astype(str).isin(involved_accounts_hex)]

    if not filtered_accounts_df.empty:
        output_path_step3 = os.path.join(PROCESSED_DIR, '3_filtered_accounts.csv')
        filtered_accounts_df.to_csv(output_path_step3, index=False)
        print(f"✅ Step 3: Saved filtered account details to '{output_path_step3}'")

except Exception as e:
    print(f"Error in Step 3: {e}")
