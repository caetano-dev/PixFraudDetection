import pandas as pd

p1 = 'data/processed/1_filtered_normal_transactions.csv'
p2 = 'data/processed/2_filtered_laundering_transactions.csv'
p3 = 'data/processed/3_filtered_accounts.csv'
raw_path = 'data/HI-Small_Trans.csv'

standard_columns = [
    'timestamp', 'from_bank', 'from_account', 'to_bank', 'to_account',
    'amount_received', 'currency_received', 'amount_sent', 'currency_sent',
    'payment_type', 'is_laundering'
]

proc_dtype = {
    'from_bank': 'string', 'to_bank': 'string',
    'from_account': 'string', 'to_account': 'string',
    'currency_received': 'string', 'currency_sent': 'string',
    'payment_type': 'string'
}

# Read processed files with consistent types and parsed timestamps
df_n = pd.read_csv(p1, dtype=proc_dtype, parse_dates=['timestamp'], low_memory=False)
df_p = pd.read_csv(p2, dtype={**proc_dtype, 'attempt_id': 'Int64', 'attempt_type': 'string'},
                   parse_dates=['timestamp'], low_memory=False)
df_a = pd.read_csv(p3, dtype={'bank_id': 'string', 'account_id_hex': 'string', 'entity_id': 'string'},
                   low_memory=False)

# Quick contamination checks
assert df_n['is_laundering'].eq(0).all(), "Normal file contains laundering rows."
assert df_p['is_laundering'].eq(1).all(), "Laundering file contains non-laundering rows."
assert set(df_n['payment_type'].str.strip().unique()) == {'ACH'}
assert set(df_p['payment_type'].str.strip().unique()) == {'ACH'}
assert set(df_n['currency_sent'].str.strip().unique()) == {'US Dollar'}
assert set(df_n['currency_received'].str.strip().unique()) == {'US Dollar'}
assert set(df_p['currency_sent'].str.strip().unique()) == {'US Dollar'}
assert set(df_p['currency_received'].str.strip().unique()) == {'US Dollar'}

# Cross-file duplicates (should be zero)
key_cols = ['timestamp','from_bank','from_account','to_bank','to_account',
            'amount_received','currency_received','amount_sent','currency_sent','payment_type']
dupes = pd.merge(df_n[key_cols], df_p[key_cols], on=key_cols, how='inner')
print('Cross-file duplicate rows:', len(dupes))

# Read raw with explicit dtypes to avoid DtypeWarning
raw = pd.read_csv(
    raw_path,
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

# Parse/normalize raw
raw['timestamp'] = pd.to_datetime(raw['timestamp'], format='%Y/%m/%d %H:%M', errors='coerce')
raw['amount_sent'] = pd.to_numeric(raw['amount_sent'], errors='coerce')
raw['amount_received'] = pd.to_numeric(raw['amount_received'], errors='coerce')
raw['is_laundering'] = pd.to_numeric(raw['is_laundering'], errors='coerce').fillna(0).astype('int8')

def norm(s: pd.Series) -> pd.Series:
    return s.astype('string').str.strip().str.upper()

raw['payment_type_u'] = norm(raw['payment_type'])
raw['currency_sent_u'] = norm(raw['currency_sent'])
raw['currency_received_u'] = norm(raw['currency_received'])

mask_raw_pos = (
    raw['payment_type_u'].eq('ACH') &
    raw['currency_sent_u'].eq('US DOLLAR') &
    raw['currency_received_u'].eq('US DOLLAR') &
    raw['is_laundering'].eq(1)
)
raw_pos = raw.loc[mask_raw_pos].copy()

# Normalize processed positives similarly and align types
df_p['payment_type_u'] = norm(df_p['payment_type'])
df_p['currency_sent_u'] = norm(df_p['currency_sent'])
df_p['currency_received_u'] = norm(df_p['currency_received'])

# Compare amounts as integer cents to avoid float equality issues
def to_cents(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors='coerce').mul(100).round().astype('Int64')

for df in (raw_pos, df_p):
    df['amount_sent_c'] = to_cents(df['amount_sent'])
    df['amount_received_c'] = to_cents(df['amount_received'])

# Define robust join keys (we already filtered to USD/ACH, so we can omit currency/p_type from keys)
merge_cols = ['timestamp','from_bank','from_account','to_bank','to_account','amount_received_c','amount_sent_c']

p_pos = df_p.copy()  # already filtered in your pipeline

# Missing positives: those in the big CSV (raw_pos) that aren't represented in patterns-derived df_p
missing_from_patterns = (
    raw_pos[merge_cols]
    .merge(p_pos[merge_cols], on=merge_cols, how='left', indicator=True)
    .query("_merge=='left_only'")
)
print('Positives in CSV but not in patterns subset:', len(missing_from_patterns))

# Attempt type distribution (useful later)
print('Attempt types in processed positives:')
print(df_p['attempt_type'].value_counts(dropna=False))
# Accounts coverage: every account referenced exists in 3_filtered_accounts
involved = pd.unique(pd.concat([df_n['from_account'], df_n['to_account'],
                                df_p['from_account'], df_p['to_account']]).astype(str))
missing_accts = set(involved) - set(df_a['account_id_hex'].astype(str))
print('Missing account records:', len(missing_accts))
