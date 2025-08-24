import pandas as pd
import networkx as nx
from pathlib import Path
from datetime import timedelta

proc = Path('data/processed')
p_norm = proc / '1_filtered_normal_transactions.csv'
p_pos  = proc / '2_filtered_laundering_transactions.csv'
p_acct = proc / '3_filtered_accounts.csv'

def parse_ts(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.strip()
    dt = pd.to_datetime(s, format='%Y/%m/%d %H:%M', errors='coerce')
    mask = dt.isna()
    if mask.any():
        dt2 = pd.to_datetime(s[mask], format='%Y/%m/%d %H:%M:%S', errors='coerce')
        dt.loc[mask] = dt2
    return dt

def to_cents(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors='coerce').mul(100).round().astype('Int64')

df_n = pd.read_csv(p_norm, parse_dates=['timestamp'], low_memory=False)
df_p = pd.read_csv(p_pos, parse_dates=['timestamp'], low_memory=False)
df = pd.concat([df_n, df_p], ignore_index=True)
df.sort_values('timestamp', inplace=True)
df['amount_sent_c'] = to_cents(df['amount_sent'])
df['amount_received_c'] = to_cents(df['amount_received'])
df['same_bank'] = (df['from_bank'].astype(str) == df['to_bank'].astype(str))

acct = pd.read_csv(p_acct, low_memory=False, dtype={'bank_id':'string','account_id_hex':'string','entity_id':'string'})
acct.set_index('account_id_hex', inplace=True)

G = nx.MultiDiGraph(name='G_all_multi')

for acc_id, row in acct.iterrows():
    G.add_node(
        str(acc_id),
        bank_id=str(row.get('bank_id', '')),
        entity_id=str(row.get('entity_id', '')),
        entity_name=str(row.get('entity_name', ''))
    )

for col in ['from_account', 'to_account']:
    missing = set(df[col].astype(str)) - set(G.nodes)
    for acc_id in missing:
        G.add_node(str(acc_id))

# Add edges (one per transaction)
for idx, r in df.iterrows():
    u = str(r['from_account']); v = str(r['to_account'])
    ts = r['timestamp']; t_ns = int(ts.value) if pd.notna(ts) else None
    G.add_edge(
        u, v,
        key=int(idx),  # unique edge key
        timestamp=ts,
        t_ns=t_ns,
        is_laundering=int(r['is_laundering']),
        attempt_id=None if pd.isna(r.get('attempt_id', None)) else r.get('attempt_id'),
        attempt_type=r.get('attempt_type', 'UNLISTED') if r.get('attempt_type', pd.NA) is pd.NA else r.get('attempt_type'),
        amount_sent_c=r['amount_sent_c'],
        amount_received_c=r['amount_received_c'],
        amount_sent=r['amount_sent'],
        amount_received=r['amount_received'],
        from_bank=str(r['from_bank']),
        to_bank=str(r['to_bank']),
        same_bank=bool(r['same_bank']),
        payment_type=r.get('payment_type', 'ACH'),
        currency_sent=r.get('currency_sent', 'US Dollar'),
        currency_received=r.get('currency_received', 'US Dollar'),
    )

print(f"Built {G} with {G.number_of_nodes():,} nodes and {G.number_of_edges():,} edges")
pos_edges = sum(1 for _,_,_,d in G.edges(keys=True, data=True) if d.get('is_laundering',0)==1)
print(f"Positive edges: {pos_edges:,}")


def get_windowed_graph(G, start, end):
    # half-open window [start, end)
    H = nx.MultiDiGraph(name=f"window_{start}_{end}")
    for u, v, k, d in G.edges(keys=True, data=True):
        ts = d.get('timestamp')
        if ts is not None and (ts >= start) and (ts < end):
            H.add_node(u, **G.nodes[u])
            H.add_node(v, **G.nodes[v])
            H.add_edge(u, v, key=k, **d)
    return H

def aggregate_graph(G, directed=False, weight_mode='count'):
    H = nx.DiGraph() if directed else nx.Graph()
    for u, v, d in G.edges(data=True):
        a, b = (u, v) if directed else tuple(sorted((u, v)))
        w = 1 if weight_mode=='count' else (int(d.get('amount_received_c') or 0))
        if H.has_edge(a, b):
            H[a][b]['w_count'] += 1
            H[a][b]['w_amount'] += w if weight_mode!='count' else 0
        else:
            H.add_edge(a, b, w_count=1, w_amount=(0 if weight_mode=='count' else w))
    return H
