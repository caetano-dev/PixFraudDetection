import pandas as pd
import networkx as nx
from pathlib import Path

from datetime import timedelta

# Optional: PR AUC from scikit-learn
try:
    from sklearn.metrics import average_precision_score, precision_recall_curve
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False

# Optional: NetworkX's built-in Louvain (NX >= 3.0) or python-louvain fallback
try:
    from networkx.algorithms.community import louvain_communities as nx_louvain_communities
except Exception:
    nx_louvain_communities = None

# -----------------------
# Config
# -----------------------
proc = Path('data/processed')
p_norm = proc / '1_filtered_normal_transactions.csv'
p_pos  = proc / '2_filtered_laundering_transactions.csv'
p_acct = proc / '3_filtered_accounts.csv'

SAVE_GPICKLE = True
GPICKLE_PATH = proc / 'G_all_multi.gpickle'

# Window experiment settings
WINDOW_DAYS_LIST = [3, 7]
WINDOW_STRIDE_DAYS = 1
MAX_WINDOWS_PER_SETTING = 5  # set None to process all windows

# Louvain settings
LOUVAIN_RESOLUTION = 1.0
LOUVAIN_SEED = 42

# -----------------------
# Utils
# -----------------------
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

def summarize_graph(G: nx.MultiDiGraph, name='Graph'):
    pos_e = sum(1 for _,_,_,d in G.edges(keys=True, data=True) if int(d.get('is_laundering', 0)) == 1)
    print(f"{name}: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges, positive_edges={pos_e:,}")

def derive_node_labels(G: nx.Graph) -> int:
    # Set node label: positive if node touches any positive edge in this graph
    nx.set_node_attributes(G, 0, 'is_laundering_involved')
    for u, v, d in G.edges(data=True):
        if int(d.get('is_laundering', 0)) == 1:
            G.nodes[u]['is_laundering_involved'] = 1
            G.nodes[v]['is_laundering_involved'] = 1
    n_pos = sum(d.get('is_laundering_involved', 0) for _, d in G.nodes(data=True))
    return n_pos

def graph_time_range(G: nx.MultiDiGraph):
    ts = [d['timestamp'] for _,_,d in G.edges(data=True) if d.get('timestamp') is not None]
    return (min(ts), max(ts)) if ts else (None, None)

def iter_windows(start, end, window_days=3, stride_days=1):
    cur = start
    while cur < end:
        yield (cur, cur + timedelta(days=window_days))
        cur += timedelta(days=stride_days)

def get_windowed_graph(G: nx.MultiDiGraph, start, end) -> nx.MultiDiGraph:
    # half-open window [start, end)
    H = nx.MultiDiGraph(name=f"win_{start:%Y%m%d}_{end:%Y%m%d}")
    for u, v, k, d in G.edges(keys=True, data=True):
        ts = d.get('timestamp')
        if ts is not None and (ts >= start) and (ts < end):
            if u not in H: H.add_node(u, **G.nodes[u])
            if v not in H: H.add_node(v, **G.nodes[v])
            H.add_edge(u, v, key=k, **d)
    return H

def aggregate_graph(G: nx.MultiDiGraph, directed=False) -> nx.Graph:
    # Collapse multi-edges; accumulate w_count and w_amount; copy node attributes
    H = nx.DiGraph() if directed else nx.Graph()
    for u, v, d in G.edges(data=True):
        a, b = (u, v) if directed else tuple(sorted((u, v)))
        if not H.has_node(a): H.add_node(a, **G.nodes[a])
        if not H.has_node(b): H.add_node(b, **G.nodes[b])
        w_amount = int(d.get('amount_received_c') or 0)
        if H.has_edge(a, b):
            H[a][b]['w_count'] += 1
            H[a][b]['w_amount'] += w_amount
        else:
            H.add_edge(a, b, w_count=1, w_amount=w_amount)
    return H

def run_louvain(H: nx.Graph, resolution=1.0, seed=42):
    # Returns (partition: dict[node]->community_id, communities: list[set])
    if nx_louvain_communities is not None:
        comms = nx_louvain_communities(H, weight='w_amount', resolution=resolution, seed=seed)
        partition = {}
        for cid, c in enumerate(comms):
            for n in c:
                partition[n] = cid
        return partition, [set(c) for c in comms]
    else:
        # Fallback to python-louvain
        try:
            import community as community_louvain  # python-louvain
        except Exception:
            print("Louvain unavailable: install networkx>=3.0 or `pip install python-louvain`.")
            return {}, []
        partition = community_louvain.best_partition(H, weight='w_amount', random_state=seed, resolution=resolution)
        comms = {}
        for n, cid in partition.items():
            comms.setdefault(cid, set()).add(n)
        comms = list(comms.values())
        return partition, comms

def community_baseline_pr(H: nx.Graph, partition: dict):
    # Score each community by illicit fraction; assign each node its community's score; compute PR AUC
    # Ensure node labels are present on H
    if 'is_laundering_involved' not in next(iter(H.nodes(data=True)))[1]:
        # If missing, derive from edges in H (useful for per-window aggregations)
        tmp_multi = nx.MultiDiGraph()
        tmp_multi.add_nodes_from(H.nodes(data=True))
        for u, v, d in H.edges(data=True):
            # no edge labels in H, so we can't derive here; recommend deriving on the windowed MultiDiGraph before aggregation
            pass

    # Build communities list from partition
    comms = {}
    for n, cid in partition.items():
        comms.setdefault(cid, set()).add(n)

    # Compute illicit fraction per community
    pos_nodes = {n for n, d in H.nodes(data=True) if int(d.get('is_laundering_involved', 0)) == 1}
    comm_pos_frac = {}
    for cid, c in comms.items():
        k = len(c)
        p = len(c & pos_nodes)
        comm_pos_frac[cid] = (p / k) if k else 0.0

    # Node-level labels and scores
    import numpy as np
    nodes = list(H.nodes())
    y_true = np.array([1 if n in pos_nodes else 0 for n in nodes], dtype=int)
    y_score = np.array([comm_pos_frac.get(partition.get(n, -1), 0.0) for n in nodes], dtype=float)

    ap = None
    if SKLEARN_OK:
        ap = average_precision_score(y_true, y_score)
    return ap, y_true, y_score, comm_pos_frac

# -----------------------
# Build base graph
# -----------------------
# Load processed transactions
df_n = pd.read_csv(p_norm, parse_dates=['timestamp'], low_memory=False)
df_p = pd.read_csv(p_pos, parse_dates=['timestamp'], low_memory=False)

df = pd.concat([df_n, df_p], ignore_index=True)
df.sort_values('timestamp', inplace=True)
# Robust types
df['is_laundering'] = pd.to_numeric(df['is_laundering'], errors='coerce').fillna(0).astype('int8')
df['amount_sent_c'] = to_cents(df['amount_sent'])
df['amount_received_c'] = to_cents(df['amount_received'])
df['same_bank'] = (df['from_bank'].astype(str) == df['to_bank'].astype(str))

# Load accounts (dedupe just in case)
acct = pd.read_csv(
    p_acct,
    low_memory=False,
    dtype={'bank_id':'string','account_id_hex':'string','entity_id':'string'}
).drop_duplicates(subset=['account_id_hex'])
acct.set_index('account_id_hex', inplace=True)

# Build MultiDiGraph
G = nx.MultiDiGraph(name='G_all_multi')

# Nodes with attributes
for acc_id, row in acct.iterrows():
    G.add_node(
        str(acc_id),
        bank_id=str(row.get('bank_id', '')),
        entity_id=str(row.get('entity_id', '')),
        entity_name=str(row.get('entity_name', ''))
    )

# Ensure any account in edges exists as a node
for col in ['from_account', 'to_account']:
    missing = set(df[col].astype(str)) - set(G.nodes)
    if missing:
        for acc_id in missing:
            G.add_node(str(acc_id))

# Edges (one per transaction)
for idx, r in df.iterrows():
    u = str(r['from_account']); v = str(r['to_account'])
    ts = r['timestamp']; t_ns = int(ts.value) if pd.notna(ts) else None
    attempt_id = r.get('attempt_id', None)
    attempt_type = r.get('attempt_type', None)
    if pd.isna(attempt_id): attempt_id = None
    if pd.isna(attempt_type): attempt_type = 'UNLISTED'
    G.add_edge(
        u, v,
        key=int(idx),  # unique edge key
        timestamp=ts,
        t_ns=t_ns,
        is_laundering=int(r['is_laundering']),
        attempt_id=attempt_id,
        attempt_type=attempt_type,
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

summarize_graph(G, "G_all_multi")

# Node labels on full graph
n_pos_nodes = derive_node_labels(G)
print(f"Positive nodes (full period): {n_pos_nodes:,}")

def save_graph_gpickle(G, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Try NetworkX gpickle writer
    try:
        from networkx.readwrite.gpickle import write_gpickle
        write_gpickle(G, str(path))
        return
    except Exception:
        pass
    # Fallback: standard pickle
    import pickle
    with open(path, 'wb') as f:
        pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_graph_gpickle(path):
    try:
        from networkx.readwrite.gpickle import read_gpickle
        return read_gpickle(str(path))
    except Exception:
        import pickle
        with open(path, 'rb') as f:
            return pickle.load(f)
# Save for reuse
if SAVE_GPICKLE:
    save_graph_gpickle(G, GPICKLE_PATH)
    print(f"Saved G_all_multi to {GPICKLE_PATH}")

# -----------------------
# Temporal windows (3-day, 7-day)
# -----------------------
tmin, tmax = graph_time_range(G)
print(f"Time range: {tmin} → {tmax}")

for window_days in WINDOW_DAYS_LIST:
    print(f"\n-- {window_days}-day windows, stride={WINDOW_STRIDE_DAYS}d --")
    for i, (ws, we) in enumerate(iter_windows(tmin, tmax, window_days=window_days, stride_days=WINDOW_STRIDE_DAYS)):
        H_win = get_windowed_graph(G, ws, we)
        if H_win.number_of_edges() == 0:
            continue
        # Derive labels within this window (important for fair evaluation)
        pos_nodes_win = derive_node_labels(H_win)

        # Quick summary
        pos_e = sum(1 for *_ , d in H_win.edges(keys=True, data=True) if int(d.get('is_laundering',0))==1)
        print(f"[{i:03d}] {ws:%Y-%m-%d} → {we:%Y-%m-%d}: nodes={H_win.number_of_nodes():,}, edges={H_win.number_of_edges():,}, pos_edges={pos_e:,}, pos_nodes={pos_nodes_win:,}")

        # Limit the number of windows processed (for speed)
        if MAX_WINDOWS_PER_SETTING is not None and i + 1 >= MAX_WINDOWS_PER_SETTING:
            break

# -----------------------
# Community detection baseline
# -----------------------
# Full-period baseline on aggregated undirected graph
H_full = aggregate_graph(G, directed=False)
# Ensure node label exists on H_full (copied from G node attributes above)
ap_full, y_true_full, y_score_full, comm_frac_full = community_baseline_pr(H_full, run_louvain(H_full, resolution=LOUVAIN_RESOLUTION, seed=LOUVAIN_SEED)[0])

print(f"\nCommunity baseline on full period:")
print(f"Aggregated graph: {H_full.number_of_nodes():,} nodes, {H_full.number_of_edges():,} edges")
if SKLEARN_OK and ap_full is not None:
    print(f"PR AUC (node-level): {ap_full:.4f}")
else:
    print("PR AUC unavailable (scikit-learn not installed).")

# Show top communities by illicit fraction
# Build comms from partition again to get sizes
partition_full, comms_full = run_louvain(H_full, resolution=LOUVAIN_RESOLUTION, seed=LOUVAIN_SEED)
pos_nodes_set = {n for n, d in H_full.nodes(data=True) if int(d.get('is_laundering_involved', 0)) == 1}
scores = []
for cid, c in enumerate(comms_full):
    size = len(c)
    p = len(c & pos_nodes_set)
    frac = p / size if size else 0.0
    scores.append((frac, size, p, cid))
scores.sort(reverse=True)
print("Top 5 communities by illicit fraction (fraction, size, positives, cid):")
for frac, size, p, cid in scores[:5]:
    print(f"  cid={cid:>4}  frac={frac:.3f}  size={size:>6}  pos={p:>6}")

# Optional: per-window community baseline for a few windows
print("\nPer-window community baseline (first few windows per setting):")
for window_days in WINDOW_DAYS_LIST:
    print(f"\n-- {window_days}-day windows --")
    count = 0
    for ws, we in iter_windows(tmin, tmax, window_days=window_days, stride_days=WINDOW_STRIDE_DAYS):
        H_win = get_windowed_graph(G, ws, we)
        if H_win.number_of_edges() == 0:
            continue
        derive_node_labels(H_win)
        H_agg = aggregate_graph(H_win, directed=False)
        part_win, _ = run_louvain(H_agg, resolution=LOUVAIN_RESOLUTION, seed=LOUVAIN_SEED)
        ap_win, _, _, _ = community_baseline_pr(H_agg, part_win)
        if SKLEARN_OK and ap_win is not None:
            print(f"[{ws:%Y-%m-%d} → {we:%Y-%m-%d}] PR AUC: {ap_win:.4f}  (nodes={H_agg.number_of_nodes():,}, edges={H_agg.number_of_edges():,})")
        else:
            print(f"[{ws:%Y-%m-%d} → {we:%Y-%m-%d}] nodes={H_agg.number_of_nodes():,}, edges={H_agg.number_of_edges():,}")
        count += 1
        if MAX_WINDOWS_PER_SETTING is not None and count >= MAX_WINDOWS_PER_SETTING:
            break

print("\nDone.")
