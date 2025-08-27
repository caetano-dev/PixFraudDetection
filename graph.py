# aml_graph.py
import pandas as pd
import networkx as nx
import numpy as np
from pathlib import Path
from datetime import timedelta

# -----------------------
# Config
# -----------------------
proc = Path('data/processed')
p_norm = proc / '1_filtered_normal_transactions.csv'
p_pos  = proc / '2_filtered_laundering_transactions.csv'
p_acct = proc / '3_filtered_accounts.csv'

SAVE_GPICKLE = True
GPICKLE_PATH = proc / 'G_all_multi.gpickle'

WINDOW_DAYS_LIST = [3, 7]
WINDOW_STRIDE_DAYS = 1
MAX_WINDOWS_PER_SETTING = 5  # set None to process all windows

LOUVAIN_RESOLUTION = 1.0
LOUVAIN_SEED = 42

try:
    from networkx.algorithms.community import louvain_communities as nx_louvain_communities
except Exception:
    nx_louvain_communities = None

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

def get_windowed_graph_fast(df_slice: pd.DataFrame, base_graph: nx.MultiDiGraph) -> nx.MultiDiGraph:
    if len(df_slice) == 0:
        return nx.MultiDiGraph(name="win_empty")
    ts_min = df_slice['timestamp'].min()
    ts_max = df_slice['timestamp'].max()
    H = nx.MultiDiGraph(name=f"win_{ts_min:%Y%m%d}_{ts_max:%Y%m%d}")
    nodes_needed = set(df_slice['from_account'].astype(str)) | set(df_slice['to_account'].astype(str))
    for node in nodes_needed:
        if node in base_graph.nodes:
            attrs = dict(base_graph.nodes[node])
            attrs.pop('is_laundering_involved', None)
            H.add_node(node, **attrs)
        else:
            H.add_node(node)
    for idx, r in df_slice.iterrows():
        u = str(r['from_account']); v = str(r['to_account'])
        ts = r['timestamp']; t_ns = int(ts.value) if pd.notna(ts) else None
        attempt_id = r.get('attempt_id', None)
        attempt_type = r.get('attempt_type', None)
        if pd.isna(attempt_id): attempt_id = None
        if pd.isna(attempt_type): attempt_type = 'UNLISTED'
        H.add_edge(
            u, v,
            key=int(idx),
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
    return H

def get_windowed_graph(G: nx.MultiDiGraph, start, end) -> nx.MultiDiGraph:
    H = nx.MultiDiGraph(name=f"win_{start:%Y%m%d}_{end:%Y%m%d}")
    for u, v, k, d in G.edges(keys=True, data=True):
        ts = d.get('timestamp')
        if ts is not None and (ts >= start) and (ts < end):
            if u not in H: H.add_node(u, **G.nodes[u])
            if v not in H: H.add_node(v, **G.nodes[v])
            H.add_edge(u, v, key=k, **d)
    return H

def aggregate_graph(G: nx.MultiDiGraph, directed=False) -> nx.Graph:
    H = nx.DiGraph() if directed else nx.Graph()
    edge_data = {}
    for u, v, d in G.edges(data=True):
        a, b = (u, v) if directed else tuple(sorted((u, v)))
        if not H.has_node(a): H.add_node(a, **G.nodes[a])
        if not H.has_node(b): H.add_node(b, **G.nodes[b])
        key = (a, b)
        edge_data.setdefault(key, []).append(d)
    for (a, b), edge_list in edge_data.items():
        w_count = len(edge_list)
        w_amount = sum(int(d.get('amount_received_c', 0) or 0) for d in edge_list)
        w_amount_log = np.log1p(w_amount) if w_amount > 0 else 0.0
        timestamps = [d.get('timestamp') for d in edge_list if d.get('timestamp') is not None]
        first_ts = min(timestamps) if timestamps else None
        last_ts = max(timestamps) if timestamps else None
        span_seconds = (last_ts - first_ts).total_seconds() if (first_ts and last_ts) else 0
        H.add_edge(a, b, w_count=w_count, w_amount=w_amount, w_amount_log=w_amount_log,
                   first_ts=first_ts, last_ts=last_ts, span_seconds=span_seconds)
    if directed:
        for u, v in H.edges():
            H[u][v]['reciprocated'] = 1 if H.has_edge(v, u) else 0
    for node in H.nodes():
        if directed:
            in_edges = [(pred, node) for pred in H.predecessors(node)]
            out_edges = [(node, succ) for succ in H.successors(node)]
            in_amount_sum  = sum(H[u][v].get('w_amount', 0) for u, v in in_edges)
            out_amount_sum = sum(H[u][v].get('w_amount', 0) for u, v in out_edges)
            in_deg = len(in_edges); out_deg = len(out_edges)
            in_tx_count  = sum(H[u][v].get('w_count', 0) for u, v in in_edges)
            out_tx_count = sum(H[u][v].get('w_count', 0) for u, v in out_edges)
        else:
            adj_edges = [(node, nbr) if node <= nbr else (nbr, node) for nbr in H.neighbors(node)]
            total_amount = sum(H[u][v].get('w_amount', 0) for u, v in adj_edges)
            total_tx     = sum(H[u][v].get('w_count', 0) for u, v in adj_edges)
            deg = len(adj_edges)
            in_amount_sum = out_amount_sum = total_amount
            in_deg = out_deg = deg
            in_tx_count = out_tx_count = total_tx
        in_out_amount_ratio = (in_amount_sum + 1) / (out_amount_sum + 1)
        H.nodes[node].update({
            'in_amount_sum': in_amount_sum,
            'out_amount_sum': out_amount_sum,
            'in_deg': in_deg, 'out_deg': out_deg,
            'in_tx_count': in_tx_count, 'out_tx_count': out_tx_count,
            'in_out_amount_ratio': in_out_amount_ratio
        })
    return H

def run_louvain(H: nx.Graph, resolution=1.0, seed=42, weight='w_amount_log'):
    if nx_louvain_communities is not None:
        comms = nx_louvain_communities(H, weight=weight, resolution=resolution, seed=seed)
        partition = {n: cid for cid, c in enumerate(comms) for n in c}
        return partition, [set(c) for c in comms]
    else:
        try:
            import community as community_louvain  # python-louvain
        except Exception:
            print("Louvain unavailable: install networkx>=3.0 or `pip install python-louvain`.")
            return {}, []
        partition = community_louvain.best_partition(H, weight=weight, random_state=seed, resolution=resolution)
        comms = {}
        for n, cid in partition.items():
            comms.setdefault(cid, set()).add(n)
        return partition, list(comms.values())

def score_communities_unsupervised(H, comms: list, min_size=3):
    scores = {}
    for cid, nodes in enumerate(comms):
        if not nodes or len(nodes) < min_size:
            scores[cid] = 0.0
            continue
        sub = H.subgraph(nodes)
        n = len(nodes)
        max_edges = n*(n-1)/2 if n > 1 else 0
        internal_density = (sub.number_of_edges()/max_edges) if max_edges else 0.0
        try:
            avg_clust = nx.average_clustering(sub, weight='w_amount_log')
        except Exception:
            avg_clust = 0.0
        total_amount = sum(d.get('w_amount', 0) for _,_,d in sub.edges(data=True))
        amount_score = min(1.0, np.log1p(total_amount)/20)
        size_boost = 1 - np.exp(-n/10)
        scores[cid] = (0.35*internal_density + 0.2*avg_clust + 0.45*amount_score) * size_boost
    return scores

# -----------------------
# Data loading and build
# -----------------------
def load_processed():
    df_n = pd.read_csv(p_norm, parse_dates=['timestamp'], low_memory=False)
    df_p = pd.read_csv(p_pos,  parse_dates=['timestamp'], low_memory=False)
    df = pd.concat([df_n, df_p], ignore_index=True)
    df.sort_values('timestamp', inplace=True)
    df['is_laundering'] = pd.to_numeric(df['is_laundering'], errors='coerce').fillna(0).astype('int8')
    df['amount_sent_c'] = to_cents(df['amount_sent'])
    df['amount_received_c'] = to_cents(df['amount_received'])
    df['same_bank'] = (df['from_bank'].astype(str) == df['to_bank'].astype(str))
    acct = pd.read_csv(
        p_acct, low_memory=False,
        dtype={'bank_id':'string','account_id_hex':'string','entity_id':'string'}
    ).drop_duplicates(subset=['account_id_hex'])
    acct.set_index('account_id_hex', inplace=True)
    return df, acct

def build_canonical_graph(df: pd.DataFrame, acct: pd.DataFrame) -> nx.MultiDiGraph:
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
        if missing:
            for acc_id in missing:
                G.add_node(str(acc_id))
    for idx, r in df.iterrows():
        u = str(r['from_account']); v = str(r['to_account'])
        ts = r['timestamp']; t_ns = int(ts.value) if pd.notna(ts) else None
        attempt_id = r.get('attempt_id', None)
        attempt_type = r.get('attempt_type', None)
        if pd.isna(attempt_id): attempt_id = None
        if pd.isna(attempt_type): attempt_type = 'UNLISTED'
        G.add_edge(
            u, v, key=int(idx),
            timestamp=ts, t_ns=t_ns,
            is_laundering=int(r['is_laundering']),
            attempt_id=attempt_id, attempt_type=attempt_type,
            amount_sent_c=r['amount_sent_c'], amount_received_c=r['amount_received_c'],
            amount_sent=r['amount_sent'], amount_received=r['amount_received'],
            from_bank=str(r['from_bank']), to_bank=str(r['to_bank']),
            same_bank=bool(r['same_bank']),
            payment_type=r.get('payment_type', 'ACH'),
            currency_sent=r.get('currency_sent', 'US Dollar'),
            currency_received=r.get('currency_received', 'US Dollar'),
        )
    return G

def build_all(save_gpickle: bool = True):
    df, acct = load_processed()
    G = build_canonical_graph(df, acct)
    summarize_graph(G, "G_all_multi")
    n_pos_nodes = derive_node_labels(G)
    print(f"Positive nodes (full period): {n_pos_nodes:,}")
    if save_gpickle and SAVE_GPICKLE:
        save_graph_gpickle(G, GPICKLE_PATH)
        print(f"Saved G_all_multi to {GPICKLE_PATH}")
    tmin, tmax = graph_time_range(G)
    print(f"Time range: {tmin} → {tmax}")
    return df, G, tmin, tmax

def save_graph_gpickle(G, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from networkx.readwrite.gpickle import write_gpickle
        write_gpickle(G, str(path)); return
    except Exception:
        pass
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

if __name__ == "__main__":
    build_all(save_gpickle=True)