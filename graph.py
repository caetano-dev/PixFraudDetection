import pandas as pd
import networkx as nx
import numpy as np
from pathlib import Path

from datetime import timedelta

# Optional: PR AUC from scikit-learn
try:
    from sklearn.metrics import average_precision_score
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

def get_windowed_graph_fast(df_slice: pd.DataFrame, base_graph: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """
    Efficiently build a windowed graph from a pre-filtered DataFrame slice.
    This avoids iterating through all edges in the base graph.
    """
    if len(df_slice) == 0:
        return nx.MultiDiGraph(name="win_empty")
    
    # Extract time range for naming
    ts_min = df_slice['timestamp'].min()
    ts_max = df_slice['timestamp'].max()
    H = nx.MultiDiGraph(name=f"win_{ts_min:%Y%m%d}_{ts_max:%Y%m%d}")
    
    # Build nodes with attributes from base graph
    nodes_needed = set(df_slice['from_account'].astype(str)) | set(df_slice['to_account'].astype(str))
    for node in nodes_needed:
        if node in base_graph.nodes:
            attrs = dict(base_graph.nodes[node])
            attrs.pop('is_laundering_involved', None)
            H.add_node(node, **attrs)
        else:
            H.add_node(node)  # fallback for missing nodes
    
    # Add edges from the DataFrame slice
    for idx, r in df_slice.iterrows():
        u = str(r['from_account'])
        v = str(r['to_account'])
        ts = r['timestamp']
        t_ns = int(ts.value) if pd.notna(ts) else None
        attempt_id = r.get('attempt_id', None)
        attempt_type = r.get('attempt_type', None)
        if pd.isna(attempt_id): 
            attempt_id = None
        if pd.isna(attempt_type): 
            attempt_type = 'UNLISTED'
        
        H.add_edge(
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
    
    return H

def get_windowed_graph(G: nx.MultiDiGraph, start, end) -> nx.MultiDiGraph:
    # half-open window [start, end) - LEGACY function, kept for compatibility
    H = nx.MultiDiGraph(name=f"win_{start:%Y%m%d}_{end:%Y%m%d}")
    for u, v, k, d in G.edges(keys=True, data=True):
        ts = d.get('timestamp')
        if ts is not None and (ts >= start) and (ts < end):
            if u not in H: H.add_node(u, **G.nodes[u])
            if v not in H: H.add_node(v, **G.nodes[v])
            H.add_edge(u, v, key=k, **d)
    return H

def aggregate_graph(G: nx.MultiDiGraph, directed=False) -> nx.Graph:
    """
    Enhanced aggregation function with richer features.
    Collapses multi-edges and computes comprehensive edge and node features.
    """
    H = nx.DiGraph() if directed else nx.Graph()
    
    # First pass: aggregate edges
    edge_data = {}  # (u,v) -> list of edge attributes
    
    for u, v, d in G.edges(data=True):
        # For undirected graphs, ensure consistent ordering
        a, b = (u, v) if directed else tuple(sorted((u, v)))
        
        # Copy node attributes if not already present
        if not H.has_node(a): 
            H.add_node(a, **G.nodes[a])
        if not H.has_node(b): 
            H.add_node(b, **G.nodes[b])
        
        # Collect edge data
        key = (a, b)
        if key not in edge_data:
            edge_data[key] = []
        edge_data[key].append(d)
    
    # Second pass: compute aggregated edge features
    for (a, b), edge_list in edge_data.items():
        w_count = len(edge_list)
        w_amount = sum(int(d.get('amount_received_c', 0) or 0) for d in edge_list)
        w_amount_log = np.log1p(w_amount) if w_amount > 0 else 0
        
        # Temporal features
        timestamps = [d.get('timestamp') for d in edge_list if d.get('timestamp') is not None]
        first_ts = min(timestamps) if timestamps else None
        last_ts = max(timestamps) if timestamps else None
        span_seconds = (last_ts - first_ts).total_seconds() if (first_ts and last_ts) else 0
        
        H.add_edge(a, b, 
                  w_count=w_count,
                  w_amount=w_amount,
                  w_amount_log=w_amount_log,
                  first_ts=first_ts,
                  last_ts=last_ts,
                  span_seconds=span_seconds)
    
    # Third pass: add reciprocated attribute for directed graphs
    if directed:
        for u, v in H.edges():
            H[u][v]['reciprocated'] = 1 if H.has_edge(v, u) else 0
    
    # Fourth pass: compute node-level features
    for node in H.nodes():
        if directed:
            in_edges = [(pred, node) for pred in H.predecessors(node)]
            out_edges = [(node, succ) for succ in H.successors(node)]
            in_amount_sum  = sum(H[u][v].get('w_amount', 0) for u, v in in_edges)
            out_amount_sum = sum(H[u][v].get('w_amount', 0) for u, v in out_edges)
            in_deg = len(in_edges)
            out_deg = len(out_edges)
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
            'in_deg': in_deg,
            'out_deg': out_deg,
            'in_tx_count': in_tx_count,
            'out_tx_count': out_tx_count,
            'in_out_amount_ratio': in_out_amount_ratio
        })
    
    return H

# Centrality baselines and evaluation
def precision_at_k(y_true, y_score, k_frac=0.01):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    n = max(1, int(len(y_true) * k_frac))
    idx = np.argsort(-y_score)[:n]
    return float(y_true[idx].mean())

def eval_scores(nodes, y_true_dict, score_dict, k_fracs=(0.005, 0.01, 0.02)):
    y_true = np.array([y_true_dict.get(n, 0) for n in nodes], dtype=int)
    res = {}
    for name, s in score_dict.items():
        scores = np.array([s.get(n, 0.0) for n in nodes], dtype=float)
        ap = average_precision_score(y_true, scores) if SKLEARN_OK and len(set(y_true)) > 1 else None
        res[name] = {'ap': ap}
        for k in k_fracs:
            res[name][f'p@{int(k*1000)/10:.1f}%'] = precision_at_k(y_true, scores, k)
    return res

def run_centrality_baselines(H_dir: nx.DiGraph):
    # Scores
    scores = {}
    try:
        scores['pagerank_wlog'] = nx.pagerank(H_dir, weight='w_amount_log', alpha=0.9, max_iter=100, tol=1e-6)
    except Exception:
        scores['pagerank_wlog'] = {}
    try:
        hubs, auth = nx.hits(H_dir, max_iter=500, tol=1e-8, normalized=True)
        scores['hits_hub'] = hubs
        scores['hits_auth'] = auth
    except Exception:
        scores['hits_hub'] = {}; scores['hits_auth'] = {}
    # Attribute-based
    scores['in_deg'] = {n: H_dir.nodes[n].get('in_deg', 0) for n in H_dir}
    scores['out_deg'] = {n: H_dir.nodes[n].get('out_deg', 0) for n in H_dir}
    scores['in_tx'] = {n: H_dir.nodes[n].get('in_tx_count', 0) for n in H_dir}
    scores['out_tx'] = {n: H_dir.nodes[n].get('out_tx_count', 0) for n in H_dir}
    scores['in_amt'] = {n: H_dir.nodes[n].get('in_amount_sum', 0) for n in H_dir}
    scores['out_amt'] = {n: H_dir.nodes[n].get('out_amount_sum', 0) for n in H_dir}
    scores['collector'] = {n: (H_dir.nodes[n].get('in_amount_sum',0)) / (H_dir.nodes[n].get('out_amount_sum',0)+1) for n in H_dir}
    scores['distributor'] = {n: (H_dir.nodes[n].get('out_amount_sum',0)) / (H_dir.nodes[n].get('in_amount_sum',0)+1) for n in H_dir}
    return scores

def run_louvain(H: nx.Graph, resolution=1.0, seed=42, weight='w_amount_log'):
    if nx_louvain_communities is not None:
        comms = nx_louvain_communities(H, weight=weight, resolution=resolution, seed=seed)
        partition = {n: cid for cid, c in enumerate(comms) for n in c}
        return partition, [set(c) for c in comms]
    else:
        # Fallback to python-louvain
        try:
            import community as community_louvain  # python-louvain
        except Exception:
            print("Louvain unavailable: install networkx>=3.0 or `pip install python-louvain`.")
            return {}, []
        partition = community_louvain.best_partition(H, weight=weight, random_state=seed, resolution=resolution)
        comms = {}
        for n, cid in partition.items():
            comms.setdefault(cid, set()).add(n)
        comms = list(comms.values())
        return partition, comms

def score_communities_unsupervised(H, comms: list, min_size=3):
    """
    Truly unsupervised community scoring based on structural properties.
    Does NOT use ground-truth labels to avoid target leakage.
    """
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
        except:
            avg_clust = 0.0
        total_amount = sum(d.get('w_amount', 0) for _,_,d in sub.edges(data=True))
        amount_score = min(1.0, np.log1p(total_amount)/20)
        size_boost = 1 - np.exp(-n/10)  # S-shaped size factor
        scores[cid] = (0.35*internal_density + 0.2*avg_clust + 0.45*amount_score) * size_boost
    return scores

def run_seeded_pagerank_baseline(H_agg_directed: nx.DiGraph, seed_nodes: set, weight='w_amount_log', alpha=0.9):
    """
    Semi-supervised baseline using Personalized PageRank.
    Uses seed nodes (known positives) to propagate suspicion scores.
    """
    if not seed_nodes: return None
    personalization = {n: (1.0/len(seed_nodes) if n in seed_nodes else 0.0) for n in H_agg_directed}
    try:
        pr_fwd = nx.pagerank(H_agg_directed, personalization=personalization, weight=weight, alpha=alpha, max_iter=100, tol=1e-6)
        pr_rev = nx.pagerank(H_agg_directed.reverse(copy=False), personalization=personalization, weight=weight, alpha=alpha, max_iter=100, tol=1e-6)
        pagerank_scores = {n: 0.5*(pr_fwd.get(n,0.0) + pr_rev.get(n,0.0)) for n in H_agg_directed}
    except:
        return None
    
    # Evaluate on non-seed nodes only
    non_seed_nodes = [n for n in H_agg_directed.nodes() if n not in seed_nodes]
    if len(non_seed_nodes) == 0:
        return None
    
    # Get ground truth labels and PageRank scores for non-seed nodes
    y_true = []
    y_score = []
    
    for node in non_seed_nodes:
        is_positive = int(H_agg_directed.nodes[node].get('is_laundering_involved', 0))
        score = pagerank_scores.get(node, 0.0)
        y_true.append(is_positive)
        y_score.append(score)
    
    # Calculate PR AUC
    if SKLEARN_OK and len(set(y_true)) > 1:  # Need both classes
        pr_auc = average_precision_score(y_true, y_score)
        return pr_auc
    else:
        return None

def community_baseline_pr(H: nx.Graph):
    """
    DEPRECATED: This function had target leakage issues.
    Use score_communities_unsupervised or run_seeded_pagerank_baseline instead.
    """
    print("WARNING: community_baseline_pr is deprecated due to target leakage.")
    print("Use score_communities_unsupervised() for truly unsupervised scoring,")
    print("or run_seeded_pagerank_baseline() for semi-supervised evaluation.")
    return None, None, None, None

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
# Temporal windows (3-day, 7-day) - Using optimized approach
# -----------------------
tmin, tmax = graph_time_range(G)
print(f"Time range: {tmin} → {tmax}")

for window_days in WINDOW_DAYS_LIST:
    print(f"\n-- {window_days}-day windows, stride={WINDOW_STRIDE_DAYS}d --")
    for i, (ws, we) in enumerate(iter_windows(tmin, tmax, window_days=window_days, stride_days=WINDOW_STRIDE_DAYS)):
        # Use optimized window slicing
        df_slice = df[(df['timestamp'] >= ws) & (df['timestamp'] < we)]
        H_win = get_windowed_graph_fast(df_slice, G)
        
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
# Enhanced Community detection baseline
# -----------------------
# Full-period baseline on enhanced aggregated undirected graph
H_full = aggregate_graph(G, directed=False)

print(f"\nEnhanced community baseline on full period:")
print(f"Aggregated graph: {H_full.number_of_nodes():,} nodes, {H_full.number_of_edges():,} edges")

# Run Louvain community detection
partition_full, comms_full = run_louvain(H_full, resolution=LOUVAIN_RESOLUTION, seed=LOUVAIN_SEED, weight='w_amount_log')

# Use unsupervised community scoring (no target leakage)
comm_scores = score_communities_unsupervised(H_full, comms_full)

print("Unsupervised community scoring (top 5 by heuristic score):")
sorted_comms = sorted(comm_scores.items(), key=lambda x: x[1], reverse=True)
for cid, score in sorted_comms[:5]:
    size = len(comms_full[cid]) if cid < len(comms_full) else 0
    print(f"  cid={cid:>4}  score={score:.3f}  size={size:>6}")

# Semi-supervised baseline using PersonalizedPageRank
H_full_directed = aggregate_graph(G, directed=True)
pos_nodes_set = {n for n, d in H_full_directed.nodes(data=True) if int(d.get('is_laundering_involved', 0)) == 1}

if len(pos_nodes_set) > 0:
    # Use a subset of positive nodes as seeds (e.g., 20%)
    seed_size = max(1, len(pos_nodes_set) // 5)
    seed_nodes = set(sorted(pos_nodes_set)[:seed_size])
    
    pr_auc_seeded = run_seeded_pagerank_baseline(H_full_directed, seed_nodes)
    if pr_auc_seeded is not None:
        print(f"PersonalizedPageRank baseline PR-AUC: {pr_auc_seeded:.4f}")
        print(f"(Used {len(seed_nodes)} seed nodes out of {len(pos_nodes_set)} total positive nodes)")
    else:
        print("PersonalizedPageRank baseline failed (insufficient data or scikit-learn unavailable)")

# Optional: per-window enhanced analysis for a few windows
print("\nPer-window enhanced analysis (first few windows per setting):")
for window_days in WINDOW_DAYS_LIST:
    print(f"\n-- {window_days}-day windows --")
    count = 0
    for ws, we in iter_windows(tmin, tmax, window_days=window_days, stride_days=WINDOW_STRIDE_DAYS):
        # Use optimized windowing
        df_slice = df[(df['timestamp'] >= ws) & (df['timestamp'] < we)]
        H_win = get_windowed_graph_fast(df_slice, G)
        
        if H_win.number_of_edges() == 0:
            continue
        
        derive_node_labels(H_win)
        H_agg = aggregate_graph(H_win, directed=False)
        H_agg_dir = aggregate_graph(H_win, directed=True)
        
        nodes = list(H_agg_dir.nodes())
        y_true_dict = {n: int(H_agg_dir.nodes[n].get('is_laundering_involved', 0)) for n in nodes}
        scores = run_centrality_baselines(H_agg_dir)
        metrics = eval_scores(nodes, y_true_dict, scores, k_fracs=(0.005, 0.01, 0.02))
        print("  Centrality baselines:", {m: {k: round(v,4) if v is not None else None for k, v in d.items()} for m, d in metrics.items()})
        
        # Unsupervised community scoring
        part_win, comms_win = run_louvain(H_agg, resolution=LOUVAIN_RESOLUTION, seed=LOUVAIN_SEED)
        comm_scores_win = score_communities_unsupervised(H_agg, comms_win)
        avg_comm_score = np.mean(list(comm_scores_win.values())) if comm_scores_win else 0
        
        # Semi-supervised PageRank baseline
        pos_nodes_win = {n for n, d in H_agg_dir.nodes(data=True) if int(d.get('is_laundering_involved', 0)) == 1}
        pr_auc_win = None
        if len(pos_nodes_win) > 0:
            seed_size_win = max(1, len(pos_nodes_win) // 5)
            seed_nodes_win = set(sorted(pos_nodes_win)[:seed_size_win])
            pr_auc_win = run_seeded_pagerank_baseline(H_agg_dir, seed_nodes_win)
        
        print(f"[{ws:%Y-%m-%d} → {we:%Y-%m-%d}] nodes={H_agg.number_of_nodes():,}, edges={H_agg.number_of_edges():,}")
        print(f"  Avg community score: {avg_comm_score:.4f}")
        if pr_auc_win is not None:
            print(f"  PersonalizedPageRank PR-AUC: {pr_auc_win:.4f}")
        
        count += 1
        if MAX_WINDOWS_PER_SETTING is not None and count >= MAX_WINDOWS_PER_SETTING:
            break
