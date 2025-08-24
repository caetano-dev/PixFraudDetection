# evaluate_community_baseline.py
import os
from pathlib import Path
from datetime import timedelta
import numpy as np
import pandas as pd
import networkx as nx

# Optional metrics
try:
    from sklearn.metrics import average_precision_score
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False

# Optional Louvain from NetworkX (>=3.0) or python-louvain fallback
try:
    from networkx.algorithms.community import louvain_communities as nx_louvain_communities
except Exception:
    nx_louvain_communities = None

# -----------------------
# Config
# -----------------------
PROC = Path("data/processed")
P_NORM = PROC / "1_filtered_normal_transactions.csv"
P_POS = PROC / "2_filtered_laundering_transactions.csv"
P_ACCT = PROC / "3_filtered_accounts.csv"
P_GPICKLE = PROC / "G_all_multi.gpickle"
RESULTS_CSV = PROC / "community_baseline_results.csv"

# Grid for evaluation
WINDOW_SETTINGS = [
    {"name": "full", "days": None, "stride": None, "limit": 1},
    {"name": "3d", "days": 3, "stride": 1, "limit": 5},
    {"name": "7d", "days": 7, "stride": 1, "limit": 5},
]
DIRECTED_OPTIONS = [False, True]
WEIGHT_OPTIONS = ["w_amount", "w_count"]        # amount-weighted vs count-weighted
RESOLUTIONS = [0.8, 1.0, 1.2]                   # Louvain resolution sweep
TOPK_FRACS = [0.01, 0.05, 0.10]                 # 1%, 5%, 10%

# -----------------------
# IO helpers
# -----------------------
def save_graph_gpickle(G, path):
    from networkx.readwrite.gpickle import write_gpickle
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_gpickle(G, str(path))

def load_graph_gpickle(path):
    try:
        from networkx.readwrite.gpickle import read_gpickle
        return read_gpickle(str(path))
    except Exception as e:
        print(f"Failed to load gpickle: {e}")
        return None

# -----------------------
# Graph builders
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

def build_full_graph_from_csv():
    # Load processed transactions
    df_n = pd.read_csv(P_NORM, parse_dates=['timestamp'], low_memory=False)
    df_p = pd.read_csv(P_POS, parse_dates=['timestamp'], low_memory=False)
    df = pd.concat([df_n, df_p], ignore_index=True)
    df.sort_values('timestamp', inplace=True)
    df['is_laundering'] = pd.to_numeric(df['is_laundering'], errors='coerce').fillna(0).astype('int8')
    df['amount_sent_c'] = to_cents(df['amount_sent'])
    df['amount_received_c'] = to_cents(df['amount_received'])
    df['same_bank'] = (df['from_bank'].astype(str) == df['to_bank'].astype(str))

    # Accounts
    acct = pd.read_csv(P_ACCT, low_memory=False,
                       dtype={'bank_id':'string','account_id_hex':'string','entity_id':'string'}) \
               .drop_duplicates(subset=['account_id_hex']).set_index('account_id_hex')

    # Build MultiDiGraph
    G = nx.MultiDiGraph(name='G_all_multi')
    # Add nodes
    for acc_id, row in acct.iterrows():
        G.add_node(
            str(acc_id),
            bank_id=str(row.get('bank_id', '')),
            entity_id=str(row.get('entity_id', '')),
            entity_name=str(row.get('entity_name', ''))
        )
    # Ensure missing nodes exist
    for col in ['from_account', 'to_account']:
        missing = set(df[col].astype(str)) - set(G.nodes)
        for acc_id in missing:
            G.add_node(str(acc_id))
    # Add edges
    for idx, r in df.iterrows():
        u = str(r['from_account']); v = str(r['to_account'])
        ts = r['timestamp']; t_ns = int(ts.value) if pd.notna(ts) else None
        attempt_id = r.get('attempt_id', None)
        attempt_type = r.get('attempt_type', None)
        if pd.isna(attempt_id): attempt_id = None
        if pd.isna(attempt_type): attempt_type = 'UNLISTED'
        G.add_edge(
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
    return G

def derive_node_labels(G: nx.Graph) -> int:
    nx.set_node_attributes(G, 0, 'is_laundering_involved')
    for u, v, d in G.edges(data=True):
        if int(d.get('is_laundering', 0)) == 1:
            G.nodes[u]['is_laundering_involved'] = 1
            G.nodes[v]['is_laundering_involved'] = 1
    return sum(d.get('is_laundering_involved', 0) for _, d in G.nodes(data=True))

def graph_time_range(G: nx.MultiDiGraph):
    ts = [d['timestamp'] for _,_,d in G.edges(data=True) if d.get('timestamp') is not None]
    return (min(ts), max(ts)) if ts else (None, None)

def iter_windows(start, end, window_days=3, stride_days=1):
    cur = start
    while cur < end:
        yield (cur, cur + timedelta(days=window_days))
        cur += timedelta(days=stride_days)

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

def run_louvain(H: nx.Graph, resolution=1.0, seed=42, weight='w_amount'):
    # Returns partition dict[node] -> community_id, and list of communities
    if nx_louvain_communities is not None:
        comms = nx_louvain_communities(H, weight=weight, resolution=resolution, seed=seed)
        partition = {}
        for cid, c in enumerate(comms):
            for n in c:
                partition[n] = cid
        return partition, [set(c) for c in comms]
    else:
        try:
            import community as community_louvain  # python-louvain
        except Exception:
            raise RuntimeError("Louvain unavailable: install networkx>=3.0 or `pip install python-louvain`.")
        # python-louvain doesn't accept 'weight' name directly in best_partition; it reads from 'weight' edge attribute name 'weight'
        # To support arbitrary weight names, clone a view with expected 'weight' attr:
        H_w = nx.Graph() if isinstance(H, nx.Graph) and not isinstance(H, nx.DiGraph) else nx.DiGraph()
        H_w.add_nodes_from(H.nodes(data=True))
        for u, v, d in H.edges(data=True):
            w = float(d.get(weight, 1.0))
            H_w.add_edge(u, v, weight=w)
        part = community_louvain.best_partition(H_w, random_state=seed, resolution=resolution)
        comms = {}
        for n, cid in part.items():
            comms.setdefault(cid, set()).add(n)
        return part, list(comms.values())

# -----------------------
# Scoring
# -----------------------
def community_node_scores(H_agg: nx.Graph, partition: dict) -> tuple[np.ndarray, np.ndarray]:
    # Node label must be present on H_agg (copied from window MultiDiGraph before aggregation)
    pos_nodes = {n for n, d in H_agg.nodes(data=True) if int(d.get('is_laundering_involved', 0)) == 1}
    # illicit fraction per community
    comm_nodes = {}
    for n, cid in partition.items():
        comm_nodes.setdefault(cid, set()).add(n)
    comm_frac = {}
    for cid, nodes in comm_nodes.items():
        k = len(nodes)
        p = len(nodes & pos_nodes)
        comm_frac[cid] = (p / k) if k else 0.0
    nodes = list(H_agg.nodes())
    y_true = np.array([1 if n in pos_nodes else 0 for n in nodes], dtype=int)
    y_score = np.array([comm_frac.get(partition.get(n, -1), 0.0) for n in nodes], dtype=float)
    return y_true, y_score

def topk_metrics(y_true: np.ndarray, y_score: np.ndarray, fracs=(0.01, 0.05, 0.10)):
    n = len(y_true)
    order = np.argsort(-y_score)
    results = {}
    for f in fracs:
        k = max(1, int(np.ceil(f * n)))
        sel = order[:k]
        tp = int(y_true[sel].sum())
        prec = tp / k
        rec = tp / int(y_true.sum() if y_true.sum() > 0 else 1)
        results[f'precision@{int(f*100)}%'] = prec
        results[f'recall@{int(f*100)}%'] = rec
    return results

# -----------------------
# Main evaluation harness
# -----------------------
def main():
    # Load graph
    if P_GPICKLE.exists():
        G = load_graph_gpickle(P_GPICKLE)
        if G is None:
            print("Falling back to building from CSVs...")
            G = build_full_graph_from_csv()
    else:
        G = build_full_graph_from_csv()

    # Derive full-period labels once
    derive_node_labels(G)
    tmin, tmax = graph_time_range(G)
    print(f"G_all_multi: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges, time: {tmin} → {tmax}")

    records = []

    for wset in WINDOW_SETTINGS:
        name = wset["name"]
        days = wset["days"]
        stride = wset["stride"]
        limit = wset["limit"]

        # Build the list of windowed MultiDiGraphs to evaluate
        windows = []
        if days is None:
            windows = [(None, None, G)]  # full graph as a single "window"
        else:
            count = 0
            for ws, we in iter_windows(tmin, tmax, window_days=days, stride_days=stride):
                H_win = get_windowed_graph(G, ws, we)
                if H_win.number_of_edges() == 0:
                    continue
                derive_node_labels(H_win)  # window-specific labels
                windows.append((ws, we, H_win))
                count += 1
                if limit is not None and count >= limit:
                    break

        for directed in DIRECTED_OPTIONS:
            for weight in WEIGHT_OPTIONS:
                for res in RESOLUTIONS:
                    for (ws, we, H_win) in windows:
                        # Aggregate
                        H_agg = aggregate_graph(H_win, directed=directed)
                        if H_agg.number_of_edges() == 0:
                            continue

                        # Community detection
                        try:
                            partition, comms = run_louvain(H_agg, resolution=res, seed=42, weight=weight)
                        except RuntimeError as e:
                            print(f"Skipping Louvain (missing dependency): {e}")
                            continue

                        # Scores and metrics
                        y_true, y_score = community_node_scores(H_agg, partition)
                        ap = float(average_precision_score(y_true, y_score)) if SKLEARN_OK else np.nan
                        topk = topk_metrics(y_true, y_score, TOPK_FRACS)

                        rec = {
                            "window": name,
                            "start": ws.strftime("%Y-%m-%d") if ws else "full",
                            "end": we.strftime("%Y-%m-%d") if we else "full",
                            "directed": directed,
                            "weight": weight,
                            "resolution": res,
                            "nodes": int(H_agg.number_of_nodes()),
                            "edges": int(H_agg.number_of_edges()),
                            "positives": int(y_true.sum()),
                            "ap": ap,
                        }
                        rec.update(topk)
                        records.append(rec)
                        print(f"[{name} {('full' if ws is None else ws.strftime('%Y-%m-%d'))} -> "
                              f"{('full' if we is None else we.strftime('%Y-%m-%d'))}] "
                              f"dir={directed} w={weight} res={res:.1f} "
                              f"nodes={rec['nodes']:,} edges={rec['edges']:,} AP={ap:.4f}")

    # Save results
    res_df = pd.DataFrame.from_records(records)
    res_df.sort_values(["window", "start", "weight", "directed", "resolution"], inplace=True)
    PROC.mkdir(parents=True, exist_ok=True)
    res_df.to_csv(RESULTS_CSV, index=False)
    print(f"\nSaved results to {RESULTS_CSV} (rows={len(res_df):,})")

    # Optional: per-typology recall@1% on full period
    try:
        # Build aggregated full graph for the best of res=1.0, undirected, amount
        H_full = aggregate_graph(G, directed=False)
        partition, _ = run_louvain(H_full, resolution=1.0, seed=42, weight='w_amount')
        _, y_score = community_node_scores(H_full, partition)
        nodes = np.array(list(H_full.nodes()))
        order = np.argsort(-y_score)
        k = max(1, int(0.01 * len(nodes)))
        top_nodes = set(nodes[order[:k]])

        # Build node->typologies map from edge attributes in G (full period)
        node_typologies = {}
        for u, v, d in G.edges(data=True):
            t = d.get('attempt_type', 'UNLISTED')
            aid = d.get('attempt_id', None)
            if t and t != 'UNLISTED' and aid is not None:
                node_typologies.setdefault(u, set()).add(t)
                node_typologies.setdefault(v, set()).add(t)

        # Compute recall@1% per typology (node-level)
        rows = []
        by_typ = {}
        for n, tys in node_typologies.items():
            for t in tys:
                by_typ.setdefault(t, set()).add(n)

        for t, nodes_set in sorted(by_typ.items(), key=lambda x: -len(x[1]))[:15]:
            denom = len(nodes_set)
            hit = len(nodes_set & top_nodes)
            rec = hit / denom if denom else 0.0
            rows.append({"attempt_type": t, "nodes": denom, "recall@1%": rec})

        if rows:
            typ_df = pd.DataFrame(rows)
            print("\nPer-typology recall@1% (node-level, full period, top 15 by size):")
            print(typ_df.to_string(index=False))
    except Exception as e:
        print(f"Typology recall step skipped: {e}")

if __name__ == "__main__":
    main()
