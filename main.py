import numpy as np
import pandas as pd
import datetime
from pathlib import Path
from datetime import timedelta
from collections import OrderedDict
from scipy.interpolate import NearestNDInterpolator
import gc

from google.colab import drive
import graph_tool.all as gt
import igraph as ig
import leidenalg as la
from sklearn.metrics import average_precision_score

import sys
from io import StringIO

class Tee:
    """Write to both stdout and a string buffer simultaneously"""
    def __init__(self, *files):
        self.files = files

    def write(self, data):
        for f in self.files:
            f.write(data)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()

def to_cents(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors='coerce').mul(100).round().astype('Int64')

def init_base_props(G: gt.Graph):
    # Vertex props
    G.vp['name']    = G.new_vertex_property('string')
    G.vp['bank_id'] = G.new_vertex_property('string')
    G.vp['entity_id'] = G.new_vertex_property('string')
    G.vp['entity_name'] = G.new_vertex_property('string')
    G.vp['is_laundering_involved'] = G.new_vertex_property('int32_t', vals=0)
    # Edge props for aggregated graphs
    G.ep['w_count'] = G.new_edge_property('int64_t', vals=0)
    G.ep['w_amount'] = G.new_edge_property('int64_t', vals=0)
    if G.is_directed():
      G.ep['w_amount_sent'] = G.new_edge_property('int64_t', vals=0)
    G.ep['w_amount_log'] = G.new_edge_property('double', vals=0.0)

def init_agg_vertex_props(G: gt.Graph):
    G.vp['in_amount_sum'] = G.new_vertex_property('int64_t', vals=0)
    G.vp['out_amount_sum'] = G.new_vertex_property('int64_t', vals=0)
    G.vp['in_deg'] = G.new_vertex_property('int32_t', vals=0)
    G.vp['out_deg'] = G.new_vertex_property('int32_t', vals=0)
    G.vp['in_tx_count'] = G.new_vertex_property('int64_t', vals=0)
    G.vp['out_tx_count'] = G.new_vertex_property('int64_t', vals=0)
    G.vp['in_out_amount_ratio'] = G.new_vertex_property('double', vals=0.0)

def window_stats(df_slice: pd.DataFrame, exclude_nodes: set | None = None):
    """
    Fast degeneracy check using df_slice only.
    Returns:
      total_nodes, pos_nodes, neg_nodes, eval_nodes, eval_pos_nodes, eval_neg_nodes
    """
    if df_slice is None or len(df_slice) == 0:
        return 0, 0, 0, 0, 0, 0

    u = df_slice['from_account'].astype(str).to_numpy(copy=False)
    v = df_slice['to_account'].astype(str).to_numpy(copy=False)
    all_nodes = pd.unique(np.concatenate([u, v]))

    pos_mask = (df_slice['is_laundering'] == 1)
    if pos_mask.any():
        up = df_slice.loc[pos_mask, 'from_account'].astype(str).to_numpy(copy=False)
        vp = df_slice.loc[pos_mask, 'to_account'].astype(str).to_numpy(copy=False)
        pos_nodes = pd.unique(np.concatenate([up, vp]))
    else:
        pos_nodes = np.array([], dtype=all_nodes.dtype)

    total = int(all_nodes.size)
    pos = int(pos_nodes.size)
    neg = total - pos

    if exclude_nodes:
        excl = set(exclude_nodes)
        eval_nodes_set = set(all_nodes) - excl
        eval_pos_set   = set(pos_nodes) - excl
        eval_nodes = len(eval_nodes_set)
        eval_pos_nodes = len(eval_pos_set)
        eval_neg_nodes = eval_nodes - eval_pos_nodes
    else:
        eval_nodes = total
        eval_pos_nodes = pos
        eval_neg_nodes = neg

    return total, pos, neg, eval_nodes, eval_pos_nodes, eval_neg_nodes

def load_processed(proc):
    print("Loading processed data...")
    p_norm = proc / '1_filtered_normal_transactions.parquet'
    p_pos  = proc / '2_filtered_laundering_transactions.parquet'
    p_acct = proc / '3_filtered_accounts.parquet'
    df_n = pd.read_parquet(p_norm)
    df_p = pd.read_parquet(p_pos)
    df = pd.concat([df_n, df_p], ignore_index=True)

    # Ensure timestamp is datetime and handle potential errors
    if not np.issubdtype(df['timestamp'].dtype, np.datetime64):
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df = df.dropna(subset=['timestamp'])
    df.sort_values('timestamp', inplace=True)

    # Clean and engineer fields
    df['is_laundering'] = pd.to_numeric(df['is_laundering'], errors='coerce').fillna(0).astype('int8')
    df['amount_sent_c'] = to_cents(df['amount_sent'])
    df['amount_received_c'] = to_cents(df['amount_received'])
    df['same_bank'] = (df['from_bank'].astype(str) == df['to_bank'].astype(str))
    # Ensure string ids
    df['from_account'] = df['from_account'].astype(str)
    df['to_account'] = df['to_account'].astype(str)

    acct = pd.read_parquet(p_acct).drop_duplicates(subset=['account_id_hex'])
    acct['account_id_hex'] = acct['account_id_hex'].astype(str)
    acct.set_index('account_id_hex', inplace=True)
    return df, acct

def build_all_light(proc):
    df, acct = load_processed(proc)
    tmin, tmax = (df['timestamp'].min(), df['timestamp'].max())
    print(f"Loaded: {len(df):,} tx; accounts: {len(acct):,}")
    print(f"Time range: {tmin} → {tmax}")
    return df, acct, tmin, tmax

def iter_window_indices(ts, start, end, window_days=3, stride_days=1):
    cur = np.datetime64(start)
    end64 = np.datetime64(end)
    step = np.timedelta64(stride_days, 'D')
    wdur = np.timedelta64(window_days, 'D')
    while cur < end64:
        ws, we = cur, cur + wdur
        i0 = ts.searchsorted(ws, side='left')
        i1 = ts.searchsorted(we, side='left')
        yield i0, i1, pd.Timestamp(ws), pd.Timestamp(we)
        cur = cur + step

def aggregate_graphs(df_slice: pd.DataFrame, acct: pd.DataFrame, mode='both', include_reciprocated=False):
    """
    mode: 'both' | 'directed' | 'undirected'
    Returns: (H_und, H_dir) where one can be None depending on mode.
    """
    # Empty fallback
    if df_slice is None or len(df_slice) == 0:
        H_und = gt.Graph(directed=False); init_base_props(H_und); init_agg_vertex_props(H_und)
        H_dir = gt.Graph(directed=True);  init_base_props(H_dir); init_agg_vertex_props(H_dir)
        return (H_und if mode in ('both', 'undirected') else None,
                H_dir if mode in ('both', 'directed') else None)

    # Raw arrays
    u_str = df_slice['from_account'].astype(str).to_numpy(copy=False)
    v_str = df_slice['to_account'].astype(str).to_numpy(copy=False)
    recv  = pd.to_numeric(df_slice['amount_received_c'], errors='coerce').fillna(0).astype(np.int64).to_numpy(copy=False)
    sent  = pd.to_numeric(df_slice['amount_sent_c'],   errors='coerce').fillna(0).astype(np.int64).to_numpy(copy=False)

    # Factorize once
    all_ids = np.concatenate([u_str, v_str])
    codes, uniques = pd.factorize(all_ids, sort=False)
    n = len(uniques)
    u = codes[:len(u_str)].astype(np.int64, copy=False)
    v = codes[len(u_str):].astype(np.int64, copy=False)
    names_list = [str(x) for x in uniques]

    # Seeds for node labels (laundering involvement)
    pos_mask = (df_slice['is_laundering'] == 1).to_numpy(dtype=bool, copy=False)
    arr_involv = np.zeros(n, dtype=np.int32)
    if pos_mask.any():
        pos_codes = np.unique(np.concatenate([u[pos_mask], v[pos_mask]]))
        arr_involv[pos_codes] = 1

    sub_acct = acct.reindex(names_list)  # account metadata aligned to names_list order

    # Helper to assign vertex properties
    def set_vertex_props(G):
        name = G.vp['name']; bank = G.vp['bank_id']; entid = G.vp['entity_id']; entname = G.vp['entity_name']
        for i in range(n):
            vi = G.vertex(i)
            name[vi] = names_list[i]
            r = sub_acct.iloc[i]  # aligned by reindex
            bank_val   = r['bank_id'] if pd.notna(r.get('bank_id')) else ''
            entid_val  = r['entity_id'] if pd.notna(r.get('entity_id')) else ''
            entname_val= r['entity_name'] if pd.notna(r.get('entity_name')) else ''
            bank[vi] = str(bank_val); entid[vi] = str(entid_val); entname[vi] = str(entname_val)

    H_und = H_dir = None
    agg_dir = agg_und = None

    need_dir = (mode in ('both', 'directed'))
    need_und = (mode in ('both', 'undirected'))

    # Directed aggregation if needed or if we want to derive undirected from it
    if need_dir or mode == 'both':
        tmp = pd.DataFrame({'u': u, 'v': v, 'w_recv': recv, 'w_sent': sent})
        agg_dir = (tmp.groupby(['u','v'], sort=False, observed=False)
                    .agg(w_amount_recv=('w_recv','sum'),
                         w_amount_sent=('w_sent','sum'),
                         w_count=('u','size'))
                    .reset_index())
        agg_dir['w_amount'] = agg_dir['w_amount_recv']
        agg_dir['w_amount_log'] = np.log1p(agg_dir['w_amount']).astype(np.float64)
        if include_reciprocated and need_dir:
            edge_idx = pd.MultiIndex.from_frame(agg_dir[['u','v']])
            rev_idx  = pd.MultiIndex.from_frame(agg_dir[['v','u']])
            agg_dir['reciprocated'] = edge_idx.isin(rev_idx).astype(np.int8)

    # Undirected aggregation
    if need_und:
        if agg_dir is not None:
            a = np.minimum(agg_dir['u'].to_numpy(np.int64), agg_dir['v'].to_numpy(np.int64))
            b = np.maximum(agg_dir['u'].to_numpy(np.int64), agg_dir['v'].to_numpy(np.int64))
            tmp_ud = pd.DataFrame({'a': a, 'b': b,
                                   'w_amount': agg_dir['w_amount'].to_numpy(np.int64),
                                   'w_count':  agg_dir['w_count'].to_numpy(np.int64)})
            agg_und = (tmp_ud.groupby(['a','b'], sort=False, observed=False)
                             .agg(w_amount=('w_amount','sum'),
                                  w_count=('w_count','sum')).reset_index())
            agg_und['w_amount_log'] = np.log1p(agg_und['w_amount']).astype(np.float64)
        else:
            # No directed agg computed (mode == 'undirected'): do direct undirected groupby
            a = np.minimum(u, v); b = np.maximum(u, v)
            tmp_ud = pd.DataFrame({'a': a, 'b': b, 'amt': recv})
            sum_df = tmp_ud.groupby(['a','b'], sort=False, observed=False)['amt'].sum().rename('w_amount')
            cnt_df = tmp_ud.groupby(['a','b'], sort=False, observed=False).size().rename('w_count')
            agg_und = pd.concat([sum_df, cnt_df], axis=1).reset_index()
            agg_und['w_amount_log'] = np.log1p(agg_und['w_amount']).astype(np.float64)

    # Build graphs requested
    if need_und:
        H_und = gt.Graph(directed=False); init_base_props(H_und); init_agg_vertex_props(H_und); H_und.add_vertex(n)
        set_vertex_props(H_und)
        H_und.vp['is_laundering_involved'].a = arr_involv
        # Add edges
        edge_tbl_und = np.column_stack([
            agg_und['a'].to_numpy(np.int64),
            agg_und['b'].to_numpy(np.int64),
            agg_und['w_count'].to_numpy(np.int64),
            agg_und['w_amount'].to_numpy(np.int64),
            agg_und['w_amount_log'].to_numpy(np.float64),
        ])
        H_und.add_edge_list(edge_tbl_und, eprops=[H_und.ep['w_count'], H_und.ep['w_amount'], H_und.ep['w_amount_log']])
        # Vertex aggregates (undirected: in == out)
        a_idx = agg_und['a'].to_numpy(np.int64); b_idx = agg_und['b'].to_numpy(np.int64)
        amt   = agg_und['w_amount'].to_numpy(np.int64)
        cnt   = agg_und['w_count'].to_numpy(np.int64)
        deg_u = np.zeros(n, dtype=np.int32); amt_u = np.zeros(n, dtype=np.int64); tx_u = np.zeros(n, dtype=np.int64)
        np.add.at(deg_u, a_idx, 1); np.add.at(deg_u, b_idx, 1)
        np.add.at(amt_u, a_idx, amt); np.add.at(amt_u, b_idx, amt)
        np.add.at(tx_u,  a_idx, cnt); np.add.at(tx_u,  b_idx, cnt)
        H_und.vp['in_deg'].a = deg_u; H_und.vp['out_deg'].a = deg_u
        H_und.vp['in_amount_sum'].a = amt_u; H_und.vp['out_amount_sum'].a = amt_u
        H_und.vp['in_tx_count'].a = tx_u; H_und.vp['out_tx_count'].a = tx_u
        H_und.vp['in_out_amount_ratio'].a = np.ones(n, dtype=np.float64)

    if need_dir:
        H_dir = gt.Graph(directed=True); init_base_props(H_dir); init_agg_vertex_props(H_dir); H_dir.add_vertex(n)
        set_vertex_props(H_dir)
        H_dir.vp['is_laundering_involved'].a = arr_involv
        # Add edges
        cols = [
            agg_dir['u'].to_numpy(np.int64),
            agg_dir['v'].to_numpy(np.int64),
            agg_dir['w_count'].to_numpy(np.int64),
            agg_dir['w_amount'].to_numpy(np.int64),
            agg_dir['w_amount_sent'].to_numpy(np.int64),
            agg_dir['w_amount_log'].to_numpy(np.float64),
        ]
        edge_tbl_dir = np.column_stack(cols)
        eprops_dir = [H_dir.ep['w_count'], H_dir.ep['w_amount'], H_dir.ep['w_amount_sent'], H_dir.ep['w_amount_log']]
        if include_reciprocated and 'reciprocated' in agg_dir.columns:
            edge_tbl_dir = np.column_stack([edge_tbl_dir, agg_dir['reciprocated'].to_numpy(np.int8)])
            H_dir.ep['reciprocated'] = H_dir.new_edge_property('int8_t', vals=0)
            eprops_dir.append(H_dir.ep['reciprocated'])
        H_dir.add_edge_list(edge_tbl_dir, eprops=eprops_dir)
        # Vertex aggregates (directed)
        u_idx = agg_dir['u'].to_numpy(np.int64); v_idx = agg_dir['v'].to_numpy(np.int64)
        recv_d  = agg_dir['w_amount_recv'].to_numpy(np.int64)
        sent_d  = agg_dir['w_amount_sent'].to_numpy(np.int64)
        cnt_d   = agg_dir['w_count'].to_numpy(np.int64)
        in_amt  = np.zeros(n, dtype=np.int64); out_amt = np.zeros(n, dtype=np.int64)
        in_tx   = np.zeros(n, dtype=np.int64); out_tx  = np.zeros(n, dtype=np.int64)
        in_deg  = np.zeros(n, dtype=np.int32); out_deg = np.zeros(n, dtype=np.int32)
        np.add.at(in_amt,  v_idx, recv_d); np.add.at(out_amt, u_idx, sent_d)
        np.add.at(in_tx,   v_idx, cnt_d);  np.add.at(out_tx,  u_idx, cnt_d)
        np.add.at(in_deg,  v_idx, 1);      np.add.at(out_deg, u_idx, 1)
        H_dir.vp['in_amount_sum'].a  = in_amt;  H_dir.vp['out_amount_sum'].a = out_amt
        H_dir.vp['in_tx_count'].a    = in_tx;   H_dir.vp['out_tx_count'].a   = out_tx
        H_dir.vp['in_deg'].a         = in_deg;  H_dir.vp['out_deg'].a        = out_deg
        H_dir.vp['in_out_amount_ratio'].a = (in_amt + 1.0) / (out_amt + 1.0)

    return H_und, H_dir

def to_igraph(H: gt.Graph, use_weight=False, weight_name='w_amount_log', include_amount=True, include_amount_sent=False):
    n = H.num_vertices()

    attrs = []
    if use_weight and (weight_name in H.ep):
        attrs.append(H.ep[weight_name])
    if include_amount and ('w_amount' in H.ep):
        attrs.append(H.ep['w_amount'])
    if include_amount_sent and H.is_directed() and ('w_amount_sent' in H.ep):
        attrs.append(H.ep['w_amount_sent'])

    if attrs:
        ed = H.get_edges(eprops=attrs)
        edges = [(int(a), int(b)) for a, b in ed[:, :2].astype(int)]
        col = 2
        w = amt = amt_sent = None
        if use_weight and (weight_name in H.ep):
            w = [float(x) for x in ed[:, col].astype(float)]; col += 1
        if include_amount and ('w_amount' in H.ep):
            amt = [float(x) for x in ed[:, col].astype(float)]; col += 1
        if include_amount_sent and H.is_directed() and ('w_amount_sent' in H.ep):
            amt_sent = [float(x) for x in ed[:, col].astype(float)]; col += 1
    else:
        ed = H.get_edges()
        edges = [(int(a), int(b)) for a, b in ed.astype(int)]
        w = amt = amt_sent = None

    g = ig.Graph(n=n, edges=edges, directed=H.is_directed())
    if w is not None:
        g.es['weight'] = w
    if amt is not None:
        g.es['amount'] = amt
    if amt_sent is not None:
        g.es['amount_sent'] = amt_sent

    g.vs['name'] = [H.vp['name'][H.vertex(i)] for i in range(n)]
    return g

def get_attempt_nodes_map_df(df_slice: pd.DataFrame) -> dict:
    if df_slice is None or len(df_slice) == 0:
        return {}
    pos = df_slice['is_laundering'] == 1
    dfp = df_slice.loc[pos, ['attempt_id', 'from_account', 'to_account']].dropna(subset=['attempt_id'])
    if dfp.empty:
        return {}
    dfp['attempt_id'] = dfp['attempt_id'].astype(str)
    g = dfp.groupby('attempt_id')
    att_nodes = {}
    for att_id, grp in g:
        att_nodes[att_id] = set(grp['from_account'].astype(str)).union(set(grp['to_account'].astype(str)))
    return att_nodes

def precision_at_k(y_true, y_score, k_frac=0.01):
    y_true = np.asarray(y_true); y_score = np.asarray(y_score)
    n = max(1, int(len(y_true) * k_frac))
    idx = np.argsort(-y_score)[:n]
    return float(y_true[idx].mean())

def get_node_names(G: gt.Graph):
    return [G.vp['name'][v] for v in G.vertices()]

def vprop_to_dict(G: gt.Graph, prop_name: str):
    prop = G.vp[prop_name]
    name = G.vp['name']
    return {name[v]: prop[v] for v in G.vertices()}

def eval_scores(nodes, y_true_dict, score_dict, k_fracs=(0.005, 0.01, 0.02), exclude_nodes=None):
    if exclude_nodes is None: exclude_nodes = set()
    eval_nodes = [n for n in nodes if n not in exclude_nodes]
    y_true = np.array([int(y_true_dict.get(n, 0)) for n in eval_nodes], dtype=int)
    res = {}
    for name, s in score_dict.items():
        scores = np.array([float(s.get(n, 0.0)) for n in eval_nodes], dtype=float)
        ap = average_precision_score(y_true, scores) if SKLEARN_OK and len(set(y_true)) > 1 else None
        metrics = {'ap': ap}
        metrics['_eval_nodes'] = len(eval_nodes)
        eval_pos_count = int(np.sum(y_true))
        metrics['_eval_pos'] = eval_pos_count

        for k in k_fracs:
            metrics[f"p_at_{pct_key(k)}"] = precision_at_k(y_true, scores, k)
        order = np.argsort(-scores)
        metrics['_ranked_nodes'] = [eval_nodes[i] for i in order]
        res[name] = metrics
    return res

def run_centrality_baselines(H_dir: gt.Graph):
    scores = {}
    w = H_dir.ep.get('w_amount_log', None)
    try:
        pr = gt.pagerank(H_dir, damping=0.9, weight=w)
        names = H_dir.vp['name']
        scores['pagerank_wlog'] = {names[v]: float(pr[v]) for v in H_dir.vertices()}
    except Exception:
        scores['pagerank_wlog'] = {}

    if RUN_HITS:
        try:
            hubs, auth = gt.hits(H_dir, weight=w)
            names = H_dir.vp['name']
            scores['hits_hub'] = {names[v]: float(hubs[v]) for v in H_dir.vertices()}
            scores['hits_auth'] = {names[v]: float(auth[v]) for v in H_dir.vertices()}
        except Exception:
            scores['hits_hub'] = {}; scores['hits_auth'] = {}
    else:
        scores['hits_hub'] = {}; scores['hits_auth'] = {}

    for k_prop, name in [
        ('in_deg', 'in_deg'), ('out_deg', 'out_deg'),
        ('in_tx_count', 'in_tx'), ('out_tx_count', 'out_tx'),
        ('in_amount_sum', 'in_amt'), ('out_amount_sum', 'out_amt'),
    ]:
        try:
            scores[name] = vprop_to_dict(H_dir, k_prop)
        except Exception:
            scores[name] = {}
    try:
        in_amt = H_dir.vp['in_amount_sum']; out_amt = H_dir.vp['out_amount_sum']; names = H_dir.vp['name']
        scores['collector'] = {names[v]: float(in_amt[v]) / (float(out_amt[v]) + 1.0) for v in H_dir.vertices()}
        scores['distributor'] = {names[v]: float(out_amt[v]) / (float(in_amt[v]) + 1.0) for v in H_dir.vertices()}
    except Exception:
        scores['collector'] = {}; scores['distributor'] = {}
    return scores

def run_kcore_baselines(H_und: gt.Graph, H_dir: gt.Graph):
    scores = {}

    # Undirected coreness
    try:
        g_und = to_igraph(H_und, use_weight=False, include_amount=False, include_amount_sent=False)
        c_und = g_und.coreness(mode="ALL")
        names_und = g_und.vs['name']
        scores['kcore_und'] = {names_und[i]: int(c_und[i]) for i in range(len(names_und))}
    except Exception:
        scores['kcore_und'] = {}

    # Directed in/out coreness
    try:
        g_dir = to_igraph(H_dir, use_weight=False, include_amount=False, include_amount_sent=False)
        names_dir = g_dir.vs['name']

        try:
            c_in = g_dir.coreness(mode="in")
            scores['kcore_in'] = {names_dir[i]: int(c_in[i]) for i in range(len(names_dir))}
        except Exception:
            scores['kcore_in'] = {}

        try:
            c_out = g_dir.coreness(mode="out")
            scores['kcore_out'] = {names_dir[i]: int(c_out[i]) for i in range(len(names_dir))}
        except Exception:
            scores['kcore_out'] = {}
    except Exception:
        scores['kcore_in'] = {}
        scores['kcore_out'] = {}

    return scores

def compute_pattern_features(H_dir: gt.Graph, H_und: gt.Graph):
    names = [H_dir.vp['name'][v] for v in H_dir.vertices()]
    n = len(names)

    if n == 0:
        return {}

    in_deg = H_dir.vp['in_deg'].a.astype(float)
    out_deg = H_dir.vp['out_deg'].a.astype(float)
    in_tx = H_dir.vp['in_tx_count'].a.astype(float)
    out_tx = H_dir.vp['out_tx_count'].a.astype(float)

    fan_out = out_deg / (in_deg + 1.0)
    fan_out_norm = fan_out / (fan_out.max() + 1e-6) if fan_out.max() > 0 else np.zeros_like(fan_out)

    fan_in = in_deg / (out_deg + 1.0)
    fan_in_norm = fan_in / (fan_in.max() + 1e-6) if fan_in.max() > 0 else np.zeros_like(fan_in)

    g_dir = to_igraph(H_dir, use_weight=False, include_amount=False)

    hub_scores = np.zeros(n)
    try:
        knn_result = g_dir.knn()
        knn = knn_result[0]
        knn_array = np.array(knn, dtype=float)
        knn_mean = np.mean(knn_array)

        if knn_mean > 0.1:
            hub_scores = knn_array / knn_mean
            hub_norm = hub_scores / (hub_scores.max() + 1e-6) if hub_scores.max() > 0 else hub_scores
        else:
            total_deg = in_deg + out_deg
            hub_norm = (total_deg - total_deg.mean()) / (total_deg.std() + 1e-6)
            hub_norm = np.clip(hub_norm, 0, None)
            hub_norm = hub_norm / (hub_norm.max() + 1e-6) if hub_norm.max() > 0 else hub_norm
    except (AttributeError, IndexError, ValueError, TypeError) as e:
        total_deg = in_deg + out_deg
        if total_deg.std() > 0:
            hub_norm = (total_deg - total_deg.mean()) / total_deg.std()
            hub_norm = np.clip(hub_norm, 0, None)
            hub_norm = hub_norm / (hub_norm.max() + 1e-6) if hub_norm.max() > 0 else hub_norm
        else:
            hub_norm = np.zeros(n)
    
    g_und = to_igraph(H_und, use_weight=False, include_amount=False)
    tree_scores = np.zeros(n)

    SAMPLE_THRESHOLD = 10000
    if n > SAMPLE_THRESHOLD:
        sample_size = min(SAMPLE_THRESHOLD, n)
        sample_indices = np.random.choice(n, size=sample_size, replace=False)

        valid_count = 0 
        for i in sample_indices:
            try:
                neighbors = g_und.neighborhood(i, order=2)
                if len(neighbors) > 2:
                    subg = g_und.induced_subgraph(neighbors)
                    nodes_local = subg.vcount()
                    edges_local = subg.ecount()
                    if nodes_local > 1:
                        ideal_edges = nodes_local - 1
                        tree_scores[i] = np.exp(-abs(edges_local - ideal_edges) / nodes_local)
                        if tree_scores[i] > 0: 
                            valid_count += 1
            except:
                tree_scores[i] = 0.0

        if sample_size < n:
            total_deg = in_deg + out_deg
            
            all_indices = np.arange(n)
            non_sampled_mask = np.ones(n, dtype=bool)
            non_sampled_mask[sample_indices] = False
            non_sampled_indices = all_indices[non_sampled_mask]

            valid_mask = tree_scores[sample_indices] > 0
            if valid_mask.sum() > 3:
                try:
                    sample_features = np.column_stack([
                        in_deg[sample_indices][valid_mask],
                        out_deg[sample_indices][valid_mask],
                        total_deg[sample_indices][valid_mask]
                    ])
                    sample_scores = tree_scores[sample_indices][valid_mask]

                    interp = NearestNDInterpolator(sample_features, sample_scores)

                    if len(non_sampled_indices) > 0:
                        non_sampled_features = np.column_stack([
                            in_deg[non_sampled_indices],
                            out_deg[non_sampled_indices],
                            total_deg[non_sampled_indices]
                        ])
                        tree_scores[non_sampled_indices] = interp(non_sampled_features)
                except Exception:
                    max_sampled = tree_scores[sample_indices].max()
                    if max_sampled > 0:
                        mean_sampled_deg = total_deg[sample_indices].mean()
                        deg_ratios = total_deg[non_sampled_indices] / (mean_sampled_deg + 1.0)
                        tree_scores[non_sampled_indices] = max_sampled * np.exp(-np.abs(deg_ratios - 1.0))
            else:
              if valid_count > 0:
                  mean_valid = tree_scores[sample_indices][valid_mask].mean()
                  tree_scores[non_sampled_indices] = mean_valid * 0.5
    else:
        for i in range(n):
            try:
                neighbors = g_und.neighborhood(i, order=2)
                if len(neighbors) > 2:
                    subg = g_und.induced_subgraph(neighbors)
                    nodes_local = subg.vcount()
                    edges_local = subg.ecount()
                    if nodes_local > 1:
                        ideal_edges = nodes_local - 1
                        tree_scores[i] = np.exp(-abs(edges_local - ideal_edges) / nodes_local)
            except:
                tree_scores[i] = 0.0

    tree_scores = np.asarray(tree_scores, dtype=float)
    tree_norm = tree_scores / (tree_scores.max() + 1e-6) if tree_scores.max() > 0 else np.zeros(n)

    tx_velocity = (in_tx + out_tx) / (in_deg + out_deg + 1.0)
    tx_vel_norm = tx_velocity / (tx_velocity.max() + 1e-6) if tx_velocity.max() > 0 else np.zeros_like(tx_velocity)

    asymmetry = np.abs(in_deg - out_deg) / (in_deg + out_deg + 1.0)

    pattern_score = (
        0.25 * fan_out_norm +
        0.25 * fan_in_norm +
        0.20 * hub_norm +
        0.15 * tree_norm +
        0.10 * tx_vel_norm +
        0.05 * asymmetry
    )

    return {names[i]: float(pattern_score[i]) for i in range(n)}

def ensemble_scores(score_dict_list, weights=None):
    if not score_dict_list:
        return {}

    score_dict_list = [sd for sd in score_dict_list if sd]
    if not score_dict_list:
        return {}

    if weights is None:
        weights = [1.0 / len(score_dict_list)] * len(score_dict_list)

    weight_sum = sum(weights)
    if weight_sum > 0:
        weights = [w / weight_sum for w in weights]

    all_nodes = set()
    for sd in score_dict_list:
        all_nodes.update(sd.keys())

    normalized = []
    default_scores = []
    for sd in score_dict_list:
        if not sd:
            normalized.append({})
            default_scores.append(0.0)
            continue
        scores_arr = np.array(list(sd.values()))
        min_s, max_s = scores_arr.min(), scores_arr.max()
        if max_s > min_s:
            norm_sd = {k: (v - min_s) / (max_s - min_s) for k, v in sd.items()}
            default_scores.append(0.0)
        else:
            norm_sd = {k: 0.5 for k in sd.keys()}
            default_scores.append(0.5)
        normalized.append(norm_sd)

    ensemble = {}
    for node in all_nodes:
        score = sum(w * norm.get(node, default) for w, norm, default in zip(weights, normalized, default_scores))
        ensemble[node] = score

    return ensemble

def create_ensemble_methods(score_dict):
    ensembles = {}

    if all(k in score_dict for k in ['in_deg', 'pagerank_wlog', 'in_tx']):
        ensembles['ensemble_top3'] = ensemble_scores([
            score_dict['in_deg'],
            score_dict['pagerank_wlog'],
            score_dict['in_tx']
        ], weights=[0.40, 0.35, 0.25])

    diverse_keys = ['in_deg', 'pagerank_wlog', 'kcore_in', 'in_tx']
    if all(k in score_dict for k in diverse_keys):
        ensembles['ensemble_diverse'] = ensemble_scores([
            score_dict[k] for k in diverse_keys
        ], weights=[0.30, 0.30, 0.20, 0.20])

    if 'pattern_features' in score_dict:
        pattern_keys = ['in_deg', 'pagerank_wlog', 'pattern_features']
        if all(k in score_dict for k in pattern_keys):
            ensembles['ensemble_pattern'] = ensemble_scores([
                score_dict[k] for k in pattern_keys
            ], weights=[0.40, 0.35, 0.25])

    ultimate_keys = ['in_deg', 'pagerank_wlog', 'in_tx', 'kcore_in']
    if 'pattern_features' in score_dict:
        ultimate_keys.append('pattern_features')
    if all(k in score_dict for k in ultimate_keys):
        if len(ultimate_keys) == 5:
            weights = [0.25, 0.25, 0.20, 0.15, 0.15]
        else:
            weights = [0.30, 0.30, 0.20, 0.20]
        ensembles['ensemble_ultimate'] = ensemble_scores([
            score_dict[k] for k in ultimate_keys
        ], weights=weights)

    if 'seeded_pr' in score_dict and score_dict.get('seeded_pr'):
        seeded_keys = ['in_deg', 'pagerank_wlog', 'seeded_pr']
        if 'pattern_features' in score_dict:
            seeded_keys.append('pattern_features')
        if all(k in score_dict for k in seeded_keys):
            if len(seeded_keys) == 4:
                weights = [0.35, 0.30, 0.20, 0.15]
            else:
                weights = [0.45, 0.35, 0.20]
            ensembles['ensemble_seeded'] = ensemble_scores([
                score_dict[k] for k in seeded_keys
            ], weights=weights)

    return ensembles

def membership_to_comms(membership, names):
    k = max(membership) + 1 if membership else 0
    comms = [set() for _ in range(k)]
    for i, cid in enumerate(membership):
        comms[cid].add(names[i])
    return comms

def score_communities_for_laundering(g, membership, min_size=3, amount_attr='amount'):
    n = g.vcount()
    if n == 0:
        return {}

    memb = np.asarray(membership, dtype=np.int64)
    K = int(memb.max()) + 1 if memb.size else 0
    size_by_c = np.bincount(memb, minlength=K)

    E = np.array(g.get_edgelist(), dtype=np.int64)
    if E.size == 0:
        return {cid: 0.0 for cid in range(K)}

    has_amount = amount_attr in g.es.attributes()
    w = np.asarray(g.es[amount_attr], dtype=float) if has_amount else np.ones(E.shape[0], dtype=float)

    cid_u = memb[E[:, 0]]
    cid_v = memb[E[:, 1]]
    mask_intra = (cid_u == cid_v)

    e_intra = np.bincount(cid_u[mask_intra], minlength=K)
    amt_intra = np.bincount(cid_u[mask_intra], weights=w[mask_intra], minlength=K)

    degrees = np.array(g.degree())

    scores = {}
    for cid in range(K):
        n_c = int(size_by_c[cid])
        if n_c < min_size:
            scores[cid] = 0.0
            continue

        comm_nodes = np.where(memb == cid)[0]

        ideal_tree_edges = n_c - 1
        actual_edges = e_intra[cid]
        if actual_edges == 0:
            tree_score = 0.0
        else:
            tree_score = np.exp(-abs(actual_edges - ideal_tree_edges) / max(1, n_c))

        max_edges = n_c * (n_c - 1) / 2.0
        density = float(e_intra[cid]) / max_edges if max_edges > 0 else 0.0
        sparsity_score = 1.0 - density

        comm_degrees = degrees[comm_nodes]
        degree_std = np.std(comm_degrees) if len(comm_degrees) > 1 else 0.0
        degree_mean = np.mean(comm_degrees) if len(comm_degrees) > 0 else 0.0
        hub_score = min(1.0, degree_std / (degree_mean + 1.0))

        amount_score = min(1.0, float(np.log1p(amt_intra[cid]) / 20.0))

        optimal_size = 8
        size_score = np.exp(-abs(n_c - optimal_size) / 10.0)

        scores[cid] = (
            0.35 * tree_score +
            0.25 * sparsity_score +
            0.15 * hub_score +
            0.15 * amount_score +
            0.10 * size_score
        )

    return scores

def compute_communities_fast(H_agg, resolution=0.2, seed=42):
    g_und = to_igraph(H_agg, use_weight=(COMMUNITY_WEIGHTED and ('w_amount_log' in H_agg.ep)), include_amount=True, include_amount_sent=False)
    names_und = g_und.vs['name']
    has_weight = 'weight' in g_und.es.attributes()
    has_amount = 'amount' in g_und.es.attributes()
    try:
        ig.random.seed(seed)
    except Exception:
        pass
    np.random.seed(seed)

    cl_louv = g_und.community_multilevel(weights=g_und.es['weight'] if has_weight else None)
    memb_louv = cl_louv.membership
    comms_louv = membership_to_comms(memb_louv, names_und)
    scores_louv = score_communities_for_laundering(
        g_und, memb_louv, min_size=3,
        amount_attr='amount' if has_amount else ('weight' if has_weight else None))
    avg_louv = float(np.mean(list(scores_louv.values()))) if scores_louv else 0.0

    part = la.RBConfigurationVertexPartition(
        g_und,
        weights='weight' if has_weight else None,
        resolution_parameter=resolution
    )
    opt = la.Optimiser()
    opt.set_rng_seed(seed)
    opt.optimise_partition(part)
    memb_leid = part.membership
    comms_leid = membership_to_comms(memb_leid, names_und)
    scores_leid = score_communities_for_laundering(
        g_und, memb_leid, min_size=3,
        amount_attr='amount' if has_amount else ('weight' if has_weight else None))
    avg_leid = float(np.mean(list(scores_leid.values()))) if scores_leid else 0.0

    ranked_cache = {}
    for tag, comms, scores in [('louvain', comms_louv, scores_louv), ('leiden', comms_leid, scores_leid)]:
        if scores:
            comm_order = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            total_nodes = g_und.vcount()
            acc = set()
            for kf in K_FRACS:
                target = max(1, int(total_nodes * kf))
                acc.clear()
                for cid, _score in comm_order:
                    acc |= comms[cid]
                    if len(acc) >= target:
                        break
                ranked_cache[(tag, kf)] = list(acc)

    return {
        'louvain': {'comms': comms_louv, 'scores': scores_louv, 'avg': avg_louv},
        'leiden':  {'comms': comms_leid, 'scores': scores_leid, 'avg': avg_leid},
        'ranked_cache': ranked_cache
    }

def get_seeded_pagerank_scores(H_agg_dir: gt.Graph, seed_nodes: set, weight=None, alpha=None):
    if not seed_nodes:
        return {}

    if PPR_MAX_SEEDS is not None and len(seed_nodes) > PPR_MAX_SEEDS:
        seed_nodes = set(list(sorted(seed_nodes))[:PPR_MAX_SEEDS])

    use_weight = (weight is not None) and (weight in H_agg_dir.ep)

    g = to_igraph(
        H_agg_dir,
        use_weight=use_weight,
        weight_name=weight if use_weight else 'w_amount_log',
        include_amount=False,
        include_amount_sent=False
    )

    seeds_idx = g.vs.select(name_in=list(seed_nodes)).indices
    if len(seeds_idx) == 0:
        return {}

    mode_out = ig.OUT
    keep = set()
    for lst in g.neighborhood(seeds_idx, order=PPR_HOPS, mode=mode_out):
        keep.update(lst)
    if PPR_BIDIR:
        for lst in g.neighborhood(seeds_idx, order=PPR_HOPS, mode=ig.IN):
            keep.update(lst)
    keep.update(seeds_idx)

    keep_idx = list(keep)
    if PPR_MAX_NODES is not None and len(keep_idx) > PPR_MAX_NODES:
        deg = g.degree(keep_idx, mode=ig.ALL)
        order = np.argsort(-np.asarray(deg))
        cap = max(1, PPR_MAX_NODES - len(seeds_idx))
        selected = set(seeds_idx) | {keep_idx[i] for i in order[:cap]}
        keep_idx = list(selected)

    sub = g.induced_subgraph(keep_idx)
    sub_names = sub.vs['name']
    sub_seeds_idx = sub.vs.select(name_in=list(seed_nodes)).indices
    if len(sub_seeds_idx) == 0:
        return {name: 0.0 for name in sub_names}

    reset = np.zeros(sub.vcount(), dtype=float)
    reset[sub_seeds_idx] = 1.0 / len(sub_seeds_idx)

    alpha_eff = PPR_ALPHA if alpha is None else alpha
    weights_key = 'weight' if (use_weight and ('weight' in sub.es.attributes())) else None
    pr = sub.personalized_pagerank(damping=alpha_eff, reset=reset, weights=weights_key, directed=True)
    return {name: float(score) for name, score in zip(sub_names, pr)}

def attempt_coverage(nodes_ranked, attempt_nodes_map: dict, k_frac=0.01):
    if not attempt_nodes_map:
        return None
    N = len(nodes_ranked); k = max(1, int(N * k_frac))
    top = set(nodes_ranked[:k])
    covered = sum(1 for nodes in attempt_nodes_map.values() if top & nodes)
    return covered / max(1, len(attempt_nodes_map))

def pretty_metrics(results: dict):
    def is_num(x):
        return isinstance(x, (int, float, np.integer, np.floating))
    out = {}
    for method, metr in results.items():
        out[method] = {}
        for k, v in metr.items():
            if str(k).startswith('_'):
                continue
            if v is None:
                out[method][k] = None
            elif is_num(v):
                out[method][k] = round(float(v), 4)
            else:
                out[method][k] = v
    return out

def pct_key(kf: float) -> str:
    return f"{kf*100:.1f}pct"

def add_random_baseline(dfm: pd.DataFrame) -> pd.DataFrame:
    cols = list(dfm.columns)
    out_rows = []
    for (wd, ws, we), group in dfm.groupby(['window_days', 'ws', 'we']):
        r = group.iloc[0]
        base = {c: r.get(c, np.nan) for c in cols}
        base['method'] = 'random'
        eval_nodes = int(r.get('eval_nodes', r.get('nodes', 0)) or 0)
        eval_pos = int(r.get('eval_pos_nodes', r.get('pos_nodes', 0)) or 0)
        prev_eval = (eval_pos / eval_nodes) if eval_nodes else np.nan

        base['prevalence_eval'] = prev_eval
        base['ap'] = prev_eval
        for kf in K_FRACS:
            p_at_key = f"p_at_{pct_key(kf)}"
            attcov_key = f"attcov_at_{pct_key(kf)}"
            base[p_at_key] = prev_eval
            base[attcov_key] = kf

        base['eval_nodes'] = eval_nodes
        base['eval_pos_nodes'] = eval_pos

        out_rows.append(base)

    rand_df = pd.DataFrame(out_rows)
    for col in dfm.columns:
        if col not in rand_df.columns:
            rand_df[col] = np.nan
    rand_df = rand_df[cols]
    return pd.concat([dfm, rand_df], ignore_index=True)

def run_analysis(drive_base, window_days_list, stride_days, max_windows, leiden_resolution, louvain_seed, community_weighted, run_hits, run_kcore, run_pattern_features, run_ensembles, metrics_dir, results_csv, eval_exclude_seeds, k_fracs, seed_cutoff_frac, ppr_alpha, ppr_hops, ppr_bidir, ppr_max_nodes, ppr_max_seeds, sklearn_ok):
    global WINDOW_DAYS_LIST, WINDOW_STRIDE_DAYS, MAX_WINDOWS_PER_SETTING, LEIDEN_RESOLUTION, LOUVAIN_SEED, COMMUNITY_WEIGHTED, RUN_HITS, RUN_KCORE, RUN_PATTERN_FEATURES, RUN_ENSEMBLES, METRICS_DIR, RESULTS_CSV, EVAL_EXCLUDE_SEEDS, K_FRACS, SEED_CUTOFF_FRAC, PPR_ALPHA, PPR_HOPS, PPR_BIDIR, PPR_MAX_NODES, PPR_MAX_SEEDS, SKLEARN_OK
    WINDOW_DAYS_LIST = window_days_list
    WINDOW_STRIDE_DAYS = stride_days
    MAX_WINDOWS_PER_SETTING = max_windows
    LEIDEN_RESOLUTION = leiden_resolution
    LOUVAIN_SEED = louvain_seed
    COMMUNITY_WEIGHTED = community_weighted
    RUN_HITS = run_hits
    RUN_KCORE = run_kcore
    RUN_PATTERN_FEATURES = run_pattern_features
    RUN_ENSEMBLES = run_ensembles
    METRICS_DIR = metrics_dir
    RESULTS_CSV = results_csv
    EVAL_EXCLUDE_SEEDS = eval_exclude_seeds
    K_FRACS = k_fracs
    SEED_CUTOFF_FRAC = seed_cutoff_frac
    PPR_ALPHA = ppr_alpha
    PPR_HOPS = ppr_hops
    PPR_BIDIR = ppr_bidir
    PPR_MAX_NODES = ppr_max_nodes
    PPR_MAX_SEEDS = ppr_max_seeds
    SKLEARN_OK = sklearn_ok

    proc = drive_base
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    output_buffer = StringIO()
    original_stdout = sys.stdout
    sys.stdout = Tee(original_stdout, output_buffer)

    df, acct, tmin, tmax = build_all_light(proc)
    ts = df['timestamp'].to_numpy()

    for window_days in WINDOW_DAYS_LIST:
        print(f"\n-- {window_days}-day windows, stride={WINDOW_STRIDE_DAYS}d --")
        for i, (i0, i1, ws, we) in enumerate(iter_window_indices(ts, tmin, tmax, window_days=window_days, stride_days=WINDOW_STRIDE_DAYS)):
            df_slice = df.iloc[i0:i1]
            if df_slice.empty:
                continue
            nodes_win = pd.unique(np.concatenate([df_slice['from_account'].to_numpy(), df_slice['to_account'].to_numpy()]))
            pos_e = int((df_slice['is_laundering'] == 1).sum())
            pos_nodes_win = len(pd.unique(np.concatenate([
                df_slice.loc[df_slice['is_laundering']==1, 'from_account'].to_numpy(),
                df_slice.loc[df_slice['is_laundering']==1, 'to_account'].to_numpy()
            ])))
            print(f"[{i:03d}] {ws:%Y-%m-%d} → {we:%Y-%m-%d}: nodes={len(nodes_win):,}, edges={len(df_slice):,}, pos_edges={pos_e:,}, pos_nodes={pos_nodes_win:,}")
            if MAX_WINDOWS_PER_SETTING is not None and i + 1 >= MAX_WINDOWS_PER_SETTING:
                break

    H_full, _ = aggregate_graphs(df, acct, mode='undirected')
    print(f"\nCommunity baselines on full period:")
    print(f"Aggregated graph: {H_full.num_vertices():,} nodes, {H_full.num_edges():,} edges")

    comm_full = compute_communities_fast(H_full, resolution=LEIDEN_RESOLUTION, seed=LOUVAIN_SEED)

    print("Louvain communities (top 5 by heuristic score):")
    scores_louv = comm_full['louvain']['scores']
    comms_louv = comm_full['louvain']['comms']
    for cid, score in sorted(scores_louv.items(), key=lambda x: x[1], reverse=True)[:5]:
        size = len(comms_louv[cid]) if cid < len(comms_louv) else 0
        print(f"  LVN cid={cid:>4}  score={score:.3f}  size={size:>6}")

    print("Leiden communities (top 5 by heuristic score):")
    scores_leid = comm_full['leiden']['scores']
    comms_leid = comm_full['leiden']['comms']
    for cid, score in sorted(scores_leid.items(), key=lambda x: x[1], reverse=True)[:5]:
        size = len(comms_leid[cid]) if cid < len(comms_leid) else 0
        print(f"  LDN cid={cid:>4}  score={score:.3f}  size={size:>6}")

    if tmin is None or tmax is None:
        raise RuntimeError("Time range unavailable; cannot build seeds.")
    T = tmin + (tmax - tmin) * SEED_CUTOFF_FRAC
    df_seed = df[(df['timestamp'] >= tmin) & (df['timestamp'] < T)]
    seed_nodes_global = set(pd.unique(np.concatenate([
        df_seed.loc[df_seed['is_laundering']==1, 'from_account'].astype(str).to_numpy(),
        df_seed.loc[df_seed['is_laundering']==1, 'to_account'].astype(str).to_numpy()
    ])))
    print(f"Global seeds cutoff T={T} | seed_nodes={len(seed_nodes_global)}")
    print(f"PPR config: alpha={PPR_ALPHA}, hops={PPR_HOPS}, bidir={PPR_BIDIR}")

    rows = []
    print("\nProcessing windows (metrics + preview):")
    for window_days in WINDOW_DAYS_LIST:
        print(f"\n-- {window_days}-day windows --")
        skipped = 0
        count = 0

        for i0, i1, ws, we in iter_window_indices(ts, tmin, tmax, window_days, WINDOW_STRIDE_DAYS):
            df_slice = df.iloc[i0:i1]
            if df_slice.empty:
                continue

            eval_exclude = seed_nodes_global if (EVAL_EXCLUDE_SEEDS and ws >= T and seed_nodes_global) else set()

            total_nodes, total_pos_nodes, total_neg_nodes, eval_nodes_sp, eval_pos_sp, eval_neg_sp = window_stats(df_slice, exclude_nodes=eval_exclude)

            if eval_nodes_sp == 0 or eval_neg_sp <= 0:
                skipped += 1
                print(f"[{ws:%Y-%m-%d} → {we:%Y-%m-%d}] Skipping window (no negatives in eval set).")
                continue

            H_agg, H_agg_dir = aggregate_graphs(df_slice, acct, mode='both', include_reciprocated=False)

            nodes = get_node_names(H_agg_dir)
            y_true_dict = vprop_to_dict(H_agg_dir, 'is_laundering_involved')
            pos_nodes_graph = int(sum(int(y_true_dict.get(n, 0)) for n in nodes))
            att_nodes_map = get_attempt_nodes_map_df(df_slice)
            if eval_exclude:
                att_nodes_map_filtered = {
                    att_id: nodes_set
                    for att_id, nodes_set in att_nodes_map.items()
                    if nodes_set - eval_exclude
                }
            else:
                att_nodes_map_filtered = att_nodes_map

            score_dict = run_centrality_baselines(H_agg_dir)
            if RUN_KCORE:
                score_dict.update(run_kcore_baselines(H_agg, H_agg_dir))

            if RUN_PATTERN_FEATURES:
                try:
                    pattern_scores = compute_pattern_features(H_agg_dir, H_agg)
                    score_dict['pattern_features'] = pattern_scores
                except Exception as e:
                    print(f"  Warning: Pattern features failed: {e}")
                    score_dict['pattern_features'] = {}

            results = eval_scores(nodes, y_true_dict, score_dict, k_fracs=K_FRACS, exclude_nodes=eval_exclude)

            seeded_scores = None
            if ws >= T and seed_nodes_global and eval_nodes_sp > 0 and eval_neg_sp > 0:
                seeded_scores = get_seeded_pagerank_scores(H_agg_dir, seed_nodes_global, weight='w_amount_log', alpha=PPR_ALPHA)
                score_dict['seeded_pr'] = seeded_scores
                seeded_res = eval_scores(
                    nodes, y_true_dict, {'seeded_pr': seeded_scores},
                    k_fracs=K_FRACS, exclude_nodes=eval_exclude
                )
                results.update(seeded_res)

            if RUN_ENSEMBLES:
                try:
                    ensemble_methods = create_ensemble_methods(score_dict)
                    if ensemble_methods:
                        ensemble_res = eval_scores(
                            nodes, y_true_dict, ensemble_methods,
                            k_fracs=K_FRACS, exclude_nodes=eval_exclude
                        )
                        results.update(ensemble_res)
                except Exception as e:
                    print(f"  Warning: Ensemble creation failed: {e}")

            comm = compute_communities_fast(H_agg, resolution=LEIDEN_RESOLUTION, seed=LOUVAIN_SEED)
            comm_ranked_nodes_cache = comm['ranked_cache']
            avg_comm_score_louv = comm['louvain']['avg']
            avg_comm_score_leid = comm['leiden']['avg']

            print(f"[{ws:%Y-%m-%d} → {we:%Y-%m-%d}] nodes={H_agg.num_vertices():,}, edges={H_agg.num_edges():,}")

            top_methods = {}
            for method in ['in_deg', 'pagerank_wlog', 'pattern_features', 'ensemble_ultimate', 'ensemble_top3']:
                if method in results:
                    top_methods[method] = results[method]
            if top_methods:
                print("  Top methods:", pretty_metrics(top_methods))

            print(f"  Avg community score (Louvain): {avg_comm_score_louv:.4f} | (Leiden): {avg_comm_score_leid:.4f}")

            if seeded_scores is not None:
                eval_nodes_list = [n for n in nodes if n not in eval_exclude]
                y_true_eval = [int(y_true_dict.get(n, 0)) for n in eval_nodes_list]
                y_score_eval = [seeded_scores.get(n, 0.0) for n in eval_nodes_list]
                if SKLEARN_OK and len(set(y_true_eval)) > 1:
                    pr_auc_win = average_precision_score(y_true_eval, y_score_eval)
                    print(f"  PersonalizedPageRank PR-AUC: {pr_auc_win:.4f}")

            base = {
                'window_days': window_days, 'ws': ws, 'we': we,
                'nodes': int(H_agg_dir.num_vertices()), 'edges': int(H_agg_dir.num_edges()),
                'pos_nodes': pos_nodes_graph
            }
            for method, m in results.items():
                eval_nodes_count = int(m.get('_eval_nodes', len(nodes)))
                eval_pos_count = int(m.get('_eval_pos', sum(y_true_dict.get(n, 0) for n in nodes)))
                eval_neg_count = eval_nodes_count - eval_pos_count
                if eval_nodes_count <= 0 or eval_neg_count <= 0:
                    continue

                row = dict(base)
                row['method'] = method
                row['ap'] = m.get('ap', None)
                row['eval_nodes'] = eval_nodes_count
                row['eval_pos_nodes'] = eval_pos_count
                row['prevalence_eval'] = eval_pos_count / eval_nodes_count

                for kf in K_FRACS:
                    p_at_key = f"p_at_{pct_key(kf)}"
                    attcov_key = f"attcov_at_{pct_key(kf)}"
                    row[p_at_key] = m.get(p_at_key, None)
                    ranked_nodes = m.get('_ranked_nodes', [])
                    cov = attempt_coverage(ranked_nodes, att_nodes_map_filtered, k_frac=kf)
                    row[attcov_key] = cov
                rows.append(row)

            for tag in ['louvain', 'leiden']:
                if any((tag, kf) in comm_ranked_nodes_cache for kf in K_FRACS):
                    comm_scores = comm[tag]['scores']
                    comm_list = comm[tag]['comms']

                    node_scores = {}
                    comm_rank_order = sorted(comm_scores.items(), key=lambda x: x[1], reverse=True)

                    node_to_degree = {}
                    for v in H_agg_dir.vertices():
                        node_name = H_agg_dir.vp['name'][v]
                        node_to_degree[node_name] = float(H_agg_dir.vp['in_deg'][v])

                    max_degree = max(node_to_degree.values()) if node_to_degree else 1.0

                    for rank, (cid, comm_score) in enumerate(comm_rank_order):
                        base_score = comm_score
                        for node in (comm_list[cid] if cid < len(comm_list) else []):
                            degree_norm = node_to_degree.get(node, 0.0) / (max_degree + 1e-6)
                            node_scores[node] = base_score + 0.01 * degree_norm

                    if node_scores:
                        comm_eval = eval_scores(
                            nodes, y_true_dict,
                            {f'communities_unsup_{tag}': node_scores},
                            k_fracs=K_FRACS,
                            exclude_nodes=eval_exclude
                        )

                        metrics = comm_eval[f'communities_unsup_{tag}']

                        row = dict(base)
                        row['method'] = f'communities_unsup_{tag}'
                        row['ap'] = metrics.get('ap', None)
                        row['eval_nodes'] = metrics.get('_eval_nodes', len(nodes))
                        row['eval_pos_nodes'] = metrics.get('_eval_pos', 0)
                        row['prevalence_eval'] = row['eval_pos_nodes'] / row['eval_nodes'] if row['eval_nodes'] > 0 else np.nan

                        ranked_nodes = metrics.get('_ranked_nodes', [])

                        for kf in K_FRACS:
                            p_at_key = f"p_at_{pct_key(kf)}"
                            attcov_key = f"attcov_at_{pct_key(kf)}"

                            row[p_at_key] = metrics.get(p_at_key, None)
                            
                            cov = attempt_coverage(ranked_nodes, att_nodes_map_filtered, k_frac=kf)
                            row[attcov_key] = cov

                        rows.append(row)
            count += 1
            if MAX_WINDOWS_PER_SETTING is not None and count >= MAX_WINDOWS_PER_SETTING:
                break

            del H_agg, H_agg_dir, score_dict, results, comm, seeded_scores, att_nodes_map, nodes, y_true_dict, att_nodes_map_filtered, comm_ranked_nodes_cache
            if 'pattern_scores' in locals():
                del pattern_scores
            if 'ensemble_methods' in locals():
                del ensemble_methods
            if 'comm_eval' in locals():
                del comm_eval
            if 'node_scores' in locals(): 
                del node_scores
            gc.collect()

        if skipped:
            print(f"Skipped {skipped} degenerate windows for {window_days}-day setting.")

    df_metrics = pd.DataFrame(rows)

    df_metrics = add_random_baseline(df_metrics)

    df_metrics['prevalence'] = df_metrics['pos_nodes'] / df_metrics['nodes']
    for kf in K_FRACS:
        p_at_key = f"p_at_{pct_key(kf)}"
        if p_at_key in df_metrics.columns:
            df_metrics[f'lift_{p_at_key}'] = df_metrics[p_at_key] / df_metrics['prevalence']
            df_metrics[f'lift_eval_{p_at_key}'] = df_metrics[p_at_key] / df_metrics['prevalence_eval']

    chk = (df_metrics.groupby(['window_days','ws','we'])
           .agg(nodes_nunique=('nodes','nunique'),
                pos_nodes_nunique=('pos_nodes','nunique'))
           .reset_index())
    bad = chk[(chk.nodes_nunique != 1) | (chk.pos_nodes_nunique != 1)]
    if not bad.empty:
        print("WARNING: nodes/pos_nodes inconsistent across methods:")
        print(bad.to_string(index=False))

    random_rows = df_metrics[df_metrics.method == 'random']
    col = f'lift_eval_p_at_{pct_key(0.01)}'
    if not random_rows.empty and col in random_rows.columns:
        random_lift_median = random_rows[col].median()
        if pd.notna(random_lift_median) and abs(random_lift_median - 1.0) > 0.05:
            print(f"WARNING: Random baseline {col} median = {random_lift_median:.3f}, expected ≈ 1.0")

    seeded_rows = df_metrics[df_metrics.method == 'seeded_pr']
    if not seeded_rows.empty:
        high_prev = seeded_rows[seeded_rows.prevalence_eval > 0.5]
        if not high_prev.empty:
            print(f"WARNING: {len(high_prev)} seeded_pr rows have prevalence_eval > 0.5 (potentially degenerate)")

    df_metrics.to_csv(RESULTS_CSV, index=False)
    print(f"\nSaved per-window metrics to {RESULTS_CSV}")

    if not df_metrics.empty:
        summary = (df_metrics
                   .groupby(['window_days', 'method'])
                   .agg(ap_median=('ap','median'),
                        p01_median=(f'p_at_{pct_key(0.01)}','median'),
                        lift_p01_median=(f'lift_p_at_{pct_key(0.01)}','median'),
                        lift_eval_p01_median=(f'lift_eval_p_at_{pct_key(0.01)}','median'),
                        attcov01_median=(f'attcov_at_{pct_key(0.01)}','median'),
                        prevalence_median=('prevalence','median'),
                        windows=('ws','count'))
                   .reset_index()
                   .sort_values(['window_days', 'ap_median'], ascending=[True, False]))

        print("\n" + "="*80)
        print("FINAL RESULTS SUMMARY (median across windows)")
        print("="*80)
        print(summary.to_string(index=False))

        print("\n" + "="*80)
        print("TOP PERFORMING METHODS")
        print("="*80)
        for wd in WINDOW_DAYS_LIST:
            wd_summary = summary[summary['window_days'] == wd].head(10)
            print(f"\n{wd}-day windows (Top 10):")
            print(wd_summary[['method', 'ap_median', 'p01_median', 'lift_eval_p01_median', 'attcov01_median']].to_string(index=False))

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    print("\n" + "="*80)
    print("FEATURE ABLATION ANALYSIS")
    print("="*80)

    methods_without_pattern = ['in_deg', 'pagerank_wlog', 'in_tx', 'ensemble_diverse']
    methods_with_pattern = ['pattern_features', 'ensemble_pattern', 'ensemble_ultimate']

    ablation_summary = summary[summary['window_days'] == 7]

    print("\nBaseline methods (no pattern features):")
    baseline_perf = ablation_summary[ablation_summary['method'].isin(methods_without_pattern)]
    print(baseline_perf[['method', 'ap_median', 'p01_median']].to_string(index=False))

    print("\nPattern-enhanced methods:")
    pattern_perf = ablation_summary[ablation_summary['method'].isin(methods_with_pattern)]
    print(pattern_perf[['method', 'ap_median', 'p01_median']].to_string(index=False))

    best_baseline = baseline_perf['ap_median'].max()
    best_pattern = pattern_perf['ap_median'].max()
    improvement = (best_pattern / best_baseline - 1) * 100

    print(f"\nPattern features improvement: {improvement:+.1f}%")
    if improvement < 5:
        print("⚠️  WARNING: Pattern features provide <5% improvement. May not be worth complexity.")

    best_methods = summary[summary['method'].isin([
        'ensemble_ultimate', 'ensemble_pattern', 'in_deg', 'pagerank_wlog', 'random'
    ])]

    print("\nBest Methods Comparison (7-day windows):")
    best_7day = best_methods[best_methods['window_days'] == 7][
        ['method', 'ap_median', 'p01_median', 'attcov01_median']
    ].sort_values('ap_median', ascending=False)

    best_7day_renamed = best_7day.rename(columns={
        'ap_median': 'AP',
        'p01_median': 'P@1%',
        'attcov01_median': 'Coverage@1%'
    })

    print(best_7day_renamed.to_string(index=False))

    print("\nKey Findings:")
    best_row = best_7day.iloc[0]
    baseline_row = best_7day[best_7day['method'] == 'random'].iloc[0]

    ap_improvement = (best_row['ap_median'] / baseline_row['ap_median'] - 1) * 100
    p_improvement = (best_row['p01_median'] / baseline_row['p01_median'] - 1) * 100

    print(f"1. Best method: {best_row['method']}")
    print(f"2. AP improvement over random: {ap_improvement:.1f}%")
    print(f"3. Precision@1% improvement: {p_improvement:.1f}%")
    print(f"4. Laundering schemes detected: {best_row['attcov01_median']*100:.1f}%")
    print(f"5. Investigation efficiency: {best_row['p01_median']/baseline_row['p01_median']:.1f}x lift")

    sys.stdout = original_stdout

    console_output = output_buffer.getvalue()
    output_buffer.close()

    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    LOG_FILE = METRICS_DIR / f"metrics_log_{timestamp}.txt"

    with open(LOG_FILE, 'w', encoding='utf-8') as f:
        f.write(console_output)

    print(f"\n{'='*80}")
    print(f"Console output saved to: {LOG_FILE}")
    print(f"Metrics CSV saved to: {RESULTS_CSV}")
    print(f"{'='*80}")

if __name__ == "__main__":
    drive.mount('/content/drive', force_remount=False)

    # HI Small
    print("Running HI Small")
    run_analysis(
        drive_base=Path('/content/drive/MyDrive/AML/processed/HI_Small/US_Dollar'),
        window_days_list=[3, 7],
        stride_days=1,
        max_windows=None,
        leiden_resolution=0.2,
        louvain_seed=42,
        community_weighted=True,
        run_hits=True,
        run_kcore=True,
        run_pattern_features=True,
        run_ensembles=True,
        metrics_dir=Path('/content/drive/MyDrive/AML/processed/HI_Small/US_Dollar/metrics'),
        results_csv=Path('/content/drive/MyDrive/AML/processed/HI_Small/US_Dollar/metrics/window_metrics.csv'),
        eval_exclude_seeds=True,
        k_fracs=(0.005, 0.01, 0.02),
        seed_cutoff_frac=0.2,
        ppr_alpha=0.85,
        ppr_hops=3,
        ppr_bidir=False,
        ppr_max_nodes=50000,
        ppr_max_seeds=1000,
        sklearn_ok=True
    )

    # LI Small
    print("Running LI Small")
    run_analysis(
        drive_base=Path('/content/drive/MyDrive/AML/processed/LI_Small/US_Dollar'),
        window_days_list=[3, 7],
        stride_days=1,
        max_windows=None,
        leiden_resolution=0.2,
        louvain_seed=42,
        community_weighted=True,
        run_hits=True,
        run_kcore=True,
        run_pattern_features=True,
        run_ensembles=True,
        metrics_dir=Path('/content/drive/MyDrive/AML/processed/LI_Small/US_Dollar/metrics'),
        results_csv=Path('/content/drive/MyDrive/AML/processed/LI_Small/US_Dollar/metrics/window_metrics.csv'),
        eval_exclude_seeds=True,
        k_fracs=(0.005, 0.01, 0.02),
        seed_cutoff_frac=0.2,
        ppr_alpha=0.85,
        ppr_hops=3,
        ppr_bidir=False,
        ppr_max_nodes=50000,
        ppr_max_seeds=1000,
        sklearn_ok=True
    )

    # HI Large
    print("Running HI Large")
    run_analysis(
        drive_base=Path('/content/drive/MyDrive/AML/processed/HI_Large/US_Dollar'),
        window_days_list=[3, 7],
        stride_days=1,
        max_windows=None,
        leiden_resolution=0.2,
        louvain_seed=42,
        community_weighted=True,
        run_hits=True,
        run_kcore=True,
        run_pattern_features=True,
        run_ensembles=True,
        metrics_dir=Path('/content/drive/MyDrive/AML/processed/HI_Large/US_Dollar/metrics'),
        results_csv=Path('/content/drive/MyDrive/AML/processed/HI_Large/US_Dollar/metrics/window_metrics.csv'),
        eval_exclude_seeds=True,
        k_fracs=(0.005, 0.01, 0.02),
        seed_cutoff_frac=0.2,
        ppr_alpha=0.85,
        ppr_hops=3,
        ppr_bidir=False,
        ppr_max_nodes=50000,
        ppr_max_seeds=1000,
        sklearn_ok=True
    )
