from sklearn.metrics import average_precision_score
SKLEARN_OK = True

"""
# Import graph utilities and config
from graph import (
    proc, p_norm, p_pos, p_acct,
    WINDOW_DAYS_LIST, WINDOW_STRIDE_DAYS,
    LOUVAIN_RESOLUTION, LOUVAIN_SEED,
    build_all, iter_windows, get_windowed_graph_fast,
    derive_node_labels, aggregate_graph, run_louvain, score_communities_unsupervised
)
"""

# -----------------------
# Metrics configs
# -----------------------
METRICS_DIR = proc / "metrics"
METRICS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_CSV = METRICS_DIR / "window_metrics.csv"

K_FRACS = (0.005, 0.01, 0.02)  # 0.5%, 1%, 2% for precision@k and attempt coverage
SEED_CUTOFF_FRAC = 0.2         # first 20% of timeline for global seeds

MAX_WINDOWS_PER_SETTING = None

# -----------------------
# Metrics helpers
# -----------------------
def precision_at_k(y_true, y_score, k_frac=0.01):
    y_true = np.asarray(y_true); y_score = np.asarray(y_score)
    n = max(1, int(len(y_true) * k_frac))
    idx = np.argsort(-y_score)[:n]
    return float(y_true[idx].mean())

def eval_scores(nodes, y_true_dict, score_dict, k_fracs=(0.005, 0.01, 0.02), exclude_nodes=None):
    if exclude_nodes is None: exclude_nodes = set()
    eval_nodes = [n for n in nodes if n not in exclude_nodes]
    y_true = np.array([y_true_dict.get(n, 0) for n in eval_nodes], dtype=int)
    res = {}
    for name, s in score_dict.items():
        scores = np.array([s.get(n, 0.0) for n in eval_nodes], dtype=float)
        ap = average_precision_score(y_true, scores) if SKLEARN_OK and len(set(y_true)) > 1 else None
        metrics = {'ap': ap}
        metrics['_eval_nodes'] = len(eval_nodes)
        metrics['_eval_pos'] = int(y_true.sum())
        for k in k_fracs:
            metrics[f"p_at_{int(k*1000)/10:.1f}pct"] = precision_at_k(y_true, scores, k)
        order = np.argsort(-scores)
        metrics['_ranked_nodes'] = [eval_nodes[i] for i in order]
        res[name] = metrics
    return res

def run_centrality_baselines(H_dir: nx.DiGraph):
    scores = {}
    try:
        scores['pagerank_wlog'] = nx.pagerank(H_dir, weight='w_amount_log', alpha=0.9, max_iter=100, tol=1e-6)
    except Exception:
        scores['pagerank_wlog'] = {}
    try:
        hubs, auth = nx.hits(H_dir, max_iter=500, tol=1e-8, normalized=True)
        scores['hits_hub'] = hubs; scores['hits_auth'] = auth
    except Exception:
        scores['hits_hub'] = {}; scores['hits_auth'] = {}
    scores['in_deg'] = {n: H_dir.nodes[n].get('in_deg', 0) for n in H_dir}
    scores['out_deg'] = {n: H_dir.nodes[n].get('out_deg', 0) for n in H_dir}
    scores['in_tx'] = {n: H_dir.nodes[n].get('in_tx_count', 0) for n in H_dir}
    scores['out_tx'] = {n: H_dir.nodes[n].get('out_tx_count', 0) for n in H_dir}
    scores['in_amt'] = {n: H_dir.nodes[n].get('in_amount_sum', 0) for n in H_dir}
    scores['out_amt'] = {n: H_dir.nodes[n].get('out_amount_sum', 0) for n in H_dir}
    scores['collector'] = {n: (H_dir.nodes[n].get('in_amount_sum',0)) / (H_dir.nodes[n].get('out_amount_sum',0)+1) for n in H_dir}
    scores['distributor'] = {n: (H_dir.nodes[n].get('out_amount_sum',0)) / (H_dir.nodes[n].get('in_amount_sum',0)+1) for n in H_dir}
    return scores

def get_attempt_nodes_map(H_win: nx.MultiDiGraph):
    att_nodes = {}
    for u, v, k, d in H_win.edges(keys=True, data=True):
        if int(d.get('is_laundering', 0)) != 1:
            continue
        att_id = d.get('attempt_id', None)
        if att_id is None or (isinstance(att_id, float) and np.isnan(att_id)):
            continue
        att_nodes.setdefault(att_id, set()).update([u, v])
    return att_nodes

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
    
def get_seeded_pagerank_scores(H_agg_dir: nx.DiGraph, seed_nodes: set, weight='w_amount_log', alpha=0.9):
    if not seed_nodes:
        return {}
    personalization = {n: (1.0/len(seed_nodes) if n in seed_nodes else 0.0) for n in H_agg_dir}
    try:
        pr_fwd = nx.pagerank(H_agg_dir, personalization=personalization, weight=weight, alpha=alpha, max_iter=100, tol=1e-6)
        pr_rev = nx.pagerank(H_agg_dir.reverse(copy=False), personalization=personalization, weight=weight, alpha=alpha, max_iter=100, tol=1e-6)
        return {n: 0.5*(pr_fwd.get(n,0.0) + pr_rev.get(n,0.0)) for n in H_agg_dir}
    except Exception:
        return {}

# -----------------------
# Build graph and data
# -----------------------
df, G, tmin, tmax = build_all(save_gpickle=True)

# -----------------------
# Temporal windows quick summary (kept for parity)
# -----------------------
for window_days in WINDOW_DAYS_LIST:
    print(f"\n-- {window_days}-day windows, stride={WINDOW_STRIDE_DAYS}d --")
    for i, (ws, we) in enumerate(iter_windows(tmin, tmax, window_days=window_days, stride_days=WINDOW_STRIDE_DAYS)):
        df_slice = df[(df['timestamp'] >= ws) & (df['timestamp'] < we)]
        H_win = get_windowed_graph_fast(df_slice, G)
        if H_win.number_of_edges() == 0:
            continue
        pos_nodes_win = derive_node_labels(H_win)
        pos_e = sum(1 for *_ , d in H_win.edges(keys=True, data=True) if int(d.get('is_laundering',0))==1)
        print(f"[{i:03d}] {ws:%Y-%m-%d} → {we:%Y-%m-%d}: nodes={H_win.number_of_nodes():,}, edges={H_win.number_of_edges():,}, pos_edges={pos_e:,}, pos_nodes={pos_nodes_win:,}")
        if MAX_WINDOWS_PER_SETTING is not None and i + 1 >= MAX_WINDOWS_PER_SETTING:
            break

# -----------------------
# Full-period community baseline and seeded PR (parity with earlier prints)
# -----------------------
H_full = aggregate_graph(G, directed=False)
print(f"\nEnhanced community baseline on full period:")
print(f"Aggregated graph: {H_full.number_of_nodes():,} nodes, {H_full.number_of_edges():,} edges")
partition_full, comms_full = run_louvain(H_full, resolution=LOUVAIN_RESOLUTION, seed=LOUVAIN_SEED, weight='w_amount_log')
comm_scores = score_communities_unsupervised(H_full, comms_full)
print("Unsupervised community scoring (top 5 by heuristic score):")
sorted_comms = sorted(comm_scores.items(), key=lambda x: x[1], reverse=True)
for cid, score in sorted_comms[:5]:
    size = len(comms_full[cid]) if cid < len(comms_full) else 0
    print(f"  cid={cid:>4}  score={score:.3f}  size={size:>6}")

H_full_directed = aggregate_graph(G, directed=True)
pos_nodes_set = {n for n, d in H_full_directed.nodes(data=True) if int(d.get('is_laundering_involved', 0)) == 1}
if len(pos_nodes_set) > 0:
    seed_size = max(1, len(pos_nodes_set) // 5)
    seed_nodes = set(sorted(pos_nodes_set)[:seed_size])
    # Compute seeded PR-AUC on full period (non-leakage variant is below for windows)
    pr_scores_full = get_seeded_pagerank_scores(H_full_directed, seed_nodes, weight='w_amount_log', alpha=0.9)
    nodes_full = list(H_full_directed.nodes())
    y_true_full = [int(H_full_directed.nodes[n].get('is_laundering_involved', 0)) for n in nodes_full if n not in seed_nodes]
    y_score_full = [pr_scores_full.get(n, 0.0) for n in nodes_full if n not in seed_nodes]
    if SKLEARN_OK and len(set(y_true_full)) > 1:
        pr_auc_full = average_precision_score(y_true_full, y_score_full)
        print(f"PersonalizedPageRank baseline PR-AUC: {pr_auc_full:.4f}")
        print(f"(Used {len(seed_nodes)} seed nodes out of {len(pos_nodes_set)} total positive nodes)")

# -----------------------
# Build fixed time-based seeds (no same-window leakage)
# -----------------------
if tmin is None or tmax is None:
    raise RuntimeError("Time range unavailable; cannot build seeds.")

T = tmin + (tmax - tmin) * SEED_CUTOFF_FRAC
df_seed = df[(df['timestamp'] >= tmin) & (df['timestamp'] < T)]
H_seed = get_windowed_graph_fast(df_seed, G)
derive_node_labels(H_seed)
H_seed_dir = aggregate_graph(H_seed, directed=True)
seed_nodes_global = {n for n, d in H_seed_dir.nodes(data=True) if int(d.get('is_laundering_involved', 0)) == 1}
print(f"Global seeds cutoff T={T} | seed_nodes={len(seed_nodes_global)}")

# -----------------------
# Per-window enhanced analysis (prints)
# -----------------------
print("\nPer-window enhanced analysis (first few windows per setting):")
for window_days in WINDOW_DAYS_LIST:
    print(f"\n-- {window_days}-day windows --")
    count = 0
    for ws, we in iter_windows(tmin, tmax, window_days=window_days, stride_days=WINDOW_STRIDE_DAYS):
        df_slice = df[(df['timestamp'] >= ws) & (df['timestamp'] < we)]
        H_win = get_windowed_graph_fast(df_slice, G)
        if H_win.number_of_edges() == 0:
            continue
        derive_node_labels(H_win)
        H_agg = aggregate_graph(H_win, directed=False)
        H_agg_dir = aggregate_graph(H_win, directed=True)

        # Centralities
        nodes = list(H_agg_dir.nodes())
        y_true_dict = {n: int(H_agg_dir.nodes[n].get('is_laundering_involved', 0)) for n in nodes}
        score_dict = run_centrality_baselines(H_agg_dir)
        results = eval_scores(nodes, y_true_dict, score_dict, k_fracs=(0.005, 0.01, 0.02))
        print("  Centrality baselines:", pretty_metrics(results))

        # Communities
        _, comms_win = run_louvain(H_agg, resolution=LOUVAIN_RESOLUTION, seed=LOUVAIN_SEED, weight='w_amount_log')
        comm_scores_win = score_communities_unsupervised(H_agg, comms_win)
        avg_comm_score = np.mean(list(comm_scores_win.values())) if comm_scores_win else 0

        # Seeded PR (time-based seeds)
        pr_auc_win = None
        if ws >= T and seed_nodes_global:
            pr_scores = get_seeded_pagerank_scores(H_agg_dir, seed_nodes_global, weight='w_amount_log', alpha=0.9)
            eval_nodes = [n for n in nodes if n not in seed_nodes_global]
            y_true = [y_true_dict[n] for n in eval_nodes]
            y_score = [pr_scores.get(n, 0.0) for n in eval_nodes]
            if SKLEARN_OK and len(set(y_true)) > 1:
                pr_auc_win = average_precision_score(y_true, y_score)

        print(f"[{ws:%Y-%m-%d} → {we:%Y-%m-%d}] nodes={H_agg.number_of_nodes():,}, edges={H_agg.number_of_edges():,}")
        print(f"  Avg community score: {avg_comm_score:.4f}")
        if pr_auc_win is not None:
            print(f"  PersonalizedPageRank PR-AUC: {pr_auc_win:.4f}")

        count += 1
        if MAX_WINDOWS_PER_SETTING is not None and count >= MAX_WINDOWS_PER_SETTING:
            break

# -----------------------
# Full per-window metrics -> CSV
# -----------------------
rows = []
for window_days in WINDOW_DAYS_LIST:
    count = 0
    for ws, we in iter_windows(tmin, tmax, window_days=window_days, stride_days=WINDOW_STRIDE_DAYS):
        df_slice = df[(df['timestamp'] >= ws) & (df['timestamp'] < we)]
        H_win = get_windowed_graph_fast(df_slice, G)
        if H_win.number_of_edges() == 0:
            continue
        derive_node_labels(H_win)
        H_agg = aggregate_graph(H_win, directed=False)
        H_agg_dir = aggregate_graph(H_win, directed=True)
        nodes = list(H_agg_dir.nodes())
        y_true_dict = {n: int(H_agg_dir.nodes[n].get('is_laundering_involved', 0)) for n in nodes}
        att_nodes_map = get_attempt_nodes_map(H_win)

        score_dict = run_centrality_baselines(H_agg_dir)
        results = eval_scores(nodes, y_true_dict, score_dict, k_fracs=K_FRACS)

        if ws >= T and seed_nodes_global:
            seeded_scores = get_seeded_pagerank_scores(H_agg_dir, seed_nodes_global, weight='w_amount_log', alpha=0.9)
            seeded_res = eval_scores(nodes, y_true_dict, {'seeded_pr': seeded_scores}, k_fracs=K_FRACS, exclude_nodes=seed_nodes_global)
            results.update(seeded_res)

        _, comms_win = run_louvain(H_agg, resolution=LOUVAIN_RESOLUTION, seed=LOUVAIN_SEED, weight='w_amount_log')
        comm_scores_win = score_communities_unsupervised(H_agg, comms_win)
        comm_ranked_nodes_cache = {}
        if comm_scores_win:
            comm_order = sorted(comm_scores_win.items(), key=lambda x: x[1], reverse=True)
            total_nodes = len(H_agg)
            acc = set()
            for kf in K_FRACS:
                target = max(1, int(total_nodes * kf))
                acc.clear()
                for cid, _score in comm_order:
                    acc |= set(comms_win[cid])
                    if len(acc) >= target:
                        break
                comm_ranked_nodes_cache[kf] = list(acc)

        base = {
            'window_days': window_days, 'ws': ws, 'we': we,
            'nodes': H_agg_dir.number_of_nodes(), 'edges': H_agg_dir.number_of_edges(),
            'pos_nodes': int(sum(y_true_dict.values()))
        }
        for method, m in results.items():
            row = dict(base); row['method'] = method; row['ap'] = m.get('ap', None)
            
            # Track evaluation population (different from full population for seeded methods)
            eval_nodes_count = m.get('_eval_nodes', len(nodes))
            eval_pos_count = m.get('_eval_pos', int(sum(y_true_dict.values())))
            row['eval_nodes'] = eval_nodes_count
            row['eval_pos_nodes'] = eval_pos_count
            row['prevalence_eval'] = (eval_pos_count / eval_nodes_count) if eval_nodes_count > 0 else np.nan
            
            for kf in K_FRACS:
                key = f"p_at_{int(kf*1000)/10:.1f}pct"
                row[key] = m.get(key, None)
                ranked_nodes = m.get('_ranked_nodes', [])
                cov = attempt_coverage(ranked_nodes, att_nodes_map, k_frac=kf)
                row[f"attcov_at_{int(kf*100)}pct"] = cov
            rows.append(row)

        if comm_ranked_nodes_cache:
            row = dict(base); row['method'] = 'communities_unsup'; row['ap'] = None
            row['eval_nodes'] = base['nodes']  # Communities use full population
            row['eval_pos_nodes'] = base['pos_nodes']
            row['prevalence_eval'] = base['pos_nodes'] / base['nodes'] if base['nodes'] > 0 else np.nan
            for kf in K_FRACS:
                row[f"p_at_{int(kf*1000)/10:.1f}pct"] = None
                cov = attempt_coverage(comm_ranked_nodes_cache[kf], att_nodes_map, k_frac=1.0)
                name = f"{int(kf*1000)/10:.1f}pct" 
                row[f"attcov_at_{name}"] = cov
            rows.append(row)

        count += 1
        if MAX_WINDOWS_PER_SETTING is not None and count >= MAX_WINDOWS_PER_SETTING:
            break

df_metrics = pd.DataFrame(rows)

def add_random_baseline(dfm: pd.DataFrame) -> pd.DataFrame:
    cols = list(dfm.columns)
    rows = []
    for _, r in dfm.groupby(['window_days', 'ws', 'we']).head(1).iterrows():
        base = {c: r.get(c, None) for c in cols}
        base['method'] = 'random'
        # prevalence_eval is already present for this row (full-pop for centralities)
        prev_eval = base.get('prevalence_eval')
        if prev_eval is None or pd.isna(prev_eval):
            prev_eval = (base.get('pos_nodes', 0) / base.get('nodes', 1)) if base.get('nodes', 0) else np.nan
            base['prevalence_eval'] = prev_eval
        base['ap'] = prev_eval  # AP of random ≈ prevalence
        
        for pcol in ['p_at_0.5pct', 'p_at_1.0pct', 'p_at_2.0pct']:
            base[pcol] = prev_eval
            pct = pcol.split('_at_')[1].replace('.0','')
            base[f'attcov_at_{pct}'] = None
        rows.append(base)
    rand_df = pd.DataFrame(rows, columns=cols)
    return pd.concat([dfm, rand_df], ignore_index=True)

df_metrics = add_random_baseline(df_metrics)

# Compute prevalence and lift metrics
df_metrics['prevalence'] = df_metrics['pos_nodes'] / df_metrics['nodes']
for col in ['p_at_0.5pct', 'p_at_1.0pct', 'p_at_2.0pct']:
    if col in df_metrics.columns:
        df_metrics[f'lift_{col}'] = df_metrics[col] / df_metrics['prevalence']
        df_metrics[f'lift_eval_{col}'] = df_metrics[col] / df_metrics['prevalence_eval']
# -----------------------
# Validation checks
# -----------------------
# Check that within each window, nodes and pos_nodes are identical across methods
chk = (df_metrics.groupby(['window_days','ws','we'])
       .agg(nodes_nunique=('nodes','nunique'),
            pos_nodes_nunique=('pos_nodes','nunique'))
       .reset_index())
bad = chk[(chk.nodes_nunique != 1) | (chk.pos_nodes_nunique != 1)]
if not bad.empty:
    print("WARNING: nodes/pos_nodes inconsistent across methods:")
    print(bad.to_string(index=False))

# Sanity check for random baseline: lift should be ≈ 1
random_rows = df_metrics[df_metrics.method == 'random']
if not random_rows.empty:
    random_lift_median = random_rows['lift_p_at_1.0pct'].median()
    if abs(random_lift_median - 1.0) > 0.05:
        print(f"WARNING: Random baseline lift_p_at_1.0pct median = {random_lift_median:.3f}, expected ≈ 1.0")

# Seeded PR sanity: prevalence_eval should be reasonable
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
                        p01_median=('p_at_1.0pct','median'),
                        lift_p01_median=('lift_p_at_1.0pct','median'),
                        lift_eval_p01_median=('lift_eval_p_at_1.0pct','median'),
                        attcov01_median=('attcov_at_1.0pct','median'),
                        prevalence_median=('prevalence','median'),
                        windows=('ws','count'))
               .reset_index()
               .sort_values(['window_days', 'ap_median'], ascending=[True, False]))
    print("\nSummary (median across windows):")
    print(summary.to_string(index=False))
