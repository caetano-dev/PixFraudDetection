import pandas as pd
import networkx as nx
from pathlib import Path

def analyze_combined_graph(data_dir: str = "data/HI_Small"):
    data_path = Path(data_dir)
    normal_file = data_path / "1_filtered_normal_transactions.parquet"
    laundering_file = data_path / "2_filtered_laundering_transactions.parquet"
    
    print(f"Loading files from {data_dir}...")
    
    # 1. Load BOTH files
    try:
        df_normal = pd.read_parquet(normal_file)
        df_laundering = pd.read_parquet(laundering_file)
    except FileNotFoundError:
        print("Error: Could not find the parquet files. Make sure you ran process_aml_data.py first.")
        return

    print(f"  - Normal Transactions: {len(df_normal):,}")
    print(f"  - Laundering Transactions: {len(df_laundering):,}")
    
    # 2. Merge into ONE DataFrame
    print("Merging datasets...")
    df_combined = pd.concat([df_normal, df_laundering], ignore_index=True)
    total_tx = len(df_combined)
    print(f"  - Combined Total: {total_tx:,}")

    # 3. Build the Unified Directed Graph
    print("\nBuilding Unified Graph (this might take a moment)...")
    # We use the combined DF so edges can connect Normal <-> Laundering
    G = nx.from_pandas_edgelist(
        df_combined, 
        source='from_account', 
        target='to_account', 
        edge_attr='is_laundering',
        create_using=nx.DiGraph
    )
    
    # 4. Component Analysis (Global Topology)
    wcc = list(nx.weakly_connected_components(G))
    largest_wcc = max(wcc, key=len)
    percent_in_lcc = (len(largest_wcc) / len(G)) * 100
    
    print("\n--- UNIFIED GRAPH HEALTH ---")
    print(f"Total Nodes: {len(G):,}")
    print(f"Isolated Components: {len(wcc):,}")
    print(f"Largest Component Size: {len(largest_wcc):,} ({percent_in_lcc:.2f}%)")
    
    if percent_in_lcc > 80:
        print("SUCCESS: The graph is well-connected! Algorithms will work.")
    else:
        print("WARNING: Graph is still fragmented. Check your raw data.")

    # 5. Laundering Chain Health (Re-evaluated on the Unified Graph)
    print("\n--- LAUNDERING CHAIN HEALTH (Unified) ---")
    
    # Identify all nodes involved in ANY laundering transaction
    laundering_nodes = set(df_laundering['from_account']).union(set(df_laundering['to_account']))
    
    dead_ends = 0
    sources = 0
    pass_through = 0
    total_laundering_nodes = 0
    
    for node in laundering_nodes:
        if node not in G: continue
        total_laundering_nodes += 1
        
        # In the unified graph, does this node have ANY incoming or outgoing edges?
        # (Even if they are 'normal' transactions, they provide flow for PageRank)
        in_degree = G.in_degree(node)
        out_degree = G.out_degree(node)
        
        if in_degree > 0 and out_degree > 0:
            pass_through += 1
        elif in_degree > 0 and out_degree == 0:
            dead_ends += 1
        elif in_degree == 0 and out_degree > 0:
            sources += 1
            
    print(f"Nodes involved in Laundering: {total_laundering_nodes:,}")
    print(f"Pass-through (Healthy Flow): {pass_through:,}")
    print(f"Dead Ends (Stuck Money): {dead_ends:,}")
    print(f"Sources (Origin Points): {sources:,}")

    # Interpretation
    if pass_through > dead_ends:
        print("\nCONCLUSION: Connectivity is GOOD. Money is flowing through these accounts.")
    else:
        print("\nCONCLUSION: Connectivity is BAD. Most laundering nodes are still isolated.")

if __name__ == "__main__":
    analyze_combined_graph()
